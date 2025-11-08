from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import re
from email.utils import parseaddr
from datetime import datetime, timedelta, timezone

from .store import _run, upsert_prospect_by_email  # reuse Neo4j runner
from .gmail_client import fetch_unseen_since
from .calendar_client import find_free_slots, suggest_windows
from .config import settings  # ← use settings, not bare names

INBOX_ADDR = os.getenv("ECO_LOCAL_INBOX_ADDRESS", "ECOLocal@ecodia.au")
AUTO_UPSERT_INBOUND = os.getenv("ECO_LOCAL_AUTO_UPSERT_INBOUND", "1").strip().lower() not in {"0", "false", "no"}

# ─────────────────────────────────────────────────────────
# Minimal, self-contained Gemini embeddings (no EOS imports)
# ─────────────────────────────────────────────────────────

_GEMINI_MODEL = (os.getenv("GEMINI_EMBEDDING_MODEL") or "gemini-embedding-001").strip()
_GEMINI_DIMS = 3072

# Prefer env override → GOOGLE_API_KEY → GEMINI_API_KEY
_GEMINI_KEY = (
    os.getenv("GOOGLE_API_KEY", "").strip()
    or settings.GOOGLE_API_KEY.strip()
    or settings.GEMINI_API_KEY.strip()
)

if os.getenv("ECO_LOCAL_EMBED_DEBUG", "1") == "1":
    # Do NOT print the key, just whether it exists.
    print(f"[embeddings] model={_GEMINI_MODEL} dims={_GEMINI_DIMS} key_present={bool(_GEMINI_KEY)}")

def _embed_vec(text: str) -> Optional[List[float]]:
    if not _GEMINI_KEY or not text or not text.strip():
        return None
    try:
        import httpx  # preferred if available
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:embedContent"
        body = {
            "model": f"models/{_GEMINI_MODEL}",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "outputDimensionality": _GEMINI_DIMS,
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": _GEMINI_KEY}
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=headers, json=body)
            if r.status_code != 200:
                # Optional: quick reason dump (403 often = wrong key/project/billing)
                try:
                    print("[embeddings] HTTP", r.status_code, r.text[:200])
                except Exception:
                    pass
                return None
            data = r.json() or {}
            emb = (data.get("embedding") or {}).get("values")
            if isinstance(emb, list) and len(emb) == _GEMINI_DIMS:
                return [float(x) for x in emb]
            embs = data.get("embeddings")
            if isinstance(embs, list) and embs and "values" in embs[0]:
                vec = embs[0]["values"]
                if isinstance(vec, list) and len(vec) == _GEMINI_DIMS:
                    return [float(x) for x in vec]
            return None
    except Exception:
        # Fallback to stdlib if httpx isn't present
        try:
            import json
            from urllib.request import Request, urlopen
            from urllib.error import URLError, HTTPError
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{_GEMINI_MODEL}:embedContent"
            body = json.dumps({
                "model": f"models/{_GEMINI_MODEL}",
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": _GEMINI_DIMS,
            }).encode("utf-8")
            req = Request(url, data=body, headers={
                "Content-Type": "application/json",
                "x-goog-api-key": _GEMINI_KEY,
            }, method="POST")
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                emb = (data.get("embedding") or {}).get("values")
                if isinstance(emb, list) and len(emb) == _GEMINI_DIMS:
                    return [float(x) for x in emb]
                embs = data.get("embeddings")
                if isinstance(embs, list) and embs and "values" in embs[0]:
                    vec = embs[0]["values"]
                    if isinstance(vec, list) and len(vec) == _GEMINI_DIMS:
                        return [float(x) for x in vec]
            return None
        except (URLError, HTTPError, Exception):
            return None

def _embed_cached(text: str) -> Optional[Tuple[float, ...]]:
    v = _embed_vec(text)
    if not v:
        return None
    try:
        return tuple(float(x) for x in v)
    except Exception:
        return None

def _trim(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

# (…rest of file stays exactly as you pasted; unchanged below…)

from .calendar_client import find_free_slots, suggest_windows
from .config import settings
# ─────────────────────────────────────────────────────────
# Transparent query builder
# ─────────────────────────────────────────────────────────
def build_semantic_query(
    *,
    subject: str = "",
    body: str = "",
    thread_text: str = "",
    max_thread_tail: int = 1200,
) -> Dict[str, Any]:
    subj = (subject or "").strip()
    bod = (body or "").strip()
    tail = (thread_text or "").strip()
    if tail:
        tail = tail[-max_thread_tail:]
    parts = {"subject": subj, "body": bod, "thread_tail": tail}
    query = " \n".join([p for p in [subj, bod, tail] if p])
    return {"parts": parts, "query": query}

# ─────────────────────────────────────────────────────────
# Low-level searches (vector → fulltext → loose fulltext)
# ─────────────────────────────────────────────────────────
def semantic_docs_raw(query: str, k: int = 5, *, loose_fulltext: bool = False) -> Dict[str, Any]:
    k = max(1, min(int(k), 100))
    qv = _embed_cached(query)
    if qv:
        cy = """
        MATCH (d:EcoLocalDoc) WHERE (d.embedding) IS NOT NULL
        WITH d, gds.similarity.cosine(d.embedding, $qvec) AS score
        ORDER BY score DESC LIMIT $k
        RETURN { id: d.id, title: d.title, text: d.text, tags: d.tags, score: score } AS doc
        """
        rows = _run(cy, {"qvec": list(qv), "k": k})
        if rows:
            return {"source": "vector", "k": k, "query": query, "docs": [r["doc"] for r in rows]}

    cy2 = """
    CALL db.index.fulltext.queryNodes('EcoLocalDocsText', $term) YIELD node, score
    RETURN { id: node.id, title: node.title, text: node.text, tags: node.tags, score: score } AS doc
    LIMIT $k
    """
    try:
        rows = _run(cy2, {"term": query, "k": k})
        if rows:
            return {"source": "fulltext", "k": k, "query": query, "docs": [r["doc"] for r in rows]}
    except Exception:
        pass

    if loose_fulltext:
        terms = [t for t in query.split() if t]
        if terms:
            loose = " OR ".join([f"{t}~ OR {t}*" for t in terms])
            try:
                rows = _run(cy2, {"term": loose, "k": k})
                if rows:
                    return {"source": "fulltext-loose", "k": k, "query": loose, "docs": [r["doc"] for r in rows]}
            except Exception:
                pass

    return {"source": "none", "k": k, "query": query, "docs": []}

def semantic_status() -> Dict[str, Any]:
    idx_rows = _run("""
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties, state
        RETURN collect({
            name: name,
            type: type,
            entityType: entityType,
            labels: labelsOrTypes,
            props: properties,
            state: state
        }) AS idx
    """)
    doc_count = _run("MATCH (d:EcoLocalDoc) RETURN count(d) AS n")[0]["n"]
    with_emb  = _run("MATCH (d:EcoLocalDoc) WHERE (d.embedding) IS NOT NULL RETURN count(d) AS n")[0]["n"]
    return {
        "docs_total": int(doc_count),
        "docs_with_embedding": int(with_emb),
        "indexes": (idx_rows[0]["idx"] if idx_rows else []),
    }

def semantic_docs(query: str, k: int = 5) -> List[Dict[str, Any]]:
    return semantic_docs_raw(query, k).get("docs", [])

def semantic_topk_for_thread(env_or_dict: Any, k: int = 5) -> List[Dict[str, Any]]:
    try:
        subject = (getattr(env_or_dict, "subject", None) or env_or_dict.get("subject") or "").strip()
    except Exception:
        subject = ""
    try:
        body = (getattr(env_or_dict, "plain_body", None) or env_or_dict.get("plain_body") or "").strip()
    except Exception:
        body = ""
    try:
        ttext = (getattr(env_or_dict, "thread_text", None) or env_or_dict.get("thread_text") or "").strip()
    except Exception:
        ttext = ""

    qb = build_semantic_query(subject=subject, body=body, thread_text=ttext, max_thread_tail=1200)
    raw = semantic_docs_raw(qb["query"] or subject or body, k=k)
    docs = raw.get("docs", []) or []

    out: List[Dict[str, Any]] = []
    for d in docs:
        out.append({
            "id": d.get("id"),
            "title": _trim(d.get("title"), 140),
            "tags": d.get("tags") or [],
            "snippet": _trim(d.get("text"), 1600),
            "score": float(d.get("score")) if d.get("score") is not None else None,
        })
    return out

# ─────────────────────────────────────────────────────────
# Thread context for LLMs
# ─────────────────────────────────────────────────────────
def thread_context(thread_id: str) -> str:
    if not thread_id:
        return ""
    try:
        rows = _run(
            """
            MATCH (t:Thread {id:$tid})<-[:IN_THREAD]-(m:Message)
            WITH m ORDER BY m.date ASC
            RETURN m { .* } AS msg
            LIMIT 30
            """,
            {"tid": thread_id},
        )
        if rows:
            lines: List[str] = []
            for r in rows:
                msg = r.get("msg") or {}
                when = (msg.get("date") or "").strip()
                frm = (msg.get("from") or "").strip()
                sub = (msg.get("subject") or "").strip()
                snip = (msg.get("snippet") or msg.get("plain") or "").strip()
                lines.append(f"[{when}] {frm} - {sub}\n{snip}\n")
            return "\n".join(lines)
    except Exception:
        pass
    return f"(thread:{thread_id})"

# ─────────────────────────────────────────────────────────
# Calendar helpers
# ─────────────────────────────────────────────────────────
def _parse_days(hint: Union[str, int]) -> int:
    if isinstance(hint, int):
        return max(1, hint)
    m = re.search(r"(\d+)", str(hint))
    return int(m.group(1)) if m else 10

def calendar_free_slots(
    range_hint: Union[str, int] = "next 10 days",
    hold_minutes: int = 30,
    *,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    days = _parse_days(range_hint)
    return find_free_slots(
        range_days=days,
        hold_minutes=hold_minutes,
        trace_id=trace_id,
    )

def calendar_suggest_windows(
    *,
    lookahead_days: int = 21,
    duration_min: int = 30,
    work_hours: Tuple[int, int] = (9, 17),
    weekdays: Optional[set[int]] = None,  # 0=Mon ... 6=Sun
    min_gap_min: int = 15,
    hold_padding_min: int = 10,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if weekdays is None:
        weekdays = {0, 1, 2, 3, 4}
    now = datetime.now(timezone.utc).astimezone()
    end = now + timedelta(days=lookahead_days)
    return suggest_windows(
        start=now,
        end=end,
        duration_min=duration_min,
        work_hours=work_hours,
        days=weekdays,
        min_gap_min=min_gap_min,
        hold_padding_min=hold_padding_min,
        trace_id=trace_id,
    )

def gmail_fetch_recent_unseen(user: str = INBOX_ADDR, label: str = "INBOX", since_minutes: int = 65):
    return fetch_unseen_since(minutes=since_minutes, label=label)

# ─────────────────────────────────────────────────────────
# Slot diversification helpers
# ─────────────────────────────────────────────────────────
def _dt_from_iso(dt_iso: str) -> datetime:
    try:
        dt = datetime.fromisoformat(dt_iso)
    except Exception:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc).astimezone()
    return dt

def diversify_slots(
    slots: List[Dict[str, Any]],
    *,
    max_total: int = 5,
    per_day_max: int = 2,
    min_gap_minutes: int = 60,
) -> List[Dict[str, Any]]:
    if not slots:
        return []
    norm: List[Tuple[datetime, Dict[str, Any]]] = []
    for s in slots:
        try:
            dt = _dt_from_iso(s["start"])
            norm.append((dt, s))
        except Exception:
            continue
    norm.sort(key=lambda x: x[0])

    from datetime import timedelta as _td
    chosen: List[Dict[str, Any]] = []
    seen_per_day: dict[str, int] = {}
    last_time_by_day: dict[str, datetime] = {}
    gap = _td(minutes=min_gap_minutes)

    for dt, s in norm:
        day_key = dt.date().isoformat()
        if seen_per_day.get(day_key, 0) >= per_day_max:
            continue
        last_dt = last_time_by_day.get(day_key)
        if last_dt and (dt - last_dt) < gap:
            continue
        chosen.append(s)
        seen_per_day[day_key] = seen_per_day.get(day_key, 0) + 1
        last_time_by_day[day_key] = dt
        if len(chosen) >= max_total:
            break
    return chosen

def calendar_free_slots_diverse(
    range_hint: Union[str, int] = "next 14 days",
    hold_minutes: int = 30,
    *,
    max_total: int = 5,
    per_day_max: int = 2,
    min_gap_minutes: int = 60,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    raw = calendar_free_slots(range_hint, hold_minutes, trace_id=trace_id)
    return diversify_slots(raw, max_total=max_total, per_day_max=per_day_max, min_gap_minutes=min_gap_minutes)

# ─────────────────────────────────────────────────────────
# Prospect mapping (thread → prospect)
# ─────────────────────────────────────────────────────────
def _norm_email(addr: Optional[str]) -> Optional[str]:
    if not addr:
        return None
    parsed = parseaddr(addr)[1]
    a = (parsed or addr).strip().lower()
    if "<" in a and ">" in a:
        a = a.split("<", 1)[1].split(">", 1)[0]
    return a or None

def match_prospect_by_thread(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tid = message.get("thread_id") or message.get("threadId") or ""
    from_email = _norm_email(message.get("from"))
    from_name = (message.get("from_name") or "").strip() or None

    if tid:
        rows = _run(
            """
            MATCH (t:Thread {id:$tid})<-[:IN_THREAD]-(p:Prospect)
            RETURN p LIMIT 1
            """,
            {"tid": tid},
        )
        if rows:
            return rows[0]["p"]

    if from_email:
        rows = _run("MATCH (p:Prospect {email:$email}) RETURN p LIMIT 1", {"email": from_email})
        if rows:
            _run(
                """
                MERGE (t:Thread {email:$email})
                  ON CREATE SET t.created_at = datetime()
                SET t.last_inbound_at = datetime()
                FOREACH (_ IN CASE WHEN $tid <> '' THEN [1] ELSE [] END |
                  SET t.id = coalesce(t.id, $tid)
                )
                WITH t
                MATCH (p:Prospect {email:$email})
                MERGE (p)-[:IN_THREAD]->(t)
                """,
                {"email": from_email, "tid": tid},
            )
            return rows[0]["p"]

        if AUTO_UPSERT_INBOUND:
            node = upsert_prospect_by_email(from_email, from_name)
            _run(
                """
                MERGE (t:Thread {email:$email})
                  ON CREATE SET t.created_at = datetime()
                SET t.last_inbound_at = datetime()
                FOREACH (_ IN CASE WHEN $tid <> '' THEN [1] ELSE [] END |
                  SET t.id = coalesce(t.id, $tid)
                )
                WITH t
                MATCH (p:Prospect {id:$pid})
                MERGE (p)-[:IN_THREAD]->(t)
                """,
                {"email": from_email, "pid": node["id"], "tid": tid},
            )
            return node

    return None
