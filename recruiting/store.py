# recruiting/store.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Dict, Any, Iterable, List, Optional
import json  # ← add this

import re
import os

from neo4j import GraphDatabase
from .config import settings

# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

_driver = GraphDatabase.driver(
    settings.NEO4J_URI,
    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
)


def _run(cy: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Thin helper around a single-session run.
    NOTE: If you hit transient locks, consider adding a simple retry loop here.
    """
    with _driver.session() as s:
        rs = s.run(cy, **(params or {}))
        return [r.data() for r in rs]


TEST_EMAIL_ENV = "ECO_LOCAL_TEST_EMAILS"  # comma-separated

def _test_email_whitelist() -> list[str]:
    """
    Optional test-only whitelist.
    If ECO_LOCAL_TEST_EMAILS is set (comma-separated emails),
    any selector that supports it will ONLY return those emails.
    """
    raw = os.getenv(TEST_EMAIL_ENV, "").strip()
    if not raw:
        return []
    return [e.strip().lower() for e in raw.split(",") if e.strip()]

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _norm_email(e: Optional[str]) -> Optional[str]:
    if not e:
        return None
    e = e.strip().lower()
    return e if EMAIL_RE.match(e) else None


def _domain_from(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if "@" in v:
        v = v.split("@", 1)[1]
    # strip scheme/path
    v = re.sub(r"^https?://", "", v).split("/", 1)[0]
    return v.lstrip(".") or None


# ──────────────────────────────────────────────────────────────────────────────
# Prospect upsert / scoring
# ──────────────────────────────────────────────────────────────────────────────

def upsert_prospect(p: Any) -> Dict[str, Any]:
    """
    Idempotently upsert a Prospect. If email is present, MERGE by email;
    otherwise MERGE by domain.

    On CREATE we set:
      - id=randomUUID()
      - qualified=true
      - qualification_score=0.90
      - outreach_started=false
      - attempt_count=0

    Returns a dict with at least {id, email, domain, name}.
    """
    # Accept dataclass / object / dict
    try:
        data = asdict(p) if hasattr(p, "__dataclass_fields__") else {
            "name": getattr(p, "name", None),
            "email": getattr(p, "email", None),
            "domain": getattr(p, "domain", None),
            "city": getattr(p, "city", None),
            "state": getattr(p, "state", None),
            "country": getattr(p, "country", None),
            "phone": getattr(p, "phone", None),
            "source": getattr(p, "source", "seed"),
        }
    except Exception:
        data = dict(p or {})

    email = _norm_email(data.get("email"))
    domain = _domain_from(data.get("domain") or email)
    name = (data.get("name") or "").strip() or None
    city = (data.get("city") or "").strip() or None
    state = (data.get("state") or "").strip() or None
    country = (data.get("country") or "").strip() or None
    phone = (data.get("phone") or "").strip() or None
    source = (data.get("source") or "seed").strip() or "seed"

    # Ensure constraints (safe if already present)
    try:
        ensure_dedupe_constraints()
    except Exception:
        pass

    if email:
        cypher = """
        MERGE (p:Prospect {email: $email})
          ON CREATE SET
            p.id                   = randomUUID(),
            p.created_at           = datetime(),
            p.source               = $source,
            p.qualified            = true,
            p.qualification_score  = 0.90,
            p.qualification_reason = 'seed',
            p.outreach_started     = false,
            p.attempt_count        = 0
          ON MATCH SET
            p.source = coalesce(p.source, $source)
        SET
          p.name       = coalesce($name, p.name),
          p.domain     = coalesce($domain, p.domain),
          p.city       = coalesce($city, p.city),
          p.state      = coalesce($state, p.state),
          p.country    = coalesce($country, p.country),
          p.phone      = coalesce($phone, p.phone),
          p.updated_at = datetime()
        RETURN p.id as id, p.email as email, p.domain as domain, p.name as name
        """
        params = {
            "email": email,
            "domain": domain,
            "name": name,
            "city": city,
            "state": state,
            "country": country,
            "phone": phone,
            "source": source,
        }
    elif domain:
        cypher = """
        MERGE (p:Prospect {domain: $domain})
          ON CREATE SET
            p.id                   = randomUUID(),
            p.created_at           = datetime(),
            p.source               = $source,
            p.qualified            = true,
            p.qualification_score  = 0.90,
            p.qualification_reason = 'seed',
            p.outreach_started     = false,
            p.attempt_count        = 0
          ON MATCH SET
            p.source = coalesce(p.source, $source)
        SET
          p.name       = coalesce($name, p.name),
          p.city       = coalesce($city, p.city),
          p.state      = coalesce($state, p.state),
          p.country    = coalesce($country, p.country),
          p.phone      = coalesce($phone, p.phone),
          p.updated_at = datetime()
        RETURN p.id as id, p.email as email, p.domain as domain, p.name as name
        """
        params = {
            "domain": domain,
            "name": name,
            "city": city,
            "state": state,
            "country": country,
            "phone": phone,
            "source": source,
        }
    else:
        raise ValueError("upsert_prospect requires at least an email or a domain")

    rows = _run(cypher, params) or []
    return dict(rows[0]) if rows else {"id": None, "email": email, "domain": domain, "name": name}


def qualify_basic(prospect_id: str, *, score: float, reason: str = "") -> Dict[str, Any]:
    """
    Minimal scoring helper used by the orchestrator.
    Clamps score to [0,1], stamps reason & updated_at, sets qualified flag at 0.5+.
    """
    cy = """
    MATCH (p:Prospect {id:$pid})
    SET p.qualification_score  = toFloat($score),
        p.qualification_reason = $reason,
        p.qualified            = CASE WHEN toFloat($score) >= 0.5 THEN true ELSE false END,
        p.updated_at           = datetime()
    RETURN p { .id, .email, .domain, .qualification_score, .qualification_reason, .qualified } AS p
    """
    s = max(0.0, min(1.0, float(score)))
    rs = _run(cy, {"pid": prospect_id, "score": s, "reason": reason})
    return rs[0]["p"] if rs else {}


# ──────────────────────────────────────────────────────────────────────────────
# Settings / gates
# ──────────────────────────────────────────────────────────────────────────────

def get_first_touch_quota() -> int:
    return settings.ECO_LOCAL_FIRST_TOUCH_QUOTA


def get_followup_days() -> list[int]:
    """
    Ordered list of follow-up offsets in days, e.g. [3,7,14]
    """
    return [int(x) for x in settings.ECO_LOCAL_FOLLOWUP_DAYS.split(",") if x.strip()]


def get_max_attempts() -> int:
    return int(settings.ECO_LOCAL_MAX_ATTEMPTS)


# ──────────────────────────────────────────────────────────────────────────────
# Prospect selection
# ──────────────────────────────────────────────────────────────────────────────
def select_prospects_for_first_touch(limit: int) -> list[Dict[str, Any]]:
    whitelist = _test_email_whitelist()
    params: Dict[str, Any] = {"limit": int(limit)}

    if whitelist:
        cy = """
        MATCH (p:Prospect)
        WHERE coalesce(p.qualified, false) = true
          AND coalesce(p.outreach_started, false) = false
          AND p.email IS NOT NULL
          AND toLower(p.email) IN $whitelist
        OPTIONAL MATCH (p)-[:IN_THREAD]->(t:Thread)
        RETURN p{.*, thread_id: coalesce(p.thread_id, t.id)} AS p
        ORDER BY p.qualification_score DESC
        LIMIT $limit
        """
        params["whitelist"] = whitelist
    else:
        cy = """
        MATCH (p:Prospect)
        WHERE coalesce(p.qualified, false) = true
          AND coalesce(p.outreach_started, false) = false
          AND p.email IS NOT NULL
        OPTIONAL MATCH (p)-[:IN_THREAD]->(t:Thread)
        RETURN p{.*, thread_id: coalesce(p.thread_id, t.id)} AS p
        ORDER BY p.qualification_score DESC
        LIMIT $limit
        """

    return [r["p"] for r in _run(cy, params)]
def select_prospects_for_followups(target: date, followup_days: list[int]) -> list[Dict[str, Any]]:
    whitelist = _test_email_whitelist()
    params: Dict[str, Any] = {
        "t": target.isoformat(),
        "days": [int(d) for d in followup_days],
        "max_attempts": get_max_attempts(),
    }

    if whitelist:
        params["whitelist"] = whitelist

    cy = """
    WITH date($t) AS tgt, $days AS days
    MATCH (p:Prospect)
    WHERE coalesce(p.outreach_started, false) = true
      AND coalesce(p.won, false) = false
      AND coalesce(p.unsubscribed, false) = false
      AND p.email IS NOT NULL
      AND coalesce(p.attempt_count, 0) < $max_attempts
    """

    if whitelist:
        cy += """
      AND toLower(p.email) IN $whitelist
        """

    cy += """
    WITH p, tgt, days, coalesce(p.attempt_count,0) AS a

    WITH p, tgt, days, a,
         (p.next_followup_at IS NOT NULL) AS has_next,
         CASE WHEN a <= 0 THEN 0 ELSE a - 1 END AS idx

    WITH p, tgt,
         CASE
           WHEN has_next
             THEN date(datetime(p.next_followup_at))
           ELSE
             date(datetime(p.last_outreach_at)) +
             duration({
               days: toInteger(
                 CASE
                   WHEN idx < size(days) THEN days[idx]
                   ELSE days[size(days)-1]
                 END
               )
             })
         END AS due_date

    WHERE due_date = tgt
    OPTIONAL MATCH (p)-[:IN_THREAD]->(t:Thread)
    RETURN p{.*, thread_id: coalesce(p.thread_id, t.id)} AS p
    """
    return [r["p"] for r in _run(cy, params)]




# ──────────────────────────────────────────────────────────────────────────────
# Modern follow-up helpers (datetime-robust)
# ──────────────────────────────────────────────────────────────────────────────

def list_followups_due(max_attempts: int, followup_days_csv: str) -> List[Dict[str, Any]]:
    """
    Returns prospects with next_followup_at due now (datetime-accurate).
    If next_followup_at is null, this function ignores them (schedule it when sending).
    """
    cy = """
    WITH datetime() AS now, toInteger($maxA) AS maxA
    MATCH (p:Prospect)
    WHERE coalesce(p.unsubscribed,false) = false
      AND coalesce(p.won,false) = false
      AND coalesce(p.attempt_count,0) < maxA
      AND p.email IS NOT NULL
      AND p.next_followup_at IS NOT NULL
      AND datetime(p.next_followup_at) <= now
    RETURN p {.*, email: p.email} AS p
    ORDER BY p.next_followup_at ASC
    LIMIT 500
    """
    return [r["p"] for r in _run(cy, {"maxA": int(max_attempts)})]


def mark_followup_sent(email: str, followup_days_csv: str) -> Dict[str, Any]:
    """
    Increments attempt_count, stamps last_outreach_at, and computes the next_followup_at
    using the follow-up schedule defined in followup_days_csv (e.g. "3,7,14").

    FIX: list comprehension syntax corrected to `[x IN list | expr]`.
    """
    cy = """
    WITH datetime() AS now, split($days_csv, ",") AS raw
    // sanitize → ints; drop blanks/whitespace
    WITH now, [x IN raw WHERE trim(x) <> ""] AS days_txt
    WITH now, [x IN days_txt | toInteger(trim(x))] AS days

    MATCH (p:Prospect {email:$email})
    WITH p, now, days, coalesce(p.attempt_count,0) AS a

    // For attempt index a (0-based), schedule offset days[min(a, size-1)].
    WITH p, now, days, a,
         CASE
           WHEN size(days) = 0 THEN NULL
           WHEN a < size(days) THEN toInteger(days[a])
           ELSE toInteger(days[size(days)-1])
         END AS addDays

    SET p.attempt_count    = a + 1,
        p.last_outreach_at = now,
        p.updated_at       = now,
        p.outreach_started = true,
        p.next_followup_at = CASE
                               WHEN size(days) = 0 THEN NULL
                               WHEN a + 1 >= size(days) THEN NULL
                               ELSE now + duration({days:addDays})
                             END
    RETURN p { .email, .attempt_count, .next_followup_at } AS p
    """
    rs = _run(cy, {"email": email, "days_csv": followup_days_csv})
    return rs[0]["p"] if rs else {}


# ──────────────────────────────────────────────────────────────────────────────
# Runsheet bookkeeping
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunItem:
    email: str
    subject: str
    body: str
    type: str  # e.g. "first" | "followup1" | "followup2" | "final"


def create_run(target: date) -> Dict[str, Any]:
    cy = """
    MERGE (r:ECOLocalRun {date: date($d)})
    ON CREATE SET r.created_at = datetime()
    RETURN r
    """
    return _run(cy, {"d": target.isoformat()})[0]["r"]


def attach_draft_email(run, prospect: Dict[str, Any], kind: str, subject: str, body: str) -> None:
    """
    Idempotently attach/update a draft for (run_date, kind, email, subject).
    """
    cy = """
    MATCH (p:Prospect {id: $pid})
    MERGE (m:Draft { run_date: date($d), kind: $kind, email: p.email, subject: $subject })
    SET m.body = $body,
        m.created_at = coalesce(m.created_at, datetime()),
        m.updated_at = datetime()
    """
    _run(
        cy,
        {
            "d": run["date"],
            "pid": prospect["id"],
            "kind": kind,
            "subject": subject,
            "body": body,
        },
    )


def freeze_run(run) -> None:
    cy = "MERGE (r:ECOLocalRun {date: date($d)}) SET r.frozen = true, r.frozen_at = datetime()"
    _run(cy, {"d": run["date"]})


def iter_run_items(target: date) -> Iterable[RunItem]:
    """
    Fetch drafts by the run_date (canonical path).
    """
    cy = """
    MATCH (m:Draft {run_date: date($d)})
    RETURN m.kind AS kind, m.email AS email, m.subject AS subject, m.body AS body
    """
    rows = _run(cy, {"d": target.isoformat()})
    for r in rows:
        yield RunItem(email=r["email"], subject=r["subject"], body=r["body"], type=r["kind"])


def mark_sent(item: RunItem, message_id: str) -> None:
    """
    - Mark draft as sent
    - Upsert a Thread keyed by (email)
    - Update Prospect: outreach_started, last_outreach_at = now
      IMPORTANT: Do NOT increment attempt_count here.
      The send pipeline should call mark_followup_sent() AFTER this,
      which handles increment + next_followup_at in one place.
    """
    cy = """
    // 1) Mark draft
    MATCH (m:Draft {email: $email, subject: $subject})
    SET m.sent = true,
        m.sent_at = datetime(),
        m.message_id = $mid

    // 2) Thread linkage
    WITH m
    MERGE (t:Thread {email: m.email})
      ON CREATE SET t.created_at = datetime()
    SET t.last_outbound_date = date(datetime()),
        t.last_outbound_at   = datetime()
    MERGE (m)-[:IN_THREAD]->(t)

    // 3) Update Prospect (basic stamps ONLY - no attempt_count here)
    WITH m, t
    MATCH (p:Prospect {email: m.email})
    SET p.outreach_started = true,
        p.last_outreach_at = datetime(),
        p.updated_at       = datetime()
    """
    _run(cy, {"email": item.email, "subject": item.subject, "mid": message_id})


# ──────────────────────────────────────────────────────────────────────────────
# Inbound/update helpers
# ──────────────────────────────────────────────────────────────────────────────

def mark_signup_payload(payload: Dict[str, Any]) -> None:
    email = (payload or {}).get("email")
    if not email:
        return
    cy = """
    MATCH (p:Prospect {email: $email})
    SET p.won = true, p.won_at = datetime(), p.updated_at = datetime()
    """
    _run(cy, {"email": email})


def mark_unsubscribe(prospect: Dict[str, Any]) -> None:
    cy = """
    MATCH (p:Prospect {id: $pid})
    SET p.unsubscribed = true, p.unsubscribed_at = datetime(), p.updated_at = datetime()
    """
    _run(cy, {"pid": prospect["id"]})


def mark_reply_won(prospect: Dict[str, Any]) -> None:
    cy = "MATCH (p:Prospect {id: $pid}) SET p.won = true, p.won_at = datetime(), p.updated_at = datetime()"
    _run(cy, {"pid": prospect["id"]})

def _to_primitive(value: Any) -> Any:
    """
    Recursively convert value into Neo4j/JSON-friendly primitives:
    - str, int, float, bool, None
    - lists / dicts of those
    - dataclasses → dicts
    - datetime/date → isoformat
    - everything else → str(value)
    """
    from dataclasses import is_dataclass

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value):
        return {k: _to_primitive(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_primitive(v) for v in value]
    # fallback: string representation
    return str(value)
def log_reply_draft(
    prospect: Dict[str, Any],
    *,
    subject: str,
    html: str,
    metadata: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record that we generated/sent a reply (draft or final) to a prospect, and link it to their Thread.

    - Thread is keyed by Prospect.email.
    - We keep a synthetic message key (subject+timestamp) for now; SES message-id can be patched later.
    - We store subject + html + metadata_json (stringified JSON) on the Reply node.
    """
    synthetic_mid = f"{subject}|{datetime.utcnow().isoformat()}"

    # Ensure metadata is fully primitive before serialising
    meta_prim = _to_primitive(metadata or {})
    metadata_json = json.dumps(meta_prim, ensure_ascii=False)

    cy = """
    MATCH (p:Prospect {id:$pid})
    MERGE (t:Thread {email: p.email})
      ON CREATE SET t.created_at = datetime()
    SET t.last_outbound_at   = datetime(),
        t.last_outbound_date = date(datetime()),
        t.id                 = coalesce(t.id, $thread_id)

    MERGE (m:Reply {message_id: $mid})
      ON CREATE SET m.created_at = datetime()
    SET m.subject       = $subject,
        m.html          = $html,
        m.metadata_json = $metadata_json,
        m.updated_at    = datetime()

    MERGE (p)-[:IN_THREAD]->(t)
    MERGE (m)-[:IN_THREAD]->(t)
    RETURN m
    """

    rows = _run(
        cy,
        {
            "pid": prospect["id"],
            "mid": synthetic_mid,
            "subject": subject,
            "html": html,
            "metadata_json": metadata_json,
            "thread_id": thread_id or "",
        },
    )
    return rows[0]["m"] if rows else {}


def book_event(prospect: Dict[str, Any], slot: Dict[str, Any]) -> None:
    """
    Persist a calendar record (currently modeled as a HOLD).
    slot: {"start": "...", "end": "...", "tz": "..."}
    """
    cy = """
    MATCH (p:Prospect {id:$pid})
    MERGE (h:CalendarHold {
      start:$start, end:$end, tz:coalesce($tz,'')
    })
    ON CREATE SET h.created_at = datetime(), h.status = 'hold'
    SET h.updated_at = datetime()
    MERGE (p)-[:HAS_HOLD]->(h)
    """
    _run(
        cy,
        {
            "pid": prospect["id"],
            "start": slot.get("start"),
            "end": slot.get("end"),
            "tz": slot.get("tz"),
        },
    )


def upsert_prospect_by_email(email: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create (or update) a minimal Prospect on first inbound from an unknown sender.
    Defaults: qualified=true, moderate score.
    """
    cy = """
    MERGE (p:Prospect {email:$email})
    ON CREATE SET p.id = randomUUID(),
                  p.created_at = datetime(),
                  p.qualified = true,
                  p.qualification_score = 0.6,
                  p.qualification_reason = 'inbound'
    SET p.name = coalesce($name, p.name),
        p.updated_at = datetime()
    RETURN p
    """
    return _run(cy, {"email": email, "name": name})[0]["p"]

def log_inbound_email(prospect: Dict[str, Any], message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist an inbound email node and link it to the Prospect's Thread.
    Creates/updates a Thread (by email) and sets Thread.id if a Gmail thread_id is supplied.
    Also mirrors the Gmail thread id onto Prospect.thread_id for outreach / context tools.
    Returns the created/merged inbound email projection.
    """
    gid = (message.get("id") or message.get("gmail_id") or "").strip()
    tid = (message.get("thread_id") or message.get("threadId") or "").strip()
    subj = (message.get("subject") or "").strip()
    snippet = (message.get("snippet") or "").strip()
    body = (message.get("body_text") or "").strip()
    frm = (message.get("from") or "").strip()

    cy = """
    MATCH (p:Prospect {id:$pid})

    // Thread by email; set thread id if provided
    MERGE (t:Thread {email: p.email})
      ON CREATE SET t.created_at = datetime()
    SET t.last_inbound_at = datetime()
    FOREACH (_ IN CASE WHEN $tid <> '' THEN [1] ELSE [] END |
      SET t.id = coalesce(t.id, $tid),
          p.thread_id = coalesce(p.thread_id, $tid)
    )

    // Create inbound email node keyed by Gmail id if available, else by (email+subject+timestamp)
    MERGE (m:InboundEmail {
      key: CASE
              WHEN $gid <> '' THEN $gid
              ELSE p.email + '|' + coalesce($subj,'') + '|' + toString(datetime())
           END
    })
      ON CREATE SET m.created_at = datetime()
    SET m.gmail_id     = CASE WHEN $gid <> '' THEN $gid ELSE m.gmail_id END,
        m.thread_id    = CASE WHEN $tid <> '' THEN $tid ELSE m.thread_id END,
        m.from_raw     = $from,
        m.subject      = $subj,
        m.snippet      = $snippet,
        m.body_text    = $body,
        m.received_at  = coalesce(m.received_at, datetime()),
        m.updated_at   = datetime()

    MERGE (p)-[:IN_THREAD]->(t)
    MERGE (m)-[:IN_THREAD]->(t)

    RETURN m { .key, .gmail_id, .thread_id, .subject, .received_at } AS inbound
    """
    rows = _run(
        cy,
        {
            "pid": prospect["id"],
            "gid": gid,
            "tid": tid,
            "from": frm,
            "subj": subj,
            "snippet": snippet,
            "body": body,
        },
    )
    return rows[0]["inbound"] if rows else {}



def close_driver() -> None:
    try:
        _driver.close()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Seen registry (dedupe across discovery runs)
# ──────────────────────────────────────────────────────────────────────────────

def _norm_domain(d: Optional[str]) -> Optional[str]:
    if not d:
        return None
    d = d.strip().lower()
    return d[4:] if d.startswith("www.") else d


def ensure_followup_props() -> None:
    cy = """
    MATCH (p:Prospect)
    SET p.next_followup_at = coalesce(p.next_followup_at, null),
        p.attempt_count    = coalesce(p.attempt_count, 0)
    """
    _run(cy)


def ensure_dedupe_constraints() -> None:
    """
    Idempotent uniqueness constraints:
      - Seen registries
      - Prospect email/domain
    Safe to call often.
    """
    stmts = [
        "CREATE CONSTRAINT seen_domain  IF NOT EXISTS FOR (n:SeenTarget) REQUIRE n.domain IS UNIQUE",
        "CREATE CONSTRAINT seen_email   IF NOT EXISTS FOR (n:SeenEmail)  REQUIRE n.email  IS UNIQUE",
        "CREATE CONSTRAINT prospect_em  IF NOT EXISTS FOR (p:Prospect)   REQUIRE p.email  IS UNIQUE",
        "CREATE CONSTRAINT prospect_dom IF NOT EXISTS FOR (p:Prospect)   REQUIRE p.domain IS UNIQUE",
    ]
    for cy in stmts:
        try:
            _run(cy)
        except Exception:
            # ignore if edition/permissions don't allow constraint ops
            pass


def has_prospect(*, email: Optional[str] = None, domain: Optional[str] = None) -> bool:
    email = _norm_email(email)
    domain = _norm_domain(domain)
    if email:
        rows = _run("MATCH (p:Prospect {email:$e}) RETURN 1 AS one LIMIT 1", {"e": email})
        if rows:
            return True
    if domain:
        rows = _run("MATCH (p:Prospect {domain:$d}) RETURN 1 AS one LIMIT 1", {"d": domain})
        if rows:
            return True
    return False


def has_seen_candidate(*, domain: Optional[str], email: Optional[str]) -> bool:
    domain = _norm_domain(domain)
    email = _norm_email(email)
    if email:
        rows = _run("MATCH (e:SeenEmail {email:$e}) RETURN 1 AS one LIMIT 1", {"e": email})
        if rows:
            return True
    if domain:
        rows = _run("MATCH (d:SeenTarget {domain:$d}) RETURN 1 AS one LIMIT 1", {"d": domain})
        if rows:
            return True
    return False


def mark_seen_candidate(*, domain: Optional[str], email: Optional[str], name: Optional[str] = None) -> None:
    ensure_dedupe_constraints()
    domain = _norm_domain(domain)
    email = _norm_email(email)
    if domain:
        _run(
            """
            MERGE (d:SeenTarget {domain:$d})
            ON CREATE SET d.first_seen = datetime()
            SET d.name = coalesce($name, d.name)
            """,
            {"d": domain, "name": name},
        )
    if email:
        _run(
            """
            MERGE (e:SeenEmail {email:$e})
            ON CREATE SET e.first_seen = datetime()
            SET e.name = coalesce($name, e.name)
            """,
            {"e": email, "name": name},
        )
