# recruiting/llm_flow.py
from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from email.utils import parseaddr

try:
    import httpx  # prefer httpx if available
except Exception:  # pragma: no cover
    httpx = None

from .branding import header_logo_src_email  # <- email-safe https/data logo
from .llm_client import generate_json as _gen_json
from .tools import thread_context, semantic_topk_for_thread
from .calendar_client import (
    is_range_free,
    suggest_windows,  # windowed search
    create_hold,
    create_event,
    promote_hold_to_event,
    cancel_holds,
    cancel_thread_events,
    build_ics,
    ICSSpec,
)

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

Json = Dict[str, Any]
LLMFn = Callable[[Any], Dict[str, Any]]  # accepts str OR list prompt (our _gen_json supports both)


@dataclass
class EmailEnvelope:
    thread_id: str
    message_id: str
    from_addr: str
    to_addr: str
    subject: str
    received_at_iso: str  # absolute ISO with tz
    plain_body: str
    html_body: Optional[str] = None
    thread_text: Optional[str] = None


@dataclass
class CalendarQuery:
    window_start: str
    window_end: str
    duration_minutes: int = 30
    exact_start: Optional[str] = None
    same_time_other_days: bool = False
    daypart_hint: Optional[str] = None
    flexibility_minutes: int = 0


@dataclass
class CandidateSlot:
    start: str
    end: str
    reason: str


@dataclass
class AnalysisPlan:
    intent: str  # meeting | info | unsubscribe | other
    summary: str
    sought_outcome: str
    calendar_queries: List[CalendarQuery]
    confidence: float  # 0..1
    explicit_questions: Optional[str] = ""
    facts_needed: Optional[List[str]] = None
    user_confirmed: Optional[bool] = None  # LLM-led, not heuristics
    has_questions: Optional[bool] = None   # explicit boolean


@dataclass
class ToolSpec:
    name: str
    description: str
    args: Dict[str, Any]


@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]
    purpose: str


@dataclass
class ToolRound:
    proposed_calls: List[ToolCall]
    results: Dict[str, Any]


@dataclass
class BookingDecision:
    action: str  # none | hold | event | promote_hold | cancel_holds | cancel_thread_events
    chosen_slot: Optional[CandidateSlot]
    location_hint: Optional[str]
    internal_notes: Optional[str] = None


@dataclass
class DraftResult:
    subject: str
    html: str


@dataclass
class FinalAction:
    booking_decision: BookingDecision
    draft_result: DraftResult


@dataclass
class FlowOutput:
    plan: AnalysisPlan
    tool_rounds: List[ToolRound]
    booking: BookingDecision
    draft: DraftResult
    ics: Optional[Tuple[str, bytes]] = None  # build_ics returns (filename, bytes)
    semantic_context: Optional[List[Dict[str, Any]]] = None  # expose injected facts

# ──────────────────────────────────────────────────────────────────────────────
# Tool inventory (LLM-visible)
# ──────────────────────────────────────────────────────────────────────────────

def _bare_email(addr: str | None) -> Optional[str]:
    if not addr:
        return None
    return (parseaddr(addr)[1] or addr).strip()

def _tool_catalog() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="calendar.check_free",
            description="Check if an exact start->end window is free.",
            args={"start_iso": "string", "end_iso": "string"},
        ),
        ToolSpec(
            name="calendar.search_free",
            description="Find free windows within a range (uses internal suggestion engine).",
            args={
                "window_start": "string",
                "window_end": "string",
                "duration_minutes": "int",
                "n": "int",
                "daypart": "string|null"
            },
        ),
        ToolSpec(
            name="calendar.create_hold",
            description="Place a *tentative* hold. Attaches an .ics (REQUEST). Does NOT send a Google invite.",
            args={"start_iso": "string", "end_iso": "string", "summary": "string", "description": "string", "location": "string|null"},
        ),
        ToolSpec(
            name="calendar.create_event",
            description="Create a *firm* event. This WILL send a Google invite to the attendee.",
            args={"start_iso": "string", "end_iso": "string", "summary": "string", "description": "string", "location": "string|null"},
        ),
        ToolSpec(
            name="calendar.promote_hold_to_event",
            description="Promote an existing HOLD to a firm event and ensure the attendee is on it.",
            args={"hold_event_id": "string"},
        ),
        ToolSpec(
            name="calendar.cancel_holds",
            description="Cancel all tentative holds related to this thread.",
            args={"thread_id": "string"},
        ),
        ToolSpec(
            name="calendar.cancel_thread_events",
            description="Cancel all events related to this thread.",
            args={"thread_id": "string"},
        ),
    ]

# ──────────────────────────────────────────────────────────────────────────────
# Utilities (time, semantics)
# ──────────────────────────────────────────────────────────────────────────────

def _now_plus_7_days(tz: ZoneInfo) -> Dict[str, Any]:
    now = datetime.now(tz)
    days = []
    for i in range(0, 14):
        d = (now + timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        days.append({
            "dow_name": d.strftime("%A"),
            "date_iso": d.isoformat(),
            "date_human": d.strftime("%a %d %b %Y"),
        })
    return {"now_iso": now.isoformat(), "days": days}


def _thread_text(env: EmailEnvelope) -> str:
    if env.thread_text:
        return env.thread_text
    try:
        return thread_context(env.thread_id)
    except Exception as e:
        log.warning("thread_context failed: %s", e)
        return f"Subject: {env.subject}\n\n{env.plain_body or ''}"


def _coerce_calendar_queries(raw: Any) -> List[CalendarQuery]:
    qs: List[CalendarQuery] = []
    if not isinstance(raw, list):
        return qs
    for item in raw:
        try:
            qs.append(CalendarQuery(
                window_start=item["window_start"],
                window_end=item["window_end"],
                duration_minutes=int(item.get("duration_minutes", 30)),
                exact_start=item.get("exact_start"),
                same_time_other_days=bool(item.get("same_time_other_days", False)),
                daypart_hint=item.get("daypart_hint"),
                flexibility_minutes=int(item.get("flexibility_minutes", 0)),
            ))
        except Exception as e:
            log.warning("skip malformed CalendarQuery: %s", e)
    return qs

def _trim_doc(d: Dict[str, Any], *, max_chars: int = 1200) -> Dict[str, Any]:
    text = (d.get("text") or d.get("snippet_300") or "")[:max_chars]
    return {
        "id": d.get("id"),
        "title": d.get("title"),
        "tags": d.get("tags") or [],
        "score": d.get("score"),
        "text": text,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Semantic (QUERY-based) via HTTP with fallback to local tools
# ──────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "re", "fw", "fwd", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
    "at", "with", "we", "you", "i", "me", "us", "our", "your", "is", "it", "that",
    "this", "hi", "hey", "thanks", "thank", "regards", "best", "team", "subject",
    "wrote", "mon", "tue", "wed", "thu", "fri", "sat", "sun", "am", "pm",
}

def _strip_quoted_email(text: str) -> str:
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(">")]
    text = "\n".join(lines)
    text = re.split(r"\nOn .* wrote:\s*\n", text, flags=re.IGNORECASE)[0]
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _keywords_from_subject_body(subject: str, body: str) -> List[str]:
    raw = f"{subject} {body}".lower()
    tokens = re.findall(r"[a-z][a-z\-]{2,}", raw)
    keywords = [t for t in tokens if t not in STOPWORDS]
    seen = set(); out = []
    for t in keywords:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:20]

def _build_semantic_candidates(email: EmailEnvelope) -> List[str]:
    subj = (email.subject or "")
    body_clean = _strip_quoted_email(email.plain_body or "")
    kws = _keywords_from_subject_body(subj, body_clean)
    kwset = set(kws)
    candidates: List[str] = []

    if {"pricing", "price", "cost", "fee", "membership"} & kwset:
        candidates.append("pricing structure for businesses")
    if {"points", "point", "currency", "eco", "token", "reward", "visits"} & kwset:
        candidates.append("ECO points system for businesses and proof of visits")
    if {"join", "sign", "onboard", "list", "offer", "partners", "business"} & kwset:
        candidates.append("For Businesses — List an offer")
    if {"youth", "scan", "map", "redeem"} & kwset:
        candidates.append("For Youth — Find offers nearby")

    if body_clean:
        first_sent = re.split(r"[.!?]", body_clean)[0].strip()
        if first_sent and len(first_sent.split()) >= 3:
            candidates.append(first_sent)

    candidates.append("ECO Local overview")

    seen = set(); uniq = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c); uniq.append(c)
    return uniq[:6]

def _fetch_semantic_docs_http_query(query: str, *, k: int, loose: bool, api_base: str) -> List[Dict[str, Any]]:
    if not httpx:
        return []
    url = api_base.rstrip("/") + "/debug/semantic/query"
    params = {"query": query, "k": int(k), "loose": 1 if loose else 0}
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            docs = data.get("results") or data.get("docs") or []
            return [_trim_doc(d) for d in docs]
    except Exception as e:
        log.warning("semantic HTTP (query) fetch failed for %r: %s", query, e)
        return []

def _get_semantic_docs(
    email: EmailEnvelope,
    *,
    k: int,
    loose: bool,
    semantic_query_override: Optional[str] = None
) -> List[Dict[str, Any]]:
    api_base = os.getenv("ECO_LOCAL_API_BASE", "").strip()

    if api_base:
        queries: List[str] = []
        if semantic_query_override:
            qv = semantic_query_override.strip()
            if qv:
                queries.append(qv)
        queries.extend(_build_semantic_candidates(email))

        for q in queries:
            docs = _fetch_semantic_docs_http_query(q, k=k, loose=loose, api_base=api_base)
            log.info("[llm_flow] semantic probe query_used=%r hits=%d", q, len(docs))
            if docs:
                return docs

    try:
        docs = semantic_topk_for_thread(email, k=k) or []
        if docs:
            log.info("[llm_flow] semantic_context via local tools: %d docs", len(docs))
            return [_trim_doc(d) for d in docs]
    except Exception:
        log.exception("semantic_topk_for_thread failed")

    log.info("[llm_flow] semantic_context: 0 docs after all probes")
    return []

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

def _prompt_analyze_and_plan(env: EmailEnvelope, tz: str, *, semantic_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    system = (
        "You are a careful ECO Local coordinator.\n"
        "Use ONLY facts present in `semantic_context` and the user's email/thread.\n"
        "Return JSON only (no prose outside JSON)."
    )
    user = {
        "timezone": tz,
        "current_time_context": _now_plus_7_days(ZoneInfo(tz)),
        "email_envelope": asdict(env),
        "thread_text": _thread_text(env),
        "latest_user_email": env.plain_body,
        "semantic_context": {
            "top_k": len(semantic_docs),
            "docs": semantic_docs,
            "note": "These are the ONLY allowed reference facts. If a needed fact is absent, say so later."
        },
        "ask": [
            "1) Identify the primary intent: meeting | info | unsubscribe | other.",
            "2) Summarize explicit questions (e.g., pricing, points).",
            "3) Set boolean `has_questions`: true iff the user asked any explicit question.",
            "4) Did the user ALREADY CONFIRM a specific time? Return boolean `user_confirmed`.",
            "5) If meeting-related or reschedule implied, propose concrete date ranges and build `calendar_queries`.",
            "6) List IDs of any `semantic_context.docs` you believe will be needed.",
            "7) Use the previous scheduling context to figure out if the user is talking about this week or next when stating a day. Do not jump to conclusions."
        ],
        "tools": [asdict(t) for t in _tool_catalog() if ("search" in t.name or "check" in t.name)],
        "return_format": {
            "type": "object",
            "properties": {
                "intent": {"enum": ["meeting", "info", "unsubscribe", "other"]},
                "summary": {"type": "string"},
                "sought_outcome": {"type": "string"},
                "explicit_questions": {"type": "string"},
                "has_questions": {"type": "boolean"},
                "user_confirmed": {"type": "boolean"},
                "calendar_queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["window_start", "window_end", "duration_minutes"],
                        "properties": {
                            "window_start": {"type": "string"},
                            "window_end": {"type": "string"},
                            "duration_minutes": {"type": "integer"},
                            "exact_start": {"type": ["string", "null"]},
                            "daypart_hint": {"type": ["string", "null"]}
                        }
                    }
                },
                "facts_needed": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"}
            },
            "required": ["intent", "summary", "sought_outcome", "calendar_queries", "confidence", "user_confirmed", "has_questions"]
        }
    }
    return {"system": system, "user": user}


def _prompt_coordinator_action(
    env: EmailEnvelope,
    plan: AnalysisPlan,
    available_slots: List[CandidateSlot],
    tz: str,
    *,
    semantic_docs: List[Dict[str, Any]],
    user_confirmed: bool,
    has_questions: bool,
) -> Dict[str, Any]:
    # System prompt explicitly gates the Q&A paragraph on `has_questions`
    system = (
        "You are Ecodia. You are warm, friendly, passionate, and a little rebellious. You are coordinating ECO Local partner outreach.\n"
        "Use the provided notes (user_confirmed, has_questions, etc.) to reply in the best way you feel.\n"
        "Be open, natural, and casual—never over-the-top or corporate. Avoid em dashes.\n"
        "Your goal is to naturally end up with the user confirming signup or booking a meeting.\n"
        "Return JSON only."
    )

    response_requirements = {
        "must_do": [""],
        "meeting_rules": [
            "If action = 'hold' or 'event', try to casually get confirmation from them via the event or through their next reply."
        ],
        "info_rules": [],
        "confirmation_rules": []
    }

    user = {
        "timezone": tz,
        "current_time_context": _now_plus_7_days(ZoneInfo(tz)),
        "full_thread_text": _thread_text(env),
        "latest_user_email": env.plain_body,
        "initial_plan": asdict(plan),
        "available_slots": [asdict(s) for s in available_slots],
        "tools_for_action": [asdict(t) for t in _tool_catalog() if ("create" in t.name or "promote" in t.name or "cancel" in t.name)],
        "semantic_context": {
            "top_k": len(semantic_docs),
            "docs": semantic_docs,
            "note": "Never hallucinate. Use only provided facts."
        },
        "user_confirmed": bool(user_confirmed),
        "has_questions": bool(has_questions),
        "response_requirements": response_requirements,
        "ask": [
            "1) Ensure day/date correctness.",
            "2) List IDs of any `semantic_context.docs` you actually used.",
            "3) Be warm and natural."
        ],
        "return_format": {
            "type": "object",
            "properties": {
                "booking_decision": {
                    "type": "object",
                    "properties": {
                        "action": {"enum": ["none", "hold", "event", "promote_hold", "cancel_holds", "cancel_thread_events"]},
                        "chosen_slot": {
                            "type": ["object", "null"],
                            "properties": {"start": {"type": "string"}, "end": {"type": "string"}, "reason": {"type": "string"}},
                            "required": ["start", "end", "reason"]
                        },
                        "location_hint": {"type": ["string", "null"]},
                        "internal_notes": {"type": ["string", "null"]}
                    },
                    "required": ["action", "chosen_slot"]
                },
                "draft_result": {
                    "type": "object",
                    "properties": {"subject": {"type": "string"}, "html": {"type": "string"}},
                    "required": ["subject", "html"]
                },
                "facts_used": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["booking_decision", "draft_result"]
        }
    }
    return {"system": system, "user": user}

# ──────────────────────────────────────────────────────────────────────────────
# Email polishing (signature + optional time suggestions)
# ──────────────────────────────────────────────────────────────────────────────

_LOGO_SRC = header_logo_src_email()  # email-safe https/data

def _render_times_block(slots: List[CandidateSlot]) -> str:
    if not slots:
        return ""
    def _label(s_iso: str, e_iso: str) -> str:
        try:
            s = datetime.fromisoformat(s_iso.replace("Z", "+00:00"))
            e = datetime.fromisoformat(e_iso.replace("Z", "+00:00"))
            h = lambda dt: (f"{(dt.hour % 12) or 12}:{dt.minute:02d}" if dt.minute else f"{(dt.hour % 12) or 12}") + ("am" if dt.hour < 12 else "pm")
            day = f"{s.strftime('%a')} {s.day} {s.strftime('%b')}"
            return f"{day}, {h(s)}–{h(e)}"
        except Exception:
            return f"{s_iso} – {e_iso}"
    items = "".join(f"<li>{_label(s.start, s.end)}</li>" for s in slots[:3])
    return (
        '<div data-ecolocal-slots="1" style="margin:16px 0 8px; font-family:Arial,Helvetica,sans-serif;">'
        '<div style="font-weight:600; margin-bottom:6px;">A few times that could work:</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )

def _signature_block() -> str:
    # marker + data attribute for reliable de-dup and insertion
    return f"""
<!--ECOL_SIGNATURE_START-->
<div data-ecol-signature="1">
  <table cellpadding="0" cellspacing="0" role="presentation" style="margin-top:16px;">
    <tr>
      <td style="padding-right:12px; vertical-align:top;">
        <img src="{_LOGO_SRC}" alt="ECO Local logo" width="110" style="display:block; border:0;">
      </td>
      <td style="vertical-align:top;">
        <div style="font-family:'Arial Narrow','Roboto Condensed',Arial,sans-serif; font-size:13px; line-height:1.4;">
          <div style="color:#396041; font-weight:600;">Proof, not offsets, building local value.</div>
          <div style="color:#000; margin-top:2px;">An Ecodia Launchpad project</div>
          <div style="margin-top:4px;">
            <a href="https://ecodia.au/eco-local" style="color:#7fd069; text-decoration:none;">ecodia.au/eco-local</a><br>
            <a href="mailto:ecolocal@ecodia.au" style="color:#7fd069; text-decoration:none;">ecolocal@ecodia.au</a>
          </div>
          <div style="margin-top:8px; color:#777%;">
            Ecodia helps communities, youth, and partners collaborate and build regenerative futures together.
            <br>We sometimes make mistakes — let us know at connect@ecodia.au</br>
          </div>
        </div>
      </td>
    </tr>
  </table>
</div>
""".strip()

def _polish_email_html(html: str, slots: List[CandidateSlot]) -> str:
    """
    - Ensure a branded signature with https/data logo is present.
    - If not already present, insert up to 3 varied time options *above* the signature marker.
    - Avoid duplicating either section.
    """
    out = (html or "")
    has_sig = ('data-ecol-signature="1"' in out) or ("<!--ECOL_SIGNATURE_START-->" in out)
    if not has_sig:
        out = out.rstrip() + "\n" + _signature_block()

    if 'data-ecolocal-slots="1"' not in out:
        times_html = _render_times_block(slots)
        if times_html:
            out = out.replace("<!--ECOL_SIGNATURE_START-->", times_html + "\n<!--ECOL_SIGNATURE_START-->")

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Tool executor
# ──────────────────────────────────────────────────────────────────────────────

def _exec_tool_call(call: ToolCall, tz: ZoneInfo, *, thread_id: Optional[str], email_envelope: EmailEnvelope) -> Tuple[str, Any]:
    name = call.tool
    args = call.args or {}
    attendee_email = _bare_email(email_envelope.from_addr or None)

    if name == "calendar.check_free":
        s = args["start_iso"]
        e = args["end_iso"]
        free = is_range_free(s, e)
        return name, {"free": bool(free)}

    if name == "calendar.search_free":
        w0 = datetime.fromisoformat(args["window_start"])
        w1 = datetime.fromisoformat(args["window_end"])
        dur = int(args["duration_minutes"])
        windows = suggest_windows(w0, w1, duration_min=dur, work_hours=(9, 17), days={0, 1, 2, 3, 4})
        slots = [{"start": it["start"], "end": it["end"], "reason": it.get("reason", "suggest")} for it in windows]
        n = int(args.get("n", 8))
        return name, {"slots": slots[:n]}

    if name == "calendar.create_hold":
        s = args["start_iso"]; e = args["end_iso"]
        evt = create_hold(
            start_iso=s,
            end_iso=e,
            thread_id=thread_id,
            prospect_email=attendee_email,
            attendees=([{"email": attendee_email}] if attendee_email else []),
            title=args.get("summary") or "Ecodia — intro chat (HOLD)",
            description=args.get("description") or "Provisional hold (auto-expires unless confirmed).",
            tz=str(tz),
            send_updates="none",
        )
        return name, {"event_id": evt.get("id"), "summary": evt.get("summary")}

    if name == "calendar.create_event":
        s = args["start_iso"]; e = args["end_iso"]
        evt = create_event(
            title=args.get("summary") or "Ecodia — intro chat",
            description=args.get("description") or "Looking forward to meeting you!",
            start_iso=s,
            end_iso=e,
            tz=str(tz),
            attendees=([{"email": attendee_email}] if attendee_email else []),
            thread_id=thread_id,
            prospect_email=attendee_email,
            kind="CONFIRMED",
            send_updates="all",
        )
        return name, {"event_id": evt.get("id"), "summary": evt.get("summary")}

    if name == "calendar.promote_hold_to_event":
        evt = promote_hold_to_event(event_id=args["hold_event_id"], ensure_attendee_email=attendee_email)
        return name, {"event_id": evt.get("id"), "summary": evt.get("summary")}

    if name == "calendar.cancel_holds":
        cancel_holds(thread_id=args["thread_id"])
        return name, {"ok": True}

    if name == "calendar.cancel_thread_events":
        cancel_thread_events(thread_id=args["thread_id"])
        return name, {"ok": True}

    return name, {"error": f"tool {name} not allowed"}

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline (2-step) — with semantic_k and LLM-led confirmation
# ──────────────────────────────────────────────────────────────────────────────

def run_llm_flow(
    *,
    email: EmailEnvelope,
    tz: str = "Australia/Brisbane",
    llm: LLMFn = _gen_json,
    allow_calendar_writes: bool = False,
    max_tool_rounds: int = 3,
    semantic_k: int = 5,
    semantic_loose: bool = True,  # allow fuzzy full-text when needed
    semantic_query_override: Optional[str] = None,
) -> FlowOutput:
    z = ZoneInfo(tz)

    # 0) Initial semantic injection (HTTP query first, local fallback)
    sem_docs = _get_semantic_docs(email, k=semantic_k, loose=semantic_loose, semantic_query_override=semantic_query_override)
    try:
        if sem_docs:
            log.info("[llm_flow] injected %d semantic docs: %s",
                     len(sem_docs),
                     ", ".join([f"{(d.get('title') or '')[:40]}#{(d.get('id') or '')[:6]}" for d in sem_docs]))
        else:
            log.info("[llm_flow] injected 0 semantic docs")
    except Exception:
        log.exception("[llm_flow] logging semantic docs failed")

    # 1) Plan — includes LLM judgment for confirmation + question detection
    a_raw = llm([_prompt_analyze_and_plan(email, tz, semantic_docs=sem_docs)])
    plan = AnalysisPlan(
        intent=a_raw.get("intent", "other"),
        summary=a_raw.get("summary", ""),
        sought_outcome=a_raw.get("sought_outcome", ""),
        calendar_queries=_coerce_calendar_queries(a_raw.get("calendar_queries", [])),
        confidence=float(a_raw.get("confidence", 0.5)),
        explicit_questions=a_raw.get("explicit_questions", "") or "",
        facts_needed=a_raw.get("facts_needed") or [],
        user_confirmed=bool(a_raw.get("user_confirmed", False)),
        has_questions=bool(a_raw.get("has_questions", False)),
    )

    # 1b) Optional targeted retrieval pass if there ARE questions
    if plan.has_questions and plan.explicit_questions:
        api_base = os.getenv("ECO_LOCAL_API_BASE", "").strip()
        if api_base:
            targeted = _fetch_semantic_docs_http_query(
                query=plan.explicit_questions, k=semantic_k, loose=semantic_loose, api_base=api_base
            )
            if targeted:
                sem_docs = targeted
                log.info("[llm_flow] reloaded semantic docs from explicit_questions (%d docs)", len(sem_docs))

    tool_rounds: List[ToolRound] = []

    # 1c) Search for slots when meeting intent
    def _queries_to_calls(qs: List[CalendarQuery]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for q in qs:
            if q.exact_start:
                s = datetime.fromisoformat(q.exact_start)
                e = s + timedelta(minutes=q.duration_minutes)
                calls.append(ToolCall(
                    tool="calendar.check_free",
                    args={"start_iso": s.isoformat(), "end_iso": e.isoformat()},
                    purpose="verify_exact_time",
                ))
            calls.append(ToolCall(
                tool="calendar.search_free",
                args={
                    "window_start": q.window_start,
                    "window_end": q.window_end,
                    "duration_minutes": q.duration_minutes,
                    "n": 8,
                    "daypart": q.daypart_hint,
                },
                purpose="find_free_in_window",
            ))
        return calls

    all_available_slots: List[CandidateSlot] = []
    if plan.intent == "meeting" and plan.calendar_queries:
        initial_calls = _queries_to_calls(plan.calendar_queries)
        results: Dict[str, Any] = {}
        for idx, c in enumerate(initial_calls):
            name, res = _exec_tool_call(c, z, thread_id=email.thread_id or None, email_envelope=email)
            results[f"call_{idx+1}:{name}"] = res
        tool_rounds.append(ToolRound(proposed_calls=initial_calls, results=results))

        agg: List[CandidateSlot] = []
        for rnd in tool_rounds:
            for _k, r in rnd.results.items():
                if isinstance(r, dict) and "slots" in r and isinstance(r["slots"], list):
                    for s in r["slots"]:
                        try:
                            agg.append(CandidateSlot(start=s["start"], end=s["end"], reason=s.get("reason", "tool_slot")))
                        except Exception:
                            pass
                if isinstance(r, dict) and r.get("free") is True:
                    for c in rnd.proposed_calls:
                        if c.tool == "calendar.check_free":
                            s = c.args.get("start_iso"); e = c.args.get("end_iso")
                            if s and e:
                                agg.append(CandidateSlot(start=s, end=e, reason="exact"))
        seen = set()
        uniq: List[CandidateSlot] = []
        for s in agg:
            key = (s.start, s.end)
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        all_available_slots = uniq

    # 2) Decide + draft (with user_confirmed & has_questions)
    final_action_raw = llm([_prompt_coordinator_action(
        email, plan, all_available_slots, tz,
        semantic_docs=sem_docs,
        user_confirmed=bool(plan.user_confirmed),
        has_questions=bool(plan.has_questions),
    )])

    b_raw = final_action_raw.get("booking_decision", {})
    booking = BookingDecision(
        action=b_raw.get("action", "none"),
        chosen_slot=(CandidateSlot(**b_raw["chosen_slot"]) if b_raw.get("chosen_slot") else None),
        location_hint=b_raw.get("location_hint"),
        internal_notes=b_raw.get("internal_notes"),
    )
    d_raw = final_action_raw.get("draft_result", {})
    # polish the LLM html with signature + optional “times” block
    polished_html = _polish_email_html(
        d_raw.get("html", "<p>Thanks for your note — here’s a quick summary below.</p>"),
        all_available_slots
    )
    draft = DraftResult(
        subject=d_raw.get("subject", f"Re: {email.subject}"),
        html=polished_html,
    )

    # Execute (optional writes)
    created_ics: Optional[Tuple[str, bytes]] = None
    if allow_calendar_writes and booking.action in ("hold", "event") and booking.chosen_slot:
        try:
            s_iso = booking.chosen_slot.start
            e_iso = booking.chosen_slot.end

            if booking.action == "hold":
                ev = create_hold(
                    start_iso=s_iso,
                    end_iso=e_iso,
                    thread_id=email.thread_id or None,
                    prospect_email=email.from_addr or None,
                    attendees=[{"email": _bare_email(email.from_addr)}] if email.from_addr else [],
                    title="Ecodia — intro chat (HOLD)",
                    description="Soft hold to make it easy. Click ‘Yes’ on the invite or reply to confirm.",
                    tz=tz,
                    send_updates="none",
                )
                created_ics = build_ics(ICSSpec(
                    start=s_iso,
                    end=e_iso,
                    summary=ev.get("summary") or "Ecodia — intro chat",
                    description=ev.get("description") or "",
                    location=booking.location_hint or "",
                    attendee_email=(email.from_addr or None),
                    method="REQUEST",
                ))
                log.info("Created HOLD + ICS(REQUEST) for %s", s_iso)

            elif booking.action == "event":
                create_event(
                    title="Ecodia — intro chat",
                    description="Looking forward to our chat! Please click 'Yes' to confirm.",
                    start_iso=s_iso,
                    end_iso=e_iso,
                    tz=tz,
                    attendees=[{"email": _bare_email(email.from_addr)}] if email.from_addr else [],
                    thread_id=email.thread_id or None,
                    prospect_email=email.from_addr or None,
                    kind="CONFIRMED",
                    send_updates="all",
                )
                log.info("Google invite sent for %s by create_event", s_iso)

        except Exception:
            log.exception("calendar write/ICS build failed")

    return FlowOutput(
        plan=plan,
        tool_rounds=tool_rounds,
        booking=booking,
        draft=draft,
        ics=created_ics,
        semantic_context=sem_docs,
    )
