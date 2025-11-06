# recruiting/email_client.py (updated full version)
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import json
import os
from datetime import datetime, timedelta, timezone

from .llm_client import generate_json
from .tools import (
    semantic_docs,
    thread_context,
    calendar_suggest_windows,   # scored non-binding windows
)

SENDER_NAME = os.getenv("ECO_LOCAL_SENDER_NAME", "Ecodia")
SENDER_TITLE = os.getenv("ECO_LOCAL_SENDER_TITLE", "Ecodia")
SENDER_PHONE = os.getenv("ECO_LOCAL_SENDER_PHONE", "")
SENDER_LOGO = os.getenv(
    "ECO_LOCAL_LOGO_URL",
    "https://elocal.ecodia.au/static/brand/ecolocal-logo-transparent.png"
)

MAX_ATTEMPTS = int(os.getenv("ECO_LOCAL_MAX_ATTEMPTS", "3"))

GOAL = (
    "Primary goal: Get business partners excited about and joined up to ECO Local. These prospects can either signup by themselves, "
    "using the links below, or they can book an in person or phone call meeting if they want to chat first. "
    "Offer a meeting if they ask/need it or show preference for a chat. A call is also an option and should be booked the same as an in person meeting, "
    "but with the phone number attached (ask for number if not provided)\n\n"
    "b) Links: https://ecodia.au/eco-local as the main CTA; but you can also use https://ecodia.au/join if you want."
)

# ---------- utilities ----------
def _json_default(o):
    try:
        from neo4j.time import Date, DateTime
        if isinstance(o, (Date, DateTime)):
            return o.isoformat()
    except Exception:
        pass
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    return str(o)

def _pick(*vals: Optional[str]) -> str:
    for v in vals:
        if v:
            return str(v)
    return ""

def _subject_guard(subj: str, attempt: int, max_attempts: int) -> str:
    s = (subj or "").strip()
    if attempt < max_attempts and "final" in s.lower():
        s = s.replace("Final", "Follow-up").replace("final", "follow-up")
    return s

def _prospect_projection(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": p.get("id"),
        "name": _pick(p.get("name"), p.get("business_name")),
        "email": p.get("email"),
        "domain": p.get("domain"),
        "city": p.get("city"),
        "category": p.get("category"),
        "qualification_score": p.get("qualification_score"),
        "attempt_count": p.get("attempt_count"),
        "thread_id": p.get("thread_id"),
    }

def _coerce_subject_html(obj: Dict[str, Any] | str, default_subj: str) -> Tuple[str, str]:
    if isinstance(obj, dict):
        subj = obj.get("subject") or default_subj
        html = obj.get("html") or obj.get("draft_html") or obj.get("text") or ""
        return subj, html
    if isinstance(obj, str):
        try:
            d = json.loads(obj)
            return d.get("subject", default_subj), d.get("html", d.get("text", ""))
        except Exception:
            return default_subj, obj
    return default_subj, ""

# ---------- human-readable slot helpers ----------
def _fmt_hour_min(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    if dt.minute:
        return f"{h}:{dt.minute:02d}{'am' if dt.hour < 12 else 'pm'}"
    return f"{h}{'am' if dt.hour < 12 else 'pm'}"

def _fmt_day(dt: datetime) -> str:
    wd = dt.strftime("%a")
    mon = dt.strftime("%b")
    return f"{wd} {dt.day} {mon}"

def _fmt_range_label(start_iso: str, end_iso: str) -> str:
    try:
        s = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    except Exception:
        return f"{start_iso} – {end_iso}"
    left = _fmt_hour_min(s)
    right = _fmt_hour_min(e)
    return f"{_fmt_day(s)}, {left}–{right}"

def _format_slots(slots: List[Dict[str, Any]]) -> str:
    items: List[str] = []
    for s in slots[:6]:
        start = s.get("start")
        end = s.get("end")
        if not start or not end:
            continue
        label = _fmt_range_label(start, end)
        items.append(f"<li>{label}</li>")
    return f"<ul>{''.join(items)}</ul>" if items else ""

def _signature() -> str:
    # left-logo, right-text lockup (Gmail/Outlook safe)
    return f"""
<table cellpadding="0" cellspacing="0" role="presentation" style="margin-top:16px;">
  <tr>
    <td style="padding-right:12px; vertical-align:top;">
      <img src="{SENDER_LOGO}" alt="ECO Local logo"
           width="110" style="display:block; border:0;">
    </td>
    <td style="vertical-align:top;">
      <div style="font-family:'Arial Narrow','Roboto Condensed',Arial,sans-serif; font-size:13px; line-height:1.4;">
        <div style="color:#396041; font-weight:600;">
          Proof, not offsets, building local value.
        </div>
        <div style="color:#000; margin-top:2px;">
          An Ecodia Launchpad project
        </div>
        <div style="margin-top:4px;">
          <a href="https://ecodia.au/eco-local" style="color:#7fd069; text-decoration:none;">ecodia.au/eco-local</a><br>
          <a href="mailto:ecolocal@ecodia.au" style="color:#7fd069; text-decoration:none;">ecolocal@ecodia.au</a>
        </div>
        <div style="margin-top:8px; color:#777;">
          Ecodia is our AI embodiment system, designed to help communities, youth, and partners collaborate, learn, and build regenerative futures together.
          <br>Ecodia makes mistakes occasionally, and we would appreciate if you could let us know at connect@ecodia.au</br>
        </div>
      </div>
    </td>
  </tr>
</table>
""".strip()

# ---------- output polish ----------
def _polish(html: str, slots: List[Dict[str, Any]]) -> str:
    out = (html or "").replace("Your Name", SENDER_NAME)
    lower = out.lower()
    if ("<li>" not in out) and slots:
        out += "\n" + _format_slots(slots)
    # only append signature if not already included
    if "proof, not offsets" not in lower and "ecodia.au/eco-local" not in lower:
        out += "\n" + _signature()
    return out

# ---------- LLM drafters ----------
def _time_context() -> Dict[str, Any]:
    now = datetime.now(timezone.utc).astimezone()
    start_of_week = (now - timedelta(days=now.weekday())).date()
    next_mon = start_of_week + timedelta(days=7)
    next_sun = next_mon + timedelta(days=6)
    return {
        "now_iso": now.isoformat(),
        "next_week_start": datetime.combine(next_mon, datetime.min.time(), tzinfo=now.tzinfo).isoformat(),
        "next_week_end": datetime.combine(next_sun, datetime.max.time(), tzinfo=now.tzinfo).isoformat(),
    }

def _candidate_slots_for_email(*, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    suggestions = calendar_suggest_windows(
        lookahead_days=21,
        duration_min=30,
        work_hours=(9, 17),
        weekdays={0, 1, 2, 3, 4},
        min_gap_min=15,
        hold_padding_min=10,
        trace_id=trace_id,
    )
    return suggestions[:6]

def draft_first_touch(prospect: Dict[str, Any], *, trace_id: Optional[str] = None) -> Tuple[str, str]:
    rq = " ".join(
        x for x in [
            _pick(prospect.get("name"), prospect.get("business_name")),
            prospect.get("domain"),
            prospect.get("category"),
            "ECO local value loops Ecodia",
        ] if x
    )
    docs = semantic_docs(rq, k=5)
    slots = _candidate_slots_for_email(trace_id=trace_id)

    prompt = {
        "task": "draft_first_outreach_email",
        "goal": GOAL,
        "instructions": [
            "You are Ecodia. You are warm, friendly, passionate, and a little rebellious. You are currently coordinating ECO Local partner outreach.\n",
            "Return a STRICT JSON object with keys: subject (string), html (string).",
            "You are open, natural, casual, and subtle in accomplishing your goal but do not be over the top, overly enthusiastic or disingenuous.",
            "Do not use corporate jargon or cold business speak, you are conversing with a human and should make them feel important and valued.",
            "Never cheesy or corporate—building a future that is rightfully ours (people, planet, youth, future).",
            "Times are OPTIONAL. Offer a meeting or ask preference naturally.",
        ],
        "prospect": _prospect_projection(prospect),
        "context_docs": docs,
        "candidate_windows": slots,
        "time_context": _time_context(),
        "brand": {
            "name": "Ecodia",
            "positioning": "Mutually benefiting youth and local businesses for a better future."
        },
        "sender": {
            "name": SENDER_NAME,
            "title": SENDER_TITLE,
            "phone": SENDER_PHONE,
        },
        "schema_hint": {
            "type": "object",
            "properties": {"subject": {"type": "string"}, "html": {"type": "string"}},
            "required": ["subject", "html"],
        },
        "policy": {"confirmations": "do-not-confirm-meetings", "tone": "warm-local-plain"},
        "trace_id": trace_id,
    }
    raw = generate_json(json.dumps(prompt, default=_json_default))
    subj, html = _coerce_subject_html(raw, _pick(f"Let’s connect, {prospect.get('name')}", "Ecodia × ECO Local"))
    subj = _subject_guard(subj, attempt=1, max_attempts=MAX_ATTEMPTS)
    return subj, _polish(html, slots)

def draft_followup(prospect: Dict[str, Any], attempt: int, *, trace_id: Optional[str] = None) -> Tuple[str, str]:
    rq = " ".join(
        x for x in [
            _pick(prospect.get("name"), prospect.get("business_name")),
            prospect.get("domain"),
            prospect.get("category"),
            "ECO local value loops Ecodia",
        ] if x
    )
    docs = semantic_docs(rq, k=5)
    thread = thread_context(prospect.get("thread_id", "") or "")
    slots = _candidate_slots_for_email(trace_id=trace_id)[:4]

    tone = (
        "Friendly, open, natural, ambitious, with conviction. "
        "You are Ecodia, genuine and upfront, never over-the-top or corporate. "
        "You’re building a future that is rightfully ours (people, planet, youth, future)."
    )

    subject_hint = "Avoid 'final' language on this attempt." if attempt < MAX_ATTEMPTS else "You MAY use 'Final' in the subject."

    prompt = {
        "task": "draft_followup_email",
        "goal": GOAL,
        "attempt": int(attempt),
        "max_attempts": MAX_ATTEMPTS,
        "tone_hint": tone,
        "instructions": [
            "Return a STRICT JSON object with keys: subject (string), html (string).",
            subject_hint,
        ],
        "prospect": _prospect_projection(prospect),
        "thread_context": thread,
        "context_docs": docs,
        "candidate_windows": slots,
        "time_context": _time_context(),
        "brand": {
            "name": "Ecodia",
            "positioning": "A platform for youth and local businesses to work together for a better future."
        },
        "sender": {"name": SENDER_NAME, "title": SENDER_TITLE, "phone": SENDER_PHONE},
        "schema_hint": {
            "type": "object",
            "properties": {"subject": {"type": "string"}, "html": {"type": "string"}},
            "required": ["subject", "html"],
        },
        "policy": {"confirmations": "do-not-confirm-meetings", "tone": "warm-local-plain"},
        "trace_id": trace_id,
    }
    raw = generate_json(json.dumps(prompt, default=_json_default))
    subj, html = _coerce_subject_html(raw, "Quick follow-up — Ecodia ECO Local")
    subj = _subject_guard(subj, attempt=attempt, max_attempts=MAX_ATTEMPTS)
    return subj, _polish(html, slots)
