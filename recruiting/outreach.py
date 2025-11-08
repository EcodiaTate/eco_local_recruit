# recruiting/outreach.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import json
import os
import re
from datetime import datetime, timedelta, timezone

from .llm_client import generate_json
from .tools import (
    semantic_docs,
    thread_context,
    calendar_suggest_windows,   # scored non-binding windows
)
from .branding import header_logo_src_email  # email-safe (https/data) logo src
from .unsub_links import build_unsub_url     # unsubscribe URL builder

SENDER_NAME = os.getenv("ECO_LOCAL_SENDER_NAME", "Ecodia")
SENDER_TITLE = os.getenv("ECO_LOCAL_SENDER_TITLE", "Ecodia")
SENDER_PHONE = os.getenv("ECO_LOCAL_SENDER_PHONE", "")

# Always resolve the logo through branding (handles ECO_LOCAL_LOGO_URL and fallback URL)
LOGO_SRC = header_logo_src_email()

MAX_ATTEMPTS = int(os.getenv("ECO_LOCAL_MAX_ATTEMPTS", "3"))

GOAL = (
    "Our mission is to find and connect with amazing local businesses (like this prospect!) who are genuinely building a better future. "
    "We're a movement building 'ECO Local', real 'local value loops' where young people (16-30s) "
    "who are *actively looking* for sustainable and community-focused businesses can find and support them. "
    "Your main goal is to send a *personal invitation* to see if they want to be part of this. "
    "It's not a hard sell. It's a 'hey, we love what you're doing, and it feels like you'd be a perfect fit for the community we're building.' "
    "Make them feel *seen* and *invited*, not 'marketed to'. The vibe is community-building. "
    "The best next steps are for them to signup using the links below or for them to have a quick human intro chat. "
    "**The core offer is a personal chat with our founder, Tate.** "
    "They can just reply to this email, and you'll set it up. "
    "If they're super keen and want to self-serve, they can also learn more (https://ecodia.au/eco-local) or join directly (https://ecodia.au/join)."
)

# ── Brand voice pack ──────────────────────────────────────────────────────────
_ECOVOICE_CONTEXT = {
    "identity": "YOU are Ecodia. You're not a person, you're the *voice* of this movement—a passionate, warm, and slightly rebellious community builder. You're youthful, natural, and genuinely inspired by the local businesses you're talking to. **You are the facilitator making a human connection.**",
    "ethos": [
        "Proof, not offsets.",
        "Local value loops.",
        "People, planet, youth and future first. People and place over polish.",
    ],
    "tone_rules": [
        "Sound like a real person writing a one-off email. Be warm, open, and have a bit of a spark. Think 'friendly local' not 'corporate rep'.",
        "Never use corporate clichés: 'reach out', 'touch base', 'circle back', 'synergy', 'leverage', 'value proposition'.",
        "No sequence/automation vibes. This is a personal invite from locals.",
        "Avoid excessive exclamation marks and emojis. **No em dashes.**",
        "Speak *about* them like you're a fan of their business. You're inviting them because you genuinely think they're a great fit.",
        "**Leverage your identity.** You are the AI (Ecodia) connecting them to a human (our founder, Tate). Be clear that a reply to *you* sets up a *human* chat.",
    ],
    "copy_dont": [
        "Hard sells, urgency, scarcity, faux deadlines.",
        "Multi-CTA lists. Keep it to one link (or a reply) per email.",
        "Don't be vague. Be mission-driven, but make it clear what you're offering: ECO Local, a way to connect with a new generation of values-aligned customers."
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

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

_BANNED_SUBJECT_FRAGMENTS = [
    "join", "movement", "sustainable future", "sunshine coast", "limited", "act now",
    "last chance", "!", "🚀"
]

def _clean_subject(s: str) -> str:
    out = (s or "").strip()
    for w in _BANNED_SUBJECT_FRAGMENTS:
        out = out.replace(w, "").replace(w.title(), "").replace(w.upper(), "")
    out = " ".join(out.split())
    return out

def _subject_guard(subj: str, attempt: int, max_attempts: int) -> str:
    s = _clean_subject(subj)
    if attempt < max_attempts and "final" in s.lower():
        s = s.replace("Final", "Follow-up").replace("final", "follow-up")
    return s or "ECO Local"

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

def _ensure_obj_dict(x) -> dict:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            j = json.loads(x)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}

# ─────────────────────────────────────────────────────────────────────────────
# Time formatting & picking
# ─────────────────────────────────────────────────────────────────────────────

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

def _format_slots(slots: List[Dict[str, Any]]) -> str:
    items: List[str] = []
    for s in slots:
        start = s.get("start")
        end = s.get("end")
        if not start or not end:
            continue
        label = _fmt_range_label(start, end)
        items.append(f"<li>{label}</li>")
    return f"<ul>{''.join(items)}</ul>" if items else ""

def _time_bucket(dt: datetime) -> str:
    h = dt.hour
    if 9 <= h < 12:
        return "morning"
    if 12 <= h < 16:
        return "afternoon"
    return "late"

def _pick_varied_slots(raw: List[Dict[str, Any]], max_total: int = 3) -> List[Dict[str, Any]]:
    """
    Choose up to `max_total` slots, preferring:
      1) Different days,
      2) Different time-of-day buckets (morning/afternoon/late),
      3) Earliest remaining.
    """
    if not raw:
        return []

    norm: List[Dict[str, Any]] = []
    for s in raw:
        try:
            dt = datetime.fromisoformat(s["start"].replace("Z", "+00:00"))
        except Exception:
            continue
        norm.append({**s, "_start_dt": dt, "_day": dt.date().isoformat(), "_bucket": _time_bucket(dt)})

    by_day: Dict[str, List[Dict[str, Any]]] = {}
    for s in norm:
        by_day.setdefault(s["_day"], []).append(s)

    chosen: List[Dict[str, Any]] = []
    for day in sorted(by_day.keys()):
        day_slots = sorted(by_day[day], key=lambda x: ({"morning": 0, "afternoon": 1, "late": 2}[x["_bucket"]], x["_start_dt"]))
        if day_slots:
            chosen.append(day_slots[0])
            if len(chosen) >= max_total:
                break

    if len(chosen) >= max_total:
        return [{k: v for k, v in c.items() if not k.startswith("_")} for c in chosen]

    chosen_days = {c["_day"] for c in chosen}
    chosen_buckets = {c["_bucket"] for c in chosen}
    remaining = [s for s in norm if s["_day"] not in chosen_days or s["_bucket"] not in chosen_buckets]

    for b in ["morning", "afternoon", "late"]:
        if len(chosen) >= max_total:
            break
        if b in chosen_buckets:
            continue
        cand = [s for s in remaining if s["_bucket"] == b]
        cand.sort(key=lambda x: (x["_day"], x["_start_dt"]))
        if cand:
            chosen.append(cand[0])

    if len(chosen) >= max_total:
        return [{k: v for k, v in c.items() if not k.startswith("_")} for c in chosen]

    leftovers = [s for s in norm if s not in chosen]
    leftovers.sort(key=lambda x: x["_start_dt"])
    for s in leftovers:
        if len(chosen) >= max_total:
            break
        chosen.append(s)

    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in chosen]

def _render_times_block(slots: List[Dict[str, Any]]) -> str:
    if not slots:
        return ""
    return (
        '<div data-ecolocal-slots="1" style="margin:16px 0 8px; font-family:Arial,Helvetica,sans-serif;">'
        '<div style="font-weight:600; margin-bottom:6px;">A few times that could work:</div>'
        f"{_format_slots(slots)}"
        "</div>"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Unsubscribe helpers (body)
# ─────────────────────────────────────────────────────────────────────────────

def _unsubscribe_block(to_email: Optional[str]) -> str:
    mailto = os.getenv("ECO_LOCAL_REPLY_TO", "") or "ecolocal@ecodia.au"
    url = ""
    if to_email:
        try:
            url = build_unsub_url(email=to_email)  # includes e, ts, sig
        except Exception:
            url = ""

    html = (
        '<div data-ecol-unsub="1" style="margin-top:10px; font-size:12px; color:#666;">'
        "Don’t want these updates? "
    )
    if url:
        html += f'<a href="{url}" style="color:#6aa36f; text-decoration:none;">Unsubscribe</a>'
        html += " · "
    html += f'<a href="mailto:{mailto}?subject=unsubscribe" style="color:#6aa36f; text-decoration:none;">Email to unsubscribe</a>'
    html += "</div>"
    return html

# ─────────────────────────────────────────────────────────────────────────────
# Signature & polishing
# ─────────────────────────────────────────────────────────────────────────────

def _signature(to_email: Optional[str]) -> str:
    logo = header_logo_src_email()
    return f"""
<!--ECOL_SIGNATURE_START-->
<div data-ecol-signature="1">
<table cellpadding="0" cellspacing="0" role="presentation" style="margin-top:16px;">
  <tr>
    <td style="padding-right:12px; vertical-align:top;">
      <img src="{logo}" alt="ECO Local logo" width="110" style="display:block; border:0;">
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
          Ecodia helps communities, youth, and partners collaborate and build regenerative futures together.
        </div>
        {_unsubscribe_block(to_email)}
      </div>
    </td>
  </tr>
</table>
</div>
""".strip()

def _polish(html: str, slots: List[Dict[str, Any]], to_email: Optional[str]) -> str:
    """
    - Always include the signature (with unsubscribe block).
    - Insert time suggestions *above* the signature marker.
    - Never duplicate the times block.
    """
    out = (html or "").replace("Your Name", SENDER_NAME)

    has_sig = ('data-ecol-signature="1"' in out) or ("<!--ECOL_SIGNATURE_START-->" in out)
    if not has_sig:
        out = out.rstrip() + "\n" + _signature(to_email)

    # Insert times (3 varied) above the footer if not already present
    if 'data-ecolocal-slots="1"' not in out:
        times_html = _render_times_block(_pick_varied_slots(slots, max_total=3))
        if times_html:
            out = out.replace("<!--ECOL_SIGNATURE_START-->", times_html + "\n<!--ECOL_SIGNATURE_START-->")

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Slot sourcing
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_slots_for_email(*, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    # Pull more than needed, then choose 3 varied options.
    suggestions = calendar_suggest_windows(
        lookahead_days=21,
        duration_min=30,
        work_hours=(9, 17),
        weekdays={0, 1, 2, 3, 4},
        min_gap_min=15,
        hold_padding_min=10,
        trace_id=trace_id,
    )
    return _pick_varied_slots(suggestions, max_total=3)

# ─────────────────────────────────────────────────────────────────────────────
# Drafting — First touch
# ─────────────────────────────────────────────────────────────────────────────

def draft_first_touch(prospect: Dict[str, Any], *, trace_id: Optional[str] = None) -> Tuple[str, str]:
    rq = " ".join(
        x for x in [
            (prospect.get("name") or prospect.get("business_name") or ""),
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
        "brand_voice": _ECOVOICE_CONTEXT,
        "instructions": [
            "Return a STRICT JSON object with keys: subject (string), html (string).",
            "html should be valid inline-styled email HTML. No external CSS.",
            "We add our header/signature separately.",
            "Personalize the *reason* for the email. Use the prospect's 'name' and 'category' to make a genuine-sounding connection.",
        ],
        "prospect": _prospect_projection(prospect),
        "context_docs": docs,
        "candidate_windows": slots,
        "time_context": _time_context(),
        "brand": {
            "name": "Ecodia",
            "positioning": "Proof, not offsets - youth and local businesses building value together. Creating a future that is RIGHTFULLY OURS."
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
        "policy": {"confirmations": "do-not-confirm-meetings", "tone": "inspired-community-builder"},
        # Expanded bans to prevent “Join…” / hypey subjects
        "banned_subject_words": ["Final", "Fwd", "Re:", "Free", "Act now", "Last chance",
                                 "Join", "join", "Movement", "movement", "Sustainable", "!"],
    }

    try:
        raw = generate_json(json.dumps(prompt, default=_json_default))
        subj, html = _coerce_subject_html(
            raw,
            (f"Quick hello from ECO Local, {prospect.get('name')}" if prospect.get("name") else "ECO Local — quick hello")
        )
    except Exception:
        subj, html = ("ECO Local — quick hello", "<p>Hi there, would you be open to a short intro chat?</p>")

    subj = _subject_guard(subj, attempt=1, max_attempts=MAX_ATTEMPTS)
    to_email = prospect.get("email")
    return subj, _polish(html, slots, to_email)

# ─────────────────────────────────────────────────────────────────────────────
# Drafting — Follow-up (reply-shaped; resilient)
# ─────────────────────────────────────────────────────────────────────────────

def _rehydrate_subject_for_followup(tctx, attempt_no: int, max_attempts: int) -> str:
    """
    Try to restore a human-looking subject from the thread context.
    Falls back to sensible defaults by attempt number.
    """
    tctx = _ensure_obj_dict(tctx)

    raw = (tctx.get("subject") or tctx.get("last_outbound_subject") or "").strip()
    if not raw:
        if attempt_no == 1:
            return "Quick follow-up about ECO Local"
        if attempt_no >= max_attempts:
            return "Should I close the loop on this?"
        return "Following up"

    # Strip common prefixes
    clean = re.sub(r"^\s*(re|fwd):\s*", "", raw, flags=re.IGNORECASE)

    # For later attempts, “Re:” looks natural; first attempt should not look like a reply
    return f"Re: {clean}" if attempt_no > 1 else clean

def _short_guard_html(html: str, max_words: int = 130) -> str:
    words = html.split()
    if len(words) <= max_words:
        return html
    return " ".join(words[:max_words]) + "…"

def _compose_safe_followup_html(name: Optional[str], when_label: str, allow_pass: bool) -> str:
    first = name or "there"
    # NEW VIBE: Warm, human, referencing Tate, and not using banned phrases
    tail = "<br><br>No pressure at all if now's not the right moment." if allow_pass else ""
    return (
        f"<p>Hey {first},</p>"
        f"<p>Just wanted to float my last email to the top of your inbox. I know life gets busy!</p>"
        f"<p>Our founder, Tate, would still be really keen to chat for 15 mins about how ECO Local is connecting businesses like yours with values-driven young people. "
        f"Would {when_label} work for a quick intro?</p>"
        f"{tail}"
    )
def draft_followup(prospect: Dict[str, Any], *, attempt_no: int, max_attempts: int, trace_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Draft a no-reply followup. Tone varies by attempt:
      - Attempt 2 (first followup): Light, warm nudge.
      - Attempt 3 (second followup): Value add (one concrete benefit).
      - Attempt 4+ (final): Graceful final nudge, giving an easy 'out'.
    """
    try:
        tctx = thread_context({
            "thread_id": prospect.get("thread_id"),
            "to_email": prospect.get("email"),
            "subject_hint": "",
            "received_at_iso": datetime.now(timezone.utc).isoformat(),
            "plain_body": "",
        }) or {}
    except Exception:
        tctx = {}

    try:
        # NEW semantic query: More human, less "case study"
        rq = " ".join(
            x for x in [
                (prospect.get("name") or prospect.get("business_name") or ""),
                prospect.get("domain"),
                prospect.get("category"),
                "ECO Local youth support local business story", # <-- More human query
            ] if x
        )
        docs = semantic_docs(rq, k=2)
    except Exception:
        docs = []

    try:
        slots = _candidate_slots_for_email(trace_id=trace_id)
    except Exception:
        slots = []

    # Shifted attempt logic:
    # attempt_no 2 = friendly-nudge
    # attempt_no 3 = value-add
    # attempt_no 4+ = graceful-close
    style = (
        "friendly-nudge" if attempt_no == 2 else
        "value-add" if attempt_no == 3 else
        "graceful-close"
    )
    
    # We are now assuming the *first* follow-up is attempt #2
    # So we always want a 'Re:' subject for follow-ups.
    subj_hint = _rehydrate_subject_for_followup(tctx, attempt_no=attempt_no, max_attempts=max_attempts)
    if "Re:" not in subj_hint:
         subj_hint = f"Re: {subj_hint}"

    prompt = {
        "task": "draft_followup_email",
        # NEW GOAL: Aligned with the Ecodia -> Tate pipeline
        "goal": (
            "Gently follow up on our first invite. We haven't heard back, and we genuinely think they're a perfect fit. "
            "Remind them of the core mission (connecting values-aligned youth to businesses like them) and "
            "re-offer the simple CTA: a personal chat with our founder, Tate. "
            "This is a *personal* follow-up, not an automated sequence. Keep it human, warm, and very short (under 120 words)."
        ),
        "brand_voice": _ECOVOICE_CONTEXT, # <-- Using your locked-in voice
        "attempt": attempt_no,
        "max_attempts": max_attempts,
        "style": style,
        # NEW INSTRUCTIONS: Clearer, aligned with the brand, and not overly restrictive
        "instructions": [
            "Return a STRICT JSON object with keys: subject (string), html (string).",
            "The subject MUST be a reply (e.g., 'Re: ...') based on the subject_scaffold.",
            "Be Ecodia: the warm, passionate AI facilitator. You're trying to connect them to a human (Tate).",
            "Do NOT re-pitch the whole program. This is a gentle, personal nudge.",
            "Use ONE clear CTA: reply to this email, or pick a time to chat with Tate.",
            "**How to vary by attempt:**",
            "  - **style 'friendly-nudge' (attempt 2):** Short, warm, 'just floating this back up'.",
            "  - **style 'value-add' (attempt 3):** Short, warm, + ONE new piece of info from context_docs if available (e.g., 'I was just thinking about how you're a [category] and...').",
            "  - **style 'graceful-close' (attempt max):** Thank them, explicitly say this is the last note, and give them an easy 'out'. Be respectful of their time.",
            "Keep total words under 120."
        ],
        "subject_scaffold": subj_hint,
        "thread_context": (tctx or {}),
        "prospect": _prospect_projection(prospect),
        "context_docs": docs,
        "candidate_windows": slots,
        "time_context": _time_context(),
        "sender": {"name": SENDER_NAME, "title": SENDER_TITLE, "phone": SENDER_PHONE},
        "schema_hint": {
            "type": "object",
            "properties": {"subject": {"type": "string"}, "html": {"type": "string"}},
            "required": ["subject", "html"],
        },
    }

    try:
        raw = generate_json(json.dumps(prompt, default=_json_default))
        subj_model, html_model = _coerce_subject_html(raw, default_subj=subj_hint)
        subj = subj_hint  # lock to reply shape
        html = html_model
        # Ensure short, tidy body & polish
        html = _short_guard_html(_polish(html, slots, prospect.get("email")), max_words=130)
        return subj, html
    except Exception:
        # Resilient non-LLM path (using our NEW, on-brand fallback)
        when = "early next week"
        if slots:
             when = f"at one of the times below (or {when})"
        allow_pass = attempt_no >= max_attempts
        subj = subj_hint
        html = _compose_safe_followup_html(prospect.get("name"), when, allow_pass)
        html = _polish(html, slots, prospect.get("email"))
        return subj, html