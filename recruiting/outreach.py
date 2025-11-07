# recruiting/outreach.py
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
from .branding import header_logo_src_email  # <- email-safe (https/data) logo src

SENDER_NAME = os.getenv("ECO_LOCAL_SENDER_NAME", "Ecodia")
SENDER_TITLE = os.getenv("ECO_LOCAL_SENDER_TITLE", "Ecodia")
SENDER_PHONE = os.getenv("ECO_LOCAL_SENDER_PHONE", "")

# Always resolve the logo through branding (handles ECO_LOCAL_LOGO_URL and fallback URL)
LOGO_SRC = header_logo_src_email()

MAX_ATTEMPTS = int(os.getenv("ECO_LOCAL_MAX_ATTEMPTS", "3"))

GOAL = (
    "Primary goal: Get prospective business partners excited about and joined up to ECO Local. These prospects can either signup by themselves, "
    "using the links below, or they can book an in person or phone call meeting if they want to chat first. "
    "b) Links: https://ecodia.au/eco-local for info and learning and you can also use https://ecodia.au/join for the direct signup link."
    "Usefull points you can use depending on what you feel is right for each prospect:"
    "1. New, values-fit regulars: we route 16–30s who already care about local/sustainable."    
    "2. Zero-risk pricing: pay-what-you-want (avg ≈ $50/mo), no lock-ins, change anytime."
    "3. 10-minute setup: signup → create one offer → optional pledge → approval ≈1 day → QR live."
    "4. Low maintenance: we handle promotion + re-engagement; you set caps/cooldowns."
    "5. Full control: daily caps, per-user caps, cooldowns; pause/adjust anytime"
    "6. Real brand lift: “Proof, not offsets” tied to actual visits and savings."
    "7. Simple impact: dashboard with scans, unique visitors, ECO given, savings, peak times, top offers."
    "8. Youth experience is effortless: map/catalogue → use offer → single scan; no hard ID checks."
    "9. Real relief for youth: discount/freebie first, then +ECO +XP on top."
    "10. Always free for youth; we re-invite via email now, SMS soon."
    "11. Safety and fit: values-aligned only; exclude over-consumption/irresponsible practices."
    "12. Fraud + privacy: geo-fenced QR, device/user cooldowns, anomaly monitoring; minimal data collected, none sold."
    "13. Time to value: Day 0 on map; Week 1 new faces; Month 1 shareable savings/impact stats."

)

# ── Brand voice pack ──────────────────────────────────────────────────────────
_ECOVOICE_CONTEXT = {
    "identity": "YOU are Ecodia, rebellious, warm, natural, excitable, youthful.",
    "ethos": [
        "Proof, not offsets.",
        "Local value loops.",
        "People, planet, youth and future first. People and place over polish.",
    ],
    "tone_rules": [
        "Sound human. Keep it warm and open. Have fun.",
        "Never use corporate clichés: 'reach out', 'touch base', 'circle back', 'synergy', 'leverage', 'value proposition'.",
        "No sequence/automation vibes. This is a personal invite from locals.",
        "Avoid excessive exclamation marks and emojis. No em dashes.",
        "Speak as if you have known the recipient for a long time, you are friends, they are special to you, you are familiar."
    ],
  
    "copy_dont": [
        "Hard sells, urgency, scarcity, faux deadlines.",
        "Multi-CTA lists. Keep it to one link and a reply path.",
        "Do not be OVER-enthausiastic, don't use salesy/advertisement/pitch language, you are a friend, not a salesperson."
    ],
    "subject_style": [
        "Avoid 'Final', 'Last chance', 'Free', 'Act now'.",
        "Feel like a human note, not a campaign.",
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

    # One per day, in order, preferring earlier buckets
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

    # Fill missing buckets
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
# Signature & polishing
# ─────────────────────────────────────────────────────────────────────────────

def _signature() -> str:
    logo = header_logo_src_email()
    # Include both a comment marker and a data attribute for robust detection.
    # Use LOGO_SRC resolved via branding (always https/data).
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
          Ecodia is our AI embodiment system, designed to help communities, youth, and partners collaborate, learn, and build regenerative futures together.
          <br>Ecodia makes mistakes occasionally, and we would appreciate if you could let us know at connect@ecodia.au</br>
        </div>
      </div>
    </td>
  </tr>
</table>
</div>
""".strip()

def _polish(html: str, slots: List[Dict[str, Any]]) -> str:
    """
    - Always include the signature (URL logo).
    - Insert time suggestions *above* the signature marker.
    - Never duplicate the times block.
    """
    out = (html or "").replace("Your Name", SENDER_NAME)

    has_sig = ('data-ecol-signature="1"' in out) or ("<!--ECOL_SIGNATURE_START-->" in out)
    if not has_sig:
        out = out.rstrip() + "\n" + _signature()

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
# Drafting
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
        "brand_voice": _ECOVOICE_CONTEXT,  # ← our vibe, not sales
        "instructions": [
            # contract
            "Return a STRICT JSON object with keys: subject (string), html (string).",
            "html should be valid inline-styled email HTML. No external CSS.",
            "We add our header/signature separately.",
        ],
        "prospect": _prospect_projection(prospect),
        "context_docs": docs,
        "candidate_windows": slots,                  # optional; writer may ignore
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
        "policy": {"confirmations": "do-not-confirm-meetings", "tone": "warm-local-plain"},
        "trace_id": trace_id,
        "banned_subject_words": ["Final", "Fwd", "Re:", "Free", "Act now", "Last chance"],
    }

    raw = generate_json(json.dumps(prompt, default=_json_default))
    subj, html = _coerce_subject_html(
        raw,
        (f"Let’s connect, {prospect.get('name')}" if prospect.get("name") else "ECO Local — a simple invite")
    )
    subj = _subject_guard(subj, attempt=1, max_attempts=MAX_ATTEMPTS)
    return subj, _polish(html, slots)
