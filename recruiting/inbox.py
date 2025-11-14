# recruiting/inbox.py
from __future__ import annotations

import asyncio
import hashlib
import logging
import re

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from email.utils import parseaddr  # NEW: for bare email extraction

from .branding import header_logo_src_email
from .config import settings
from .gmail_client import fetch_unseen_since
from .sender import send_email
from .store import (
    _run,
    book_event,          # kept for future use
    log_inbound_email,
    log_reply_draft,
    mark_reply_won,
    mark_unsubscribe,
)
from .tools import match_prospect_by_thread
from .llm_flow import run_llm_flow, EmailEnvelope

log = logging.getLogger(__name__)
_level = getattr(logging, (os.getenv("ECO_LOCAL_LOG_LEVEL") or "INFO").upper(), logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=_level)
else:
    logging.getLogger().setLevel(_level)

# ──────────────────────────────────────────────────────────────────────────────
# Signature helpers (URL or CID)
# ──────────────────────────────────────────────────────────────────────────────

_LOGO_CID = "ecolocal-logo"


def _resolve_app_root() -> Path:
    try:
        return Path(os.getenv("APP_ROOT") or Path(__file__).resolve().parents[1])
    except Exception:
        return Path(__file__).resolve().parents[1]


def _load_logo_bytes_from_path(raw: str | None) -> Optional[bytes]:
    """
    If `raw` is a filesystem path (absolute or relative), read bytes.
    Relative paths resolve from APP_ROOT. Returns None if not found.
    """
    if not raw:
        return None
    try:
        p = Path(raw)
        if not p.is_absolute():
            p = _resolve_app_root() / raw.lstrip("/\\")
        if p.is_file():
            return p.read_bytes()
    except Exception:
        log.exception("[inbox] failed reading logo file from path=%s", raw)
    return None


def _detect_logo_source() -> dict:
    """
    Decide how we should render the logo in the signature.

    Precedence:
      - ECO_LOCAL_LOGO_URL starts with 'https://' or 'http://' -> public URL.
      - ECO_LOCAL_LOGO_URL starts with 'data:'                 -> data URL.
      - ECO_LOCAL_LOGO_URL is a path                           -> load bytes, use CID.
      - No env or unreadable path                              -> try default file as CID,
                                                                 else fallback to public URL.
    """
    raw = (os.getenv("ECO_LOCAL_LOGO_URL") or "").strip().strip('"').strip("'")

    if raw.startswith(("https://", "http://")):
        return {"use_cid": False, "cid": _LOGO_CID, "bytes": None, "public_src": raw}

    if raw.startswith("data:"):
        return {"use_cid": False, "cid": _LOGO_CID, "bytes": None, "public_src": raw}

    if raw:
        b = _load_logo_bytes_from_path(raw)
        if b:
            return {"use_cid": True, "cid": _LOGO_CID, "bytes": b, "public_src": None}
        log.warning("[inbox] ECO_LOCAL_LOGO_URL path not found or unreadable: %s", raw)

    default_path = str(_resolve_app_root() / "static" / "brand" / "ecolocal-logo-transparent.png")
    b = _load_logo_bytes_from_path(default_path)
    if b:
        return {"use_cid": True, "cid": _LOGO_CID, "bytes": b, "public_src": None}

    return {
        "use_cid": False,
        "cid": _LOGO_CID,
        "bytes": None,
        "public_src": "https://elocal.ecodia.au/static/brand/ecolocal-logo-transparent.png",
    }


_LOGO_SPEC = _detect_logo_source()


def signature_inline_images() -> List[dict]:
    """
    Inline images for the email send. Only include when using CID mode.
    """
    if _LOGO_SPEC.get("use_cid") and _LOGO_SPEC.get("bytes"):
        return [{"cid": _LOGO_CID, "bytes": _LOGO_SPEC["bytes"]}]
    return []


def _signature_html() -> str:
    logo = header_logo_src_email()
    return f"""
<table cellpadding="0" cellspacing="0" role="presentation" style="margin-top:16px;">
  <tr>
    <td style="padding-right:12px; vertical-align:top;">
      <img src="{logo}" alt="ECO Local logo" width="110" style="display:block; border:0;">
    </td>
    <td style="vertical-align:top;">
      <div style="font-family:'Arial Narrow','Roboto Condensed',Arial,sans-serif; font-size:13px; line-height:1.4;">
        <div style="color:#396041; font-weight:600;">Proof, not offsets, building local value.</div>
        <div style="color:#000; margin-top:2px;">An Ecodia Launchpad project</div>
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


def _append_signature_if_missing(html: str) -> str:
    low = (html or "").lower()
    if any(k in low for k in ("proof, not offsets", "ecodia.au/eco-local", "ecolocal@ecodia.au")):
        return html
    return (html or "") + "\n" + _signature_html()


def _to_html(s: str) -> str:
    """
    Convert plain text → HTML, or lightly post-process existing HTML.
    """
    t = (s or "").strip()
    if "<" in t and ">" in t:
        return _append_signature_if_missing(t)

    paras = [seg.strip() for seg in t.split("\n\n") if seg.strip()]
    if not paras:
        return _signature_html()

    def _wrap(ptext: str) -> str:
        return "<p>" + ptext.replace("\n", "<br>") + "</p>"

    if len(paras) == 1:
        return _append_signature_if_missing(_wrap(paras[0]))
    return _append_signature_if_missing("".join(_wrap(ptext) for ptext in paras))


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers for metadata
# ──────────────────────────────────────────────────────────────────────────────

def _to_dict_safe(x):
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    return getattr(x, "__dict__", {}) or {}


def _list_tool_rounds_safe(flow):
    rounds = getattr(flow, "tool_rounds", None) or []
    out = []
    for rnd in rounds:
        proposed = []
        for c in getattr(rnd, "proposed_calls", None) or []:
            proposed.append(_to_dict_safe(c))
        results = getattr(rnd, "results", None) or {}
        out.append({"proposed_calls": proposed, "results_keys": list(results.keys())})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Gmail idempotency helpers
# ──────────────────────────────────────────────────────────────────────────────

def _message_key(m: Dict[str, Any]) -> str:
    for k in ("id", "message_id", "gmail_id"):
        if m.get(k):
            return str(m[k])
    basis = f"{(m.get('thread_id') or m.get('threadId') or '').strip()}|{(m.get('date') or m.get('internal_date') or '').strip()}|{(m.get('subject') or '').strip()}"
    return "hash:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()


def _already_processed(external_id: str) -> bool:
    rows = _run("MATCH (ie:InboundEmail {external_id:$k}) RETURN 1 LIMIT 1", {"k": external_id})
    return bool(rows)


def _record_processed(external_id: str, thread_id: Optional[str]) -> None:
    _run(
        """
        MERGE (ie:InboundEmail {external_id:$k})
          ON CREATE SET ie.created_at = datetime()
        SET ie.thread_id = coalesce(ie.thread_id, $tid)
        """,
        {"k": external_id, "tid": (thread_id or "")},
    )


def _mark_gmail_processed_best_effort(message: Dict[str, Any]) -> None:
    mid = message.get("id") or message.get("message_id") or message.get("gmail_id")
    try:
        from .gmail_client import mark_processed_message  # type: ignore
        if mid:
            log.info("[inbox] marking Gmail processed id=%s", mid)
            mark_processed_message(mid)
            return
    except Exception:
        log.debug("[inbox] mark_processed_message not available")

    try:
        from .gmail_client import mark_read  # type: ignore
        if mid:
            log.info("[inbox] marking Gmail read id=%s", mid)
            mark_read(mid)
    except Exception:
        log.debug("[inbox] mark_read not available")


# ──────────────────────────────────────────────────────────────────────────────
# Envelope extraction for the LLM-first flow
# ──────────────────────────────────────────────────────────────────────────────

def _parse_iso(dt_value: Optional[str | int | float]) -> str:
    tz = ZoneInfo(settings.LOCAL_TZ or "Australia/Brisbane")
    if dt_value is None or dt_value == "":
        return datetime.now(tz).isoformat()
    try:
        if isinstance(dt_value, (int, float)):
            sec = float(dt_value) / (1000.0 if dt_value > 10_000_000_000 else 1.0)
            return datetime.fromtimestamp(sec, tz=tz).isoformat()
        s = str(dt_value)
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(tz).isoformat()
    except Exception:
        pass
    return datetime.now(tz).isoformat()


def _extract_envelope(message: Dict[str, Any]) -> EmailEnvelope:
    received_iso = (
        message.get("internal_date_iso")
        or message.get("date_iso")
        or _parse_iso(message.get("internal_date") or message.get("date"))
    )
    return EmailEnvelope(
        thread_id=(message.get("thread_id") or message.get("threadId") or "").strip(),
        # prefer RFC id for threading; fall back to Gmail id
        message_id=(message.get("rfc_message_id") or message.get("id") or message.get("message_id") or "").strip(),
        from_addr=(message.get("from") or message.get("sender") or "").strip(),
        to_addr=(message.get("to") or settings.GSUITE_IMPERSONATED_USER or "").strip(),
        subject=(message.get("subject") or "").strip(),
        received_at_iso=received_iso,
        plain_body=(message.get("body_text") or message.get("snippet") or "").strip(),
        html_body=message.get("body_html"),
        thread_text=None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Address helper
# ──────────────────────────────────────────────────────────────────────────────

def _bare_email(addr: Optional[str]) -> str:
    """
    Extract a bare email address from something like 'Name <user@example.com>'.
    Falls back to the original string if parsing fails.
    """
    if not addr:
        return ""
    email = (parseaddr(addr)[1] or "").strip()
    return email or addr.strip()


# ──────────────────────────────────────────────────────────────────────────────
# LLM-driven triage + reply
# ──────────────────────────────────────────────────────────────────────────────

def triage_and_update(message: Dict[str, Any]) -> Dict[str, Any]:
    trace_id = f"m-{int(time.time() * 1000)}"
    external_id = _message_key(message)
    thread_id = (message.get("thread_id") or message.get("threadId") or "").strip()

    log.info("[inbox] triage start external_id=%s thread=%s", external_id, thread_id)

    if _already_processed(external_id):
        log.info("[inbox] skip: already processed external_id=%s", external_id)
        return {"action": "skip_already_processed", "external_id": external_id}

    prospect = match_prospect_by_thread(message)

    _record_processed(external_id, thread_id)
    log.info("[inbox] recorded processed external_id=%s", external_id)

    inbound = {}
    try:
        if prospect:
            inbound = log_inbound_email(prospect, message)
    except Exception:
        log.exception("[inbox] failed to log inbound email trace=%s", trace_id)

    # Basic text-based unsubscribe detection
    body_lc = ((message.get("body_text") or "") + " " + (message.get("snippet") or "")).lower()
    if any(k in body_lc for k in ("unsubscribe me", "stop emailing", "remove me from")) and prospect:
        mark_unsubscribe(prospect)
        _mark_gmail_processed_best_effort(message)
        return {
            "action": "unsubscribe",
            "prospect": prospect.get("email"),
            "inbound": inbound,
            "external_id": external_id,
        }

    envelope = _extract_envelope(message)

    allow_writes = os.getenv("ECO_LOCAL_ALLOW_CAL_WRITES", "true").lower() in ("1", "true", "yes")
    tz = settings.LOCAL_TZ or "Australia/Brisbane"
    semantic_k = int(os.getenv("ECO_LOCAL_SEMANTIC_K", "8"))

    t0 = time.perf_counter()
    flow = None
    try:
        flow = run_llm_flow(
            email=envelope,
            tz=tz,
            allow_calendar_writes=allow_writes,
            max_tool_rounds=int(os.getenv("ECO_LOCAL_TOOL_ROUNDS", "3")),
            semantic_k=semantic_k,
        )
        log.info(
            "[inbox] flow finished: intent=%s action=%s",
            getattr(flow.plan, "intent", "?"),
            getattr(flow.booking, "action", "?"),
        )
    except Exception:
        log.exception("[inbox] LLM flow failed; skipping send")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not flow or not getattr(flow, "draft", None):
        _mark_gmail_processed_best_effort(message)
        return {"action": "llm_flow_failed", "external_id": external_id, "trace_id": trace_id}

    # ── Basic send fields ─────────────────────────────────────────────────────
    # ALWAYS reply to the inbound sender; prospect.email is for bookkeeping only.
    raw_from = (envelope.from_addr or "").strip()
    to_addr = _bare_email(raw_from)
    subject = (flow.draft.subject or "").strip()
    html_raw = (flow.draft.html or "").strip()
    plan_intent = getattr(flow.plan, "intent", None)
    plan_has_questions = bool(getattr(flow.plan, "has_questions", False))
    booking_action = getattr(flow.booking, "action", None)

    log.debug(
        "[inbox] draft from LLM: intent=%s has_questions=%s booking_action=%s "
        "subj=%r html_preview=%r",
        plan_intent,
        plan_has_questions,
        booking_action,
        subject,
        html_raw[:200],
    )

    # If the LLM effectively produced no body, treat as no-send.
    if not html_raw or _html_plain_len(html_raw) == 0:
        log.warning(
            "[inbox] suppressing reply: empty html from flow; trace=%s intent=%s",
            trace_id,
            plan_intent,
        )
        _mark_gmail_processed_best_effort(message)
        return {
            "action": "no_send",
            "reason": "empty_html_from_flow",
            "external_id": external_id,
            "trace_id": trace_id,
        }

    # Additional safety: 'other' + no explicit questions + no booking → don't auto-reply.
    if plan_intent == "other" and not plan_has_questions and booking_action in (None, "", "none"):
        log.info(
            "[inbox] suppressing auto-reply for 'other' intent with no questions; trace=%s",
            trace_id,
        )
        _mark_gmail_processed_best_effort(message)
        return {
            "action": "no_send",
            "reason": "other_intent_no_questions",
            "external_id": external_id,
            "trace_id": trace_id,
        }

    # If HTML is totally missing, we genuinely have nothing to send
    if not to_addr or not html_raw:
        log.warning(
            "[inbox] no valid reply produced (missing to or html); "
            "trace=%s raw_from=%r to_addr=%r subj=%r html_len=%d booking_action=%s plan_intent=%s",
            trace_id,
            raw_from,
            to_addr,
            subject,
            len(html_raw),
            getattr(flow.booking, "action", None) if flow else None,
            getattr(flow.plan, "intent", None) if flow else None,
        )
        _mark_gmail_processed_best_effort(message)
        return {
            "action": "no_send",
            "reason": "missing_to_or_html",
            "external_id": external_id,
            "trace_id": trace_id,
        }

    # Local subject fallback (belt-and-braces)
    if not subject:
        base = (envelope.subject or "").strip()
        if base and not base.lower().startswith("re:"):
            subject = f"Re: {base}"
        else:
            subject = base or "Quick reply from ECO Local"

    log.info(
        "[inbox] preparing reply trace=%s to=%r subject=%r booking_action=%s",
        trace_id,
        to_addr,
        subject,
        getattr(flow.booking, "action", None),
    )

    # Rephrase “attached” → “below” when there are no non-ICS attachments
    if "attached" in html_raw.lower() or "attachment" in html_raw.lower():
        html_raw = html_raw.replace("attached", "below").replace("Attachment", "Summary")

    html = _to_html(html_raw)
    ics_attachment = getattr(flow, "ics", None)

    # ── Persist semantic facts into reply metadata (short form) ───────────────
    try:
        sem = getattr(flow, "semantic_context", None) or []
        sem_meta = [
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "score": d.get("score"),
                "snippet_200": (d.get("snippet") or d.get("text") or "")[:200],
            }
            for d in sem
        ]

        meta = {
            "trace_id": trace_id,
            "plan": _to_dict_safe(getattr(flow, "plan", None)),
            "booking": _to_dict_safe(getattr(flow, "booking", None)),
            "tool_rounds": _list_tool_rounds_safe(flow),
            "semantic_context": sem_meta,
        }

        if (os.getenv("ECO_LOCAL_TRACE_SEMANTICS") or "").lower() in {"1", "true", "yes"}:
            log.info("[inbox] semantic_context k=%d", len(sem_meta))
            for i, d in enumerate(sem_meta, 1):
                log.info(
                    "  #%d  id=%s  score=%s  title=%s",
                    i,
                    d.get("id"),
                    d.get("score"),
                    (d.get("title") or "")[:120],
                )

        # This just logs to Neo; failure here should NEVER block sending
        if prospect:
            log_reply_draft(
                prospect=prospect,
                subject=subject,
                html=html,
                metadata=meta,
                thread_id=envelope.thread_id or "",
            )
    except Exception:
        log.exception("[inbox] failed to log reply draft")

    # ── Send email via SES ────────────────────────────────────────────────────
    ses_message_id = ""
    try:
        reply_mid = message.get("rfc_message_id") or envelope.message_id or None
        refs = [reply_mid] if reply_mid else None

        ses_message_id = send_email(
            to=to_addr,
            subject=subject,
            html=html,
            inline_images=signature_inline_images(),  # URL mode => [], CID mode => [{'cid','bytes'}]
            reply_to_message_id=reply_mid,
            references=refs,
            ics=ics_attachment,
            # These may be None; sender.py handles that.
            list_unsubscribe_url=getattr(flow, "unsubscribe_url", None),
            extra_headers=getattr(flow, "list_unsub_headers", None) or None,
        )
        log.info(
            "[inbox] sent reply to=%s subject=%s ses_message_id=%s",
            to_addr,
            subject,
            ses_message_id,
        )
    except Exception:
        log.exception("[inbox] failed to send reply email")

    # ── Mark WON based on LLM action only ─────────────────────────────────────
    try:
        if prospect and getattr(flow.booking, "action", None) == "event":
            mark_reply_won(prospect)
    except Exception:
        log.exception("[inbox] mark_reply_won failed")

    _mark_gmail_processed_best_effort(message)

    return {
        "action": "llm_flow",
        "external_id": external_id,
        "trace_id": trace_id,
        "thread_id": envelope.thread_id,
        "prospect": (prospect or {}).get("email"),
        "elapsed_ms": elapsed_ms,
        "plan": {
            "intent": getattr(flow.plan, "intent", None),
            "summary": getattr(flow.plan, "summary", None),
            "confidence": getattr(flow.plan, "confidence", None),
        },
        "booking": {
            "action": getattr(flow.booking, "action", None),
        },
        "ses_message_id": ses_message_id,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Fetch & poll
# ──────────────────────────────────────────────────────────────────────────────

def fetch_new_messages(minutes: int = 65, label: str = "INBOX") -> List[Dict[str, Any]]:
    return fetch_unseen_since(minutes=minutes, label=label)


def _sync_await(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        def _runner():
            return asyncio.run(coro)
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_runner).result()
    return asyncio.run(coro)


async def _poll_async() -> int:
    msgs = fetch_unseen_since(minutes=65, label="INBOX")
    processed = 0
    for m in msgs:
        try:
            triage_and_update(m)
        except Exception:
            log.exception("[inbox] triage failed on a message")
        processed += 1
    return processed


def hourly_inbox_poll() -> int:
    return _sync_await(_poll_async())

def _html_plain_len(s: str) -> int:
    """
    Very rough length of visible text. Used to avoid sending 'nothing but a signature'.
    """
    if not s:
        return 0
    txt = re.sub(r"<[^>]+>", " ", s)
    txt = re.sub(r"\s+", " ", txt)
    return len(txt.strip())
