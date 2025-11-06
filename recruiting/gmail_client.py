from __future__ import annotations

import base64
import logging
import os
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Tuple, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import settings
from .sa_loader import load_sa_credentials

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
# We need modify scope so we can remove UNREAD and add labels.
_GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Label we add after processing (created if missing).
PROCESSED_LABEL_NAME = os.getenv("ECO_LOCAL_GMAIL_PROCESSED_LABEL", "ECO Local/Processed")

# Default label to fetch from (system label works by name: INBOX, SENT, etc.)
DEFAULT_FETCH_LABEL = os.getenv("ECO_LOCAL_GMAIL_FETCH_LABEL", "INBOX")

log = logging.getLogger(__name__)
_level = (getattr(logging, (os.getenv("ECO_LOCAL_LOG_LEVEL") or "INFO").upper(), logging.INFO))
if not logging.getLogger().handlers:
    logging.basicConfig(level=_level)
else:
    logging.getLogger().setLevel(_level)

# Cache name->id for labels
_LABEL_CACHE: Dict[str, str] = {}


# ─────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────
def _build_gmail_service():
    creds = load_sa_credentials(
        scopes=_GMAIL_SCOPES,
        subject=settings.GSUITE_IMPERSONATED_USER,
    )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ─────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────
def _decode_part_data(data: Optional[str]) -> str:
    if not data:
        return ""
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_bodies(payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Recursively walk payload parts to collect text/plain and text/html bodies.
    Returns (plain, html).
    """
    plain, html = "", ""
    if not payload:
        return plain, html

    mime = payload.get("mimeType", "") or ""
    body_data = (payload.get("body") or {}).get("data")
    parts = payload.get("parts") or []

    if parts:
        for p in parts:
            p_plain, p_html = _extract_bodies(p)
            # Prefer the *first* text/plain and text/html encountered
            if p_plain and not plain:
                plain = p_plain
            if p_html and not html:
                html = p_html
            if plain and html:
                break
        return plain, html

    # Single-part message
    if body_data:
        decoded = _decode_part_data(body_data)
        if mime.startswith("text/plain"):
            plain = decoded
        elif mime.startswith("text/html"):
            html = decoded

    return plain, html


def _headers_to_dict(headers: List[Dict[str, str]]) -> Dict[str, str]:
    return {h.get("name", "").lower(): h.get("value", "") for h in (headers or [])}


def _list_labels(svc) -> Dict[str, str]:
    global _LABEL_CACHE
    if _LABEL_CACHE:
        return _LABEL_CACHE
    res = svc.users().labels().list(userId="me").execute() or {}
    mapping: Dict[str, str] = {}
    for lab in res.get("labels", []):
        mapping[lab["name"]] = lab["id"]
    _LABEL_CACHE = mapping
    return mapping


def _ensure_label_id(svc, name: str) -> str:
    labs = _list_labels(svc)
    if name in labs:
        return labs[name]
    created = svc.users().labels().create(
        userId="me",
        body={"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"},
    ).execute()
    _LABEL_CACHE[name] = created["id"]
    log.info("[gmail] created label %r id=%s", name, created["id"])
    return created["id"]


# ─────────────────────────────────────────────────────────
# Public: fetch + mark helpers
# ─────────────────────────────────────────────────────────
def fetch_unseen_since(
    minutes: int = 65,
    label: str = DEFAULT_FETCH_LABEL,
    *,
    max_pages: int = 5,
    page_size: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch unread, recent messages for the impersonated user.
    Uses Gmail search syntax: is:unread newer_than:{minutes}m label:{label} -from:me

    Returns list of dicts with keys:
      id, gmail_id, message_id, thread_id, from, to, subject, snippet,
      body_text, body_html, internal_date (ms since epoch, str), internal_date_iso (ISO 8601),
      date (RFC-2822 if available)
    """
    svc = _build_gmail_service()

    query = f"is:unread newer_than:{int(minutes)}m -from:me"
    if label:
        query += f" label:{label}"

    results: List[Dict[str, Any]] = []
    page_token: Optional[str] = None
    pages = 0

    try:
        while True:
            if pages >= max_pages:
                break
            pages += 1

            resp = (
                svc.users()
                .messages()
                .list(userId="me", q=query, pageToken=page_token, maxResults=page_size)
                .execute()
                or {}
            )
            msgs = resp.get("messages", []) or []
            page_token = resp.get("nextPageToken")

            for m in msgs:
                try:
                    full = (
                        svc.users()
                        .messages()
                        .get(userId="me", id=m["id"], format="full")
                        .execute()
                        or {}
                    )
                except HttpError:
                    log.exception("[gmail] get message failed id=%s", m.get("id"))
                    continue

                payload = full.get("payload", {}) or {}
                headers = payload.get("headers", []) or []
                # ... after headers parsed:
                hdr = _headers_to_dict(headers)
                body_plain, body_html = _extract_bodies(payload)

                message_id_hdr = hdr.get("message-id", "")  # ← real RFC Message-ID

                internal_ms = full.get("internalDate", "")
                date_hdr = hdr.get("date", "")
                internal_iso = ""
                try:
                    if internal_ms:
                        ms = int(internal_ms)
                        from datetime import datetime, timezone as _tz
                        internal_iso = (
                            parsedate_to_datetime(date_hdr).isoformat()
                            if date_hdr else
                            datetime.fromtimestamp(ms/1000.0, tz=_tz.utc).astimezone().isoformat()
                        )
                    elif date_hdr:
                        internal_iso = parsedate_to_datetime(date_hdr).isoformat()
                except Exception:
                    internal_iso = ""

                results.append({
                    "id": full.get("id"),
                    "gmail_id": full.get("id"),
                    "message_id": full.get("id"),        # legacy
                    "rfc_message_id": message_id_hdr,    # ← NEW
                    "thread_id": full.get("threadId"),
                    "from": hdr.get("from",""),
                    "to": hdr.get("to",""),
                    "subject": hdr.get("subject",""),
                    "snippet": full.get("snippet",""),
                    "body_text": body_plain or "",
                    "body_html": body_html or "",
                    "internal_date": internal_ms,
                    "internal_date_iso": internal_iso,
                    "date": date_hdr,
                })


            if not page_token:
                break

    except HttpError as e:
        log.exception("[gmail] list/read HttpError: %s", e)

    log.info("[gmail] fetched unread=%d label=%s since=%dm", len(results), label or "(any)", minutes)
    return results


def mark_read(message_id: str) -> None:
    """Remove UNREAD from a message (no-op if already read)."""
    try:
        svc = _build_gmail_service()
        svc.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()
        log.debug("[gmail] mark_read id=%s", message_id)
    except HttpError:
        log.exception("[gmail] mark_read failed id=%s", message_id)


def mark_processed_message(message_id: str, *, add_label: Optional[str] = PROCESSED_LABEL_NAME) -> None:
    """
    Remove UNREAD and add the ECO Local/Processed label (created if missing).
    Safe to call multiple times.
    """
    try:
        svc = _build_gmail_service()
        add_ids: List[str] = []
        if add_label:
            lab_id = _ensure_label_id(svc, add_label)
            add_ids.append(lab_id)
        svc.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["UNREAD"], "addLabelIds": add_ids or None},
        ).execute()
        log.debug("[gmail] mark_processed id=%s label=%s", message_id, add_label)
    except HttpError:
        log.exception("[gmail] mark_processed failed id=%s", message_id)
