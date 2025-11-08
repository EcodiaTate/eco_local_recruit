# recruiting/sender.py
from __future__ import annotations

import mimetypes
import os
import time
import uuid
import socket
import pathlib
import logging
import urllib.parse
from typing import List, Tuple, Optional, Dict, Union, Mapping
from email.message import EmailMessage
from email.utils import parseaddr

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import settings

# Optional: allow typing against your ICSSpec without strict import-time cycles
try:
    from .calendar_client import ICSSpec as _ICSSpec  # type: ignore
    from .calendar_client import build_ics as _build_ics_file  # type: ignore
except Exception:
    _ICSSpec = None
    _build_ics_file = None  # type: ignore

log = logging.getLogger(__name__)

_ses = boto3.client("ses", region_name=settings.SES_REGION)

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _bare(addr: str) -> str:
    return parseaddr(addr)[1] or addr

def _make_message_id(from_addr: str) -> str:
    domain = (from_addr.split("@", 1)[1] if "@" in from_addr else socket.getfqdn()) or "localhost"
    return f"<{int(time.time())}.{uuid.uuid4().hex}@{domain}>"

def _save_eml_if_enabled(raw_bytes: bytes, subject: str) -> None:
    if os.getenv("ECO_LOCAL_SAVE_EML", "").lower() not in {"1", "true", "yes"}:
        return
    outdir = pathlib.Path(os.getenv("ECO_LOCAL_EML_DIR", "./outbox_debug"))
    outdir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in (subject or "") if ch.isalnum() or ch in (" ", "-", "_")).strip()[:80] or "email"
    path = outdir / f"{int(time.time())}_{safe}.eml"
    try:
        path.write_bytes(raw_bytes)
        log.info("[ses] wrote debug EML: %s", str(path))
    except Exception:
        log.exception("[ses] failed to write debug EML")

def _coerce_ics(ics: Optional[Union[Tuple[str, bytes], object]]) -> Optional[Tuple[str, bytes]]:
    """
    Accept either (filename, bytes) or an ICSSpec-like object and return (filename, bytes).
    If ICSSpec is provided but build_ics isn't available, we skip attaching.
    """
    if not ics:
        return None
    if isinstance(ics, tuple) and len(ics) == 2 and isinstance(ics[0], str) and isinstance(ics[1], (bytes, bytearray)):
        return (ics[0], bytes(ics[1]))
    if _ICSSpec and _build_ics_file and hasattr(ics, "start") and hasattr(ics, "end"):
        try:
            fn, content = _build_ics_file(ics)  # type: ignore[arg-type]
            return (fn, content)
        except Exception:
            return None
    return None
# recruiting/sender.py  (only the changed parts shown; replace the helper)
from .unsub_links import build_unsub_url  # NEW

def _default_list_unsub_values(to_email: str) -> tuple[Optional[str], Optional[str]]:
    """
    Produce RFC-compliant List-Unsubscribe values.
    URL is a signed one-click link; mailto falls back to reply-to or sender.
    """
    url = None
    try:
        url = build_unsub_url(email=to_email, thread_id=None)  # signed: e,ts,sig
    except Exception:
        url = None

    mailto_target = getattr(settings, "ECO_LOCAL_REPLY_TO", None) or settings.SES_SENDER_EMAIL
    mailto = f"mailto:{mailto_target}?subject=unsubscribe"
    return url, mailto


# ─────────────────────────────────────────────────────────
# MIME construction
# ─────────────────────────────────────────────────────────

def _add_calendar_alternative(alt_container: EmailMessage, *, ics_bytes: bytes) -> None:
    alt_container.add_attachment(
        ics_bytes,
        maintype="text",
        subtype="calendar",
        cte="base64",
        params={"method": "REQUEST", "charset": "UTF-8"},
        filename="invite.ics",
    )

def _attach_calendar_part(container: EmailMessage, *, ics_name: str, ics_bytes: bytes) -> None:
    cal = EmailMessage()
    cal.set_content(
        ics_bytes,
        maintype="text",
        subtype="calendar",
        cte="base64",
        params={"method": "REQUEST", "charset": "UTF-8"},
    )
    cal["Content-Disposition"] = f'attachment; filename="{ics_name}"'
    container.attach(cal)

def _build_mime(
    *,
    to: str,
    subject: str,
    html: str,
    attachments: Optional[List[Tuple[str, str, bytes]]],
    inline_images: Optional[List[Dict[str, bytes | str]]],
    reply_to_message_id: Optional[str],
    references: Optional[List[str]],
    ics: Optional[Union[Tuple[str, bytes], object]],
    list_unsubscribe_url: Optional[str] = None,
    list_unsubscribe_mailto: Optional[str] = None,
    extra_headers: Optional[Mapping[str, str]] = None,   # ← NEW: arbitrary custom headers
) -> bytes:
    """
    Build a message with:
      root: multipart/mixed
        - (if inline) multipart/related
            - multipart/alternative (text/plain + text/html)
            - inline images (Content-ID)
          (else) multipart/alternative directly under root
        - attachments (incl. optional ICS)

    Adds RFC-compliant List-Unsubscribe headers:
      List-Unsubscribe: <https://...>, <mailto:...>
      List-Unsubscribe-Post: List-Unsubscribe=One-Click (only when https URL exists)
    """
    root = EmailMessage()
    root["Subject"] = subject
    root["From"] = settings.SES_SENDER_EMAIL
    root["To"] = to
    if getattr(settings, "ECO_LOCAL_REPLY_TO", None):
        root["Reply-To"] = settings.ECO_LOCAL_REPLY_TO

    root["Message-ID"] = _make_message_id(settings.SES_SENDER_EMAIL)

    # Threading headers
    if reply_to_message_id and "@" in reply_to_message_id:
        root["In-Reply-To"] = f"<{reply_to_message_id.strip('<>')}>"
    if references:
        refs_norm = " ".join(
            f"<{r.strip('<>')}>"
            for r in references
            if r and "@" in r
        )
        if refs_norm:
            root["References"] = refs_norm

    # List-Unsubscribe headers
    bare_to = _bare(to)
    url, mailto = (
        (list_unsubscribe_url, list_unsubscribe_mailto)
        if (list_unsubscribe_url or list_unsubscribe_mailto)
        else _default_list_unsub_values(bare_to)
    )
    parts = []
    if url:
        parts.append(f"<{url}>")
    if mailto:
        parts.append(f"<{mailto}>")
    if parts:
        # Avoid duplicates; overwrite if already present via extra_headers
        if root.get("List-Unsubscribe"):
            del root["List-Unsubscribe"]
        root["List-Unsubscribe"] = ", ".join(parts)
        # One-click only valid with HTTPS URL endpoint
        if url:
            if root.get("List-Unsubscribe-Post"):
                del root["List-Unsubscribe-Post"]
            root["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"

    # Apply any caller-supplied custom headers last (can overwrite above intentionally)
    if extra_headers:
        for k, v in extra_headers.items():
            if not k:
                continue
            if root.get(k):
                del root[k]
            root[k] = str(v)

    root.make_mixed()

    # multipart/alternative (text/plain + text/html)
    alt = EmailMessage()
    alt.set_content("This email includes an HTML version.", charset="utf-8")
    # Send HTML as base64 to avoid quoted-printable damage to attributes
    alt.add_alternative(html or "<html></html>", subtype="html", charset="utf-8", cte="base64")

    if inline_images:
        # Ensure multipart/related so CIDs resolve
        related = EmailMessage()
        related.make_related()
        related.attach(alt)

        for img in (inline_images or []):
            cid = (str(img.get("cid") or "inline").strip() or "inline").strip("<>")
            data: Optional[bytes] = None
            path = img.get("path")

            if "bytes" in img and isinstance(img["bytes"], (bytes, bytearray)):
                data = bytes(img["bytes"])
            elif path:
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except Exception:
                    data = None
            if not data:
                continue

            guessed = "image/png"
            if path:
                g = mimetypes.guess_type(path)[0]
                if g:
                    guessed = g
            maintype, subtype = guessed.split("/", 1)

            img_part = EmailMessage()
            img_part.set_content(
                data,
                maintype=maintype,
                subtype=subtype,
                cte="base64",
            )
            # Ensure single Content-Disposition header, set to inline
            if img_part.get("Content-Disposition"):
                del img_part["Content-Disposition"]
            img_part.add_header("Content-Disposition", "inline", filename=(path or f"{cid}.img"))
            img_part.add_header("Content-ID", f"<{cid}>")
            img_part.add_header("Content-Location", f"cid:{cid}")
            img_part.add_header("X-Attachment-Id", cid)

            related.attach(img_part)

        root.attach(related)
    else:
        root.attach(alt)

    # ICS both inline (alternative) and as an attachment so most clients behave well
    ics_tup = _coerce_ics(ics)
    if ics_tup:
        _add_calendar_alternative(alt, ics_bytes=ics_tup[1])
        _attach_calendar_part(root, ics_name=ics_tup[0], ics_bytes=ics_tup[1])

    for (fname, mime, content) in (attachments or []):
        maintype, subtype = (mime.split("/", 1) if "/" in mime else ("application", "octet-stream"))
        root.add_attachment(content, maintype=maintype, subtype=subtype, filename=fname)

    raw = root.as_bytes()
    _save_eml_if_enabled(raw, subject)
    return raw

# ─────────────────────────────────────────────────────────
# SES send
# ─────────────────────────────────────────────────────────

def send_email(
    *,
    to: str,
    subject: str,
    html: str,
    attachments: Optional[List[Tuple[str, str, bytes]]] = None,
    inline_images: Optional[List[Dict[str, bytes | str]]] = None,
    reply_to_message_id: Optional[str] = None,
    references: Optional[List[str]] = None,
    ics: Optional[Union[Tuple[str, bytes], object]] = None,
    # Allow callers to force specific List-Unsubscribe values
    list_unsubscribe_url: Optional[str] = None,
    list_unsubscribe_mailto: Optional[str] = None,
    # NEW: arbitrary additional headers (e.g., X-Campaign, List-ID, etc.)
    extra_headers: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Send via SES.
    - Raw path for anything that needs threading/attachments/ICS/inline images/custom headers.
    - Simple path for HTML-only mail (note: cannot stamp custom headers on simple path).
    Adds List-Unsubscribe headers automatically if possible.
    """
    try:
        bare_to = _bare(to)
        cfg_set = os.getenv("SES_CONFIGURATION_SET")
        bcc_debug = os.getenv("ECO_LOCAL_BCC_DEBUG")

        use_raw = bool(inline_images or attachments or reply_to_message_id or references or ics or extra_headers
                       or list_unsubscribe_url or list_unsubscribe_mailto)

        if use_raw:
            raw = _build_mime(
                to=to,
                subject=subject,
                html=html,
                attachments=attachments,
                inline_images=inline_images,
                reply_to_message_id=reply_to_message_id,
                references=references,
                ics=ics,
                list_unsubscribe_url=list_unsubscribe_url,
                list_unsubscribe_mailto=list_unsubscribe_mailto,
                extra_headers=extra_headers,
            )
            kwargs: Dict[str, Union[str, Dict, List]] = {
                "Source": settings.SES_SENDER_EMAIL,
                "Destinations": [bare_to] + ([bcc_debug] if bcc_debug else []),
                "RawMessage": {"Data": raw},
            }
            if cfg_set:
                kwargs["ConfigurationSetName"] = cfg_set
            resp = _ses.send_raw_email(**kwargs)
        else:
            msg = {
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Html": {"Data": html, "Charset": "UTF-8"}},
            }
            dest: Dict[str, List[str]] = {"ToAddresses": [bare_to]}
            if bcc_debug:
                dest["BccAddresses"] = [bcc_debug]
            kwargs2: Dict[str, Union[str, Dict, List]] = {
                "Source": settings.SES_SENDER_EMAIL,
                "Destination": dest,
                "Message": msg,
                "ReplyToAddresses": (
                    [settings.ECO_LOCAL_REPLY_TO] if getattr(settings, "ECO_LOCAL_REPLY_TO", None) else []
                ),
            }
            # NB: Simple path cannot set custom headers, so List-Unsubscribe will only be present on the raw path.
            if cfg_set:
                kwargs2["ConfigurationSetName"] = cfg_set
            resp = _ses.send_email(**kwargs2)

        mid = (resp or {}).get("MessageId", "")
        log.info(
            "[ses] send ok from=%s to=%s env_to=%s subject=%r message_id=%s",
            settings.SES_SENDER_EMAIL,
            to,
            bare_to,
            subject,
            mid,
        )
        return mid
    except (BotoCoreError, ClientError):
        log.exception("[ses] send failed to=%s subject=%r", to, subject)
        raise
