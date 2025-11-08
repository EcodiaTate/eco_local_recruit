# recruiting/unsub_link.py
from __future__ import annotations

import hmac, hashlib, os, time, urllib.parse
from typing import Optional, Dict

def _secret() -> bytes:
    key = os.getenv("UNSUB_SIGNING_SECRET") or os.getenv("ECO_LOCAL_UNSUB_SIGNING_SECRET")
    if not key:
        # Strongly recommended to set in env. We fall back so dev doesn't explode.
        key = "dev-only-override-me"
    return key.encode("utf-8")

def _base_url() -> str:
    # Prefer dash path to match Next: /eco-local/business/unsubscribe
    base = (
        os.getenv("UNSUB_BASE_URL")
        or os.getenv("ECO_LOCAL_UNSUB_BASE_URL")
        or "https://ecodia.au/eco-local/business/unsubscribe"
    )
    return base

def _canonical_string(email: str, ts: str) -> str:
    # Canonical form MUST match the server:
    # lowercased+trimmed email and integer seconds timestamp
    return f"{email.strip().lower()}|{ts}"

def _sign(email: str, ts: str) -> str:
    msg = _canonical_string(email, ts).encode("utf-8")
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()

def build_unsub_url(*, email: str, thread_id: Optional[str] = None, ts: Optional[int] = None) -> str:
    """
    Returns a fully signed one-click URL:
      <BASE>?e=<email>&ts=<unix>&sig=<hmac>[&t=<thread_id>]
    HMAC = sha256( secret, "<lower(email)>|<ts>" )
    NOTE: thread_id is NOT part of the signature (only attached as a non-auth param).
    """
    if not email:
        raise ValueError("email required")
    ts_val = int(ts or time.time())
    sig = _sign(email, str(ts_val))

    q = {
        "e": email,              # keep original case in URL (server lowercases)
        "ts": str(ts_val),
        "sig": sig,
    }
    if thread_id:
        q["t"] = thread_id  # optional, **not** signed

    qs = urllib.parse.urlencode(q, quote_via=urllib.parse.quote)
    return f"{_base_url()}?{qs}"

def build_list_unsub_headers(email: str) -> Dict[str, str]:
    """
    Optional convenience for your sender:
      - Adds List-Unsubscribe + List-Unsubscribe-Post (RFC 8058 one-click)
      - Includes a mailto fallback
    Env (optional):
      - ECO_LOCAL_UNSUB_MAILTO (default: ECOLocal@ecodia.au)
    """
    mailto = os.getenv("ECO_LOCAL_UNSUB_MAILTO", "ECOLocal@ecodia.au").strip()
    url = build_unsub_url(email=email)
    return {
        "List-Unsubscribe": f"<mailto:{mailto}?subject=unsubscribe>, <{url}>",
        "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
    }
