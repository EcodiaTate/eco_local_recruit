from __future__ import annotations
import os

try:
    import httpx  # optional; we fail open if not present
except Exception:
    httpx = None

# Default: point at a globally public URL you control
DEFAULT_LOGO_URL = "https://storage.googleapis.com/ecodia-brand-assets/ecolocal-logo-transparent.png"

def _looks_url(s: str) -> bool:
    return s.startswith(("https://", "http://", "data:"))

def _probe_ok(url: str, timeout: float = 2.0) -> bool:
    if not httpx:
        return True  # fail open
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as c:
            r = c.head(url)
            if r.status_code >= 400:
                r = c.get(url, headers={"Range": "bytes=0-0"})
            return 200 <= r.status_code < 400
    except Exception:
        return False

def header_logo_src_email() -> str:
    """
    Return an email-safe img src:
      - Prefer ECO_LOCAL_LOGO_URL if it's http(s)/data:
      - Optionally probe availability (ECO_LOCAL_LOGO_PROBE=1)
      - Otherwise fall back to DEFAULT_LOGO_URL
    """
    raw = (os.getenv("ECO_LOCAL_LOGO_URL") or "").strip().strip('"').strip("'")
    if _looks_url(raw):
        if os.getenv("ECO_LOCAL_LOGO_PROBE", "").lower() in {"1", "true", "yes"}:
            return raw if _probe_ok(raw) else DEFAULT_LOGO_URL
        return raw
    return DEFAULT_LOGO_URL

# Web UI variants can reuse the same
def header_logo_src() -> str:
    return header_logo_src_email()
