from __future__ import annotations
import os

DEFAULT_LOGO_URL = "https://elocal.ecodia.au/static/brand/ecolocal-logo-transparent.png"

def header_logo_src_email() -> str:
    """
    For emails: always return an email-safe src (https:// or data:).
    - ECO_LOCAL_LOGO_URL respected when it starts with http(s) or data:
    - otherwise fall back to DEFAULT_LOGO_URL
    """
    raw = (os.getenv("ECO_LOCAL_LOGO_URL") or "").strip().strip('"').strip("'")
    if raw.startswith(("https://", "http://", "data:")):
        return raw
    return DEFAULT_LOGO_URL

# If you later want a non-email variant for web UI you can expand here:
def header_logo_src() -> str:
    return header_logo_src_email()
