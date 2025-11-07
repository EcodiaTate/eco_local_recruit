# branding.py
from __future__ import annotations
import os, pathlib

DEFAULT_LOGO_URL = "https://elocal.ecodia.au/static/brand/ecolocal-logo-transparent.png"

def header_logo_src_email() -> str:
    raw = (os.getenv("ECO_LOCAL_LOGO_URL") or "").strip().strip('"').strip("'")
    if raw.startswith(("https://", "http://", "data:")):
        return raw
    return DEFAULT_LOGO_URL

def email_logo_payload(mode_env: str = "ECO_LOCAL_EMAIL_INLINE"):
    """
    Returns (src, inline_images) for email use.
    - src is a string suitable for <img src="...">
    - inline_images is a list of {"cid","bytes","filename","content_type"} or None
    Modes: url | cid | none  (default url)
    """
    mode = (os.getenv(mode_env) or "url").strip().lower()
    if mode == "cid":
        path = pathlib.Path("/app/static/brand/ecolocal-logo-transparent.png")
        try:
            b = path.read_bytes()
            return ("cid:ecolocal-logo", [{
                "cid": "ecolocal-logo",
                "bytes": b,
                "filename": "ecolocal-logo.png",
                "content_type": "image/png",
            }])
        except Exception:
            # fallback to URL if bytes missing
            pass
    if mode == "none":
        return ("", None)
    return (header_logo_src_email(), None)

# utils for email html
import re

_CID_SIG = re.compile(r'src=["\']cid:ecolocal-logo["\']', re.I)

def normalize_sig_logo(html: str, *, is_inline_cid: bool) -> str:
    if is_inline_cid:
        return html
    return _CID_SIG.sub(f'src="{header_logo_src_email()}"', html)
