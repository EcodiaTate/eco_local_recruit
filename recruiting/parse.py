# recruiting/parse.py
from __future__ import annotations
import os, re, json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")

_TEXT_EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)

# Providers we often see for small businesses; we keep them but rank domain emails higher.
_COMMON_FREE_PROVIDERS = (
    "@gmail.com", "@outlook.com", "@hotmail.com", "@yahoo.com", "@live.com",
)

def _regex_contacts(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text("\n", strip=True)
    emails = sorted(set(_TEXT_EMAIL_RE.findall(text)))
    phones = sorted(set(re.findall(r"(?:\+?\d[\d\s().-]{6,}\d)", text)))
    # JSON-LD extraction
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            blocks = data if isinstance(data, list) else [data]
            for b in blocks:
                if isinstance(b, dict):
                    em = b.get("email")
                    if em and isinstance(em, str):
                        emails.append(em)
                    # contactPoint can be list/dict
                    cp = b.get("contactPoint")
                    cps = cp if isinstance(cp, list) else [cp] if cp else []
                    for c in cps:
                        if isinstance(c, dict):
                            em2 = c.get("email")
                            if em2 and isinstance(em2, str):
                                emails.append(em2)
        except Exception:
            continue
    return {"emails": sorted(set(emails)), "phones": phones}


def _normalize_domain(url_or_domain: Optional[str]) -> Optional[str]:
    if not url_or_domain:
        return None
    if "://" in url_or_domain:
        u = urlparse(url_or_domain)
        host = (u.netloc or "").lower()
    else:
        host = (url_or_domain or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _email_rank(e: str, target_domain: Optional[str]) -> int:
    """
    Lower is better.
    0 → exact business domain
    1 → generic inbox on business domain (info@, hello@, contact@ ...)
    2 → staff/owner email on business domain
    3 → free provider (gmail/outlook/etc.)
    4 → other domain
    """
    el = e.lower()
    if not target_domain:
        # Prefer generics first if we don't know domain
        if el.startswith(("contact@", "hello@", "info@", "enquiries@", "admin@", "team@", "office@", "support@", "sales@")):
            return 1
        return 3 if any(el.endswith(p) for p in _COMMON_FREE_PROVIDERS) else 4

    if el.endswith("@" + target_domain):
        # Split generic vs personal
        if el.startswith(("contact@", "hello@", "info@", "enquiries@", "admin@", "team@", "office@", "support@", "sales@")):
            return 1
        # exact domain, but not generic
        return 2

    if any(el.endswith(p) for p in _COMMON_FREE_PROVIDERS):
        return 3
    # Other domain (e.g., accountant/host)
    return 4


def _pick_best_email(emails: List[str], domain: Optional[str]) -> Optional[str]:
    if not emails:
        return None
    domain = (domain or "").strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    ranked = sorted(emails, key=lambda e: _email_rank(e, domain))
    return ranked[0] if ranked else None


# --- Minimal LLM harness (OpenAI first, Gemini fallback) ----------------------
def _call_llm_struct(system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    # Try OpenAI JSON mode
    if _OPENAI_KEY:
        try:
            import requests
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4.1-mini",
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "temperature": 0.2,
                },
                timeout=60,
            )
            resp.raise_for_status()
            js = resp.json()
            content = js["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            pass
    # Try Gemini
    if _GEMINI_KEY:
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={_GEMINI_KEY}"
            payload = {
                "contents": [{"parts": [{"text": f"{system}\n\n---\n{user}"}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            }
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            text = (data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text") or "{}")
            return json.loads(text)
        except Exception:
            pass
    return {}


def extract_contacts(
    html: str | None,
    *,
    domain: str | None = None,
    extra_emails: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Returns a normalized dict:
    {
      "best_email": str|None,
      "emails": [..],
      "phones": [..],
      "name": str|None,
      "address": str|None,
      "hours": str|None,
      "about": str|None,
      "social": {"facebook":..., "instagram":..., "x":..., "linkedin":...},
      "sustainability_signals": [str]
    }
    """
    html = html or ""
    soup = BeautifulSoup(html, "html.parser")

    # Fast regex + JSON-LD baseline
    baseline = _regex_contacts(html)

    # Build compact, token-sane page sample for LLM
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    metas = " ".join([m.get("content", "") for m in soup.find_all("meta", attrs={"name": re.compile(r"description|og:description", re.I)})])
    body_sample = (soup.get_text("\n", strip=True) or "")[:4000]

    system = "You extract structured business contact & profile info for a local directory. Be accurate; return strict JSON."
    user = f"""
PAGE_TITLE: {title}
DOMAIN: {domain or ""}
META: {metas}
BODY_SAMPLE:
{body_sample}

Return JSON with keys:
best_email, emails[], phones[], name, address, hours, about,
social{{facebook,instagram,x,linkedin}}, sustainability_signals[].
Prefer 'contact'/'hello' style emails if no owner emails exist.
"""

    schema = {}  # model handles JSON format; we validate below
    data = _call_llm_struct(system, user, schema) or {}

    # Merge + normalize
    emails_raw = (data.get("emails") or []) + baseline.get("emails", []) + (extra_emails or [])
    # dedupe, keep order
    seen, ordered = set(), []
    for e in emails_raw:
        el = (e or "").strip().lower()
        if el and "@" in el and el not in seen:
            seen.add(el); ordered.append(e.strip())

    # Return both phones: LLM + baseline
    phones = sorted(set((data.get("phones") or []) + baseline.get("phones", [])))

    # Compute best email with ranking
    best_email = _pick_best_email(ordered, _normalize_domain(domain))

    social = data.get("social") or {}
    out = {
        "best_email": best_email,
        "emails": ordered,
        "phones": phones,
        "name": (data.get("name") or title or (domain or "")).strip() or None,
        "address": (data.get("address") or None),
        "hours": (data.get("hours") or None),
        "about": (data.get("about") or None),
        "social": {
            "facebook": social.get("facebook"),
            "instagram": social.get("instagram"),
            "x": social.get("x") or social.get("twitter"),
            "linkedin": social.get("linkedin"),
        },
        "sustainability_signals": data.get("sustainability_signals") or [],
        "raw_features": {"title": title, "meta": (metas or "")[:500]},
    }
    return out
    