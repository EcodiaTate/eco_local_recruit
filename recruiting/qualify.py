# recruiting/qualify.py
from __future__ import annotations
import os, json, re
from typing import Any, Dict, Tuple, Optional

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")

_CLEAN_ENABLED = os.getenv("ECO_LOCAL_CLEAN_NAME", "1").strip().lower() in {"1","true","yes","on"}
_CLEAN_MAX = int(os.getenv("ECO_LOCAL_CLEAN_NAME_MAXLEN", "60"))

RUBRIC = """
You score ECO Local suitability for a SINGLE business.

If ANY KILLSWITCH triggers, STOP and output STRICT JSON with all components 0.0 and:
{"score":0.0,"reason":"Excluded: <why>","components":{"local_fit":0.0,"community_values":0.0,"youth_access":0.0,"offer_fit":0.0,"redemption_readiness":0.0,"site_quality":0.0}}

KILLSWITCH (hard exclude):
- Gambling-first: casino, pokies, betting, sportsbook, lotteries, tipping sites.
- Adult-only: strip clubs, escorts, sex shops; vape/tobacco-only; shisha lounges.
- Predatory finance: payday lenders, pawn-only, crypto exchanges/brokers, get-rich/MLM.
- Platforms/aggregators/“big web”: Uber Eats, DoorDash, Menulog, TripAdvisor, Booking, Expedia, Shopify/Wix/Squarespace THEMSELVES (listing pages of other businesses).
- Pure directories/SEO spam/parked or template placeholder sites, obvious scams.
- Major global fast-food/coffee chains (e.g., McDonald’s, KFC, Starbucks, Subway) UNLESS a local franchise page with local activity.
- Online-only stores with no local address/hours/visit flow.

Edge handling:
- Social-only pages for a real local venue are NOT killswitch; they lower site_quality/redemption_readiness.
- Bars/pubs with food/daytime trade can be eligible; liquor-only/nightclub-only is killswitch.

Output STRICT JSON only:
{
  "score": <0..1>,
  "reason": "<<=140 chars>>",
  "components": {
    "local_fit": <0..1>,
    "community_values": <0..1>,
    "youth_access": <0..1>,
    "offer_fit": <0..1>,
    "redemption_readiness": <0..1>,
    "site_quality": <0..1>,
    "notes": "<optional>"
  }
}

Use only provided fields; if unknown → lower band. Clamp [0,1].

WEIGHTS:
local_fit .35, community_values .25, youth_access .10,
offer_fit .15, redemption_readiness .10, site_quality .05

SCORE:
score = .35*local_fit + .25*community_values + .10*youth_access + .15*offer_fit + .10*redemption_readiness + .05*site_quality
Round to 2 decimals. Keep reason ≤140 chars. JSON only.
"""

# === New: name cleaning ===
CLEAN_NAME_SPEC = """
You are normalizing a BUSINESS DISPLAY NAME for a CRM list.

Goal: Return the concise official trading name (brand name) only.
- Max {max_len} characters.
- No city/suburb/region unless part of brand.
- No category descriptors (e.g., "restaurant", "cafe", "barber") unless brand includes them.
- No taglines, promos, pipes/dashes tails (e.g., " | Sunshine Coast", " – Best in Noosa").
- Keep legal suffix only if present in brand (Pty Ltd, Ltd); otherwise omit.
- Title-case sensible; preserve stylized caps if brand uses it (e.g., "iFixit", "eBike Co").
- Remove parentheses/brackets containing locations or promos.

Return STRICT JSON only:
{{"name":"<cleaned>"}}
"""

def _call_llm(system: str, user: str, *, json_mode: bool = False) -> Dict[str, Any]:
    # OpenAI first
    if _OPENAI_KEY:
        try:
            import requests
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": 0.0,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
                json=payload, timeout=45,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content) if json_mode else {"text": content}
        except Exception:
            pass
    # Gemini fallback
    if _GEMINI_KEY:
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={_GEMINI_KEY}"
            text = f"{system}\n---\n{user}"
            payload = {"contents":[{"parts":[{"text": text}]}]}
            if json_mode:
                payload["generationConfig"] = {"responseMimeType": "application/json"}
            r = requests.post(url, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            txt = (data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text") or "{}")
            return json.loads(txt) if json_mode else {"text": txt}
        except Exception:
            pass
    return {}

# Heuristic cleaner used when LLM off/unavailable
_SEP_RE = re.compile(r"\s*(\||–|—|-|:)\s*")
_LOC_PAREN_RE = re.compile(r"\s*[\(\[]\s*(sunshine coast|noosa|maroochydore|mooloolaba|caloundra|kawana|peregian|nambour|coolum|australia|qld)\s*[\)\]]\s*", re.I)
_BAD_TRAILERS = re.compile(r"\s*\b(sunshine coast|qld|australia|official site|home|menu|contact|book now|shop online)\b\s*$", re.I)
_GENERIC_CATS = re.compile(r"\b(restaurant|cafe|coffee|bakery|barber|butcher|florist|plumbing|electrical|landscap|cleaning|laundry|laundromat|vet|physio|yoga|martial arts|gym|studio|market|bookstore|fishmonger|hardware|tool hire|garden centre|greengrocer|brewery|distillery|e-?bike|repair|device|phone|computer|outdoor gear|delicatessen|deli|cheesemonger|vegan|vegetarian)\b", re.I)

def _tidy_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def _titlecase_soft(s: str) -> str:
    # Keep weird brand caps mostly intact; only normalize obvious all-caps
    if s.isupper():
        return s.title()
    return s

def _heuristic_clean_name(name: str) -> str:
    n = name or ""
    n = _tidy_spaces(n)
    # take leading segment before separators if trailing looks like location/tagline
    parts = _SEP_RE.split(n)
    if len(parts) >= 3:
        # parts like [lead, sep, tail, sep, tail2...]; keep first segment if tails look like fluff
        lead = parts[0]
        tail = " ".join(parts[2::2])  # join tails
        if _BAD_TRAILERS.search(tail) or len(lead) >= 2:
            n = lead
    # strip location parentheses
    n = _LOC_PAREN_RE.sub("", n)
    # strip obvious trailers
    n = _BAD_TRAILERS.sub("", n)
    n = _tidy_spaces(n)
    # if what's left is just a category word, keep original (avoid "Bakery")
    if _GENERIC_CATS.search(n) and len(n.split()) <= 3:
        pass
    # clamp length
    if len(n) > _CLEAN_MAX:
        n = n[:_CLEAN_MAX].rstrip()
    return _titlecase_soft(n)

def clean_business_name(*, name: Optional[str], domain: Optional[str], city: Optional[str], parsed: Dict[str, Any] | None = None) -> Optional[str]:
    """
    Returns a concise brand/trading name. Uses LLM if enabled/available; falls back to heuristic.
    """
    raw = (name or "").strip() or (domain or "").strip()
    if not raw:
        return None
    # fast heuristic pass first (cheap & good)
    base = _heuristic_clean_name(raw)

    if not _CLEAN_ENABLED:
        return base or raw

    # Try LLM for a final polish; include a couple of hints
    about  = (parsed or {}).get("about") or ""
    title  = (parsed or {}).get("title") or ""
    user = f"""
RAW_NAME: {raw}
DOMAIN: {domain or ""}
CITY: {city or ""}
HOMEPAGE_TITLE: {title}
ABOUT_SNIPPET: {about[:300]}
"""
    data = _call_llm(
        CLEAN_NAME_SPEC.format(max_len=_CLEAN_MAX),
        user,
        json_mode=True,
    ) or {}
    cleaned = (data.get("name") or "").strip()
    # Validate and clamp
    if cleaned:
        cleaned = _tidy_spaces(_LOC_PAREN_RE.sub("", cleaned))
        cleaned = _BAD_TRAILERS.sub("", cleaned)
        if len(cleaned) > _CLEAN_MAX:
            cleaned = cleaned[:_CLEAN_MAX].rstrip()
        # avoid returning just a generic category
        if _GENERIC_CATS.fullmatch(cleaned.lower()):
            return base or raw
        return cleaned
    return base or raw

def _safe_float(x: Any, default: float) -> float:
    try:
        f = float(x)
        if f != f:  # NaN
            return default
        return max(0.0, min(1.0, f))
    except Exception:
        return default

def _call_scoring_llm(system: str, user: str) -> Dict[str, Any]:
    # isolate to keep scoring durable if clean-name path changes
    return _call_llm(system, user, json_mode=True)

def score_prospect(prospect_node: Dict[str, Any], parsed: Dict[str, Any] | None) -> Tuple[float, str]:
    """
    Returns (score, reason) using the rubric weights.
    """
    p = parsed or {}
    name = prospect_node.get("name") or prospect_node.get("domain") or "Unknown"
    domain = prospect_node.get("domain") or ""
    city = prospect_node.get("city") or ""
    emails = ", ".join(p.get("emails") or [])
    phones = ", ".join(p.get("phones") or [])
    about  = p.get("about") or ""
    sust   = ", ".join(p.get("sustainability_signals") or [])

    user = f"""
NAME: {name}
CITY: {city}
DOMAIN: {domain}
EMAILS: {emails}
PHONES: {phones}
ABOUT: {about}
SUSTAINABILITY_SIGNALS: {sust}
"""

    data = _call_scoring_llm(RUBRIC, user) or {}
    comps = data.get("components") or {}

    local_fit            = _safe_float(comps.get("local_fit"), 0.6)
    community_values     = _safe_float(comps.get("community_values"), 0.5)
    youth_access         = _safe_float(comps.get("youth_access"), 0.5)
    offer_fit            = _safe_float(comps.get("offer_fit"), 0.5)
    redemption_readiness = _safe_float(comps.get("redemption_readiness"), 0.5)
    site_quality         = _safe_float(comps.get("site_quality"), 0.7)

    score = (
        0.35*local_fit
        + 0.25*community_values
        + 0.10*youth_access
        + 0.15*offer_fit
        + 0.10*redemption_readiness
        + 0.05*site_quality
    )
    score = round(score, 2)
    reason = (data.get("reason") or "Locally relevant with workable redemption path.").strip()[:140]
    return (score, reason)
