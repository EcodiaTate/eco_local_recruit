# recruiting/qualify.py
from __future__ import annotations
import os, json
from typing import Any, Dict, Tuple

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")

RUBRIC = """
You score ECO Local suitability for a SINGLE business.

If ANY KILLSWITCH triggers, STOP and output STRICT JSON with all components 0.0 and:
{"score":0.0,"reason":"Excluded: <why>","components":{"local_fit":0.0,"community_values":0.0,"youth_access":0.0,"offer_fit":0.0,"redemption_readiness":0.0,"site_quality":0.0}}

KILLSWITCH (hard exclude):
- Gambling-first: casino, pokies, betting, sportsbook, lotteries, tipping sites.
- Adult-only: strip clubs, escorts, sex shops; vape/tobacco-only; shisha lounges.
- Predatory finance: payday lenders, pawn-only, crypto exchanges/brokers, get-rich/MLM.
- Platforms/aggregators/“big web”: Uber Eats, DoorDash, Menulog, TripAdvisor, Booking, Expedia, Shopify/Wix/Squarespace THEMSELVES (listing pages of other businesses). (Note: a café listed on these is OK; the platform as the “business” is excluded.)
- Pure directories/SEO spam/parked or template placeholder sites, obvious scams.
- Major global fast-food/coffee chains (e.g., McDonald’s, KFC, Starbucks, Subway) UNLESS the page clearly represents a locally owned franchise with local community activity (then allow, score accordingly).
- Online-only stores with no local address/hours/visit flow.

Edge handling:
- Social-only pages for a real local venue (FB/IG/Linktree) are NOT killswitch; they just lower site_quality/redemption_readiness.
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

Use only provided fields; if unknown → lower band. Clamp each to [0,1].

WEIGHTS:
local_fit .35, community_values .25, youth_access .10,
offer_fit .15, redemption_readiness .10, site_quality .05

BANDS (pick nearest):
local_fit: 1.0 local/indie + city ties; 0.8 small local chain/strong local; 0.6 nat’l but locally engaged; 0.4 nat’l minimal local; 0.2 online/weak; 0.0 none.
community_values: + for community/youth/local produce/reuse/inclusion; cap ≤0.2 on clear contra (gambling-first, payday, adult-only, exclusion). 1.0 strong & recent; 0.7 clear; 0.5 light; 0.2 vague/mixed; 0.0 none/contra.
youth_access: 1.0 clearly youth/family friendly; 0.7 generally welcoming; 0.5 neutral; 0.2 adult-leaning; 0.0 adult-only.
offer_fit: 1.0 explicit deals/small-ticket; 0.7 clear small-ticket; 0.5 plausible; 0.2 high-ticket only; 0.0 none.
redemption_readiness: 1.0 clear hours/address/contact + simple flow; 0.7 minor gaps; 0.5 thin but usable; 0.2 confusing; 0.0 chaotic.
site_quality: 1.0 working/https/clear info; 0.8 small gaps; 0.5 thin or social-only but current; 0.2 outdated/broken; 0.0 placeholder.

SCORE (exact):
score = .35*local_fit + .25*community_values + .10*youth_access + .15*offer_fit + .10*redemption_readiness + .05*site_quality
Round to 2 decimals. Keep reason plain, ≤140 chars. JSON only.

"""

def _call_llm(system: str, user: str) -> Dict[str, Any]:
    if _OPENAI_KEY:
        try:
            import requests
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {_OPENAI_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4.1-mini",
                    "response_format": {"type": "json_object"},
                    "messages": [{"role":"system","content":system},{"role":"user","content":user}],
                    "temperature": 0.0,
                },
                timeout=45,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:
            pass
    if _GEMINI_KEY:
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={_GEMINI_KEY}"
            payload = {
                "contents":[{"parts":[{"text": f"{system}\n---\n{user}"}]}],
                "generationConfig":{"responseMimeType":"application/json"}
            }
            r = requests.post(url, json=payload, timeout=45)
            r.raise_for_status()
            data = r.json()
            text = (data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text") or "{}")
            return json.loads(text)
        except Exception:
            pass
    return {}


def _safe_float(x: Any, default: float) -> float:
    try:
        f = float(x)
        if f != f:  # NaN
            return default
        return max(0.0, min(1.0, f))
    except Exception:
        return default


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

    data = _call_llm(RUBRIC, user) or {}
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
