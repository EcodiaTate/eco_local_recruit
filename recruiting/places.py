# recruiting/places.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

PLACES_KEY = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GENERAL_GOOGLE_API_KEY")

_USER_AGENT = "EcodiaEcoLocal/places (https://ecodia.au)"
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)


def _client() -> httpx.Client:
    return httpx.Client(timeout=_TIME_TIMEOUT(), headers={"User-Agent": _USER_AGENT})


def _TIME_TIMEOUT() -> httpx.Timeout:
    return _TIMEOUT


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


def text_search(query: str, city: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Google Places Text Search. Returns basic candidates with place_id and formatted address.
    """
    if not PLACES_KEY:
        return []
    q = f"{query} in {city}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": q,
        "key": PLACES_KEY,
        "region": "AU",
        "language": "en",
    }
    out: List[Dict[str, Any]] = []
    with _client() as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    for it in (data.get("results") or [])[:limit]:
        out.append({
            "name": it.get("name"),
            "place_id": it.get("place_id"),
            "formatted_address": it.get("formatted_address"),
            "rating": it.get("rating"),
            "user_ratings_total": it.get("user_ratings_total"),
            "types": it.get("types"),
        })
    return out


def place_details(place_id: str) -> Dict[str, Any]:
    """
    Place Details for website/phone/location. Email is NOT provided by Places.
    """
    if not PLACES_KEY:
        return {}
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": PLACES_KEY,
        "region": "AU",
        "language": "en",
        "fields": (
            "name,website,formatted_phone_number,international_phone_number,"
            "formatted_address,geometry,types,url"
        ),
    }
    with _client() as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    result = data.get("result") or {}
    # Normalise website → domain to simplify downstream matching
    website = (result.get("website") or "").strip() or None
    if website and website.startswith("http"):
        result["domain"] = _normalize_domain(website)
    else:
        result["domain"] = None
    return result
