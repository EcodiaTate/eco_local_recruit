# recruiting/profile.py
from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime
from .store import _run

def upsert_profile_for_prospect(prospect_node: Dict[str, Any], *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Persists a Profile node with parsed fields and links it to the Prospect.
    `extra` may include: parsed (dict from parse.extract_contacts), source (str)
    """
    p = extra or {}
    parsed = p.get("parsed") or {}
    source = p.get("source") or "places"
    pid = prospect_node["id"]

    cy = """
    MATCH (pros:Prospect {id:$pid})
    MERGE (prof:Profile {prospect_id: $pid})
      ON CREATE SET prof.id = randomUUID(), prof.created_at = datetime()
    SET prof.updated_at = datetime(),
        prof.name = coalesce($name, prof.name),
        prof.domain = coalesce($domain, prof.domain),
        prof.best_email = coalesce($best_email, prof.best_email),
        prof.emails = coalesce($emails, prof.emails),
        prof.phones = coalesce($phones, prof.phones),
        prof.address = coalesce($address, prof.address),
        prof.hours = coalesce($hours, prof.hours),
        prof.about = coalesce($about, prof.about),
        prof.social = coalesce($social, prof.social),
        prof.sustainability_signals = coalesce($sust, prof.sustainability_signals),
        prof.source = $source
    MERGE (pros)-[:HAS_PROFILE]->(prof)
    RETURN pros, prof
    """
    params = {
        "pid": pid,
        "name": parsed.get("name"),
        "domain": prospect_node.get("domain"),
        "best_email": parsed.get("best_email"),
        "emails": parsed.get("emails") or [],
        "phones": parsed.get("phones") or [],
        "address": parsed.get("address"),
        "hours": parsed.get("hours"),
        "about": parsed.get("about"),
        "social": parsed.get("social") or {},
        "sust": parsed.get("sustainability_signals") or [],
        "source": source,
    }
    rows = _run(cy, params)
    return {"prospect": rows[0]["pros"], "profile": rows[0]["prof"]}
