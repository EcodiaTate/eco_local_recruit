# eco_local/seed_cli.py
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# Load env vars from .env if present (works without python-dotenv too)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Local imports (seed_cli.py lives in /eco-local)
# ──────────────────────────────────────────────────────────────────────────────
# Use relative package imports so this works when invoked as:
#   python -m eco_local.seed_cli
# and also when imported by orchestrator_cli as `from ..seed_cli import ...`
from recruiting import store                   # eco_local/recruiting/store.py
from recruiting.config import settings         # eco_local/recruiting/config.py


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _norm_email(e: Optional[str]) -> Optional[str]:
    if not e:
        return None
    e = e.strip()
    return e.lower() if EMAIL_RE.match(e) else None

def _domain_from(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if "@" in v:
        v = v.split("@", 1)[1]
    # strip scheme/path
    v = re.sub(r"^https?://", "", v).split("/", 1)[0]
    return v.lstrip(".") or None


# ──────────────────────────────────────────────────────────────────────────────
# Public API used by orchestrator_cli
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProspectInput:
    name: Optional[str] = None
    email: Optional[str] = None
    domain: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    phone: Optional[str] = None
    source: Optional[str] = "seed"

def upsert_prospect(p: Any) -> Dict[str, Any]:
    """
    Idempotently upsert a Prospect. If email is present, we MERGE by email;
    otherwise we MERGE by domain.

    On CREATE we set:
      - id=randomUUID()
      - qualified=true
      - qualification_score=0.90
      - outreach_started=false
      - attempt_count=0

    Returns a dict with at least {id, email, domain, name}.
    """
    # Accept dataclass / object / dict
    try:
        data = asdict(p) if hasattr(p, "__dataclass_fields__") else {
            "name": getattr(p, "name", None),
            "email": getattr(p, "email", None),
            "domain": getattr(p, "domain", None),
            "city": getattr(p, "city", None),
            "state": getattr(p, "state", None),
            "country": getattr(p, "country", None),
            "phone": getattr(p, "phone", None),
            "source": getattr(p, "source", "seed"),
        }
    except Exception:
        data = dict(p or {})

    email = _norm_email(data.get("email"))
    domain = _domain_from(data.get("domain") or email)
    name = (data.get("name") or "").strip() or None
    city = (data.get("city") or "").strip() or None
    state = (data.get("state") or "").strip() or None
    country = (data.get("country") or "").strip() or None
    phone = (data.get("phone") or "").strip() or None
    source = (data.get("source") or "seed").strip() or "seed"

    # Ensure constraints (safe if already present)
    try:
        store.ensure_dedupe_constraints()
    except Exception:
        pass

    if email:
        cypher = """
        MERGE (p:Prospect {email: $email})
          ON CREATE SET
            p.id                   = randomUUID(),
            p.created_at           = datetime(),
            p.source               = $source,
            p.qualified            = true,
            p.qualification_score  = 0.90,
            p.qualification_reason = 'seed',
            p.outreach_started     = false,
            p.attempt_count        = 0
          ON MATCH SET
            p.source = coalesce(p.source, $source)
        SET
          p.name       = coalesce($name, p.name),
          p.domain     = coalesce($domain, p.domain),
          p.city       = coalesce($city, p.city),
          p.state      = coalesce($state, p.state),
          p.country    = coalesce($country, p.country),
          p.phone      = coalesce($phone, p.phone),
          p.updated_at = datetime()
        RETURN p.id as id, p.email as email, p.domain as domain, p.name as name
        """
        params = {
            "email": email,
            "domain": domain,
            "name": name,
            "city": city,
            "state": state,
            "country": country,
            "phone": phone,
            "source": source,
        }
    elif domain:
        cypher = """
        MERGE (p:Prospect {domain: $domain})
          ON CREATE SET
            p.id                   = randomUUID(),
            p.created_at           = datetime(),
            p.source               = $source,
            p.qualified            = true,
            p.qualification_score  = 0.90,
            p.qualification_reason = 'seed',
            p.outreach_started     = false,
            p.attempt_count        = 0
          ON MATCH SET
            p.source = coalesce(p.source, $source)
        SET
          p.name       = coalesce($name, p.name),
          p.city       = coalesce($city, p.city),
          p.state      = coalesce($state, p.state),
          p.country    = coalesce($country, p.country),
          p.phone      = coalesce($phone, p.phone),
          p.updated_at = datetime()
        RETURN p.id as id, p.email as email, p.domain as domain, p.name as name
        """
        params = {
            "domain": domain,
            "name": name,
            "city": city,
            "state": state,
            "country": country,
            "phone": phone,
            "source": source,
        }
    else:
        raise ValueError("upsert_prospect requires at least an email or a domain")

    rows = store._run(cypher, params) or []
    return dict(rows[0]) if rows else {"id": None, "email": email, "domain": domain, "name": name}

def qualify_basic(prospect_id: str, *, score: float, reason: str = "seed") -> Dict[str, Any]:
    """
    Minimal qualifier used by orchestrator_cli: attaches a numeric score & reason.
    Also ensures `qualified=true`.
    """
    score = float(score or 0.0)
    reason = (reason or "seed").strip()
    cypher = """
    MATCH (p:Prospect {id: $id})
    SET p.qualification_score = $score,
        p.qualification_reason = $reason,
        p.qualified            = true,
        p.updated_at           = datetime()
    RETURN p.id as id, p.qualification_score as score, p.qualification_reason as reason
    """
    rows = store._run(cypher, {"id": prospect_id, "score": score, "reason": reason}) or []
    return dict(rows[0]) if rows else {"id": prospect_id, "score": score, "reason": reason}


# ──────────────────────────────────────────────────────────────────────────────
# CLI (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _seed_default() -> Dict[str, Any]:
    """
    Seed a Prospect for tate@ecodia.au, idempotently, and give it a high score.
    """
    p = ProspectInput(
        name="Helen",
        email="tate@ecodia.au",
        domain="mic.qld.edu.au",
        city="Sunshine Coast",
        state="QLD",
        country="AU",
        phone=None,
        source="tate",
    )
    node = upsert_prospect(p)
    # Lift score so first-touch selection won’t filter it out.
    qualify_basic(node["id"], score=0.99, reason="seed_default")
    return node

def _cmd_seed_default(_: argparse.Namespace) -> None:
    node = _seed_default()
    print("[seed-default] id:", node.get("id"), "email:", node.get("email"), "domain:", node.get("domain"))

def _cmd_upsert(args: argparse.Namespace) -> None:
    p = ProspectInput(
        name=args.name,
        email=args.email,
        domain=args.domain,
        city=args.city,
        state=args.state,
        country=args.country,
        phone=args.phone,
        source=args.source or "seed",
    )
    node = upsert_prospect(p)
    print("[upsert] id:", node.get("id"), "email:", node.get("email"), "domain:", node.get("domain"))

def _cmd_qualify(args: argparse.Namespace) -> None:
    res = qualify_basic(args.id, score=args.score, reason=args.reason or "seed")
    print("[qualify] id:", res.get("id"), "score:", res.get("score"), "reason:", res.get("reason"))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ECO Local Seeder")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed-default", help="Seed default Prospect (tate@ecodia.au)")
    s.set_defaults(func=_cmd_seed_default)

    u = sub.add_parser("upsert", help="Upsert a Prospect (by email or domain)")
    u.add_argument("--name")
    u.add_argument("--email")
    u.add_argument("--domain")
    u.add_argument("--city")
    u.add_argument("--state")
    u.add_argument("--country")
    u.add_argument("--phone")
    u.add_argument("--source")
    u.set_defaults(func=_cmd_upsert)

    q = sub.add_parser("qualify", help="Set basic qualification score on a Prospect")
    q.add_argument("--id", required=True)
    q.add_argument("--score", type=float, required=True)
    q.add_argument("--reason", default="seed")
    q.set_defaults(func=_cmd_qualify)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    finally:
        try:
            store.close_driver()
        except Exception:
            pass

if __name__ == "__main__":
    main()
