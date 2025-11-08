# recruiting/referrals_jobs.py
from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import httpx

from .sender import send_email
from .llm_client import generate_json as _gen_json
from .store import _run  # executes Cypher and returns list[dict]
from .inbox import _signature_html as _signature_block  # rendered signature HTML
from .branding import header_logo_src_email  # unified URL/data resolver

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Config / constants
# ──────────────────────────────────────────────────────────────────────────────
BRIS_TZ = os.getenv("ECO_TZ", "Australia/Brisbane")
REFERRAL_BONUS_ECO = int(os.getenv("ECO_LOCAL_REFERRAL_BONUS", "50"))

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class YouthReferral:
    id: str
    submitted_at: datetime
    youth_id: str
    youth_name: str
    store_name: str
    location: str
    notes: Optional[str] = None

    website: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    place_id: Optional[str] = None

    # lifecycle / joins
    status: str = "submitted"
    outreach_sent_at: Optional[datetime] = None
    joined_at: Optional[datetime] = None
    partner_id: Optional[str] = None

_REF_FIELDS = {f.name for f in fields(YouthReferral)}
def _to_referral(d: Dict[str, Any]) -> YouthReferral:
    safe = {k: d.get(k) for k in _REF_FIELDS}
    return YouthReferral(**safe)  # type: ignore[arg-type]

# ──────────────────────────────────────────────────────────────────────────────
# Logo helpers (URL default, CID optional)
# ──────────────────────────────────────────────────────────────────────────────
def _app_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd()

_CID_SIG = re.compile(r'src=["\']cid:ecolocal-logo["\']', re.I)

def _normalize_signature_logo(html: str, *, is_inline_cid: bool) -> str:
    if is_inline_cid:
        return html
    url = header_logo_src_email()
    return _CID_SIG.sub(f'src="{url}"', html)

def _load_logo_bytes() -> Optional[bytes]:
    p = _app_root() / "static" / "brand" / "ecolocal-logo-transparent.png"
    try:
        return p.read_bytes() if p.is_file() else None
    except Exception:
        return None

def _pick_logo_mode() -> str:
    """
    none | url | cid
    Defaults to 'url' for real inboxes (HTTPS public).
    Dev override via ECO_LOCAL_REFERRALS_INLINE.
    """
    v = (os.getenv("ECO_LOCAL_REFERRALS_INLINE") or "").strip().lower()
    if v in ("none", "url", "cid"):
        return v
    return "url"

def _logo_src_for_email() -> Tuple[str, Optional[List[Dict[str, bytes | str]]]]:
    mode = _pick_logo_mode()
    if mode == "cid":
        b = _load_logo_bytes()
        if b:
            return "cid:ecolocal-logo", [{"cid": "ecolocal-logo", "bytes": b}]
    if mode == "none":
        return "", None
    return header_logo_src_email(), None

# ──────────────────────────────────────────────────────────────────────────────
# Constraints / CRUD (used by jobs and partner webhook)
# ──────────────────────────────────────────────────────────────────────────────
def create_constraints() -> None:
    _run("""
    CREATE CONSTRAINT referral_id IF NOT EXISTS
      FOR (r:YouthReferral) REQUIRE r.id IS UNIQUE;
    """)

def get_referral(ref_id: str) -> Optional[YouthReferral]:
    recs = _run("""MATCH (r:YouthReferral {id: $id}) RETURN r {.*} AS r""", {"id": ref_id})
    if not recs:
        return None
    return _to_referral(recs[0]["r"])

def list_unenriched(limit: int = 100) -> List[YouthReferral]:
    recs = _run("""
    MATCH (r:YouthReferral)
    WHERE r.status = 'submitted'
    RETURN r {.*} AS r
    ORDER BY r.submitted_at ASC
    LIMIT $limit
    """, {"limit": limit})
    return [_to_referral(r["r"]) for r in recs]

def update_referral(ref_id: str, updates: Dict[str, Any]) -> YouthReferral:
    sets = []
    params = {"id": ref_id}
    for k, v in updates.items():
        sets.append(f"r.{k} = ${k}")
        params[k] = v
    cy = "MATCH (r:YouthReferral {id: $id}) SET " + ", ".join(sets) + " RETURN r {.*} AS r"
    recs = _run(cy, params)
    return _to_referral(recs[0]["r"])

def list_referrals_needing_monthly_outreach(month_start: date, month_end: date) -> List[YouthReferral]:
    recs = _run("""
    MATCH (r:YouthReferral)
    WHERE date(r.submitted_at) >= date($start) AND date(r.submitted_at) <= date($end)
      AND r.status IN ['enriched', 'submitted']
    RETURN r {.*} AS r
    """, {"start": month_start.isoformat(), "end": month_end.isoformat()})
    return [_to_referral(r["r"]) for r in recs]

def mark_outreach_sent(ref_id: str) -> None:
    _run("""MATCH (r:YouthReferral {id: $id})
            SET r.status = 'outreach_sent', r.outreach_sent_at = datetime()""", {"id": ref_id})

def mark_joined_and_reward(ref_id: str, partner_id: str, bonus_eco: int) -> None:
    _run("""
    MATCH (r:YouthReferral {id: $ref_id})
    MERGE (p:BusinessPartner {id: $partner_id})
    SET r.status = 'joined', r.joined_at = datetime(), r.partner_id = $partner_id
    MERGE (y:User {id: r.youth_id})
    MERGE (y)-[:EARNED]->(:EcoTx {
      id: randomUUID(),
      created_at: datetime(),
      amount: $bonus_eco,
      kind: 'referral_bonus',
      note: 'Business joined via your referral',
      referral_id: r.id
    })
    """, {"ref_id": ref_id, "partner_id": partner_id, "bonus_eco": bonus_eco})

# ──────────────────────────────────────────────────────────────────────────────
# Enrichment via Google CSE & Places
# ──────────────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_PSE_KEY") or ""
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID") or os.getenv("GOOGLE_PSE_CX") or ""
GOOGLE_PLACES_KEY = os.getenv("GOOGLE_PLACES_KEY") or GOOGLE_API_KEY

async def cse_search(store_name: str, location: str) -> Dict[str, Any]:
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID):
        return {}
    q = f"{store_name} {location}"
    url = "https://www.googleapis.com/customsearch/v1"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params={"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": q, "num": 3})
        r.raise_for_status()
        data = r.json()
    items = data.get("items") or []
    best = items[0] if items else {}
    return {"website": (best.get("link") or None), "snippet": (best.get("snippet") or None)}

async def places_lookup(store_name: str, location: str) -> Dict[str, Any]:
    if not GOOGLE_PLACES_KEY:
        return {}
    q = f"{store_name} in {location}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params={"query": q, "key": GOOGLE_PLACES_KEY, "language": "en"})
        r.raise_for_status()
        data = r.json()
    results = data.get("results") or []
    if not results:
        return {}
    first = results[0]
    place_id = first.get("place_id")
    details = {}
    if place_id:
        det_url = "https://maps.googleapis.com/maps/api/place/details/json"
        async with httpx.AsyncClient(timeout=20) as client:
            d = await client.get(det_url, params={
                "key": GOOGLE_PLACES_KEY,
                "place_id": place_id,
                "fields": "formatted_phone_number,website"
            })
            d.raise_for_status()
            det = d.json().get("result") or {}
            details = {"phone": det.get("formatted_phone_number"), "website": det.get("website")}
    return {"place_id": place_id, **details}

async def enrich_one(r: YouthReferral) -> YouthReferral:
    cse = await cse_search(r.store_name, r.location)
    plc = await places_lookup(r.store_name, r.location)
    website = plc.get("website") or cse.get("website") or r.website
    phone   = plc.get("phone") or r.phone
    place_id = plc.get("place_id") or r.place_id
    status = "enriched" if website or phone else r.status
    return update_referral(r.id, {"website": website, "phone": phone, "place_id": place_id, "status": status})

async def enrich_pending(limit: int = 100) -> List[YouthReferral]:
    pending = list_unenriched(limit=limit)
    out = []
    for r in pending:
        try:
            out.append(await enrich_one(r))
        except Exception as e:
            log.warning("[referrals] enrich failed for %s: %s", r.id, e)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Outreach email
# ──────────────────────────────────────────────────────────────────────────────
def _brand_wrap(inner_html: str) -> Tuple[str, Optional[List[Dict[str, bytes | str]]]]:
    logo_src, inline = _logo_src_for_email()
    logo_img = (f'<img src="{logo_src}" alt="ECO Local" style="height:36px;display:block;">') if logo_src else ""
    html = f"""
<div style="background:#fafdfb;padding:24px 0;">
  <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial;color:#111;">
    <tr>
      <td align="center">
        <table role="presentation" cellpadding="0" cellspacing="0" width="640" style="width:640px;max-width:96%;background:#ffffff;border:1px solid #e6f2ea;border-radius:12px;box-shadow:0 1px 2px rgba(0,0,0,0.04);overflow:hidden;">
          <tr>
            <td style="padding:20px 24px 8px 24px;">
              {logo_img}
            </td>
          </tr>
          {inner_html}
          <tr>
            <td style="padding:8px 24px 20px 24px">
              <hr style="border:none;border-top:1px solid #eef5f0;margin:8px 0 16px 0;">
              {_signature_block()}
            </td>
          </tr>
        </table>
        <div style="font-size:12px;color:#71827a;margin-top:12px">You’re receiving this because locals asked us to invite you to ECO Local.</div>
      </td>
    </tr>
  </table>
</div>
""".strip()
    return html, inline

def _pill(text: str) -> str:
    return f"""<span style="display:inline-block;background:#e7f7ee;color:#104d2d;font-size:12px;line-height:1;border-radius:999px;padding:6px 10px;margin:2px 6px 2px 0;">{text}</span>"""

def _cta(href: str, label: str = "Join ECO Local") -> str:
    return f"""
  <a href="{href}"
     style="display:inline-block;text-decoration:none;background:#2da561;color:#fff;font-weight:600;padding:12px 16px;border-radius:10px;">
     {label}
  </a>
""".strip()

def _build_referral_email_html(
    to_name: str,
    business_name: str,
    location: str,
    youth_names: List[str],
    website: Optional[str]
) -> Tuple[str, Optional[List[Dict[str, bytes | str]]]]:
    youths = ", ".join(youth_names[:5]) + (" and others" if len(youth_names) > 5 else "")
    website_hint = f" (we noticed your site: {website})" if website else ""
    perks = "".join(_pill(p) for p in ["Locally owned", "Values-driven", "Youth-friendly perks", "Foot traffic"])
    join_url = "https://elocal.ecodia.au/partner/join"

    body = f"""
  <tr>
    <td style="padding:8px 24px 0 24px">
      <h1 style="font-size:20px;line-height:1.3;margin:0 0 6px 0;">Hi {to_name or business_name},</h1>
      <p style="margin:0 0 12px 0;"><strong>{youths}</strong> asked us to invite <strong>{business_name}</strong> ({location}) to <strong>ECO Local</strong> - a youth-led, values-first network of local businesses and young regulars.</p>
      <p style="margin:0 0 8px 0;">It’s simple: list a youth-friendly perk (discount, freebie, special) and we’ll send you real local foot traffic - not vanity metrics.{website_hint}</p>
      <div style="margin:12px 0 8px 0;">{perks}</div>
    </td>
  </tr>
  <tr>
    <td style="padding:8px 24px 12px 24px">
      {_cta(join_url, "List your youth perk")}
      <div style="font-size:12px;color:#426e57;margin-top:8px;">
        Prefer email? Just reply here and we’ll set it up with you in minutes.
      </div>
    </td>
  </tr>
""".strip()

    return _brand_wrap(body)

def _build_referral_subject(business_name: str) -> str:
    return f"{business_name} ✽ locals are asking for you on ECO Local"

def _llm_craft_salutation(business_name: str, website: Optional[str]) -> str:
    try:
        prompt = {
            "ask": "Infer a likely human salutation for a small local business; if unknown, return empty string.",
            "business": business_name,
            "website": website,
        }
        resp = _gen_json(
            system="Return JSON with {'name': string}.",
            user=prompt,
            schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        )
        name = (resp.get("name") or "").strip()
        return name if name and len(name) <= 40 else ""
    except Exception:
        return ""

def _gather_youth_names_for(business_name: str, location: str) -> List[str]:
    recs = _run("""
    MATCH (r:YouthReferral {store_name: $b, location: $loc})
    RETURN r.youth_name AS youth_name
    """, {"b": business_name, "loc": location})
    names = sorted({rec["youth_name"] for rec in recs if rec.get("youth_name")})
    return names[:12]

def _best_to_email(website: Optional[str]) -> Optional[str]:
    # Hook for website→email heuristic if enabled later.
    return None

def _send_referral_outreach(r: YouthReferral) -> None:
    youth_names = _gather_youth_names_for(r.store_name, r.location) or [r.youth_name]
    to_name = _llm_craft_salutation(r.store_name, r.website)
    subject = _build_referral_subject(r.store_name)

    html, inline_images = _build_referral_email_html(
        to_name=to_name,
        business_name=r.store_name,
        location=r.location,
        youth_names=youth_names,
        website=r.website,
    )
    html = _normalize_signature_logo(html, is_inline_cid=bool(inline_images))

    to_email = (r.email or _best_to_email(r.website) or "").strip()
    if not to_email:
        log.warning("[referrals] no recipient for id=%s store=%s loc=%s", r.id, r.store_name, r.location)
        return

    mid = send_email(
        to=to_email,
        subject=subject,
        html=html,
        attachments=None,
        inline_images=inline_images,
        reply_to_message_id=None,
        references=None,
        ics=None,
    )
    log.info("[referrals] send ok id=%s ses_mid=%s inline=%s", r.id, mid, "cid" if inline_images else "url/none")
    mark_outreach_sent(r.id)

# ──────────────────────────────────────────────────────────────────────────────
# Batch jobs / webhooks (called by your jobs router)
# ──────────────────────────────────────────────────────────────────────────────
def month_bounds(d: date, tz: str = BRIS_TZ) -> Tuple[date, date]:
    start = d.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
    return start, end

def monthly_referral_outreach(run_for_month: Optional[date] = None) -> Dict[str, Any]:
    tz = ZoneInfo(BRIS_TZ)
    today = datetime.now(tz).date()
    target_month = run_for_month or today
    start, end = month_bounds(target_month, BRIS_TZ)

    # 1) Enrich anything still 'submitted'
    import asyncio
    asyncio.run(enrich_pending(limit=200))

    # 2) Gather and send
    targets = list_referrals_needing_monthly_outreach(start, end)
    sent = 0
    for r in targets:
        try:
            _send_referral_outreach(r)
            sent += 1
        except Exception as e:
            log.warning("[referrals] outreach failed for %s: %s", r.id, e)

    return {"month": target_month.isoformat(), "range": [start.isoformat(), end.isoformat()], "attempted": len(targets), "sent": sent}

def mark_partner_joined(referral_id: str, partner_id: str, bonus_eco: Optional[int] = None) -> Dict[str, Any]:
    bonus = REFERRAL_BONUS_ECO if bonus_eco is None else bonus_eco
    mark_joined_and_reward(referral_id, partner_id, bonus)
    return {"ok": True, "referral_id": referral_id, "partner_id": partner_id, "bonus_eco": bonus}
