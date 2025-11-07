# app/main.py
from __future__ import annotations

import os
import sys
import json
import time
import platform
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from random import Random

try:
    import httpx  # lightweight fetch for hosted config
except Exception:
    httpx = None  # fall back to defaults if missing

from dotenv import load_dotenv; load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from recruiting.runsheet import hourly_inbox_poll
from recruiting.config import settings
from recruiting.debug_semantic import router as debug_semantic_router
from recruiting.referrals_router import router as referrals_router
from recruiting.timezones import resolve_tz
from recruiting import orchestrator_cli as oc
from recruiting.referral_jobs import router as dev_router

# ──────────────────────────────────────────────────────────────────────────────
# Boot logs (handy in Cloud Run)
# ──────────────────────────────────────────────────────────────────────────────
print("PYTHON:", sys.executable)
print("VERSION:", sys.version)
print("PLATFORM:", platform.platform())

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("ECO_LOCAL_LOG_LEVEL", "INFO"))


# ──────────────────────────────────────────────────────────────────────────────
# App & CORS
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="ECO Local Recruit Service", version="1.0.0", docs_url="/")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
]
env_origins = os.getenv("ECO_LOCAL_CORS_ORIGINS")
if env_origins:
    ALLOWED_ORIGINS.extend([o.strip() for o in env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.include_router(debug_semantic_router)
app.include_router(referrals_router)
app.include_router(dev_router)

# ──────────────────────────────────────────────────────────────────────────────
# Static (robust for local + prod)
# ──────────────────────────────────────────────────────────────────────────────
APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent
PKG_ROOT = APP_DIR.parent
REPO_ROOT = PKG_ROOT.parent

ENV_STATIC = os.getenv("ECO_LOCAL_STATIC_DIR")
CANDIDATES = [
    ENV_STATIC,
    str(PKG_ROOT / "static"),
    str(APP_DIR / "static"),
    str(REPO_ROOT / "static"),
]
STATIC_DIR = next((p for p in CANDIDATES if p and os.path.isdir(p)), None)
if not STATIC_DIR:
    STATIC_DIR = str(PKG_ROOT / "static")
    os.makedirs(STATIC_DIR, exist_ok=True)
    log.info("Created local static directory at %s", STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
log.info("Mounted static directory at /static from %s", STATIC_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# Timezone / utilities
# ──────────────────────────────────────────────────────────────────────────────
LOCAL_TZ = resolve_tz(settings.LOCAL_TZ or "Australia/Brisbane")

def _days_since_epoch_au(dt: Optional[date] = None) -> int:
    tz = LOCAL_TZ
    now = datetime.now(tz) if dt is None else datetime(dt.year, dt.month, dt.day, tzinfo=tz)
    epoch = datetime(1970, 1, 1, tzinfo=tz)
    return (now - epoch).days

# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "service": "eco_local-recruit", "env": settings.ENV}

# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator passthroughs (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/eco_local/recruit/outreach/build")
def outreach_build(
    date_iso: str | None = None,
    freeze: bool = True,
    allow_fallback: bool = False,
):
    argv = ["build"]
    if date_iso: argv += ["--date", date_iso]
    if freeze:   argv += ["--freeze"]
    if allow_fallback: argv += ["--allow-fallback"]
    try:
        oc.invoke(argv)
        return {"ok": True, "argv": argv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eco_local/recruit/outreach/send")
def outreach_send(date_iso: str | None = None):
    argv = ["send"]
    if date_iso: argv += ["--date", date_iso]
    try:
        oc.invoke(argv)
        return {"ok": True, "argv": argv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eco_local/recruit/discover")
async def discover(req: Request):
    body = await req.json()
    query = body.get("query")
    city = body.get("city")
    limit = str(body.get("limit", 10))
    require_email = body.get("require_email", None)  # None = use default
    if not query or not city:
        raise HTTPException(status_code=400, detail="query and city are required")
    argv = ["discover", "--query", query, "--city", city, "--limit", limit]
    if require_email is True:
        argv += ["--require-email"]
    elif require_email is False:
        argv += ["--no-require-email"]
    try:
        oc.invoke(argv)
        return {"ok": True, "argv": argv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eco_local/recruit/inbox/poll")
def poll_inbox() -> Dict[str, int]:
    processed = hourly_inbox_poll()
    return {"processed": processed}

@app.post("/eco_local/recruit/signup/webhook")
async def signup_webhook(req: Request) -> Dict[str, bool]:
    payload = await req.json()
    from recruiting.store import mark_signup_payload
    mark_signup_payload(payload)
    return {"ok": True}

# ──────────────────────────────────────────────────────────────────────────────
# Rotation & Plan loaders (URL/file/env → fallback defaults)
# ──────────────────────────────────────────────────────────────────────────────
# Flat rotation cache
_ROTATION_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "etag": None}
_ROTATION_TTL_SECONDS = int(os.getenv("ECO_LOCAL_DISCOVER_ROTATION_TTL", "600"))

# Plan (generator) cache
_PLAN_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "etag": None}
_PLAN_TTL_SECONDS = int(os.getenv("ECO_LOCAL_DISCOVER_PLAN_TTL", "600"))

DEFAULT_ROTATION: List[Dict[str, Any]] = [
    {"query": "sunshine coast cafes",   "city": "Sunshine Coast", "limit": 60},
    {"query": "brisbane thrift stores", "city": "Brisbane",        "limit": 60},
    {"query": "organic farms",          "city": "Sunshine Coast",  "limit": 60},
    {"query": "zero waste shops",       "city": "Brisbane",        "limit": 60},
    {"query": "farmers markets",        "city": "Sunshine Coast",  "limit": 60},
]

def _load_json_file(path: str) -> Optional[Any]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _fetch_json_url(url: str, cache: Dict[str, Any], ttl: int) -> Optional[Any]:
    if not httpx:
        return None
    now = time.time()
    if cache["data"] and (now - cache["ts"] < ttl):
        return cache["data"]
    headers = {}
    if cache.get("etag"):
        headers["If-None-Match"] = cache["etag"]
    try:
        resp = httpx.get(url, headers=headers, timeout=6.0)
        if resp.status_code == 304 and cache["data"]:
            cache["ts"] = now
            return cache["data"]
        resp.raise_for_status()
        data = resp.json()
        cache["data"] = data
        cache["ts"] = now
        cache["etag"] = resp.headers.get("ETag")
        return data
    except Exception:
        return None

def _load_rotation_plan(*, force_refresh: bool = False) -> List[Dict[str, Any]]:
    # 1) file
    if not force_refresh:
        data = _load_json_file(os.getenv("ECO_LOCAL_DISCOVER_ROTATION_FILE", "/config/discover_rotation.json"))
        if isinstance(data, list) and data:
            return data
    # 2) env
    raw = os.getenv("ECO_LOCAL_DISCOVER_ROTATION")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    # 3) URL
    url = os.getenv("ECO_LOCAL_DISCOVER_ROTATION_URL", "https://elocal.ecodia.au/config/discover_rotation.json")
    data = _fetch_json_url(url, _ROTATION_CACHE, _ROTATION_TTL_SECONDS)
    if isinstance(data, list) and data:
        return data
    # 4) default
    return DEFAULT_ROTATION

def _load_generator_plan(*, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    # 1) file
    if not force_refresh:
        data = _load_json_file(os.getenv("ECO_LOCAL_DISCOVER_PLAN_FILE", "/config/discover_plan.json"))
        if isinstance(data, dict) and data:
            return data
    # 2) env
    raw = os.getenv("ECO_LOCAL_DISCOVER_PLAN_JSON")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass
    # 3) URL
    url = os.getenv("ECO_LOCAL_DISCOVER_PLAN_URL", "https://elocal.ecodia.au/config/discover_query_plan_v3.json")
    data = _fetch_json_url(url, _PLAN_CACHE, _PLAN_TTL_SECONDS)
    if isinstance(data, dict) and data:
        return data
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Generator utilities (plan → list[ {query, city, limit} ])
# ──────────────────────────────────────────────────────────────────────────────
def _weighted_pick(rng: Random, items: List[Tuple[str, float]]) -> str:
    total = sum(w for _, w in items)
    if total <= 0:
        return items[0][0]
    x = rng.random() * total
    acc = 0.0
    for v, w in items:
        acc += w
        if x <= acc:
            return v
    return items[-1][0]

def _norm_mix(mix: List[Dict[str, Any]]) -> List[Tuple[str, float, List[str]]]:
    out: List[Tuple[str, float, List[str]]] = []
    for m in mix or []:
        name = str(m.get("name") or "")
        w = float(m.get("weight") or 0.0)
        tpls = [str(t) for t in (m.get("templates") or []) if t]
        if name and w > 0 and tpls:
            out.append((name, w, tpls))
    return out

def _synth_queries_from_plan(plan: Dict[str, Any], day_seed: Optional[str], batch_n: int) -> List[Dict[str, Any]]:
    """
    Compose N (query, city, limit) items deterministically from the plan.
    """
    # seed
    if day_seed:
        try:
            sd = date.fromisoformat(day_seed)
            seed_val = int(sd.strftime("%Y%m%d"))
        except Exception:
            seed_val = _days_since_epoch_au()
    else:
        seed_val = _days_since_epoch_au()
    rng = Random(seed_val)

    # buckets
    buckets = plan.get("buckets") or {}
    nouns_core = list(buckets.get("nouns_core") or []) + list(buckets.get("nouns_explore") or [])
    modifiers = list(buckets.get("modifiers") or [])

    geo = plan.get("geo") or {}
    region = str(geo.get("region") or "Sunshine Coast")
    suburbs_top = list(geo.get("suburbs_top") or [])
    suburbs_all = list(geo.get("suburbs_all") or [])

    # mix weights
    mix_spec = _norm_mix(plan.get("mix") or [
        {"name": "region_mod_noun", "weight": 0.50, "templates": ["{modifier} {noun}"]},
        {"name": "suburb_noun",     "weight": 0.35, "templates": ["{noun}"]},
        {"name": "suburb_mod_noun", "weight": 0.15, "templates": ["{modifier} {noun}"]},
    ])
    mix_items = [(name, w) for (name, w, _tpls) in mix_spec]

    def pick_from(lst: List[str]) -> str:
        return lst[rng.randrange(len(lst))]

    items: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, str]] = set()

    # how many suburb vs region
    # we'll bias toward top suburbs but sprinkle others
    for _ in range(max(1, batch_n)):
        beam = _weighted_pick(rng, mix_items)
        tpl_list = [tpls for (nm, _w, tpls) in mix_spec if nm == beam][0]
        tpl = pick_from(tpl_list)
        noun = pick_from(nouns_core)
        mod  = pick_from(modifiers) if "{modifier}" in tpl else None

        if beam.startswith("region"):
            city = region
        elif beam.startswith("suburb"):
            # 70% from top suburbs, 30% from wider list
            if suburbs_top and (rng.random() < 0.70):
                city = pick_from(suburbs_top)
            else:
                pool = (suburbs_all or suburbs_top or [region])
                city = pick_from(pool)
        else:
            city = region

        q = tpl.replace("{noun}", noun).replace("{modifier}", mod or "").strip()
        key = (q.lower(), city.lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)

        items.append({"query": q, "city": city, "limit": int(plan.get("constraints", {}).get("limit_default", 60))})

    return items

# ──────────────────────────────────────────────────────────────────────────────
# Rotate endpoint (supports flat rotation OR generator plan), with batching
# ──────────────────────────────────────────────────────────────────────────────
@app.api_route("/eco_local/recruit/discover/rotate", methods=["GET", "POST"])
async def discover_rotate(
    request: Request,
    day_seed: str | None = Query(default=None, description="ISO date seed, e.g. 2025-11-07"),
    offset: int = Query(default=0, description="Offset window for flat rotation mode (can be negative)"),
    require_email: bool | None = Query(default=None, description="Override require-email gate"),
    limit: int | None = Query(default=None, description="Override per-run limit (applies to all items)"),
    refresh: int = Query(default=0, description="Force refresh of remote rotation/plan (1=yes)"),
    batch_n: int = Query(default=30, description="How many items to process in this call"),
    dry_run: int = Query(default=0, description="If 1, don't invoke discover; just return the batch"),
):
    """
    Behavior:
      - If a generator PLAN is available (URL/file/env), synthesize `batch_n` items from it.
      - Else, consume a contiguous `batch_n` window from the flat rotation list deterministically per day.
    """

    # Merge JSON body over query params
    body: Dict[str, Any] = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
    day_seed = body.get("day_seed", day_seed)
    offset   = int(body.get("offset", offset))
    require_email = body.get("require_email", require_email)
    limit    = body.get("limit", limit)
    refresh  = int(body.get("refresh", refresh or 0))
    batch_n  = int(body.get("batch_n", batch_n))
    dry_run  = int(body.get("dry_run", dry_run))

    # Prefer PLAN if available
    plan = _load_generator_plan(force_refresh=bool(refresh))
    picked: List[Dict[str, Any]] = []
    source = "plan" if plan else "rotation"

    if plan:
        items = _synth_queries_from_plan(plan, day_seed, batch_n*2)  # synth a few extra to avoid duplicates
        # enforce unique and truncate to batch_n
        seen: set[Tuple[str, str]] = set()
        for it in items:
            k = (it["query"].lower(), it["city"].lower())
            if k in seen: continue
            seen.add(k); picked.append(it)
            if len(picked) >= batch_n: break
        # allow override of per-run limit
        if limit:
            for it in picked: it["limit"] = int(limit)
    else:
        # Flat rotation mode
        plan_list = _load_rotation_plan(force_refresh=bool(refresh))
        if not plan_list:
            raise HTTPException(status_code=500, detail="Rotation plan is empty")

        days = _days_since_epoch_au(date.fromisoformat(day_seed)) if day_seed else _days_since_epoch_au()
        base_idx = (days * batch_n + offset) % len(plan_list)

        for k in range(max(1, batch_n)):
            idx = (base_idx + k) % len(plan_list)
            chosen = plan_list[idx]
            query = str(chosen.get("query") or "").strip()
            city  = str(chosen.get("city") or "").strip()
            if not query or not city:
                continue
            picked.append({
                "query": query,
                "city": city,
                "limit": int(limit or chosen.get("limit") or 60)
            })

    if dry_run:
        return {"ok": True, "dry_run": True, "source": source, "batch": picked}

    # Execute sequentially
    results = {"ok": True, "source": source, "processed": 0, "errors": 0, "items": []}
    for item in picked:
        argv = ["discover", "--query", item["query"], "--city", item["city"], "--limit", str(item["limit"])]
        if require_email is True:
            argv += ["--require-email"]
        elif require_email is False:
            argv += ["--no-require-email"]

        try:
            oc.invoke(argv)
            results["processed"] += 1
            results["items"].append({"argv": argv, "status": "ok"})
        except Exception as e:
            results["errors"] += 1
            results["items"].append({"argv": argv, "status": "error", "detail": str(e)})

    results["meta"] = {
        "batch_n": batch_n,
        "day_seed": day_seed,
        "offset": offset,
        "refreshed": bool(refresh),
    }
    return results
