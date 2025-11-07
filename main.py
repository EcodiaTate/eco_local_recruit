# app/main.py
from __future__ import annotations

import os
import sys
import json
import platform
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any, List
# app/main.py (add near the other imports)
import time
from typing import Tuple
try:
    import httpx  # FastAPI stacks usually have it; if not, `pip install httpx`
except Exception:
    httpx = None  # we'll fail soft and fall back to defaults

from dotenv import load_dotenv; load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from zoneinfo import ZoneInfo
from pathlib import Path

from recruiting.runsheet import (
    hourly_inbox_poll,
)
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

# Allow your Next.js dev host(s). Add others as needed or control via env.
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
]
env_origins = os.getenv("ECO_LOCAL_CORS_ORIGINS")
if env_origins:
    # Comma-separated, e.g. "https://elocal.ecodia.au,https://admin.ecodia.au"
    ALLOWED_ORIGINS.extend([o.strip() for o in env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,     # must be exact origins, not "*", if using credentials
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
APP_DIR = APP_FILE.parent                    # .../eco_local/app
PKG_ROOT = APP_DIR.parent                    # .../eco_local
REPO_ROOT = PKG_ROOT.parent                  # .../EcodiaOS

ENV_STATIC = os.getenv("ECO_LOCAL_STATIC_DIR")

CANDIDATES = [
    ENV_STATIC,                              # explicit override wins
    str(PKG_ROOT / "static"),                # prefer eco_local/static  ✅
    str(APP_DIR / "static"),                 # optional: app/static
    str(REPO_ROOT / "static"),               # fallback: EcodiaOS/static
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

def _parse_date_iso(date_iso: Optional[str]) -> date:
    if not date_iso:
        return datetime.now(LOCAL_TZ).date()
    try:
        if len(date_iso) == 10:
            return datetime.fromisoformat(date_iso).date()
        return datetime.fromisoformat(date_iso).astimezone(LOCAL_TZ).date()
    except Exception:
        return datetime.now(LOCAL_TZ).date()

# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "service": "eco_local-recruit", "env": settings.ENV}

# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator endpoints (existing)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/eco_local/recruit/outreach/build")
def outreach_build(
    date_iso: str | None = None,
    freeze: bool = True,
    allow_fallback: bool = False,
):
    argv = ["build"]
    if date_iso:
        argv += ["--date", date_iso]
    if freeze:
        argv += ["--freeze"]
    if allow_fallback:
        argv += ["--allow-fallback"]
    try:
        oc.invoke(argv)
        return {"ok": True, "argv": argv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eco_local/recruit/outreach/send")
def outreach_send(date_iso: str | None = None):
    argv = ["send"]
    if date_iso:
        argv += ["--date", date_iso]
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
# NEW: Rotating discovery endpoint
# ──────────────────────────────────────────────────────────────────────────────
"""
Rotation plan source (first found wins):

1) ENV var ECO_LOCAL_DISCOVER_ROTATION as JSON list:
   [
     {"query": "sunshine coast cafes", "city": "Sunshine Coast", "limit": 60},
     {"query": "brisbane thrift stores", "city": "Brisbane", "limit": 60},
     {"query": "organic farms", "city": "Sunshine Coast", "limit": 60}
   ]

2) Default rotation baked in below.

Selection:
- default index = days since 1970-01-01 in Australia/Brisbane modulo len(plan)
- override with ?index=2 or ?day_seed=2025-11-07 or ?offset=+1
"""
# app/main.py (keep your DEFAULT_ROTATION as-is; we’ll still use it)
DEFAULT_ROTATION: List[Dict[str, Any]] = [
    {"query": "sunshine coast cafes",   "city": "Sunshine Coast", "limit": 60},
    {"query": "brisbane thrift stores", "city": "Brisbane",        "limit": 60},
    {"query": "organic farms",          "city": "Sunshine Coast",  "limit": 60},
    {"query": "zero waste shops",       "city": "Brisbane",        "limit": 60},
    {"query": "farmers markets",        "city": "Sunshine Coast",  "limit": 60},
]
# app/main.py (NEW: small cache + loader that tries file -> env -> URL -> default)
_ROTATION_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "etag": None}
_ROTATION_TTL_SECONDS = int(os.getenv("ECO_LOCAL_DISCOVER_ROTATION_TTL", "600"))  # 10 min default

def _load_rotation_plan(*, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Priority:
      1) File (ECO_LOCAL_DISCOVER_ROTATION_FILE), if present and valid JSON
      2) Env var ECO_LOCAL_DISCOVER_ROTATION (JSON)
      3) Remote URL ECO_LOCAL_DISCOVER_ROTATION_URL (JSON)  ← your prod: https://elocal.ecodia.au/config/discover_rotation.json
      4) DEFAULT_ROTATION (in code)
    Caches URL result for _ROTATION_TTL_SECONDS unless force_refresh=True.
    """
    # 1) File override
    path = os.getenv("ECO_LOCAL_DISCOVER_ROTATION_FILE", "/config/discover_rotation.json")
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                plan = json.load(f)
                if isinstance(plan, list) and plan:
                    return plan
    except Exception:
        log.warning("Rotation file invalid; ignoring: %s", path)

    # 2) Env var override
    raw = os.getenv("ECO_LOCAL_DISCOVER_ROTATION")
    if raw:
        try:
            plan = json.loads(raw)
            if isinstance(plan, list) and plan:
                return plan
        except Exception:
            log.warning("ECO_LOCAL_DISCOVER_ROTATION invalid JSON; using fallbacks.")

    # 3) Remote URL
    url = os.getenv(
        "ECO_LOCAL_DISCOVER_ROTATION_URL",
        "https://elocal.ecodia.au/config/discover_rotation.json"
    )
    now = time.time()
    if not force_refresh and _ROTATION_CACHE["data"] and (now - _ROTATION_CACHE["ts"] < _ROTATION_TTL_SECONDS):
        return _ROTATION_CACHE["data"]

    if httpx and url:
        headers = {}
        if _ROTATION_CACHE.get("etag"):
            headers["If-None-Match"] = _ROTATION_CACHE["etag"]
        try:
            resp = httpx.get(url, headers=headers, timeout=5.0)
            if resp.status_code == 304 and _ROTATION_CACHE["data"]:
                _ROTATION_CACHE["ts"] = now
                return _ROTATION_CACHE["data"]
            resp.raise_for_status()
            plan = resp.json()
            if isinstance(plan, list) and plan:
                _ROTATION_CACHE["data"] = plan
                _ROTATION_CACHE["ts"] = now
                _ROTATION_CACHE["etag"] = resp.headers.get("ETag")
                return plan
            else:
                log.warning("Rotation URL returned non-list or empty JSON; falling back. url=%s", url)
        except Exception as e:
            log.warning("Failed to fetch rotation from URL (%s): %s", url, e)

    # 4) In-code default
    return DEFAULT_ROTATION



def _days_since_epoch_au(dt: Optional[date] = None) -> int:
    tz = LOCAL_TZ
    now = datetime.now(tz) if dt is None else datetime(dt.year, dt.month, dt.day, tzinfo=tz)
    epoch = datetime(1970, 1, 1, tzinfo=tz)
    return (now - epoch).days

def _pick_rotation(plan: List[Dict[str, Any]], *, index: Optional[int], day_seed: Optional[str], offset: int) -> Dict[str, Any]:
    if not plan:
        raise HTTPException(status_code=500, detail="Rotation plan is empty")
    if index is not None:
        idx = index % len(plan)
    else:
        seed_date: Optional[date] = None
        if day_seed:
            try:
                seed_date = datetime.fromisoformat(day_seed).date()
            except Exception:
                try:
                    seed_date = date.fromisoformat(day_seed)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid day_seed format (use YYYY-MM-DD)")
        base = _days_since_epoch_au(seed_date)
        idx = base % len(plan)
        if offset:
            idx = (idx + offset) % len(plan)
    return plan[idx]
# app/main.py (tweak your rotate endpoint to allow refresh=?)
@app.api_route("/eco_local/recruit/discover/rotate", methods=["GET", "POST"])
async def discover_rotate(
    request: Request,
    index: int | None = Query(default=None, description="Pick specific rotation index"),
    day_seed: str | None = Query(default=None, description="ISO date to seed rotation, e.g. 2025-11-07"),
    offset: int = Query(default=0, description="Offset from seeded index (can be negative)"),
    require_email: bool | None = Query(default=None, description="Override require-email gate"),
    limit: int | None = Query(default=None, description="Override per-run limit"),
    refresh: int = Query(default=0, description="Force refresh of remote rotation (1=yes)"),
):
    body: Dict[str, Any] = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}

    # allow overrides from body as well
    index = body.get("index", index)
    day_seed = body.get("day_seed", day_seed)
    offset = int(body.get("offset", offset))
    require_email = body.get("require_email", require_email)
    limit = body.get("limit", limit)
    refresh = int(body.get("refresh", refresh or 0))

    plan = _load_rotation_plan(force_refresh=bool(refresh))
    chosen = _pick_rotation(plan, index=index, day_seed=day_seed, offset=offset)

    query = str(chosen.get("query") or "").strip()
    city = str(chosen.get("city") or "").strip()
    if not query or not city:
        raise HTTPException(status_code=500, detail="Chosen rotation item missing query or city")

    final_limit = int(limit or chosen.get("limit") or 10)

    argv = ["discover", "--query", query, "--city", city, "--limit", str(final_limit)]
    if require_email is True:
        argv += ["--require-email"]
    elif require_email is False:
        argv += ["--no-require-email"]

    try:
        oc.invoke(argv)
        return {
            "ok": True,
            "argv": argv,
            "picked": {"query": query, "city": city, "limit": final_limit},
            "meta": {
                "plan_len": len(plan),
                "index": index,
                "day_seed": day_seed,
                "offset": offset,
                "refreshed": bool(refresh),
                "source": (
                    "file" if os.path.isfile(os.getenv("ECO_LOCAL_DISCOVER_ROTATION_FILE", "/config/discover_rotation.json"))
                    else ("env" if os.getenv("ECO_LOCAL_DISCOVER_ROTATION") else "url_or_default")
                ),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
