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
import traceback

# ---------------- Optional deps ----------------
try:
    import httpx
except Exception:
    httpx = None  # type: ignore

# ---------------- Env ----------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- FastAPI ----------------
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------- Local imports ----------------
from recruiting.runsheet import hourly_inbox_poll
from recruiting.config import settings
from recruiting.debug_semantic import router as debug_semantic_router
from recruiting.referrals_router import router as referrals_router
from recruiting.timezones import resolve_tz
from recruiting import orchestrator_cli as oc
from recruiting.referral_jobs import router as dev_router

# ---------------- Startup diag ----------------
print("PYTHON:", sys.executable)
print("VERSION:", sys.version)
print("PLATFORM:", platform.platform())

log = logging.getLogger("eco_local.app")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("ECO_LOCAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

app = FastAPI(title="ECO Local Recruit Service", version="1.0.0", docs_url="/")

# ---------------- CORS ----------------
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

# ---------------- Routers ----------------
app.include_router(debug_semantic_router)
app.include_router(referrals_router)
app.include_router(dev_router)

# ---------------- Static mount ----------------
APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent            # /app/app
PKG_ROOT = APP_DIR.parent            # /app
REPO_ROOT = PKG_ROOT.parent          # /

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

# ---------------- TZ ----------------
from recruiting.timezones import resolve_tz
LOCAL_TZ = resolve_tz(settings.LOCAL_TZ or "Australia/Brisbane")

def _days_since_epoch_au(dt: Optional[date] = None) -> int:
    tz = LOCAL_TZ
    now = datetime.now(tz) if dt is None else datetime(dt.year, dt.month, dt.day, tzinfo=tz)
    epoch = datetime(1970, 1, 1, tzinfo=tz)
    return (now - epoch).days

# ---------------- Health ----------------
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "service": "eco_local-recruit", "env": settings.ENV}

@app.get("/readyz")
def readyz() -> Dict[str, Any]:
    return {"ok": True, "ts": time.time()}

# ---------------- Outreach (build/send) ----------------
@app.post("/eco-local/recruit/outreach/build")
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
        log.error("[outreach/build] failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eco-local/recruit/outreach/send")
def outreach_send(date_iso: str | None = None):
    argv = ["send"]
    if date_iso:
        argv += ["--date", date_iso]
    try:
        oc.invoke(argv)
        return {"ok": True, "argv": argv}
    except Exception as e:
        log.error("[outreach/send] failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Simple discover proxy ----------------
@app.post("/eco-local/recruit/discover")
async def discover(req: Request):
    body = await _safe_json(req)
    query = body.get("query")
    city = body.get("city")
    limit = str(body.get("limit", 10))
    require_email = body.get("require_email", None)
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
        log.error("[discover] failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Inbox ----------------
@app.post("/eco-local/recruit/inbox/poll")
def poll_inbox() -> Dict[str, int]:
    processed = hourly_inbox_poll()
    return {"processed": processed}

# ---------------- Webhook ----------------
@app.post("/eco-local/recruit/signup/webhook")
async def signup_webhook(req: Request) -> Dict[str, bool]:
    payload = await _safe_json(req)
    from recruiting.store import mark_signup_payload
    mark_signup_payload(payload)
    return {"ok": True}

# ──────────────────────────────────────────────────────────────────────────────
# PLAN-ONLY GENERATOR (single stream) + time-budgeted rotate
# ──────────────────────────────────────────────────────────────────────────────
_PLAN_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "etag": None}
_PLAN_TTL_SECONDS = int(os.getenv("ECO_LOCAL_DISCOVER_PLAN_TTL", "600"))

async def _safe_json(req: Request) -> Dict[str, Any]:
    try:
        return await req.json()
    except Exception:
        return {}

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
    except Exception as e:
        log.warning("[plan] fetch url failed (%s): %s", url, e)
        return None

def _load_generator_plan(*, force_refresh: bool = False) -> Dict[str, Any]:
    default_file = "/config/discover_plan.json" if os.path.isdir("/config") else ""
    path = os.getenv("ECO_LOCAL_DISCOVER_PLAN_FILE", default_file or "")
    try:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
    except Exception as e:
        log.warning("[plan] file load failed (%s): %s", path, e)

    raw = os.getenv("ECO_LOCAL_DISCOVER_PLAN_JSON")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data:
                return data
        except Exception as e:
            log.warning("[plan] env JSON invalid: %s", e)

    url = os.getenv("ECO_LOCAL_DISCOVER_PLAN_URL", "/config/discover_rotation.json")
    data = _fetch_json_url(url, _PLAN_CACHE, _PLAN_TTL_SECONDS)
    if isinstance(data, dict) and data:
        return data

    raise HTTPException(
        status_code=500,
        detail="Discovery plan unavailable (file/env/url). Set ECO_LOCAL_DISCOVER_PLAN_* or mount /config/discover_plan.json",
    )

def _days_seed_value(day_seed: Optional[str]) -> int:
    if day_seed:
        try:
            sd = date.fromisoformat(day_seed)
            return int(sd.strftime("%Y%m%d"))
        except Exception:
            pass
    return _days_since_epoch_au()

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
    seed_val = _days_seed_value(day_seed)
    rng = Random(seed_val)

    buckets = plan.get("buckets") or {}
    nouns_core = list(buckets.get("nouns_core") or []) + list(buckets.get("nouns_explore") or [])
    modifiers  = list(buckets.get("modifiers") or [])
    if not nouns_core:
        raise HTTPException(status_code=500, detail="Plan missing buckets.nouns_core")
    if not modifiers:
        modifiers = ["local","organic","sustainable","ethical","refill","zero waste","bulk"]

    geo = plan.get("geo") or {}
    region = str(geo.get("region") or "Sunshine Coast")
    suburbs_top = list(geo.get("suburbs_top") or [])
    suburbs_all = list(geo.get("suburbs_all") or [])

    mix_spec = _norm_mix(plan.get("mix") or [
        {"name": "region_mod_noun", "weight": 0.50, "templates": ["{modifier} {noun}"]},
        {"name": "suburb_noun",     "weight": 0.35, "templates": ["{noun}"]},
        {"name": "suburb_mod_noun", "weight": 0.15, "templates": ["{modifier} {noun}"]},
    ])
    if not mix_spec:
        raise HTTPException(status_code=500, detail="Plan mix is empty")

    mix_items = [(name, w) for (name, w, _tpls) in mix_spec]
    def tpl_list_for(name: str) -> List[str]:
        for (nm, _w, tpls) in mix_spec:
            if nm == name:
                return tpls
        return ["{noun}"]

    def pick_from(lst: List[str]) -> str:
        return lst[rng.randrange(len(lst))]

    batch: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    default_limit = int(plan.get("constraints", {}).get("limit_default", 60))

    while len(batch) < max(1, batch_n):
        beam = _weighted_pick(rng, mix_items)
        tpl  = pick_from(tpl_list_for(beam))
        noun = pick_from(nouns_core)
        mod  = pick_from(modifiers) if "{modifier}" in tpl else None

        if beam.startswith("region"):
            city = region
        elif beam.startswith("suburb"):
            if suburbs_top and (rng.random() < 0.70):
                city = pick_from(suburbs_top)
            else:
                pool = (suburbs_all or suburbs_top or [region])
                city = pick_from(pool)
        else:
            city = region

        q = tpl.replace("{noun}", noun).replace("{modifier}", mod or "").strip()
        key = (q.lower(), city.lower())
        if key in seen:
            continue
        seen.add(key)
        batch.append({"query": q, "city": city, "limit": default_limit})

    return batch

# single-stream endpoint (bigger & faster by default) with time budget + waves
@app.api_route("/eco-local/recruit/discover/rotate", methods=["GET", "POST"])
async def discover_rotate(
    request: Request,
    day_seed: str | None = Query(default=None, description="ISO date seed, e.g. 2025-11-07"),
    require_email: bool | None = Query(default=None, description="Override require-email gate"),
    limit: int | None = Query(default=None, description="Override per-run limit (applies to all items)"),
    refresh: int = Query(default=0, description="Force refresh of plan (1=yes)"),
    batch_n: int = Query(default=200, description="How many items to process"),
    default_limit: int = Query(default=50, description="Per-item limit when 'limit' is not provided"),
    workers: int = Query(default=24, ge=1, le=64, description="Max concurrent item runners"),
    job_budget_s: int = Query(default=25*60, ge=60, le=29*60, description="Hard wall-clock budget (seconds)"),
    dry_run: int = Query(default=0, description="If 1, don't invoke; just return the batch"),
):
    import asyncio
    from time import monotonic

    body: Dict[str, Any] = {}
    if request.method == "POST":
        body = await _safe_json(request)

    # Body can override query params
    day_seed      = body.get("day_seed", day_seed)
    require_email = body.get("require_email", require_email)
    limit         = body.get("limit", limit)
    refresh       = int(body.get("refresh", refresh or 0))
    batch_n       = int(body.get("batch_n", batch_n))
    dry_run       = int(body.get("dry_run", dry_run))
    default_limit = int(body.get("default_limit", default_limit))
    workers       = int(body.get("workers", workers))
    job_budget_s  = int(body.get("job_budget_s", job_budget_s))

    # Guardrails
    batch_n = max(1, min(batch_n, 1000))
    default_limit = max(1, min(default_limit, 500))

    try:
        plan = _load_generator_plan(force_refresh=bool(refresh))
        picked = _synth_queries_from_plan(plan, day_seed, batch_n)

        # Apply limit defaults/overrides
        if limit is not None:
            for it in picked:
                it["limit"] = int(limit)
        else:
            for it in picked:
                if not it.get("limit"):
                    it["limit"] = default_limit

        if dry_run:
            return {"ok": True, "source": "plan", "dry_run": True, "batch": picked}

        results = {"ok": True, "source": "plan", "processed": 0, "errors": 0, "items": []}

        # Concurrency + time budget
        sem = asyncio.Semaphore(workers)
        t0 = monotonic()

        async def run_one(it: Dict[str, Any]):
            argv = ["discover", "--query", it["query"], "--city", it["city"], "--limit", str(it["limit"])]
            if require_email is True:
                argv += ["--require-email"]
            elif require_email is False:
                argv += ["--no-require-email"]
            async with sem:
                await asyncio.to_thread(oc.invoke, argv)
                return {"argv": argv, "status": "ok"}

        # process in waves so we can exit on budget
        WAVE = max(1, workers * 2)
        attempted = 0
        for i in range(0, len(picked), WAVE):
            # budget left?
            elapsed = time.monotonic() - t0
            if elapsed >= job_budget_s:
                break
            remaining = max(1, int(job_budget_s - elapsed))

            batch = picked[i : i + WAVE]
            tasks = [asyncio.create_task(run_one(it)) for it in batch]

            try:
                # Drain this wave with remaining budget
                for coro in asyncio.as_completed(tasks, timeout=remaining):
                    try:
                        item_res = await coro
                        results["items"].append(item_res)
                        results["processed"] += 1
                        attempted += 1
                        if attempted % 10 == 0:
                            log.info("[rotate] progress attempted=%d processed=%d elapsed=%ds",
                                     attempted, results["processed"], int(time.monotonic() - t0))
                    except Exception as e:
                        results["items"].append({"status": "error", "detail": str(e)})
                        results["errors"] += 1
            except asyncio.TimeoutError:
                for t in tasks:
                    t.cancel()
                break

        results["meta"] = {
            "batch_n": batch_n,
            "day_seed": day_seed,
            "refreshed": bool(refresh),
            "workers": workers,
            "default_limit": default_limit,
            "override_limit": limit,
            "job_budget_s": job_budget_s,
            "elapsed_s": int(time.monotonic() - t0),
            "attempted": attempted,
        }
        return results

    except HTTPException:
        raise
    except Exception as e:
        log.error("[rotate] failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
