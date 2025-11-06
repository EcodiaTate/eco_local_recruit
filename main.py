# app/main.py
from __future__ import annotations

import os
import sys
import platform
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any

from dotenv import load_dotenv; load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from zoneinfo import ZoneInfo

from recruiting.runsheet import (
 
    hourly_inbox_poll,
)
from recruiting.config import settings
from recruiting.debug_semantic import router as debug_semantic_router
from recruiting.timezones import resolve_tz
from recruiting import orchestrator_cli as oc


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


# ──────────────────────────────────────────────────────────────────────────────
# Static (robust/optional)
# ──────────────────────────────────────────────────────────────────────────────
# Default to baked-in image path; override with ECO_LOCAL_STATIC_DIR
STATIC_DIR = os.getenv("ECO_LOCAL_STATIC_DIR", "/app/static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    log.info("Mounted static directory at /static from %s", STATIC_DIR)
else:
    log.warning("Static dir not found (%s); skipping mount", STATIC_DIR)


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
# Orchestrator endpoints
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
    # Keep it simple: mark “won” if payload indicates signup=true; extend as needed.
    from recruiting.store import mark_signup_payload
    mark_signup_payload(payload)
    return {"ok": True}
