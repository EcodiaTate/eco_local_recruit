from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

# These live in the eco_local repo (your recruiting package)
from recruiting.referrals import (
    monthly_referral_outreach,
    enrich_pending,
    mark_partner_joined,
)

router = APIRouter(prefix="/referrals", tags=["referrals"])


# ─────────────────────────────────────────────────────────────
# Simple health/ping (handy for Cloud Scheduler test)
# ─────────────────────────────────────────────────────────────
@router.get("/jobs/ping", summary="Ping for scheduler/health")
def ping():
    return {"ok": True, "service": "eco_local.referrals", "ping": "pong"}


# ─────────────────────────────────────────────────────────────
# Job: Enrich any pending 'submitted' referrals (idempotent)
# Useful if you want a daily job separate from the monthly send
# ─────────────────────────────────────────────────────────────
class EnrichReq(BaseModel):
    limit: int = Field(200, ge=1, le=2000, description="Max rows to enrich in one run")


@router.post("/jobs/enrich-pending", summary="Enrich any submitted referrals")
def job_enrich_pending(body: EnrichReq):
    # This wraps the coroutine to run now
    enriched = enrich_pending(limit=body.limit)
    # enrich_pending returns a coroutine in recruiting.referrals; run it here
    import asyncio
    results = asyncio.run(enriched)
    return {
        "ok": True,
        "enriched": len(results),
    }


# ─────────────────────────────────────────────────────────────
# Job: Monthly referral outreach (Cloud Scheduler target)
# - Enriches anything still 'submitted'
# - Sends outreach emails for that month’s referrals
# - Accepts optional `month` (YYYY-MM-DD, any day within target month)
# ─────────────────────────────────────────────────────────────
class MonthlyOutreachReq(BaseModel):
    month: Optional[date] = Field(
        default=None,
        description="Any date inside the target month; defaults to current month in server TZ.",
    )


@router.post("/jobs/monthly-outreach", summary="Run monthly referral outreach")
def job_monthly_outreach(body: MonthlyOutreachReq):
    result = monthly_referral_outreach(run_for_month=body.month)
    # result has: { month, range: [start,end], attempted, sent }
    return {"ok": True, **result}


# ─────────────────────────────────────────────────────────────
# Webhook: Mark business joined + issue ECO bonus to referrer
# - Call this from your partner signup completion handler
# ─────────────────────────────────────────────────────────────
class PartnerJoinedIn(BaseModel):
    referral_id: str
    partner_id: str
    bonus_eco: Optional[int] = Field(
        default=None,
        description="Override the default ECO bonus if needed",
    )


@router.post("/webhooks/partner-joined", summary="Flip referral to joined and award ECO")
def partner_joined(body: PartnerJoinedIn):
    res = mark_partner_joined(
        referral_id=body.referral_id,
        partner_id=body.partner_id,
        bonus_eco=body.bonus_eco,
    )
    return res
