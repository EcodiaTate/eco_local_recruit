# app/routers/referral_jobs.py  (eco_local app)
from fastapi import APIRouter
from pydantic import BaseModel
import os
from typing import Optional

from recruiting.referrals import get_referral, _send_referral_outreach

router = APIRouter(prefix="/referrals", tags=["referrals"])

class SendOneReq(BaseModel):
    referral_id: str
    to_override: Optional[str] = None
    bcc: Optional[str] = None
    inline: Optional[str] = None  # none|url|cid

@router.post("/dev/send-one", summary="Dev: send a single referral email now")
def dev_send_one(body: SendOneReq):
    if body.to_override is not None:
        os.environ["ECO_LOCAL_REFERRALS_TO_OVERRIDE"] = body.to_override
    if body.bcc is not None:
        os.environ["ECO_LOCAL_REFERRALS_BCC"] = body.bcc
    if body.inline is not None:
        os.environ["ECO_LOCAL_REFERRALS_INLINE"] = body.inline

    r = get_referral(body.referral_id)
    if not r:
        return {"ok": False, "error": "not_found"}
    _send_referral_outreach(r)
    return {"ok": True, "id": r.id}
