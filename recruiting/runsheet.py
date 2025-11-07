# recruiting/runsheet.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Dict

from .store import (
    get_first_touch_quota,
    get_followup_days,
    select_prospects_for_first_touch,
    select_prospects_for_followups,
    create_run,
    attach_draft_email,
    freeze_run,
    iter_run_items,
    mark_sent,
)
from .outreach import draft_first_touch
from .sender import send_email

@dataclass
class BuildResult:
    created: bool
    counts: Dict[str, int]

def send_runsheet_for_date(target: date) -> Dict[str, int]:
    sent_counts: Dict[str, int] = {"first": 0, "followups": 0, "final": 0}
    for item in list(iter_run_items(target)):
        msg_id = send_email(to=item.email, subject=item.subject, html=item.body)
        mark_sent(item, msg_id)
        bucket = "followups" if item.type == "followup1" else item.type
        sent_counts[bucket] = sent_counts.get(bucket, 0) + 1
    return sent_counts

def hourly_inbox_poll() -> int:
    from .inbox import hourly_inbox_poll as _poll
    return _poll()
