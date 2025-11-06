# recruiting/time_windows.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import settings
from .calendar_client import (
    suggest_windows as _suggest_windows,
    find_free_slots as _find_free_slots,       # kept for compatibility
    is_range_free as _is_range_free,
    find_free_windows as _find_free_windows,   # NEW: continuous, merged windows
)

def _tz() -> ZoneInfo:
    return ZoneInfo(settings.LOCAL_TZ or "Australia/Brisbane")

def calendar_suggest_windows(
    *,
    lookahead_days: int = 21,
    duration_min: int = 30,
    work_hours: Tuple[int, int] = (9, 17),
    weekdays: set[int] = {0, 1, 2, 3, 4},
    min_gap_min: int = 15,
    hold_padding_min: int = 10,
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Thin wrapper used by the inbox LLM prompt. Returns top-scored candidate windows.
    Model is free to choose ANY sub-range within the continuous free_windows exposed separately.
    """
    tz = _tz()
    start = datetime.now(tz)
    end = start + timedelta(days=lookahead_days)
    return _suggest_windows(
        start=start,
        end=end,
        duration_min=duration_min,
        work_hours=work_hours,
        days=weekdays,
        min_gap_min=min_gap_min,
        hold_padding_min=hold_padding_min,
        calendar_id=calendar_id,
        trace_id=trace_id,
    )

def nearest_same_day_options(
    *,
    target_start_iso: str,
    n: int = 3,
    duration_min: int = 30,
    work_hours: Tuple[int, int] = (9, 17),
    weekdays: set[int] = {0, 1, 2, 3, 4},
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Given a requested start time, return up to `n` nearest FREE slots on that same day.
    Uses continuous free windows (non-grid), then chooses nearest sub-ranges of `duration_min`.
    """
    tz = _tz()
    try:
        req = datetime.fromisoformat(target_start_iso.replace("Z", "+00:00")).astimezone(tz)
    except Exception:
        req = datetime.fromisoformat(target_start_iso).replace(tzinfo=tz)

    if req.weekday() not in weekdays:
        return []

    # Find continuous free windows for that day, then evaluate distances
    day_start = req.replace(hour=work_hours[0], minute=0, second=0, microsecond=0)
    day_end   = req.replace(hour=work_hours[1], minute=0, second=0, microsecond=0)

    windows = _find_free_windows(
        range_days=0,
        daily_windows=[work_hours],
        calendar_id=calendar_id,
        trace_id=trace_id,
    )

    same_day_windows: List[Tuple[datetime, datetime]] = []
    for w in windows:
        try:
            sdt = datetime.fromisoformat(w["start"].replace("Z", "+00:00")).astimezone(tz)
            edt = datetime.fromisoformat(w["end"].replace("Z", "+00:00")).astimezone(tz)
        except Exception:
            continue
        if sdt.date() != req.date():
            continue
        # clamp to work-hours
        sdt = max(sdt, day_start)
        edt = min(edt, day_end)
        if edt > sdt:
            same_day_windows.append((sdt, edt))

    # For each window, consider the slot whose midpoint is nearest the requested midpoint
    out: List[Dict[str, Any]] = []
    half = timedelta(minutes=duration_min/2)
    target_mid = req + half
    for sdt, edt in same_day_windows:
        # If requested slot fully fits, include it first
        if sdt <= req and (req + timedelta(minutes=duration_min)) <= edt:
            cand = {"start": req.isoformat(), "end": (req + timedelta(minutes=duration_min)).isoformat(), "tz": str(tz)}
            if _is_range_free(cand["start"], cand["end"], calendar_id=calendar_id, trace_id=trace_id):
                out.append(cand)
        # Otherwise, pick a slot centered near target_mid but clamped to the window
        mid_start = max(sdt, min(target_mid - half, edt - timedelta(minutes=duration_min)))
        mid_end = mid_start + timedelta(minutes=duration_min)
        if mid_end <= edt:
            cand = {"start": mid_start.isoformat(), "end": mid_end.isoformat(), "tz": str(tz)}
            if _is_range_free(cand["start"], cand["end"], calendar_id=calendar_id, trace_id=trace_id):
                out.append(cand)
        if len(out) >= n:
            break

    # As a fallback, you could sample a couple more spread across the windows (omitted for simplicity)
    return out[:n]
