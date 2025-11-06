from __future__ import annotations
from datetime import timedelta, timezone
from typing import Optional

def resolve_tz(key: str, *, fallback_offset_hours: int = 10):
    """
    Robust ZoneInfo resolver for Windows.
    - Tries zoneinfo with tzdata installed.
    - Falls back to a fixed UTC offset if still unavailable.
    """
    # 1) Try zoneinfo straight
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(key)
    except Exception:
        pass

    # 2) If tzdata is installed, retry (import ensures the DB is available)
    try:
        import tzdata  # noqa: F401
        from zoneinfo import ZoneInfo
        return ZoneInfo(key)
    except Exception:
        pass

    # 3) Last-ditch: fixed offset (Brisbane = UTC+10, no DST most of QLD)
    return timezone(timedelta(hours=fallback_offset_hours))
