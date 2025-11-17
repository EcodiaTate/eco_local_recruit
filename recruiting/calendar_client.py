from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import logging
import time

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import settings
from .sa_loader import load_sa_credentials

log = logging.getLogger(__name__)
_level = (getattr(logging, (getattr(settings, "ECO_LOCAL_LOG_LEVEL", None) or "INFO").upper(), logging.INFO))
if not logging.getLogger().handlers:
    logging.basicConfig(level=_level)
else:
    logging.getLogger().setLevel(_level)

__all__ = [
    "build_calendar_context",
    "find_free_slots",
    "find_free_windows",
    "is_range_free",
    "nearest_n_free",
    "suggest_windows",
    "ICSSpec",
    "build_ics",
    "create_event",
    "create_hold",
    "update_event_kind",
    "patch_event_time",
    "delete_event",
    "find_thread_holds",
    "find_thread_confirmed",
    "list_holds",
    "cancel_holds",
    "promote_hold_to_event",
    "cancel_other_thread_holds",
    "supersede_thread_booking",
    "cancel_thread_events",
    "cleanup_stale_holds",
]

# ──────────────────────────────────────────────────────────────────────────────
# Constants / globals
# ──────────────────────────────────────────────────────────────────────────────

_CAL_SCOPES = ["https://www.googleapis.com/auth/calendar"]

_FB_CACHE: Dict[Tuple[str, str, str], Tuple[float, List[Tuple[datetime, datetime]]]] = {}
_FB_CACHE_TTL_S = 5.0

def _now_perf() -> float:
    return time.perf_counter()

def _fb_invalidate(cal_id: Optional[str] = None) -> None:
    if cal_id is None:
        _FB_CACHE.clear()
        return
    for k in list(_FB_CACHE.keys()):
        if k[0] == cal_id:
            _FB_CACHE.pop(k, None)

# ──────────────────────────────────────────────────────────────────────────────
# Service builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_calendar_service():
    log.debug("[calendar] building Google Calendar service (impersonate=%s)", settings.GSUITE_IMPERSONATED_USER)
    t0 = _now_perf()
    creds = load_sa_credentials(
        scopes=_CAL_SCOPES,
        subject=settings.GSUITE_IMPERSONATED_USER,
    )
    svc = build("calendar", "v3", credentials=creds, cache_discovery=False)
    log.debug("[calendar] service built in %.1fms", (_now_perf() - t0) * 1000)
    return svc

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _tz() -> ZoneInfo:
    return ZoneInfo(settings.LOCAL_TZ or "Australia/Brisbane")

def _iso_utc_z(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _sanitize_window(
    start: datetime,
    end: Optional[datetime],
    *,
    tz: ZoneInfo,
    min_seconds: int = 60,
    default_days: int = 21,
) -> tuple[datetime, datetime]:
    s = start.astimezone(tz)
    e = (end or (s + timedelta(days=default_days))).astimezone(tz)
    if e <= s:
        e = s + timedelta(seconds=max(min_seconds, 1))
    return s, e

def _in_business_days(dt_: datetime) -> bool:
    return dt_.weekday() in (0, 1, 2, 3, 4)

def _subtract_busy_from_window(
    window_start: datetime,
    window_end: datetime,
    busy_blocks: List[Tuple[datetime, datetime]],
) -> List[Tuple[datetime, datetime]]:
    free: List[Tuple[datetime, datetime]] = [(window_start, window_end)]
    for b_start, b_end in busy_blocks:
        new_free: List[Tuple[datetime, datetime]] = []
        for f_start, f_end in free:
            if b_end <= f_start or b_start >= f_end:
                new_free.append((f_start, f_end))
                continue
            if b_start > f_start:
                new_free.append((f_start, min(b_start, f_end)))
            if b_end < f_end:
                new_free.append((max(b_end, f_start), f_end))
        free = [(s, e) for (s, e) in new_free if e > s]
    return free

def _chunk_free_intervals(
    free_intervals: List[Tuple[datetime, datetime]],
    hold_minutes: int,
) -> List[Tuple[datetime, datetime]]:
    out: List[Tuple[datetime, datetime]] = []
    delta = timedelta(minutes=hold_minutes)
    for s, e in free_intervals:
        cur = s
        while cur + delta <= e:
            out.append((cur, cur + delta))
            cur = cur + delta
    return out

def _parse_iso(s: str, tz: ZoneInfo) -> datetime:
    try:
        return datetime.fromisoformat(s).astimezone(tz)
    except Exception:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(tz)

def _to_local(s: str, tz: ZoneInfo) -> datetime:
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
    else:
        dt = dt.astimezone(tz)
    return dt

def _daily_windows_for(day: datetime, windows: List[Tuple[int, int]]) -> List[Tuple[datetime, datetime]]:
    return [(day.replace(hour=a, minute=0, second=0, microsecond=0),
             day.replace(hour=b, minute=0, second=0, microsecond=0)) for (a, b) in windows]

from email.utils import parseaddr

def _bare(addr: Optional[str]) -> Optional[str]:
    if not addr:
        return None
    return (parseaddr(addr)[1] or addr).strip() or None

# ──────────────────────────────────────────────────────────────────────────────
# Freebusy (memoized) + window suggestion
# ──────────────────────────────────────────────────────────────────────────────

def _freebusy(
    cal,
    cal_id: str,
    time_min: datetime,
    time_max: Optional[datetime],
    tz: ZoneInfo,
    *,
    trace_id: Optional[str] = None
) -> List[Tuple[datetime, datetime]]:
    tmin, tmax = _sanitize_window(time_min, time_max, tz=tz)
    key = (cal_id, _iso_utc_z(tmin), _iso_utc_z(tmax))
    now = time.time()
    cached = _FB_CACHE.get(key)
    if cached and (now - cached[0]) <= _FB_CACHE_TTL_S:
        log.debug("[calendar] freebusy cache hit cal=%s window=%s→%s trace=%s", cal_id, key[1], key[2], trace_id)
        return cached[1]

    body = {"timeMin": key[1], "timeMax": key[2], "items": [{"id": cal_id}], "timeZone": str(tz)}
    t0 = _now_perf()
    try:
        fb = cal.freebusy().query(body=body).execute() or {}
    except HttpError as e:
        log.error(
            "[calendar] freebusy query failed cal=%s trace=%s error=%s body=%s",
            cal_id, trace_id, e, body, exc_info=True
        )
        return []
    finally:
        log.debug("[calendar] freebusy query cal=%s took %.1fms trace=%s", cal_id, (_now_perf() - t0) * 1000, trace_id)

    busy: List[Tuple[datetime, datetime]] = []
    raw = (fb.get("calendars", {}).get(cal_id, {}) or {}).get("busy", []) or []
    for b in raw:
        start = _parse_iso(b["start"], tz)
        end = _parse_iso(b["end"], tz)
        if end > start:
            busy.append((start, end))

    _FB_CACHE[key] = (now, busy)
    return busy

def suggest_windows(
    start: datetime,
    end: datetime,
    *,
    duration_min: int = 30,
    work_hours: Tuple[int, int] = (9, 17),
    days: set[int] = {0, 1, 2, 3, 4},
    min_gap_min: int = 15,
    hold_padding_min: int = 10,
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    tz = _tz()
    w_start, w_end = _sanitize_window(start, end, tz=tz, default_days=21)
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    cal = _build_calendar_service()

    busy_blocks = _freebusy(cal, cal_id, w_start, w_end, tz, trace_id=trace_id)

    daily: List[Tuple[datetime, datetime]] = []
    cur = w_start.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= w_end:
        if cur.weekday() in days:
            daily.append((cur.replace(hour=work_hours[0], minute=0),
                          cur.replace(hour=work_hours[1], minute=0)))
        cur = cur + timedelta(days=1)

    slot_delta = timedelta(minutes=duration_min)
    results: List[Dict[str, Any]] = []
    for day_start, day_end in daily:
        day_busy = [(max(bs, day_start), min(be, day_end)) for (bs, be) in busy_blocks if not (be <= day_start or bs >= day_end)]
        free_intervals = _subtract_busy_from_window(day_start, day_end, day_busy)

        pad = timedelta(minutes=hold_padding_min)
        for fs, fe in free_intervals:
            ps, pe = fs + pad, fe - pad
            if pe <= ps:
                continue
            cur_s = max(ps, w_start)
            while cur_s + slot_delta <= min(pe, w_end):
                mid_pref = day_start + (day_end - day_start) / 2
                midday_score = 1.0 / (1.0 + abs((cur_s + slot_delta/2 - mid_pref).total_seconds())/3600.0)
                soon_score = 1.0 / (1.0 + max(0.0, (cur_s - w_start).total_seconds())/86400.0)
                score = midday_score * 0.6 + soon_score * 0.4
                results.append({
                    "start": cur_s.isoformat(),
                    "end": (cur_s + slot_delta).isoformat(),
                    "tz": str(tz),
                    "score": round(score, 3),
                    "reason": "Within work hours; avoids conflicts; balanced daytime placement",
                })
                cur_s += timedelta(minutes=max(duration_min, min_gap_min))

    results.sort(key=lambda r: (-r["score"], r["start"]))
    return results[:12]

# ──────────────────────────────────────────────────────────────────────────────
# Free windows and slots
# ──────────────────────────────────────────────────────────────────────────────

def build_calendar_context(
    range_days: int = 21,
    hold_minutes: int = 30,
    *,
    daily_windows: List[Tuple[int, int]] = [(9, 12), (13, 17)],
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    tz = _tz()
    windows = find_free_windows(
        range_days=range_days,
        daily_windows=daily_windows,
        calendar_id=calendar_id,
        trace_id=trace_id,
    )
    slots = find_free_slots(
        range_days=range_days,
        hold_minutes=hold_minutes,
        daily_windows=daily_windows,
        calendar_id=calendar_id,
        trace_id=trace_id,
    )
    now_local = datetime.now(tz)

    readable_windows = []
    for w in windows[:24]:
        try:
            s = datetime.fromisoformat(w["start"].replace("Z","+00:00")).astimezone(tz)
            e = datetime.fromisoformat(w["end"].replace("Z","+00:00")).astimezone(tz)
            readable_windows.append(f"{s.strftime('%a %d %b')}, {s.strftime('%-I:%M%p').lower()} - {e.strftime('%-I:%M%p').lower()}")
        except Exception:
            pass

    return {
        "tz": str(tz),
        "now_iso": now_local.isoformat(),
        "calendar_id": calendar_id or settings.ECO_LOCAL_GCAL_ID,
        "free_windows": windows,
        "free_slots": slots,
        "window_days": range_days,
        "slot_size_minutes": hold_minutes,
        "daily_windows": daily_windows,
        "FREE_WINDOWS_readable": readable_windows,
    }

def find_free_windows(
    range_days: int = 10,
    *,
    daily_windows: List[Tuple[int, int]] = [(9, 12), (13, 17)],
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    tz = _tz()
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    cal = _build_calendar_service()

    now_local = datetime.now(tz).replace(microsecond=0)
    time_min = now_local
    time_max = (now_local + timedelta(days=range_days)).replace(microsecond=0)

    busy_blocks = _freebusy(cal, cal_id, time_min, time_max, tz, trace_id=trace_id)

    windows: List[Dict[str, Any]] = []
    for i in range(range_days + 1):
        day = (time_min + timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        if not _in_business_days(day):
            continue

        for w_start, w_end in _daily_windows_for(day, daily_windows):
            if w_end <= now_local:
                continue
            day_busy = [(max(bs, w_start), min(be, w_end)) for (bs, be) in busy_blocks if not (be <= w_start or bs >= w_end)]
            free_intervals = _subtract_busy_from_window(w_start, w_end, day_busy)
            for s, e in free_intervals:
                if e > now_local:
                    windows.append({"start": s.isoformat(), "end": e.isoformat(), "tz": str(tz)})

    log.info(
        "[calendar] windows %s→%s tz=%s cal=%s busy=%d windows=%d (range_days=%d) trace=%s",
        time_min.isoformat(),
        time_max.isoformat(),
        tz,
        cal_id,
        len(busy_blocks),
        len(windows),
        range_days,
        trace_id,
    )
    return windows

def find_free_slots(
    range_days: int = 10,
    hold_minutes: int = 30,
    *,
    daily_windows: List[Tuple[int, int]] = [(9, 12), (13, 17)],
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    tz = _tz()
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    cal = _build_calendar_service()

    now_local = datetime.now(tz).replace(microsecond=0)
    time_min = now_local
    time_max = (now_local + timedelta(days=range_days)).replace(microsecond=0)

    busy_blocks = _freebusy(cal, cal_id, time_min, time_max, tz, trace_id=trace_id)

    slots: List[Dict[str, Any]] = []
    for i in range(range_days + 1):
        day = (time_min + timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        if not _in_business_days(day):
            continue

        for w_start, w_end in _daily_windows_for(day, daily_windows):
            if w_end <= now_local:
                continue

            day_busy = [
                (max(bs, w_start), min(be, w_end))
                for (bs, be) in busy_blocks
                if not (be <= w_start or bs >= w_end)
            ]
            free_intervals = _subtract_busy_from_window(w_start, w_end, day_busy)
            chunked = _chunk_free_intervals(free_intervals, hold_minutes)

            for s, e in chunked:
                if e > now_local:
                    slots.append({"start": s.isoformat(), "end": e.isoformat(), "tz": str(tz)})

    log.info(
        "[calendar] slots %s→%s tz=%s cal=%s busy=%d slots=%d (range_days=%d, hold=%d) trace=%s",
        time_min.isoformat(),
        time_max.isoformat(),
        tz,
        cal_id,
        len(busy_blocks),
        len(slots),
        range_days,
        hold_minutes,
        trace_id,
    )
    return slots

def is_range_free(
    start_iso: str, end_iso: str, *, calendar_id: Optional[str] = None, trace_id: Optional[str] = None
) -> bool:
    tz = _tz()
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    cal = _build_calendar_service()

    s = _parse_iso(start_iso, tz)
    e = _parse_iso(end_iso, tz)
    if e <= s:
        return False

    fb = _freebusy(cal, cal_id, s - timedelta(hours=1), e + timedelta(hours=1), tz, trace_id=trace_id)
    for bs, be in fb:
        if not (be <= s or bs >= e):
            return False
    return True

def nearest_n_free(
    anchor_iso: str,
    n: int = 2,
    hold_minutes: int = 30,
    *,
    lookahead_days: int = 21,
    daily_windows: List[Tuple[int, int]] = [(9, 12), (13, 17)],
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    tz = _tz()
    anchor = _parse_iso(anchor_iso, tz)
    slots = find_free_slots(
        range_days=lookahead_days,
        hold_minutes=hold_minutes,
        daily_windows=daily_windows,
        calendar_id=calendar_id,
        trace_id=trace_id,
    )
    out: List[Dict[str, Any]] = []
    for s in slots:
        sdt = _parse_iso(s["start"], tz)
        if sdt >= anchor:
            if is_range_free(s["start"], s["end"], calendar_id=calendar_id, trace_id=trace_id):
                out.append(s)
        if len(out) >= n:
            break
    return out

# ──────────────────────────────────────────────────────────────────────────────
# ICS & Events
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ICSSpec:
    start: str
    end: str
    summary: str
    description: str = ""
    location: str = ""
    organizer_name: str = "Ecodia"
    organizer_email: str = "ecolocal@ecodia.au"
    attendee_email: Optional[str] = None
    method: str = "REQUEST"  # Meeting invites should be REQUEST, not PUBLISH


def _ics_escape(s: Optional[str]) -> str:
    """Escape text per RFC5545: backslash, comma, semicolon, newline."""
    if not s:
        return ""
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace(",", "\\,")
    s = s.replace(";", "\\;")
    return s

def build_ics(spec: ICSSpec) -> Tuple[str, bytes]:
    tz = _tz()
    uid = f"ecodia-{int(datetime.now(tz).timestamp())}@ecodia.au"
    dtstamp = datetime.now(tz).strftime("%Y%m%dT%H%M%S")

    def _fmt(dt_iso: str) -> str:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00")).astimezone(tz)
        return dt.strftime("%Y%m%dT%H%M%S")

    method = (spec.method or "PUBLISH").upper()
    lines = [
        "BEGIN:VCALENDAR",
        "PRODID:-//Ecodia//ECO Local Outreach//EN",
        "VERSION:2.0",
        "CALSCALE:GREGORIAN",
        f"METHOD:{method}",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART:{_fmt(spec.start)}",
        f"DTEND:{_fmt(spec.end)}",
        f"SUMMARY:{_ics_escape(spec.summary)}",
    ]
    if spec.location:
        lines.append(f"LOCATION:{_ics_escape(spec.location)}")
    if spec.description:
        lines.append(f"DESCRIPTION:{_ics_escape(spec.description)}")
    lines.append(f"ORGANIZER;CN={_ics_escape(spec.organizer_name)}:mailto:{_ics_escape(spec.organizer_email)}")
    if spec.attendee_email:
        lines.append("ATTENDEE;ROLE=REQ-PARTICIPANT;PARTSTAT=NEEDS-ACTION:mailto:" + _ics_escape(spec.attendee_email))
    lines += ["END:VEVENT", "END:VCALENDAR"]
    content = "\r\n".join(lines).encode("utf-8")
    filename = f"ecodia-meeting-{uid}.ics"
    return filename, content


def create_hold(
    *,
    title: str = "Ecodia - intro chat (HOLD)",
    description: str = "Provisional hold (auto-expires unless confirmed).",
    start_iso: str,
    end_iso: str,
    thread_id: Optional[str],
    prospect_email: Optional[str],
    expires_in_hours: int = 48,
    attendees: Optional[List[Dict[str, str]]] = None,
    tz: Optional[str] = None,
    calendar_id: Optional[str] = None,
    send_updates: str = "none",
    trace_id: Optional[str] = None,
    meeting_type: Optional[str] = None,         # NEW: "call" | "meeting"
    location_name: Optional[str] = None,        # NEW: business name or short location
) -> Dict[str, Any]:
    tz_str = tz or settings.LOCAL_TZ or "Australia/Brisbane"

    # decorate title with meeting type if provided
    mt = (meeting_type or "").lower().strip()
    typetag = "Call" if mt == "call" else ("Meeting" if mt == "meeting" else None)
    title2 = f"{title} - {typetag}" if typetag and "HOLD" in title else (f"{title} - {typetag}" if typetag else title)

    ev = create_event(
        title=title2,
        description=description,
        start_iso=start_iso,
        end_iso=end_iso,
        tz=tz_str,
        attendees=attendees,           # normalized inside create_event
        calendar_id=calendar_id,
        send_updates=send_updates,     # keep 'none' so GCal doesn't email on holds
        thread_id=thread_id,
        prospect_email=prospect_email,
        kind="HOLD",
        trace_id=trace_id,
        meeting_type=meeting_type,
        location_name=location_name,
    )
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    svc = _build_calendar_service()
    priv = (ev.get("extendedProperties") or {}).get("private", {}) or {}
    expires_at = (datetime.now(_tz()) + timedelta(hours=expires_in_hours)).isoformat()
    priv["ecoLocalHoldExpiresAt"] = expires_at
    ev["extendedProperties"] = {"private": priv}
    ev = svc.events().patch(calendarId=cal_id, eventId=ev["id"], body=ev, sendUpdates="none").execute()
    _fb_invalidate(cal_id)
    return ev

def create_event(
    title: str,
    description: str,
    start_iso: str,
    end_iso: str,
    tz: str | None = None,
    attendees: Optional[List[Dict[str, str]]] = None,
    calendar_id: Optional[str] = None,
    send_updates: str = "all",
    *,
    thread_id: Optional[str] = None,
    prospect_email: Optional[str] = None,
    kind: str = "CONFIRMED",
    trace_id: Optional[str] = None,
    meeting_type: Optional[str] = None,     # NEW: "call" | "meeting"
    location_name: Optional[str] = None,    # NEW: business name or short location
    location_full: Optional[str] = None,    # optional full address if you have it later
) -> Dict[str, Any]:
    """
    Adds:
      - body['location'] visible in calendar clients
      - extendedProperties.private:
            ecoLocalKind, ecoLocalMeetingType, ecoLocalLocationName
    """
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    tzinfo = ZoneInfo(tz or settings.LOCAL_TZ or "Australia/Brisbane")
    start = _to_local(start_iso, tzinfo)
    end = _to_local(end_iso, tzinfo)

    # Normalize attendees to bare emails to ensure Google actually emails them
    from email.utils import parseaddr
    norm_attendees: List[Dict[str, str]] = []
    for a in (attendees or []):
        e = (parseaddr((a or {}).get("email", ""))[1] or "").strip()
        if e:
            norm_attendees.append({"email": e})

    # decorate title with meeting type for confirmed events too
    mt = (meeting_type or "").lower().strip()
    typetag = "Call" if mt == "call" else ("Meeting" if mt == "meeting" else None)
    summary = f"{title} - {typetag}" if typetag and typetag not in title else title

    # choose location field
    loc = location_full or location_name or ""

    service = _build_calendar_service()
    body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": str(tzinfo)},
        "end": {"dateTime": end.isoformat(), "timeZone": str(tzinfo)},
        "attendees": norm_attendees,
        "transparency": "opaque",
        **({"location": loc} if loc else {}),
        "extendedProperties": {
            "private": {
                "ecoLocalKind": kind,
                **({"ecoLocalThread": thread_id} if thread_id else {}),
                **({"ecoLocalProspect": prospect_email} if prospect_email else {}),
                **({"ecoLocalMeetingType": mt} if mt else {}),
                **({"ecoLocalLocationName": location_name} if location_name else {}),
            }
        },
    }
    log.info(
        "[calendar] create_event %s→%s title=%r attendees=%d kind=%s mt=%s loc=%r cal=%s trace=%s",
        start, end, summary, len(body.get("attendees", [])), kind, mt or "-", loc, cal_id, trace_id
    )
    ev = service.events().insert(calendarId=cal_id, body=body, sendUpdates=send_updates).execute()
    log.debug("[calendar] event created id=%s link=%s trace=%s", ev.get("id"), ev.get("htmlLink"), trace_id)
    _fb_invalidate(cal_id)
    return ev


def update_event_kind(
    event_id: str,
    *,
    kind: str,
    calendar_id: Optional[str] = None,
    summary: Optional[str] = None,
    attendees: Optional[List[Dict[str, str]]] = None,
    send_updates: str = "all",
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    svc = _build_calendar_service()
    current = svc.events().get(calendarId=cal_id, eventId=event_id).execute()
    ep = (current.get("extendedProperties") or {}).get("private", {}) or {}
    ep["ecoLocalKind"] = kind
    if summary is not None:
        current["summary"] = summary
    if attendees is not None:
        current["attendees"] = attendees
    current.setdefault("extendedProperties", {}).setdefault("private", {}).update(ep)
    log.info("[calendar] update_event_kind id=%s kind=%s trace=%s", event_id, kind, trace_id)
    ev = svc.events().patch(calendarId=cal_id, eventId=event_id, body=current, sendUpdates=send_updates).execute()
    _fb_invalidate(cal_id)
    return ev

def patch_event_time(
    event_id: str,
    *,
    start_iso: str,
    end_iso: str,
    tz: Optional[str] = None,
    calendar_id: Optional[str] = None,
    send_updates: str = "all",
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    tzinfo = ZoneInfo(tz or settings.LOCAL_TZ or "Australia/Brisbane")
    s = _to_local(start_iso, tzinfo)
    e = _to_local(end_iso, tzinfo)
    svc = _build_calendar_service()
    cur = svc.events().get(calendarId=cal_id, eventId=event_id).execute()
    cur["start"] = {"dateTime": s.isoformat(), "timeZone": str(tzinfo)}
    cur["end"] = {"dateTime": e.isoformat(), "timeZone": str(tzinfo)}
    cur.setdefault("transparency", "opaque")
    # Preserve/roll forward our private props
    priv = (cur.get("extendedProperties") or {}).get("private", {}) or {}
    cur.setdefault("extendedProperties", {}).setdefault("private", {}).update(priv)
    log.info("[calendar] patch_event_time id=%s %s→%s trace=%s", event_id, s, e, trace_id)
    ev = svc.events().patch(calendarId=cal_id, eventId=event_id, body=cur, sendUpdates=send_updates).execute()
    _fb_invalidate(cal_id)
    return ev

def delete_event(event_id: str, *, calendar_id: Optional[str] = None, send_updates: str = "all", trace_id: Optional[str] = None) -> None:
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    svc = _build_calendar_service()
    log.info("[calendar] delete_event id=%s cal=%s trace=%s", event_id, cal_id, trace_id)
    svc.events().delete(calendarId=cal_id, eventId=event_id, sendUpdates=send_updates).execute()
    _fb_invalidate(cal_id)

# ──────────────────────────────────────────────────────────────────────────────
# Thread-scoped helpers (unchanged except for prop persistence)
# ──────────────────────────────────────────────────────────────────────────────

def _list_events_by_thread(
    *,
    thread_id: str,
    time_min_iso: str,
    time_max_iso: str,
    calendar_id: Optional[str] = None,
    kind: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    tzinfo = _tz()
    time_min = _to_local(time_min_iso, tzinfo).isoformat()
    time_max = _to_local(time_max_iso, tzinfo).isoformat()
    svc = _build_calendar_service()
    props = [f"ecoLocalThread={thread_id}"]
    if kind:
        props.append(f"ecoLocalKind={kind}")
    q_params = {
        "calendarId": cal_id,
        "timeMin": time_min,
        "timeMax": time_max,
        "singleEvents": True,
        "orderBy": "startTime",
        "privateExtendedProperty": props,
    }
    items: List[Dict[str, Any]] = []
    page_token = None
    while True:
        resp = svc.events().list(pageToken=page_token, **q_params).execute()
        items.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    log.debug("[calendar] list_by_thread thread=%s kind=%s found=%d trace=%s", thread_id, kind, len(items), trace_id)
    return items

def find_thread_holds(
    *, thread_id: str, time_min_iso: str, time_max_iso: str, calendar_id: Optional[str] = None, trace_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    return _list_events_by_thread(thread_id=thread_id, time_min_iso=time_min_iso, time_max_iso=time_max_iso, calendar_id=calendar_id, kind="HOLD", trace_id=trace_id)

def find_thread_confirmed(
    *, thread_id: str, time_min_iso: str, time_max_iso: str, calendar_id: Optional[str] = None, trace_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    return _list_events_by_thread(thread_id=thread_id, time_min_iso=time_min_iso, time_max_iso=time_max_iso, calendar_id=calendar_id, kind="CONFIRMED", trace_id=trace_id)

def list_holds(*, thread_id: str, lookback_days: int = 30, lookahead_days: int = 180, calendar_id: Optional[str] = None, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    tz = _tz()
    now = datetime.now(tz)
    return find_thread_holds(
        thread_id=thread_id,
        time_min_iso=(now - timedelta(days=lookback_days)).isoformat(),
        time_max_iso=(now + timedelta(days=lookahead_days)).isoformat(),
        calendar_id=calendar_id,
        trace_id=trace_id,
    )

def cancel_holds(*, thread_id: str, calendar_id: Optional[str] = None, trace_id: Optional[str] = None) -> Dict[str, int]:
    holds = list_holds(thread_id=thread_id, calendar_id=calendar_id, trace_id=trace_id)
    if not holds:
        log.debug("[calendar] cancel_holds skipped thread=%s reason=no_holds trace=%s", thread_id, trace_id)
        return {"deleted_holds": 0}
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    svc = _build_calendar_service()
    deleted = 0
    for h in holds:
        try:
            svc.events().delete(calendarId=cal_id, eventId=h["id"], sendUpdates="none").execute()
            deleted += 1
        except Exception:
            log.exception("[calendar] cancel_holds: delete id=%s failed trace=%s", h.get("id"), trace_id)
    log.info("[calendar] cancel_holds thread=%s deleted=%d trace=%s", thread_id, deleted, trace_id)
    _fb_invalidate(cal_id)
    return {"deleted_holds": deleted}

def _pick_best_hold_for_thread(
    *,
    thread_id: str,
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    tz = _tz()
    now = datetime.now(tz)
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID

    holds = list_holds(thread_id=thread_id, calendar_id=cal_id, trace_id=trace_id)
    if not holds:
        return None

    def _start_iso(ev: Dict[str, Any]) -> Optional[datetime]:
        try:
            s = (ev.get("start") or {}).get("dateTime") or (ev.get("start") or {}).get("date")
            if len(s) <= 10:
                dt = datetime.fromisoformat(s + "T00:00:00+00:00")
            else:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.astimezone(tz)
        except Exception:
            return None

    upcoming = []
    past = []
    for h in holds:
        sdt = _start_iso(h)
        if not sdt:
            continue
        if sdt >= now:
            upcoming.append((sdt, h))
        else:
            past.append((sdt, h))

    if upcoming:
        upcoming.sort(key=lambda x: x[0])
        return upcoming[0][1]
    if past:
        past.sort(key=lambda x: x[0], reverse=True)
        return past[0][1]
    return None

def _ensure_attendee(ev: Dict[str, Any], email: Optional[str]) -> Dict[str, Any]:
    e = _bare(email)
    if not e:
        return ev
    cur = ev.get("attendees") or []
    emails = { _bare(a.get("email","")) for a in cur if isinstance(a, dict) }
    if e not in emails:
        cur.append({"email": e})
        ev["attendees"] = cur
    return ev

def promote_hold_to_event(
    *,
    thread_id: Optional[str] = None,
    event_id: Optional[str] = None,
    new_title: Optional[str] = None,
    ensure_attendee_email: Optional[str] = None,
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    svc = _build_calendar_service()

    if event_id:
        target = svc.events().get(calendarId=cal_id, eventId=event_id).execute()
        if not target:
            raise ValueError("promote_hold_to_event: event_id not found")
    else:
        if not thread_id:
            raise ValueError("promote_hold_to_event: provide thread_id or event_id")
        target = _pick_best_hold_for_thread(thread_id=thread_id, calendar_id=cal_id, trace_id=trace_id)
        if not target:
            raise ValueError(f"promote_hold_to_event: no HOLD found for thread_id={thread_id}")
        target = svc.events().get(calendarId=cal_id, eventId=target["id"]).execute()

    priv = ((target.get("extendedProperties") or {}).get("private") or {})
    kind = priv.get("ecoLocalKind")
    if kind != "HOLD":
        log.info("[calendar] promote_hold_to_event: target not HOLD (kind=%s), returning as-is trace=%s", kind, trace_id)
        return target

    target["summary"] = new_title or target.get("summary") or "Ecodia - intro chat"
    target.setdefault("transparency", "opaque")
    target.setdefault("extendedProperties", {}).setdefault("private", {})["ecoLocalKind"] = "CONFIRMED"
    if ensure_attendee_email:
        target = _ensure_attendee(target, ensure_attendee_email)

    ev = svc.events().patch(calendarId=cal_id, eventId=target["id"], body=target, sendUpdates="all").execute()
    log.info("[calendar] promote_hold_to_event id=%s (CONFIRMED) trace=%s", ev.get("id"), trace_id)

    thread_val = ((ev.get("extendedProperties") or {}).get("private") or {}).get("ecoLocalThread")
    if thread_val:
        try:
            cancel_other_thread_holds(thread_id=thread_val, keep_event_id=ev.get("id"), calendar_id=cal_id, trace_id=trace_id)
        except Exception:
            log.exception("[calendar] cancel_other_thread_holds after promotion failed trace=%s", trace_id)

    _fb_invalidate(cal_id)
    return ev

def cancel_other_thread_holds(
    *,
    thread_id: str,
    keep_event_id: Optional[str],
    lookback_days: int = 30,
    lookahead_days: int = 60,
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> int:
    tz = _tz()
    now = datetime.now(tz)
    time_min = (now - timedelta(days=lookback_days)).isoformat()
    time_max = (now + timedelta(days=lookahead_days)).isoformat()
    holds = find_thread_holds(thread_id=thread_id, time_min_iso=time_min, time_max_iso=time_max, calendar_id=calendar_id, trace_id=trace_id)
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    count = 0
    for h in holds:
        if keep_event_id and h.get("id") == keep_event_id:
            continue
        try:
            delete_event(h["id"], calendar_id=cal_id, send_updates="none", trace_id=trace_id)
            count += 1
        except Exception:
            log.exception("[calendar] failed to delete old hold id=%s trace=%s", h.get("id"), trace_id)
    if count:
        log.info("[calendar] cancelled %d other holds for thread=%s trace=%s", count, thread_id, trace_id)
    _fb_invalidate(cal_id)
    return count

def supersede_thread_booking(
    *,
    thread_id: str,
    new_start_iso: str,
    new_end_iso: str,
    tz: Optional[str],
    prospect_email: Optional[str],
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    tzinfo = ZoneInfo(tz or settings.LOCAL_TZ or "Australia/Brisbane")
    now = datetime.now(tzinfo)
    confirmed = find_thread_confirmed(
        thread_id=thread_id,
        time_min_iso=(now - timedelta(days=7)).isoformat(),
        time_max_iso=(now + timedelta(days=90)).isoformat(),
        calendar_id=calendar_id,
        trace_id=trace_id,
    )
    if confirmed:
        eid = confirmed[0]["id"]
        ev = patch_event_time(eid, start_iso=new_start_iso, end_iso=new_end_iso, tz=str(tzinfo), calendar_id=calendar_id, send_updates="all", trace_id=trace_id)
        log.info("[calendar] superseded confirmed event id=%s -> %s..%s trace=%s", eid, new_start_iso, new_end_iso, trace_id)
        try:
            cancel_other_thread_holds(thread_id=thread_id, keep_event_id=eid, calendar_id=calendar_id, trace_id=trace_id)
        except Exception:
            log.exception("[calendar] cancel_other_thread_holds during supersede failed trace=%s", trace_id)
        return ev

    ev = create_event(
        title="Ecodia - ECO Local",
        description="Intro/chat about ECO Local",
        start_iso=new_start_iso,
        end_iso=new_end_iso,
        attendees=([{"email": prospect_email}] if prospect_email else []),
        thread_id=thread_id,
        prospect_email=prospect_email,
        location=None,
        kind="CONFIRMED",
        send_updates="all",
        trace_id=trace_id,
    )
    try:
        cancel_other_thread_holds(thread_id=thread_id, keep_event_id=ev.get("id"), calendar_id=calendar_id, trace_id=trace_id)
    except Exception:
        log.exception("[calendar] cancel_other_thread_holds after promote failed trace=%s", trace_id)
    return ev

def cancel_thread_events(
    *,
    thread_id: str,
    lookback_days: int = 30,
    lookahead_days: int = 180,
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, int]:
    tz = _tz()
    now = datetime.now(tz)
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    svc = _build_calendar_service()

    stats = {"deleted_confirmed": 0, "deleted_holds": 0}
    time_min = (now - timedelta(days=lookback_days)).isoformat()
    time_max = (now + timedelta(days=lookahead_days)).isoformat()

    for kind, key in (("CONFIRMED", "deleted_confirmed"), ("HOLD", "deleted_holds")):
        items = _list_events_by_thread(thread_id=thread_id, time_min_iso=time_min, time_max_iso=time_max, calendar_id=cal_id, kind=kind, trace_id=trace_id)
        if not items:
            continue
        for ev in items:
            try:
                svc.events().delete(calendarId=cal_id, eventId=ev["id"], sendUpdates="all").execute()
                stats[key] += 1
            except Exception:
                log.exception("[calendar] cancel_thread_events: delete id=%s failed trace=%s", ev.get("id"), trace_id)
    log.info("[calendar] cancel_thread_events thread=%s stats=%s trace=%s", thread_id, stats, trace_id)
    _fb_invalidate(cal_id)
    return stats

# ──────────────────────────────────────────────────────────────────────────────
# Hygiene
# ──────────────────────────────────────────────────────────────────────────────

def cleanup_stale_holds(
    *,
    max_past_days: int = 14,
    drop_future_if_confirmed: bool = True,
    calendar_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, int]:
    tz = _tz()
    now = datetime.now(tz)
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    svc = _build_calendar_service()

    stats = {"deleted_past_holds": 0, "deleted_holds_due_to_confirmed": 0}

    time_min = (now - timedelta(days=max_past_days)).isoformat()
    time_max = now.isoformat()
    page_token = None
    while True:
        resp = svc.events().list(
            calendarId=cal_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            privateExtendedProperty=["ecoLocalKind=HOLD"],
            pageToken=page_token,
        ).execute()
        for ev in resp.get("items", []):
            try:
                delete_event(ev["id"], calendar_id=cal_id, send_updates="none", trace_id=trace_id)
                stats["deleted_past_holds"] += 1
            except Exception:
                log.exception("[calendar] cleanup: failed delete past hold id=%s trace=%s", ev.get("id"), trace_id)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    if not drop_future_if_confirmed:
        log.info("[calendar] cleanup_stale_holds stats=%s trace=%s", stats, trace_id)
        _fb_invalidate(cal_id)
        return stats

    confirmed_map: Dict[str, List[str]] = {}
    page_token = None
    time_min = now.isoformat()
    time_max = (now + timedelta(days=120)).isoformat()
    while True:
        resp = svc.events().list(
            calendarId=cal_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            privateExtendedProperty=["ecoLocalKind=CONFIRMED"],
            pageToken=page_token,
        ).execute()
        for ev in resp.get("items", []):
            thread = ((ev.get("extendedProperties") or {}).get("private") or {}).get("ecoLocalThread")
            if thread:
                confirmed_map.setdefault(thread, []).append(ev["id"])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    for thread_id in confirmed_map.keys():
        try:
            holds = find_thread_holds(
                thread_id=thread_id,
                time_min_iso=(now - timedelta(days=30)).isoformat(),
                time_max_iso=(now + timedelta(days=180)).isoformat(),
                calendar_id=cal_id,
                trace_id=trace_id,
            )
            for h in holds:
                try:
                    delete_event(h["id"], calendar_id=cal_id, send_updates="none", trace_id=trace_id)
                    stats["deleted_holds_due_to_confirmed"] += 1
                except Exception:
                    log.exception("[calendar] cleanup: delete hold id=%s failed trace=%s", h.get("id"), trace_id)
        except Exception:
            log.exception("[calendar] cleanup: fetch holds for thread=%s failed trace=%s", thread_id, trace_id)

    log.info("[calendar] cleanup_stale_holds stats=%s trace=%s", stats, trace_id)
    _fb_invalidate(cal_id)
    return stats
