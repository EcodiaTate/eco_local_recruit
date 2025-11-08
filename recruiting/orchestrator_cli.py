# recruiting/orchestrator_cli.py
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

# ---------------- env / dotenv ----------------
try:
    from dotenv import load_dotenv, find_dotenv
    # Load from CWD first, then repo root (…/eco_local/.env)
    load_dotenv(find_dotenv(usecwd=True))
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)
except Exception:
    pass  # works without python-dotenv

# ---------------- sys.path shim (run-as-script friendliness) ----------------
import sys as _sys
_pkg_dir = Path(__file__).resolve().parents[1]   # .../eco_local
_repo_dir = _pkg_dir.parent                      # .../
if _pkg_dir.name == "eco_local" and str(_repo_dir) not in _sys.path:
    _sys.path.insert(0, str(_repo_dir))

# ---------------- local imports ----------------
from . import store
from .sender import send_email
from .inbox import hourly_inbox_poll, triage_and_update
from .calendar_client import (
    cleanup_stale_holds,
    find_free_slots,
    is_range_free,
    list_holds,
    find_thread_confirmed,
)
from .calendar_client import _tz as _cal_tz  # type: ignore
from .calendar_client import _build_calendar_service  # type: ignore
from .config import settings
from .tools import calendar_suggest_windows  # agentic, scored windows

# ---- robust discovery-module loader (scrape/parse/qualify/profile) ----
import importlib
import importlib.util

HERE = Path(__file__).resolve()
PKG_DIR = HERE.parent            # /app/recruiting
APP_ROOT = PKG_DIR.parent        # /app

def _import_by_spec(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

def _load_discover_modules():
    """
    Try to import scrape/parse/qualify/profile that live next to this file.
    Strategies:
      1) from recruiting import scrape, parse, qualify, profile
      2) from . import scrape, parse, qualify, profile
      3) direct file import from PKG_DIR/*.py
    """
    errors = []

    # 1) absolute package import
    try:
        from recruiting import scrape, parse, qualify, profile  # type: ignore
        return scrape, parse, qualify, profile
    except Exception as e:
        errors.append(("recruiting.*", repr(e)))

    # 2) package-relative
    try:
        from . import scrape, parse, qualify, profile  # type: ignore
        return scrape, parse, qualify, profile
    except Exception as e:
        errors.append((".<mods>", repr(e)))

    # 3) direct-by-file
    try:
        scr = _import_by_spec("recruiting.scrape", PKG_DIR / "scrape.py")
        par = _import_by_spec("recruiting.parse", PKG_DIR / "parse.py")
        qua = _import_by_spec("recruiting.qualify", PKG_DIR / "qualify.py")
        pro = _import_by_spec("recruiting.profile", PKG_DIR / "profile.py")
        return scr, par, qua, pro
    except Exception as e:
        errors.append(("by-spec", repr(e)))

    print("[discover] import failed across strategies.")
    print(f"[discover] __file__: {HERE}")
    print(f"[discover] cwd: {Path.cwd()}")
    print(f"[discover] sys.path[0]: {_sys.path[0] if _sys.path else ''}")
    print(f"[discover] sys.path: {_sys.path}")
    for rel in ["scrape.py", "parse.py", "qualify.py", "profile.py", "__init__.py"]:
        print(f"[discover] exists {rel}? -> {(PKG_DIR / rel).exists()}")
    for where, err in errors:
        print(f"[discover]   strategy '{where}' error: {err}")
    raise ImportError("Failed to import discovery modules via all strategies.")

_scrape, _parse, _qualify, _profile = _load_discover_modules()

# ---- robust seed helpers loader (upsert_prospect / qualify_basic) ----
def _load_seed_helpers():
    """
    Try several import paths for seed_cli; otherwise proxy to store.* if available.
    """
    # 1) recruiting.seed_cli
    try:
        from recruiting.seed_cli import upsert_prospect, qualify_basic  # type: ignore
        return upsert_prospect, qualify_basic
    except Exception:
        pass
    # 2) top-level seed_cli
    try:
        from seed_cli import upsert_prospect, qualify_basic  # type: ignore
        return upsert_prospect, qualify_basic
    except Exception:
        pass
    # 3) eco_local.seed_cli (monorepo style)
    try:
        from eco_local.seed_cli import upsert_prospect, qualify_basic  # type: ignore
        return upsert_prospect, qualify_basic
    except Exception:
        pass
    # 4) direct-by-file (repo_root/seed_cli.py or /app/seed_cli.py)
    for cand in [APP_ROOT / "seed_cli.py", APP_ROOT.parent / "seed_cli.py"]:
        if cand.exists():
            mod = _import_by_spec("seed_cli", cand)
            up = getattr(mod, "upsert_prospect", None)
            qb = getattr(mod, "qualify_basic", None)
            if callable(up) and callable(qb):
                return up, qb

    # 5) proxy to store if it exposes similar helpers
    def _store_upsert_proxy(p):
        for fname in ("upsert_prospect", "create_prospect", "upsert_or_create_prospect"):
            fn = getattr(store, fname, None)
            if callable(fn):
                return fn(p)
        raise RuntimeError(
            "No seed_cli and store lacks upsert helper. Provide seed_cli.upsert_prospect "
            "or implement store.upsert_prospect/create_prospect."
        )

    def _store_qualify_proxy(node_id: str, score: float, reason: str):
        for fname in ("qualify_basic", "set_prospect_score", "update_prospect_score"):
            fn = getattr(store, fname, None)
            if callable(fn):
                return fn(node_id, score=score, reason=reason)
        # If nothing, act as no-op
        print("[discover] qualify proxy: no store helper; skipping score update.")
        return None

    return _store_upsert_proxy, _store_qualify_proxy

upsert_prospect, qualify_basic = _load_seed_helpers()

# ---------------- flags ----------------
DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_FALLBACK_DEFAULT = os.getenv("ECO_LOCAL_ALLOW_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}

# Require email to accept a candidate (kill switch). Default ON; override with CLI.
REQUIRE_EMAIL_DEFAULT = os.getenv("ECO_LOCAL_REQUIRE_EMAIL", "1").strip().lower() in {"1","true","yes","on"}

# Pre-fit threshold for discovery acceptance (prevents low-fit upserts)
MIN_FIT_DEFAULT = float(os.getenv("ECO_LOCAL_MIN_FIT", "0.60"))

# ---------------- outreach binding ----------------
import traceback
import time

def _bind_outreach(allow_fallback: bool):
    try:
        from . import outreach as _outreach

        def _draft_first_touch(p: Dict[str, Any]) -> tuple[str, str]:
            trace_id = f"cli-{int(time.time()*1000)}"
            try:
                return _outreach.draft_first_touch(p, trace_id=trace_id)
            except TypeError:
                return _outreach.draft_first_touch(p)

        def _draft_followup(p: Dict[str, Any], attempt_no: int) -> tuple[str, str]:
            trace_id = f"cli-{int(time.time()*1000)}"
            try:
                return _outreach.draft_followup(p, attempt=attempt_no, trace_id=trace_id)
            except TypeError:
                return _outreach.draft_followup(p, attempt=attempt_no)

        return _draft_first_touch, _draft_followup, "llm"
    except Exception as e:
        print("\n[orchestrator] outreach import failed. Full traceback:")
        traceback.print_exc()
        if not allow_fallback:
            raise RuntimeError(
                "Failed to import recruiting.outreach (LLM drafters). "
                "Set ECO_LOCAL_ALLOW_FALLBACK=1 or pass --allow-fallback."
            ) from e

        def _draft_first_touch_fallback(p: Dict[str, Any]) -> tuple[str, str]:
            subj = f"Ecodia: quick intro for {p.get('name') or p.get('business_name') or 'your team'}"
            body = (
                "<p>Hey! We’re building local value loops in Ecodia - "
                "youth earn ECO for real actions, then retire ECO at local businesses.</p>"
                "<p>Would you be open to a 20–30 min intro this week?</p>"
            )
            return subj, body

        def _draft_followup_fallback(p: Dict[str, Any], attempt_no: int) -> tuple[str, str]:
            subj = f"Circling back - Ecodia × {p.get('name') or p.get('business_name') or 'your team'}"
            body = (
                "<p>Following up on my note about ECO (earn from actions → retire at local businesses). "
                "Happy to share how it works and show live results.</p>"
            )
            return subj, body

        return _draft_first_touch_fallback, _draft_followup_fallback, "fallback"

def _today_or(v: Optional[str]) -> date:
    return date.fromisoformat(v) if v else date.today()

# ---------- Small datetime helpers (CLI-only) ----------
from typing import Optional  # ensure this import exists at top

def _dt_local(s: Optional[str], *, default: Optional[datetime] = None) -> datetime:
    """
    Parse a local-friendly datetime string into a timezone-aware datetime.
    """
    tz = _cal_tz()

    if not s:
        return default or datetime.now(tz)

    low = s.strip().lower()
    now = datetime.now(tz)

    if low == "now":
        return now
    if low == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if low == "tomorrow":
        tmr = now + timedelta(days=1)
        return tmr.replace(hour=0, minute=0, second=0, microsecond=0)

    # Try full ISO first
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        # Try date-only ISO
        try:
            d = date.fromisoformat(s)
            return datetime(d.year, d.month, d.day, tzinfo=tz)
        except Exception as e:
            raise ValueError(f"Unrecognized datetime format: {s!r}") from e

    # Normalize timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def _fmt_range(start: datetime, end: datetime) -> str:
    def hhmm(dt: datetime) -> str:
        h = dt.hour % 12 or 12
        return f"{h}:{dt.minute:02d}" if dt.minute else f"{h}"
    return f"{start.strftime('%a %d %b')}, {hhmm(start)}{'am' if start.hour < 12 else 'pm'}–{hhmm(end)}{'am' if end.hour < 12 else 'pm'}"

def _print_events(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("  (none)")
        return
    for ev in rows:
        st = ev.get("start", {}).get("dateTime") or ev.get("start")
        en = ev.get("end", {}).get("dateTime") or ev.get("end")
        kind = ((ev.get("extendedProperties") or {}).get("private") or {}).get("ecoLocalKind") or "?"
        title = ev.get("summary") or "(no title)"
        try:
            sdt = datetime.fromisoformat((st or "").replace("Z", "+00:00")).astimezone(_cal_tz())
            edt = datetime.fromisoformat((en or "").replace("Z", "+00:00")).astimezone(_cal_tz())
            label = _fmt_range(sdt, edt)
        except Exception:
            label = f"{st} → {en}"
        print(f"  - {label} | {kind:9s} | {title}")

# ---------- Calendar: raw freebusy ----------
def _freebusy_blocks(start: datetime, end: datetime, *, calendar_id: Optional[str]) -> List[Tuple[datetime, datetime]]:
    svc = _build_calendar_service()
    cal_id = calendar_id or settings.ECO_LOCAL_GCAL_ID
    tz = _cal_tz()
    body = {
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "timeZone": str(tz),
        "items": [{"id": cal_id}],
    }
    fb = svc.freebusy().query(body=body).execute() or {}
    items = (fb.get("calendars", {}).get(cal_id, {}) or {}).get("busy", []) or []
    out: List[Tuple[datetime, datetime]] = []
    for b in items:
        try:
            s = datetime.fromisoformat(b["start"].replace("Z", "+00:00")).astimezone(tz)
            e = datetime.fromisoformat(b["end"].replace("Z", "+00:00")).astimezone(tz)
            if e > s:
                out.append((s, e))
        except Exception:
            pass
    return out

# ---------- Commands: calendar ----------

def cmd_slots(args: argparse.Namespace) -> None:
    suggestions = calendar_suggest_windows(
        lookahead_days=args.days,
        duration_min=args.hold,
        work_hours=(9, 17),
        weekdays={0,1,2,3,4},
        min_gap_min=15,
        hold_padding_min=10,
    )
    print(f"Suggested windows (not commitments): {len(suggestions)}")
    for s in suggestions[:10]:
        print(" -", _human_label(s))

def cmd_cal_freebusy(args: argparse.Namespace) -> None:
    tz = _cal_tz()
    start = _dt_local(args.start)
    end   = _dt_local(args.end, default=(start + timedelta(days=args.days))).replace(tzinfo=tz)
    blocks = _freebusy_blocks(start, end, calendar_id=args.calendar)
    print(f"[freebusy] calendar={args.calendar or settings.ECO_LOCAL_GCAL_ID} tz={tz} window={start.isoformat()}→{end.isoformat()}")
    if not blocks:
        print("  (no busy blocks)")
        return
    for s, e in blocks:
        print("  -", _fmt_range(s, e), f"({(e - s).seconds//60} min)")

def cmd_cal_check(args: argparse.Namespace) -> None:
    tz = _cal_tz()
    start = _dt_local(args.start)
    end   = _dt_local(args.end, default=(start + timedelta(days=args.days)))
    ok = is_range_free(start.isoformat(), end.isoformat(), trace_id="cli-check")
    print(f"[check] {('FREE' if ok else 'BUSY ')} :: {_fmt_range(start, end)}  tz={tz}")

def cmd_cal_hold(args: argparse.Namespace) -> None:
    from .calendar_client import create_hold
    tz = _cal_tz()
    start = _dt_local(args.start)
    end = _dt_local(args.end, default=(start + timedelta(minutes=args.hold)))
    ev = create_hold(
        title=args.title,
        description=args.desc,
        start_iso=start.isoformat(),
        end_iso=end.isoformat(),
        thread_id=args.thread_id,
        prospect_email=args.email,
        expires_in_hours=args.expires,
        send_updates="none",
        trace_id="cli-hold",
    )
    print("[hold] created:", ev.get("id"), ev.get("htmlLink"))

def cmd_cal_thread(args: argparse.Namespace) -> None:
    tz = _cal_tz()
    start = _dt_local(args.start) if args.start else datetime.now(tz) - timedelta(days=args.back)
    end   = _dt_local(args.end) if args.end else datetime.now(tz) + timedelta(days=args.forward)
    print(f"[thread] id={args.thread_id} window={start.isoformat()}→{end.isoformat()} tz={tz}")
    conf = find_thread_confirmed(thread_id=args.thread_id, time_min_iso=start.isoformat(), time_max_iso=end.isoformat(), trace_id="cli-thread")
    holds = list_holds(thread_id=args.thread_id, lookback_days=args.back, lookahead_days=args.forward, trace_id="cli-thread")
    print("CONFIRMED:"); _print_events(conf)
    print("HOLDS:"); _print_events(holds)

# ---------- Commands: outreach build/send ----------

def cmd_build(args: argparse.Namespace) -> None:
    allow_fallback = args.allow_fallback or ALLOW_FALLBACK_DEFAULT
    draft_first, draft_follow, mode = _bind_outreach(allow_fallback)
    print(f"[build] drafter_mode={mode} allow_fallback={allow_fallback}")

    run_date = _today_or(args.date)
    run = store.create_run(run_date)

    first_quota = store.get_first_touch_quota()
    first_prospects = store.select_prospects_for_first_touch(first_quota)
    print(f"[build] first-touch prospects: {len(first_prospects)}")
    for p in first_prospects:
        subj, body = draft_first(p)
        store.attach_draft_email(run, p, "first", subj, body)

    follow_days = store.get_followup_days()
    follow_prospects = store.select_prospects_for_followups(run_date, follow_days)
    print(f"[build] followup prospects: {len(follow_prospects)}")
    for p in follow_prospects:
        attempt_no = int(p.get("attempt_count", 0)) + 1
        subj, body = draft_follow(p, attempt_no)
        kind = f"followup{attempt_no}" if attempt_no < store.get_max_attempts() else "final"
        store.attach_draft_email(run, p, kind, subj, body)

    if args.freeze:
        store.freeze_run(run)
        print("[build] run frozen")

    print("[build] staged drafts:")
    for item in store.iter_run_items(run_date):
        print(f"  - {item.type:10s} | {item.email:30s} | {item.subject}")

def cmd_send(args: argparse.Namespace) -> None:
    run_date = _today_or(args.date)
    sent = 0
    for item in store.iter_run_items(run_date):
        if DRY_RUN:
            print(f"[DRY_SEND] {item.email} :: {item.subject}")
            message_id = f"dryrun-{datetime.utcnow().isoformat()}"
        else:
            message_id = send_email(to=item.email, subject=item.subject, html=item.body)
        store.mark_sent(item, message_id)
        sent += 1
    print(f"[send] processed={sent} dry_run={DRY_RUN}")

# ---------- Commands: inbox + holds hygiene ----------

def cmd_poll_inbox(args: argparse.Namespace) -> None:
    processed = hourly_inbox_poll()
    print(f"[inbox] processed={processed}")

def cmd_followups(args: argparse.Namespace) -> None:
    target = _today_or(args.date)
    days = store.get_followup_days()
    prospects = store.select_prospects_for_followups(target, days)
    print(f"[followups] due on {target} for days={days}: {len(prospects)}")
    for p in prospects[:20]:
        print(f" - {p.get('email')} (attempt_count={p.get('attempt_count')})")

def cmd_demo_inbound(args: argparse.Namespace) -> None:
    msg: Dict[str, Any] = {
        "from": args.from_addr,
        "subject": args.subject or "",
        "snippet": args.body or "",
        "body_text": args.body or "",
    }
    if args.action == "unsubscribe":
        msg["subject"] = msg["subject"] or "Unsubscribe please"
        msg["body_text"] = msg["body_text"] or "can you unsubscribe me?"
    elif args.action == "mark_won":
        msg["body_text"] = msg["body_text"] or "we're in"
    elif args.action == "reply":
        msg["subject"] = msg["subject"] or "How much?"
        msg["body_text"] = msg["body_text"] or "what is ECO Local cost?"
    decision = triage_and_update(msg)
    print("[demo-inbound] decision:", decision)

def cmd_cleanup_holds(args: argparse.Namespace) -> None:
    stats = cleanup_stale_holds(
        max_past_days=args.max_past_days,
        drop_future_if_confirmed=(not args.no_drop_future_if_confirmed),
    )
    print(f"[calendar] cleanup stats={stats}")

# ---------- Commands: discovery / seeding (updated with pre-fit threshold) ----------

def _heuristic_pre_fit(parsed: Dict[str, Any], domain: Optional[str]) -> float:
    """
    Lightweight pre-fit estimate when _qualify has no pre-scorer.
    """
    if not parsed:
        return 0.30
    emails = list(parsed.get("emails") or [])
    best = (parsed.get("best_email") or "").lower()
    dom = (domain or "").lower().lstrip("www.")
    def is_freemail(e: str) -> bool:
        e = e.lower()
        return any(e.endswith(x) for x in (
            "@gmail.com", "@outlook.com", "@hotmail.com", "@yahoo.com", "@icloud.com", "@proton.me", "@pm.me"
        ))
    def is_generic_local(e: str) -> bool:
        pref = (e.split("@",1)[0] if "@" in e else "").lower()
        return pref in {
            "hello","contact","info","enquiries","admin","team","office",
            "support","sales","booking","reservations","accounts","media",
            "press","jobs","careers","privacy","legal","hi","help",
        }

    score = 0.30
    if emails:
        score += 0.10
    if len(emails) >= 2:
        score += 0.05
    if parsed.get("phones"):
        score += 0.05
    if best:
        if dom and best.endswith("@"+dom):
            score += 0.30  # strong in-domain signal
            if is_generic_local(best):
                score -= 0.05
        elif is_freemail(best):
            score += 0.00  # neutral; let post-qualify decide
        else:
            score += 0.10  # non-freemail but off-domain (could be vendor)
    return max(0.0, min(1.0, score))

def _pre_fit_score(parsed: Dict[str, Any], domain: Optional[str]) -> Tuple[float, str]:
    """
    Try a pre-fit from _qualify if available (e.g., score_from_parsed / pre_score),
    else fall back to heuristic.
    """
    try:
        if hasattr(_qualify, "pre_score"):
            s, reason = _qualify.pre_score(parsed)  # type: ignore
            return float(s or 0.0), str(reason or "pre_score")
        if hasattr(_qualify, "score_from_parsed"):
            s, reason = _qualify.score_from_parsed(parsed)  # type: ignore
            return float(s or 0.0), str(reason or "score_from_parsed")
    except Exception:
        pass
    return _heuristic_pre_fit(parsed, domain), "heuristic_pre_fit"

def cmd_discover(args: argparse.Namespace) -> None:
    store.ensure_dedupe_constraints()  # make sure constraints exist

    query = args.query
    city = args.city
    limit = int(args.limit)
    require_email = args.require_email
    min_fit = float(os.getenv("ECO_LOCAL_MIN_FIT", str(MIN_FIT_DEFAULT)))

    print(f"[discover] query={query!r} city={city!r} limit={limit} min_fit={min_fit:.2f} require_email={require_email}")
    places: List[Dict[str, Any]] = _scrape.discover_places(query=query, city=city, limit=limit)
    print(f"[discover] candidates={len(places)}")

    accepted = 0
    skipped = 0
    deduped = 0

    for pl in places:
        name = pl.get("name") or "Unknown"
        domain = pl.get("domain") or pl.get("website")
        city_ = pl.get("city") or city

        # ---- HARD DEDUPE GUARDS (seen or existing prospect)
        if store.has_seen_candidate(domain=domain, email=None) or store.has_prospect(domain=domain):
            deduped += 1
            continue

        homepage_html = _scrape.fetch_homepage(domain) if domain else ""
        mailto_emails = _scrape.harvest_emails_for_domain(domain) if domain else []
        parsed = _parse.extract_contacts(
            homepage_html,
            domain=domain,
            extra_emails=mailto_emails
        ) if (homepage_html or mailto_emails) else {}
        best_email = (parsed or {}).get("best_email")

        # ---------- PRE-FIT GATE ----------
        try:
            pre_fit, pre_reason = _pre_fit_score(parsed or {}, domain)
        except Exception:
            pre_fit, pre_reason = (0.0, "pre-fit error")
        if pre_fit < min_fit:
            skipped += 1
            # mark as seen to avoid re-enqueue in future passes:
            store.mark_seen_candidate(domain=domain, email=best_email, name=name)
            print(f"  - skip (below min_fit {min_fit:.2f}): {domain or name} | pre_fit={pre_fit:.2f} ({pre_reason})")
            continue
        # ----------------------------------

        # Kill switch: require email unless overridden
        if require_email and not best_email:
            skipped += 1
            # If you still want the domain to be marked seen even when no email, uncomment below
            # store.mark_seen_candidate(domain=domain, email=None, name=name)
            print(f"  - skip (no email): {domain or name}")
            continue

        # Extra dedupe: if we've already seen or have a prospect by email, skip
        if best_email and (store.has_seen_candidate(domain=None, email=best_email) or store.has_prospect(email=best_email)):
            deduped += 1
            continue

        # Upsert prospect (include email if we have it)
        try:
            node = upsert_prospect(type("P", (), {
                "name": name,
                "email": best_email,
                "domain": domain,
                "city": city_,
                "state": None,
                "country": None,
                "phone": None,
                "source": "discover"
            })())
        except Exception as e:
            # Make the failure extremely obvious and actionable
            raise RuntimeError(
                "Failed to upsert prospect. Ensure seed_cli.upsert_prospect exists OR expose a "
                "compatible helper on store (upsert_prospect/create_prospect)."
            ) from e

        # Immediately mark “seen” so later passes won’t re-enqueue
        store.mark_seen_candidate(domain=domain, email=best_email, name=name)

        # Qualify with tuned rubric (post-upsert definitive score)
        try:
            score, reason = _qualify.score_prospect(node, parsed)
        except Exception:
            # keep at least pre-fit rather than nuking a good pre-signal
            score, reason = (max(pre_fit, 0.6), "fallback score (qualify error)")
        try:
            qualify_basic(node["id"], score=score, reason=reason)
        except Exception:
            print("[discover] warn: qualify_basic missing; score not persisted")

        # Profile node
        try:
            _profile.upsert_profile_for_prospect(node, extra={"parsed": parsed, "source": "discover"})
        except Exception:
            pass

        accepted += 1
        print(f"  - upserted: {best_email or domain or name} | pre_fit={pre_fit:.2f} | qual={score:.2f} | {reason}")

    print(f"[discover] done. accepted={accepted} skipped={skipped} deduped={deduped} require_email={require_email} min_fit={min_fit:.2f}")

def cmd_cal_promote(args: argparse.Namespace) -> None:
    from .calendar_client import promote_hold_to_event
    ev = promote_hold_to_event(
        thread_id=args.thread_id if not args.event_id else None,
        event_id=args.event_id,
        new_title=args.title,
        ensure_attendee_email=args.email,
        trace_id="cli-promote",
    )
    print("[promote] id:", ev.get("id"), "| kind:",
          ((ev.get("extendedProperties") or {}).get("private") or {}).get("ecoLocalKind"),
          "|", ev.get("htmlLink"))

# ---------- Parser ----------
def _human_label(s: Dict[str, Any]) -> str:
    try:
        from datetime import datetime as _dt
        st = _dt.fromisoformat((s.get("start") or "").replace("Z", "+00:00"))
        en = _dt.fromisoformat((s.get("end") or "").replace("Z", "+00:00"))
        hh = st.hour % 12 or 12
        hm = f"{hh}:{st.minute:02d}" if st.minute else f"{hh}"
        ap = "am" if st.hour < 12 else "pm"
        dd = st.strftime("%a %d %b")
        window = f"{dd}, {hm}{ap}–{(en.hour % 12 or 12)}{':' + str(en.minute).zfill(2) if en.minute else ''}{'am' if en.hour < 12 else 'pm'}"
    except Exception:
        window = f"{s.get('start')} → {s.get('end')}"
    score = s.get("score")
    reason = s.get("reason")
    bits = [window]
    if score is not None:
        bits.append(f"(score={score:.2f})")
    if reason:
        bits.append(f"- {reason}")
    return " ".join(bits)

def cmd_cal_supersede(args: argparse.Namespace) -> None:
    from .calendar_client import supersede_thread_booking
    start = _dt_local(args.start)
    end = _dt_local(args.end, default=(start + timedelta(minutes=args.hold)))
    ev = supersede_thread_booking(
        thread_id=args.thread_id,
        new_start_iso=start.isoformat(),
        new_end_iso=end.isoformat(),
        tz=str(_cal_tz()),
        prospect_email=args.email,
        trace_id="cli-supersede",
    )
    print("[supersede] id:", ev.get("id"), ev.get("htmlLink"))

def cmd_cal_cancel_thread(args: argparse.Namespace) -> None:
    from .calendar_client import cancel_thread_events
    stats = cancel_thread_events(thread_id=args.thread_id, trace_id="cli-cancel-thread")
    print("[cancel-thread] stats:", stats)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ECO Local Orchestrator CLI (idempotent)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("slots", help="Suggest human-friendly windows (non-binding)")
    s.add_argument("--days", type=int, default=10)
    s.add_argument("--hold", type=int, default=30)
    s.set_defaults(func=cmd_slots)

    fb = sub.add_parser("cal-freebusy", help="Show raw Google Calendar busy blocks")
    fb.add_argument("--start", type=str, default="today")
    fb.add_argument("--end", type=str)
    fb.add_argument("--days", type=int, default=14)
    fb.add_argument("--calendar", type=str)
    fb.set_defaults(func=cmd_cal_freebusy)

    sp = sub.add_parser("cal-supersede", help="Reschedule confirmed for a thread (or create)")
    sp.add_argument("--thread-id", required=True)
    sp.add_argument("--start", required=True)
    sp.add_argument("--end")
    sp.add_argument("--hold", type=int, default=30)
    sp.add_argument("--email")
    sp.set_defaults(func=cmd_cal_supersede)

    ct = sub.add_parser("cal-cancel-thread", help="Delete all HOLD/CONFIRMED for a thread")
    ct.add_argument("--thread-id", required=True)
    ct.set_defaults(func=cmd_cal_cancel_thread)

    ck = sub.add_parser("cal-check", help="Check if a specific range is free")
    ck.add_argument("--start", type=str, required=True)
    ck.add_argument("--end", type=str)
    ck.add_argument("--hold", type=int, default=30)
    ck.set_defaults(func=cmd_cal_check)

    hd = sub.add_parser("cal-hold", help="Create a HOLD event")
    hd.add_argument("--start", required=True)
    hd.add_argument("--end")
    hd.add_argument("--hold", type=int, default=30)
    hd.add_argument("--title", default="Ecodia - intro chat (HOLD)")
    hd.add_argument("--desc", default="Provisional hold (auto-expires unless confirmed).")
    hd.add_argument("--thread-id", required=True)
    hd.add_argument("--email")
    hd.add_argument("--expires", type=int, default=48)
    hd.set_defaults(func=cmd_cal_hold)

    th = sub.add_parser("cal-thread", help="List HOLD/CONFIRMED events for a thread")
    th.add_argument("--thread-id", required=True)
    th.add_argument("--back", type=int, default=30)
    th.add_argument("--forward", type=int, default=90)
    th.add_argument("--start", type=str)
    th.add_argument("--end", type=str)
    th.set_defaults(func=cmd_cal_thread)

    b = sub.add_parser("build", help="Build (idempotent) drafts for a run date")
    b.add_argument("--date", type=str)
    b.add_argument("--freeze", action="store_true")
    b.add_argument("--allow-fallback", action="store_true")
    b.set_defaults(func=cmd_build)

    pr = sub.add_parser("cal-promote", help="Promote a HOLD to CONFIRMED and notify")
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--thread-id")
    g.add_argument("--event-id")
    pr.add_argument("--title", default=None)
    pr.add_argument("--email", default=None)
    pr.set_defaults(func=cmd_cal_promote)

    se = sub.add_parser("send", help="Send drafts (or print with DRY_RUN=1)")
    se.add_argument("--date", type=str)
    se.set_defaults(func=cmd_send)

    pi = sub.add_parser("poll-inbox", help="Process Gmail inbox")
    pi.set_defaults(func=cmd_poll_inbox)

    fu = sub.add_parser("followups", help="List prospects due for followups on a date")
    fu.add_argument("--date", type=str)
    fu.set_defaults(func=cmd_followups)

    di = sub.add_parser("demo-inbound", help="Exercise triage paths without Gmail")
    di.add_argument("--action", choices=["unsubscribe", "mark_won", "reply"], required=True)
    di.add_argument("--from", dest="from_addr", required=True)
    di.add_argument("--subject", type=str)
    di.add_argument("--body", type=str)
    di.set_defaults(func=cmd_demo_inbound)

    cl = sub.add_parser("cleanup-holds", help="Delete stale holds; prune holds when a thread has a confirmed event")
    cl.add_argument("--max-past-days", type=int, default=14)
    cl.add_argument("--no-drop-future-if-confirmed", action="store_true")
    cl.set_defaults(func=cmd_cleanup_holds)

    # NEW: discovery with kill switch flag (defaults to env=ON)
    dp = sub.add_parser("discover", help="LLM-powered business discovery → parse → score → profile (with min-fit gate)")
    dp.add_argument("--query", required=True)
    dp.add_argument("--city", required=True)
    dp.add_argument("--limit", type=int, default=10)
    on = dp.add_mutually_exclusive_group()
    on.add_argument("--require-email", dest="require_email", action="store_true", default=REQUIRE_EMAIL_DEFAULT)
    on.add_argument("--no-require-email", dest="require_email", action="store_false")
    dp.set_defaults(func=cmd_discover)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    finally:
        try:
            store.close_driver()
        except Exception:
            pass

# --- Programmatic entrypoint for Cloud Run / FastAPI ---
def invoke(argv: list[str]) -> int:
    """
    Run the orchestrator CLI programmatically, e.g. invoke(["build","--date","2025-11-04","--freeze"])
    Returns 0 on success; raises on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    finally:
        try:
            store.close_driver()
        except Exception:
            pass

if __name__ == "__main__":
    main()
