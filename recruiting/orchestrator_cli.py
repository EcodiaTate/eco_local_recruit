# recruiting/orchestrator_cli.py
from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from urllib.parse import quote

# ---------------- env / dotenv ----------------
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)
except Exception:
    pass

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
from .tools import calendar_suggest_windows
from .tools import semantic_docs, semantic_topk_for_thread

from .outreach import draft_first_touch as outreach_draft_first_touch  # type: ignore
from .outreach import draft_followup as outreach_draft_followup  # type: ignore

# ---- robust discovery-module loader (scrape/parse/qualify/profile) ----
import importlib
import importlib.util
import sys as _sys

HERE = Path(__file__).resolve()
PKG_DIR = HERE.parent
APP_ROOT = PKG_DIR.parent

if str(APP_ROOT) not in _sys.path:
    _sys.path.insert(0, str(APP_ROOT))


def _import_by_spec(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _load_discover_modules():
    errors = []
    try:
        from recruiting import scrape, parse, qualify, profile  # type: ignore
        return scrape, parse, qualify, profile
    except Exception as e:
        errors.append(("recruiting.*", repr(e)))
    try:
        from . import scrape, parse, qualify, profile  # type: ignore
        return scrape, parse, qualify, profile
    except Exception as e:
        errors.append((".<mods>", repr(e)))
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

# ---------------- flags ----------------
DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_FALLBACK_DEFAULT = os.getenv("ECO_LOCAL_ALLOW_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}

REQUIRE_EMAIL_DEFAULT = os.getenv("ECO_LOCAL_REQUIRE_EMAIL", "1").strip().lower() in {"1","true","yes","on"}
MIN_FIT_DEFAULT = float(os.getenv("ECO_LOCAL_MIN_FIT", "0.60"))

import time
import traceback


def _today_or(v: Optional[str]) -> date:
    return date.fromisoformat(v) if v else date.today()

# ---------- Small datetime helpers (CLI-only) ----------

def _dt_local(s: Optional[str], *, default: Optional[datetime] = None) -> datetime:
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
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        try:
            d = date.fromisoformat(s)
            return datetime(d.year, d.month, d.day, tzinfo=tz)
        except Exception as e:
            raise ValueError(f"Unrecognized datetime format: {s!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def _fmt_range(start: datetime, end: datetime) -> str:
    def hhmm(dt: datetime) -> str:
        h = dt.hour % 12 or 12
        return f"{h}:{dt.minute:02d}" if dt.minute else f"{h}"
    return f"{start.strftime('%a %d %b')}, {hhmm(start)}{'am' if start.hour < 12 else 'pm'} - {hhmm(end)}{'am' if end.hour < 12 else 'pm'}"


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

# ---------- Commands: SEMANTIC (local-only) ----------

def _trim_doc(doc: Dict[str, Any], *, max_chars: int) -> Dict[str, Any]:
    text = (doc.get("text") or doc.get("snippet_300") or "")[:max_chars]
    return {
        "id": doc.get("id"),
        "title": doc.get("title"),
        "tags": doc.get("tags") or [],
        "score": doc.get("score"),
        "text": text,
    }

def _pp_docs(docs: List[Dict[str, Any]]) -> None:
    if not docs:
        print("  (no results)")
        return
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.get('title') or '(untitled)'}  (score={d.get('score')})  id={d.get('id')}")
        if d.get("tags"):
            print("    tags:", ", ".join(map(str, d["tags"])))
        txt = (d.get("text") or "").strip()
        if txt:
            preview = txt if len(txt) <= 280 else txt[:280] + f"... (+{len(txt)-280} chars)"
            print("    └─", preview)

def cmd_semantic_query(args: argparse.Namespace) -> None:
    q = args.query.strip()
    docs = semantic_docs(q, k=args.k) or []
    trimmed = [_trim_doc(d, max_chars=args.max_chars) for d in docs]
    print(f"[semantic-query] q={q!r} k={args.k} results={len(trimmed)} (local)")
    if args.json:
        import json
        print(json.dumps(trimmed, ensure_ascii=False, indent=2))
    else:
        _pp_docs(trimmed)

def cmd_semantic_thread(args: argparse.Namespace) -> None:
    envelope: Dict[str, Any] = {
        "thread_id": args.thread_id,
        "message_id": "",
        "from_addr": "",
        "to_addr": "",
        "subject": args.subject or "",
        "received_at_iso": datetime.now(_cal_tz()).isoformat(),
        "plain_body": args.body or "",
        "html_body": None,
        "thread_text": None,
    }
    docs = semantic_topk_for_thread(envelope, k=args.k) or []
    trimmed = [_trim_doc(d, max_chars=args.max_chars) for d in docs]
    print(f"[semantic-thread] thread_id={args.thread_id} k={args.k} results={len(trimmed)} (local)")
    if args.json:
        import json
        print(json.dumps(trimmed, ensure_ascii=False, indent=2))
    else:
        _pp_docs(trimmed)

# ---------- List-Unsubscribe helpers ----------

def _list_unsub_values(to_email: str) -> tuple[Optional[str], Optional[str]]:
    base = os.getenv("UNSUB_BASE_URL") or os.getenv("ECO_LOCAL_UNSUB_BASE_URL")
    url = None
    if base:
        sep = "&" if ("?" in base) else "?"
        url = f"{base}{sep}e={quote(to_email)}"
    mailto_target = os.getenv("ECO_LOCAL_REPLY_TO") or getattr(settings, "SES_SENDER_EMAIL", None) or "ecolocal@ecodia.au"
    mailto = f"mailto:{mailto_target}?subject=unsubscribe"
    return url, mailto

# ---------- Outreach build/send ----------

def cmd_build(args: argparse.Namespace) -> None:
    store.ensure_followup_props()
    run_date = _today_or(args.date)
    run = store.create_run(run_date)

    first_quota = store.get_first_touch_quota()
    first_prospects = store.select_prospects_for_first_touch(first_quota)
    print(f"[build] first-touch prospects: {len(first_prospects)}")
    for p in first_prospects:
        subj, body = outreach_draft_first_touch(p, trace_id="cli-build-first")
        store.attach_draft_email(run, p, "first", subj, body)

    follow_days = store.get_followup_days()
    follow_prospects = store.select_prospects_for_followups(run_date, follow_days)
    print(f"[build] followup prospects: {len(follow_prospects)}")
    max_attempts = store.get_max_attempts()
    for p in follow_prospects:
        attempt_no = int(p.get("attempt_count", 0)) + 1
        kind = f"followup{attempt_no}" if attempt_no < max_attempts else "final"
        try:
            subj, body = outreach_draft_followup(
                p,
                attempt_no=attempt_no,
                max_attempts=max_attempts,
                trace_id="cli-build-follow"
            )
        except Exception as e:
            print(f"[build][ERROR] followup draft failed for {p.get('email')} (attempt={attempt_no}/{max_attempts}): {e}")
            traceback.print_exc()
            continue
        store.attach_draft_email(run, p, kind, subj, body)

    if args.freeze:
        store.freeze_run(run)
        print("[build] run frozen")

    print("[build] staged drafts:")
    for item in store.iter_run_items(run_date):
        print(f"  - {item.type:10s} | {item.email:30s} | {item.subject}")


def cmd_send(args: argparse.Namespace) -> None:
    store.ensure_followup_props()
    run_date = _today_or(args.date)
    sent = 0
    for item in store.iter_run_items(run_date):
        to_addr = item.email
        if DRY_RUN:
            print(f"[DRY_SEND] {to_addr} :: {item.subject}")
            message_id = f"dryrun-{datetime.utcnow().isoformat()}"
        else:
            lu_url, lu_mailto = _list_unsub_values(to_addr)
            message_id = send_email(
                to=to_addr,
                subject=item.subject,
                html=item.body,
                list_unsubscribe_url=lu_url,
                list_unsubscribe_mailto=lu_mailto,
            )
        store.mark_sent(item, message_id)
        try:
            store.mark_followup_sent(to_addr, settings.ECO_LOCAL_FOLLOWUP_DAYS)
        except Exception as e:
            print(f"[send] WARN: could not schedule next follow-up for {to_addr}: {e}")
        sent += 1
    print(f"[send] processed={sent} dry_run={DRY_RUN}")

# ---------- Inbox + holds hygiene ----------

def cmd_poll_inbox(args: argparse.Namespace) -> None:
    processed = hourly_inbox_poll()
    print(f"[inbox] processed={processed}")

def cmd_followups(args: argparse.Namespace) -> None:
    target = _today_or(args.date)
    days = store.get_followup_days()
    prospects = store.select_prospects_for_followups(target, days)
    print(f"[followups] due on {target} for days={days}: {len(prospects)}")
    if not prospects:
        print("  tip: first follow-up hits after days[0] days from last_outreach_at.")
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

# ---------- Discovery (store-only) ----------

def _heuristic_pre_fit(parsed: Dict[str, Any], domain: Optional[str]) -> float:
    if not parsed:
        return 0.30
    emails = list(parsed.get("emails") or [])
    best = (parsed.get("best_email") or "").lower()
    dom = (domain or "").lower().lstrip("www.")

    def is_freemail(e: str) -> bool:
        e = e.lower()
        return any(e.endswith(x) for x in (
            "@gmail.com", "@outlook.com", "@hotmail.com", "@yahoo.com",
            "@icloud.com", "@proton.me", "@pm.me"
        ))

    def is_generic_local(e: str) -> bool:
        pref = (e.split("@",1)[0] if "@" in e else "").lower()
        return pref in {"hello","contact","info","enquiries","admin","team","office",
                        "support","sales","booking","reservations","accounts","media",
                        "press","jobs","careers","privacy","legal","hi","help"}

    score = 0.30
    if emails:
        score += 0.10
    if len(emails) >= 2:
        score += 0.05
    if parsed.get("phones"):
        score += 0.05
    if best:
        if dom and best.endswith("@"+dom):
            score += 0.30
            if is_generic_local(best):
                score -= 0.05
        elif is_freemail(best):
            score += 0.00
        else:
            score += 0.10
    return max(0.0, min(1.0, score))


def _pre_fit_score(parsed: Dict[str, Any], domain: Optional[str]) -> Tuple[float, str]:
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
    store.ensure_dedupe_constraints()

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

        try:
            if hasattr(_qualify, "clean_business_name"):
                cleaned = _qualify.clean_business_name(name=name, domain=domain, city=city_, parsed=parsed or {})
                if cleaned:
                    name = cleaned
        except Exception:
            pass

        category_hint = None
        try:
            cats = (parsed or {}).get("categories") or (parsed or {}).get("tags") or []
            if isinstance(cats, list) and cats:
                category_hint = str(cats[0])[:40]
            elif isinstance((parsed or {}).get("category"), str):
                category_hint = str(parsed.get("category"))[:40]
        except Exception:
            category_hint = None

        try:
            pre_fit, pre_reason = _pre_fit_score(parsed or {}, domain)
        except Exception:
            pre_fit, pre_reason = (0.0, "pre-fit error")
        if pre_fit < min_fit:
            skipped += 1
            store.mark_seen_candidate(domain=domain, email=best_email, name=name)
            print(f"  - skip (below min_fit {min_fit:.2f}): {domain or name} | pre_fit={pre_fit:.2f} ({pre_reason})")
            continue

        if require_email and not best_email:
            skipped += 1
            print(f"  - skip (no email): {domain or name}")
            continue

        if best_email and (store.has_seen_candidate(domain=None, email=best_email) or store.has_prospect(email=best_email)):
            deduped += 1
            continue

        node = store.upsert_prospect(type("P", (), {
            "name": name,
            "email": best_email,
            "domain": domain,
            "city": city_,
            "state": None,
            "country": None,
            "phone": None,
            "source": "discover"
        })())

        store.mark_seen_candidate(domain=domain, email=best_email, name=name)

        try:
            score, reason = _qualify.score_prospect(node, parsed)
        except Exception:
            score, reason = (max(pre_fit, 0.6), "fallback score (qualify error)")

        try:
            store.qualify_basic(node["id"], score=score, reason=reason)
        except Exception:
            print("[discover] warn: qualify_basic failed; score not persisted")

        try:
            _profile.upsert_profile_for_prospect(
                node,
                extra={
                    "parsed": parsed,
                    "source": "discover",
                    "personalize": {
                        "short_name": name,
                        "category_hint": category_hint,
                    },
                },
            )
        except Exception:
            pass

        accepted += 1
        print(f"  - upserted: {best_email or domain or name} | pre_fit={pre_fit:.2f} | qual={score:.2f} | {reason}")

    print(f"[discover] done. accepted={accepted} skipped={skipped} deduped={deduped} require_email={require_email} min_fit={min_fit:.2f}")

# ---------- Promotion / supersede / cancel ----------

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

# ---------- Simple plain-text Chat Completion (no schema) ----------

from typing import Tuple
from openai import OpenAI
from openai import BadRequestError, APIStatusError, APIError, RateLimitError

def _default_model() -> str:
    return (os.getenv("LLM_MODEL") or "gpt-4o").strip()

def _chat_text_once(
    *,
    prompt: str,
    model: str,
    system: Optional[str],
    max_tokens: int,
    temperature: Optional[float],
    force_text: bool,
    use_max_completion: bool,
) -> Tuple[str, Optional[str], bool, bool]:
    """
    Returns (text, finish_reason, bad_param_retryable, temp_retryable)
    bad_param_retryable -> we should try alternate token param or drop response_format
    temp_retryable -> we should retry with temperature=None
    """
    client = OpenAI()
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(prompt)})

    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    if force_text:
        kwargs["response_format"] = {"type": "text"}  # nudge models to return text content

    if use_max_completion:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens

    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        resp = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        choice = resp.choices[0]
        text = (getattr(choice.message, "content", "") or "")
        finish = getattr(choice, "finish_reason", None)
        return str(text), (str(finish) if finish else None), False, False
    except BadRequestError as e:
        msg = (getattr(e, "message", str(e)) or "").lower()
        bad_param_retryable = (
            "unsupported parameter" in msg
            and ("max_tokens" in msg or "max_completion_tokens" in msg or "response_format" in msg)
        )
        temp_retryable = ("unsupported value" in msg and "temperature" in msg)
        return "", None, bad_param_retryable, temp_retryable
    except (RateLimitError, APIStatusError, APIError):
        return "", None, False, False
    except Exception:
        return "", None, False, False

def _chat_text(
    *,
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    max_tokens: int = 800,
    temperature: Optional[float] = None,
) -> Tuple[str, Optional[str]]:
    mdl = (model or _default_model()).strip()

    # Try: response_format=text + max_completion_tokens
    text, finish, bad_param_retry, temp_retry = _chat_text_once(
        prompt=prompt, model=mdl, system=system, max_tokens=max_tokens,
        temperature=temperature, force_text=True, use_max_completion=True,
    )
    if text:
        return text, finish

    # If temperature not accepted, retry without temperature (same params)
    if temp_retry:
        text, finish, bad_param_retry2, _ = _chat_text_once(
            prompt=prompt, model=mdl, system=system, max_tokens=max_tokens,
            temperature=None, force_text=True, use_max_completion=True,
        )
        if text:
            return text, finish
        bad_param_retry = bad_param_retry or bad_param_retry2

    # If bad param (e.g., max_completion_tokens or response_format unsupported), try without response_format and with max_tokens
    if bad_param_retry:
        text, finish, bad_param_retry3, temp_retry3 = _chat_text_once(
            prompt=prompt, model=mdl, system=system, max_tokens=max_tokens,
            temperature=temperature, force_text=False, use_max_completion=False,
        )
        if text:
            return text, finish
        if temp_retry3:
            text, finish, *_ = _chat_text_once(
                prompt=prompt, model=mdl, system=system, max_tokens=max_tokens,
                temperature=None, force_text=False, use_max_completion=False,
            )
            if text:
                return text, finish

    # As a last nudge, add a strong system hint and try once more
    strong_sys = (system + " Reply in plain text only.") if system else "Reply in plain text only."
    text, finish, *_ = _chat_text_once(
        prompt=prompt, model=mdl, system=strong_sys, max_tokens=max_tokens,
        temperature=None, force_text=False, use_max_completion=True,
    )
    return text, finish

def cmd_chat(args: argparse.Namespace) -> None:
    # Read prompt from --prompt or file/stdin
    src = args.prompt
    if src is None:
        if args.input == "-":
            import sys as _sys
            src = _sys.stdin.read()
        elif args.input:
            p = Path(args.input)
            if not p.exists():
                print(f"error: input file not found: {p}")
                return
            src = p.read_text(encoding="utf-8")
    if not src or not str(src).strip():
        print("error: no prompt provided (use --prompt, --input FILE, or pipe via STDIN)")
        return

    text, finish = _chat_text(
        prompt=src,
        model=args.model,
        system=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    # Always print something so it's obvious if content was empty
    print(text.strip() if text and text.strip() else "(no content)")

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
        window = f"{dd}, {hm}{ap} - {(en.hour % 12 or 12)}{':' + str(en.minute).zfill(2) if en.minute else ''}{'am' if en.hour < 12 else 'pm'}"
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

    # --- Semantic (local-only) ---
    sq = sub.add_parser("semantic-query", help="Local semantic retrieval for a free-form query")
    sq.add_argument("--query", "-q", required=True)
    sq.add_argument("--k", type=int, default=5)
    sq.add_argument("--max-chars", type=int, default=600)
    sq.add_argument("--json", action="store_true", help="Print raw JSON")
    sq.set_defaults(func=cmd_semantic_query)

    st = sub.add_parser("semantic-thread", help="Local semantic retrieval for a Gmail thread id")
    st.add_argument("--thread-id", required=True)
    st.add_argument("--subject", type=str, help="Optional subject hint")
    st.add_argument("--body", type=str, help="Optional body hint")
    st.add_argument("--k", type=int, default=5)
    st.add_argument("--max-chars", type=int, default=600)
    st.add_argument("--json", action="store_true", help="Print raw JSON")
    st.set_defaults(func=cmd_semantic_thread)

    # --- Calendar / scheduling helpers ---
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

    # --- Outreach build/send ---
    b = sub.add_parser("build", help="Build (idempotent) drafts for a run date (first + followups)")
    b.add_argument("--date", type=str)
    b.add_argument("--freeze", action="store_true")
    b.set_defaults(func=cmd_build)

    pr = sub.add_parser("cal-promote", help="Promote a HOLD to CONFIRMED and notify")
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--thread-id")
    g.add_argument("--event-id")
    pr.add_argument("--title", default=None)
    pr.add_argument("--email", default=None)
    pr.set_defaults(func=cmd_cal_promote)

    se = sub.add_parser("send", help="Send drafts (first + followups) for the run date")
    se.add_argument("--date", type=str)
    se.set_defaults(func=cmd_send)

    # --- Inbox + hygiene ---
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

    # --- Discovery ---
    dp = sub.add_parser("discover", help="LLM-powered business discovery → parse → score → profile (with min-fit gate)")
    dp.add_argument("--query", required=True)
    dp.add_argument("--city", required=True)
    dp.add_argument("--limit", type=int, default=10)
    on = dp.add_mutually_exclusive_group()
    on.add_argument("--require-email", dest="require_email", action="store_true", default=REQUIRE_EMAIL_DEFAULT)
    on.add_argument("--no-require-email", dest="require_email", action="store_false")
    dp.set_defaults(func=cmd_discover)

    # --- Plain text chat completion (super simple) ---
    ch = sub.add_parser("chat", help="Plain text chat completion (no schema). Prints raw text.")
    ch.add_argument("--prompt", type=str, help="Inline prompt text. If omitted, use --input or STDIN.")
    ch.add_argument("--input", type=str, help="Read prompt from FILE or '-' for STDIN.")
    ch.add_argument("--model", type=str, help="Model name (defaults to env LLM_MODEL or gpt-4o).")
    ch.add_argument("--system", type=str, help="Optional system message (style/role).")
    ch.add_argument("--max-tokens", type=int, default=800, help="Max completion tokens (or equivalent).")
    ch.add_argument("--temperature", type=float, help="Optional temperature. Leave unset for strict models.")
    ch.set_defaults(func=cmd_chat)

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


def invoke(argv: List[str]) -> int:
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
