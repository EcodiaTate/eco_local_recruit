# recruiting/browser_fetch.py
# pyright: reportMissingImports=false
from __future__ import annotations

import re
import sys, atexit
import asyncio
import socket
from typing import List, Set, Deque, Optional, Tuple
from urllib.parse import urlparse, urljoin
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# -------------------- Regexes / Hints --------------------

_EMAIL_RE_TEXT = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)

# Broader set of labels we treat as "contact-ish"
_CONTACT_HINTS = re.compile(
    r"\b(contact|contact-us|contactus|get[-\s]?in[-\s]?touch|find[-\s]?us|visit|where|"
    r"location|locations|about|about-us|team|support|help|enquiries?|enquiry|feedback|"
    r"book|booking|reservations?)\b",
    re.I,
)

# JS string-join patterns that form emails in markup
_JS_STR_JOIN_EMAIL = re.compile(
    r"(?:['\"][a-z0-9._%+\-@]+['\"]\s*\+\s*)+['\"][a-z0-9._%+\-@]+\.[a-z]{2,}['\"]",
    re.I,
)

# -------------------- URL helpers --------------------
# --- deobfuscation + filters -----------------------------------------------

_BAD_DOMAINS = {
    # theme/demo placeholders
    "enfold-restaurant.com",
    "example.com",
    "example.org",
    "example.net",
    # policy vendors / analytics junk patterns can creep in
    "privacypolicy",
    "termsfeed",
    "cookiebot",
    "iubenda",
}

_ALLOWED_TLDS = (
    ".com",
    ".com.au",
    ".org",
    ".org.au",
    ".net",
    ".net.au",
    ".edu.au",
    ".gov.au",
    ".co",
    ".io",
)

_GENERIC_INBOX_PREFIXES = (
    "hello@",
    "contact@",
    "info@",
    "enquiries@",
    "admin@",
    "team@",
    "office@",
    "support@",
    "sales@",
    "booking@",
    "reservations@",
)

_OBFUSCATE_PATTERNS = [
    # common obfuscations -> convert to literal email text before finding
    (re.compile(r"\b(at|\[at\]|\(at\)|\sat\s|\sAT\s|\swhere\s)", re.I), "@"),
    (re.compile(r"\b(dot|\[dot\]|\(dot\)|\sdot\s|\sDOT\s|\spoint\s)", re.I), "."),
    (re.compile(r"\s*\[?\(?\s*(?:at|@)\s*\)?\]?\s*", re.I), "@"),
    (re.compile(r"\s*\[?\(?\s*(?:dot|\.)\s*\)?\]?\s*", re.I), "."),
]


def _deobfuscate_text(txt: str) -> str:
    s = txt or ""
    for pat, repl in _OBFUSCATE_PATTERNS:
        s = pat.sub(repl, s)
    # squish things like "name@ domain . com"
    s = re.sub(r"\s*@\s*", "@", s)
    s = re.sub(r"\s*\.\s*", ".", s)
    return s


def _domain_ok(email: str) -> bool:
    if "@" not in email:
        return False
    dom = email.split("@", 1)[1].lower()
    # kill obvious junk
    if any(bad in dom for bad in _BAD_DOMAINS):
        return False
    # basic TLD check
    if not any(dom.endswith(tld) for tld in _ALLOWED_TLDS):
        return False
    return True


def _order_and_filter_emails(emails: list[str], prefer_domain: str | None = None) -> list[str]:
    # de-dupe (case-insensitive) + filter
    seen, out = set(), []
    prefer_domain = (prefer_domain or "").lower().lstrip("www.")

    # sort by: exact-domain > generic inbox > others
    def score(e: str) -> tuple[int, int]:
        el = e.lower()
        exact = 1 if (prefer_domain and el.endswith("@" + prefer_domain)) else 0
        generic = 1 if any(el.startswith(g) for g in _GENERIC_INBOX_PREFIXES) else 0
        # sort descending by (exact, generic)
        return (-exact, -generic)

    cleaned = []
    for e in emails:
        e = e.strip()
        if not e or "@" not in e:
            continue
        if not _domain_ok(e):
            continue
        el = e.lower()
        if el not in seen:
            seen.add(el)
            cleaned.append(e)
    cleaned.sort(key=score)
    return cleaned


def _norm_host(u: str) -> str:
    p = urlparse(u)
    host = (p.netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def _same_origin(u: str, base: str) -> bool:
    hu, hb = _norm_host(u), _norm_host(base)
    if not hu or not hb:
        return False
    return hu == hb


def _seed_urls(domain_or_url: str) -> List[str]:
    parsed = urlparse(domain_or_url)
    host = (parsed.netloc or parsed.path or "").lower()
    host = host[4:] if host.startswith("www.") else host
    seeds = [f"https://{host}", f"http://{host}", f"https://www.{host}", f"http://www.{host}"]
    # keep order but dedupe
    out, seen = [], set()
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        return sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright not available. Install and provision Chromium:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium\n"
            f"Underlying error: {e}"
        )


# -------------------- Page extraction helpers (run inside browser) --------------------


def _emails_from_cloudflare(page) -> List[str]:
    """Decode Cloudflare __cf_email__ obfuscation."""
    js = """
    () => {
      function decode(cfhex) {
        const r = parseInt(cfhex.slice(0,2), 16);
        let out = "";
        for (let n = 2; n < cfhex.length; n += 2) {
          const charCode = parseInt(cfhex.slice(n, n+2), 16) ^ r;
          out += String.fromCharCode(charCode);
        }
        return out;
      }
      const nodes = Array.from(document.querySelectorAll("[data-cfemail], .__cf_email__"));
      const emails = [];
      for (const el of nodes) {
        const hex = el.getAttribute("data-cfemail");
        if (hex) {
          try { const e = decode(hex); if (e && e.includes("@")) emails.push(e); } catch(e) {}
        }
      }
      return emails;
    }
    """
    try:
        return page.evaluate(js) or []
    except Exception:
        return []


def _emails_from_jsonld(page) -> List[str]:
    """Pull emails from schema.org JSON-LD (Organization, LocalBusiness, etc.)."""
    js = """
    () => {
      const out = [];
      const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
      for (const s of scripts) {
        try {
          const data = JSON.parse(s.textContent || "null");
          const arr = Array.isArray(data) ? data : [data];
          for (const item of arr) {
            if (!item || typeof item !== 'object') continue;
            const push = (v) => { if (typeof v === 'string' && v.includes('@')) out.push(v); };
            push(item.email);
            if (item.contactPoint && typeof item.contactPoint === 'object') {
              if (Array.isArray(item.contactPoint)) {
                for (const cp of item.contactPoint) { if (cp) push(cp.email); }
              } else {
                push(item.contactPoint.email);
              }
            }
          }
        } catch(e) {}
      }
      return out;
    }
    """
    try:
        return page.evaluate(js) or []
    except Exception:
        return []


def _emails_from_attrs_and_forms(page) -> List[str]:
    """
    Scrape data-* attributes, aria-labels, placeholders, and form hints that may hold emails.
    """
    js = """
    () => {
      const emails = new Set();
      const re = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}/ig;

      function scanAttr(el, name) {
        try {
          const v = el.getAttribute(name);
          if (v && typeof v === 'string') {
            const m = v.match(re);
            if (m) m.forEach(x => emails.add(x));
          }
        } catch(e) {}
      }

      const all = Array.from(document.querySelectorAll("body, body *"));
      for (const el of all) {
        scanAttr(el, "data-email");
        scanAttr(el, "data-contact");
        scanAttr(el, "aria-label");
        scanAttr(el, "title");
        scanAttr(el, "placeholder");
        // inline onclick/onmouseover often hide mailtos
        scanAttr(el, "onclick");
        scanAttr(el, "onmouseover");
      }

      // form actions can be mailto:
      const forms = Array.from(document.querySelectorAll("form"));
      for (const f of forms) {
        const act = f.getAttribute("action") || "";
        if (act.toLowerCase().startsWith("mailto:")) {
          emails.add(act.slice(7).split("?")[0]);
        }
      }
      return Array.from(emails);
    }
    """
    try:
        return page.evaluate(js) or []
    except Exception:
        return []


def _emails_from_js_joins(page) -> List[str]:
    """Detect common 'string'+'join' JS patterns that form emails."""
    try:
        html = page.content()
    except Exception:
        html = ""
    out: List[str] = []
    for m in _JS_STR_JOIN_EMAIL.findall(html or ""):
        try:
            parts = re.findall(r"['\"]([^'\"]+)['\"]", m)
            if parts:
                e = "".join(parts)
                if "@" in e:
                    out.append(e)
        except Exception:
            continue
    return out


def _sanitize_mailtos(mailtos: List[str]) -> List[str]:
    out = []
    for m in mailtos or []:
        m = (m or "").strip()
        if not m:
            continue
        if m.lower().startswith("mailto:"):
            m = m[7:]
        if "?" in m:
            m = m.split("?", 1)[0]
        if "@" in m:
            out.append(m)
    return out


def _collect_internal_links(page, url: str) -> List[str]:
    try:
        hrefs = page.eval_on_selector_all("a[href]", "els => els.map(a => a.getAttribute('href') || '')") or []
    except Exception:
        hrefs = []
    links: List[str] = []
    seen_local: Set[str] = set()
    for href in hrefs:
        try:
            absu = urljoin(url, href).split("#", 1)[0]
            if not _same_origin(absu, url):
                continue
            if absu and absu not in seen_local:
                seen_local.add(absu)
                links.append(absu)
        except Exception:
            continue
    # contact-ish first
    contactish = [u for u in links if _CONTACT_HINTS.search(u)]
    others = [u for u in links if u not in contactish]
    return contactish + others


def _jump_to_common_anchors(page) -> None:
    # try to hit likely anchors so the DOM renders contact blocks (some frameworks lazy-render)
    anchors = ["#contact", "#get-in-touch", "#about", "#footer", "#find-us"]
    for a in anchors:
        try:
            page.evaluate(
                f"""
              (sel) => {{
                const el = document.querySelector(sel);
                if (el && el.scrollIntoView) el.scrollIntoView({{behavior:'instant', block:'center'}});
              }}
            """,
                a,
            )
            page.wait_for_timeout(80)
        except Exception:
            pass


def _expand_common_ui(page) -> None:
    """
    Expand accordions, <details>, tab panels, and any button that looks like "Contact/Email/Show".
    """
    js = """
    () => {
      // expand <details>
      document.querySelectorAll('details').forEach(d => { try { d.open = true; } catch(e){} });
      // click elements that reveal content
      const clickable = Array.from(document.querySelectorAll(
        "button, [role='button'], a, summary, .accordion, .accordion-button, .collapsible, .expand, .toggle"
      ));
      const KEY_HINT = /contact|email|enquir|get[-\\s]?in[-\\s]?touch|find|where|location|about|support|help|show|more|details/i;
      for (const el of clickable.slice(0, 60)) {
        const t = (el.innerText || el.textContent || el.getAttribute('aria-label') || '').trim();
        if (KEY_HINT.test(t)) { try { el.click(); } catch(e){} }
      }
      // activate first tabs if present
      const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
      for (const t of tabs.slice(0, 8)) { try { t.click(); } catch(e){} }
    }
    """
    try:
        page.evaluate(js)
        page.wait_for_timeout(150)
    except Exception:
        pass


def _scroll_and_wait(page, timeout_ms: int) -> None:
    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:
        pass
    try:
        # Basic scroll to trigger lazy DOM
        for _ in range(8):
            page.mouse.wheel(0, 1200)
            page.wait_for_timeout(120)
    except Exception:
        pass
    _jump_to_common_anchors(page)
    _expand_common_ui(page)
    # minor extra idle
    page.wait_for_timeout(120)


def _harvest_on_page(page, url: str) -> Tuple[List[str], List[str]]:
    """Return (emails, internal_links) found on the current page."""
    # mailto links
    try:
        mailtos = page.eval_on_selector_all('a[href^="mailto:"]', "els => els.map(a => a.getAttribute('href') || '')") or []
    except Exception:
        mailtos = []
    emails_from_mailtos = _sanitize_mailtos(mailtos)

    # visible text + HTML
    try:
        text = page.evaluate("() => document.body ? document.body.innerText : ''")
    except Exception:
        text = ""
    try:
        html = page.content()
    except Exception:
        html = ""
    emails = set(_EMAIL_RE_TEXT.findall(text or "")) | set(_EMAIL_RE_TEXT.findall(html or ""))
    emails |= set(emails_from_mailtos)

    # Cloudflare + JSON-LD + attribute/form hints + JS string joins
    emails |= set(_emails_from_cloudflare(page))
    emails |= set(_emails_from_jsonld(page))
    emails |= set(_emails_from_attrs_and_forms(page))
    emails |= set(_emails_from_js_joins(page))

    # deobfuscation pass
    try:
        deob = _deobfuscate_text(text or "")
        emails |= set(_EMAIL_RE_TEXT.findall(deob or ""))
    except Exception:
        pass

    # final ordering/filter (prefer same-domain inboxes)
    host = _norm_host(url)
    emails = set(_order_and_filter_emails(list(emails), prefer_domain=host))

    # same-origin iframes (light touch)
    try:
        for frame in page.frames:
            if frame is page.main_frame:
                continue
            f_url = (frame.url or "")
            if _same_origin(f_url, url):
                try:
                    f_txt = frame.evaluate("() => document.body ? document.body.innerText : ''")
                except Exception:
                    f_txt = ""
                try:
                    f_html = frame.content()
                except Exception:
                    f_html = ""
                emails |= set(_EMAIL_RE_TEXT.findall(f_txt or "")) | set(_EMAIL_RE_TEXT.findall(f_html or ""))
                # mailto in frame
                try:
                    f_mailtos = frame.eval_on_selector_all('a[href^="mailto:"]', "els => els.map(a => a.getAttribute('href') || '')") or []
                except Exception:
                    f_mailtos = []
                emails |= set(_sanitize_mailtos(f_mailtos))
    except Exception:
        pass

    # internal links (prioritise contact-ish)
    links = _collect_internal_links(page, url)

    return (list(emails), links)


# -------------------- Sync crawl core (used by thread/offload) --------------------


def _fetch_emails_with_browser_sync(
    domain_or_url: str,
    *,
    max_pages: int,
    timeout_ms: int,
) -> List[str]:
    """
    Pure SYNC implementation. Safe to run in a background thread.
    """
    # Ensure Windows can spawn subprocesses for Playwright
    _ensure_win_proactor()

    sync_playwright = _require_playwright()

    # DNS sanity: skip obviously invalid hosts quickly
    try:
        host = (_norm_host(domain_or_url) or domain_or_url).strip()
        host = host.split("/", 1)[0]
        if host:
            socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except Exception:
        return []

    seeds = _seed_urls(domain_or_url)
    visited: Set[str] = set()
    queue: Deque[str] = deque()
    emails: List[str] = []

    with sync_playwright() as pw:
        # Add container-safe flags for Cloud Run
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            java_script_enabled=True,
            ignore_https_errors=True,
            viewport={"width": 1360, "height": 1800},
            user_agent="EcodiaEcoLocal/browser (https://ecodia.au)",
        )

        # Trim heavy / noisy resources
        def _route_handler(route):
            req = route.request
            rt = (getattr(req, "resource_type", None) or "").lower()
            url = (getattr(req, "url", None) or "").lower()
            if rt in ("image", "media", "font"):
                return route.abort()
            if any(
                b in url
                for b in (
                    "google-analytics.com",
                    "googletagmanager.com",
                    "doubleclick.net",
                    "facebook.net",
                    "hotjar",
                    "segment.com",
                    "sentry.io",
                )
            ):
                return route.abort()
            return route.continue_()

        context.route("**/*", _route_handler)

        page = context.new_page()
        page.set_default_navigation_timeout(timeout_ms)
        page.set_default_timeout(timeout_ms)

        # find first loadable seed
        start_url: Optional[str] = None
        for s in seeds:
            try:
                page.goto(s, wait_until="domcontentloaded")
                start_url = page.url
                if start_url:
                    break
            except Exception:
                continue

        if not start_url:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass
            return []

        queue.append(start_url)

        try:
            while queue and (len(visited) < max_pages):
                url = queue.popleft()
                if url in visited:
                    continue
                visited.add(url)
                try:
                    page.goto(url, wait_until="domcontentloaded")
                    _scroll_and_wait(page, timeout_ms)
                    found_emails, links = _harvest_on_page(page, url)
                except Exception:
                    continue

                # accumulate unique emails
                seen_lower = {x.lower() for x in emails}
                for e in found_emails:
                    e_clean = e.strip()
                    if e_clean and "@" in e_clean and e_clean.lower() not in seen_lower:
                        emails.append(e_clean)

                # enqueue next links within budget & origin
                for nxt in links:
                    if nxt not in visited and _same_origin(nxt, start_url) and (
                        len(visited) + len(queue) < max_pages
                    ):
                        queue.append(nxt)
        finally:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass

    # Unique while preserving order & basic sanity
    seen: Set[str] = set()
    ordered: List[str] = []
    for e in emails:
        el = e.lower()
        # conservative filter to drop broken "reserv@ion" etc.
        if not re.search(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", e, re.I):
            continue
        if el not in seen:
            seen.add(el)
            ordered.append(e)
    return ordered


# -------------------- Thread offload helpers --------------------


def _ensure_win_proactor() -> None:
    if sys.platform.startswith("win"):
        try:
            if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass


# --- keep your existing executor ---
_executor: ThreadPoolExecutor | None = None


@atexit.register
def _shutdown_executor() -> None:
    global _executor
    try:
        if _executor:
            _executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _in_asyncio() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _run_in_thread(fn, *args, **kwargs):
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-fetch")
    fut = _executor.submit(fn, *args, **kwargs)
    return fut.result()


# -------------------- Public entrypoint --------------------


def fetch_emails_with_browser(
    domain_or_url: str,
    *,
    max_pages: int = 16,
    timeout_ms: int = 18000,
) -> List[str]:
    """
    Headless Playwright crawl (sync) of same-origin pages:
      - picks the first loadable seed (http/https, www/non-www)
      - navigates to contact-like links first
      - expands accordions/tabs; scrolls; visits same-origin iframes
      - extracts emails from mailto, visible text/HTML, CF, JSON-LD, attrs, JS joins
    Returns a de-duplicated list of emails found.

    IMPORTANT: If called under an active asyncio loop (e.g., FastAPI request),
    we offload the sync Playwright run into a worker thread to avoid:
      "It looks like you are using Playwright Sync API inside the asyncio loop."
    """
    # Quick DNS sanity (fast path for junk hosts)
    try:
        host = (_norm_host(domain_or_url) or domain_or_url).strip()
        host = host.split("/", 1)[0]
        if host:
            socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except Exception:
        return []

    if _in_asyncio():
        return _run_in_thread(
            _fetch_emails_with_browser_sync,
            domain_or_url,
            max_pages=max_pages,
            timeout_ms=timeout_ms,
        )
    else:
        # Ensure Windows policy locally; safe no-op on Linux/Cloud Run
        _ensure_win_proactor()
        return _fetch_emails_with_browser_sync(
            domain_or_url,
            max_pages=max_pages,
            timeout_ms=timeout_ms,
        )
