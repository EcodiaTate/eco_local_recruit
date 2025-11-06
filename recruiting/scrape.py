# pyright: reportMissingImports=false
# # recruiting/scrape.py
from __future__ import annotations

import os
import re
import io
import json
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup

# Places is optional; discovery will still work via Google CSE if Places key not set
from . import places as _places  # type: ignore

LOG = logging.getLogger("eco_local.scrape")
_DEBUG = (os.getenv("ECO_LOCAL_SCRAPE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"})

def _d(msg: str) -> None:
    if _DEBUG:
        LOG.info(msg)

# -------------------- Config / Keys --------------------

GOOGLE_API_KEY = os.getenv("GENERAL_GOOGLE_API_KEY")
GOOGLE_PSE_CX  = os.getenv("GOOGLE_PSE_CX")

USE_BROWSER         = (os.getenv("ECO_LOCAL_USE_BROWSER", "1").strip().lower() in {"1","true","yes","on"})
BROWSER_MAX_PAGES   = int(os.getenv("ECO_LOCAL_BROWSER_MAX_PAGES", "12"))
BROWSER_TIMEOUT_MS  = int(os.getenv("ECO_LOCAL_BROWSER_TIMEOUT_MS", "15000"))
GUESS_GENERIC       = (os.getenv("ECO_LOCAL_GUESS_GENERIC", "0").strip().lower() in {"1","true","yes","on"})

_USER_AGENT = "EcodiaEcoLocal/scrape (https://ecodia.au)"
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# More generous, AU-friendly path sweep
_COMMON_CONTACT_PATHS = [
    "/contact", "/contact-us", "/contacts",
    "/about", "/about-us", "/team",
    "/find-us", "/get-in-touch", "/visit-us",
    "/store-locator", "/locations", "/location",
    "/support", "/help", "/faq", "/faqs",
    "/privacy", "/privacy-policy", "/legal", "/impressum",
    "/menu", "/functions", "/events", "/catering",
]

# Third-party domains we avoid treating as first-class business websites
_THIRD_PARTY_DOMAINS = tuple([
    "resdiary", "opentable", "quandoo", "square.site", "linktr.ee",
    "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com", "linkedin.com",
    "mindbodyonline", "setmore", "acuityscheduling", "wixsite.com",
    "google.com/maps", "googleusercontent.com",
])

_GENERIC_INBOXES = ("info@", "hello@", "contact@", "enquiries@", "admin@", "office@", "support@", "sales@")


# -------------------- HTTP --------------------

def _client() -> httpx.Client:
    return httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT})


# -------------------- Domain helpers --------------------

def _is_gov_domain(domain: Optional[str]) -> bool:
    if not domain:
        return False
    d = domain.lower()
    return (".gov." in d) or d.endswith(".gov") or d.endswith(".gov.au")


def _normalize_domain(url_or_domain: Optional[str]) -> Optional[str]:
    if not url_or_domain:
        return None
    if "://" in url_or_domain:
        u = urlparse(url_or_domain)
        host = (u.netloc or "").lower()
    else:
        host = (url_or_domain or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _base_candidates(url_or_domain: Optional[str]) -> List[str]:
    """
    Always try both www and non-www, https and http.
    """
    if not url_or_domain:
        return []
    raw_host = (urlparse(url_or_domain).netloc or url_or_domain).lower()
    host = raw_host[4:] if raw_host.startswith("www.") else raw_host
    hosts = [host, f"www.{host}"]
    schemes = ["https", "http"]
    out, seen = [], set()
    for h in hosts:
        for sch in schemes:
            base = f"{sch}://{h}"
            if base not in seen:
                seen.add(base); out.append(base)
    return out


# -------------------- Discovery (Places + Google CSE) --------------------

def _pse_page(query: str, city: str, start: int, page_size: int = 10) -> List[Dict[str, Any]]:
    if not (GOOGLE_API_KEY and GOOGLE_PSE_CX):
        return []
    q = f"{query} {city} site:.au"
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_PSE_CX, "q": q, "num": page_size, "start": start}
    out: List[Dict[str, Any]] = []
    with _client() as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    for it in (data.get("items") or []):
        link = it.get("link")
        dom = _normalize_domain(link)
        if not dom or _is_gov_domain(dom):
            continue
        # Skip obvious third-party landing pages
        if any(tp in (link or "").lower() for tp in _THIRD_PARTY_DOMAINS):
            continue
        out.append({"name": it.get("title"), "domain": dom, "website": f"https://{dom}"})
    return out


def _pse_search_until(query: str, city: str, needed: int, hard_cap: int = 80) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    start = 1  # CSE starts at 1
    while len(results) < needed and start <= hard_cap:
        page = _pse_page(query, city, start)
        if not page:
            break
        results.extend(page)
        start += 10
    return results


def _places_search_until(query: str, city: str, needed: int, hard_pages: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not getattr(_places, "PLACES_KEY", None):
        return out

    with _client() as c:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": f"{query} in {city}", "key": _places.PLACES_KEY, "region": "AU", "language": "en"}
        token = None
        pages = 0
        while len(out) < needed and pages < hard_pages:
            if token:
                params["pagetoken"] = token
            r = c.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            for it in (data.get("results") or []):
                pid = it.get("place_id")
                name = it.get("name")
                if not pid or not name:
                    continue
                out.append({
                    "name": name,
                    "place_id": pid,
                    "formatted_address": it.get("formatted_address"),
                    "domain": None,
                    "website": None,
                    "rating": it.get("rating"),
                    "user_ratings_total": it.get("user_ratings_total"),
                })
                if len(out) >= needed:
                    break
            token = data.get("next_page_token")
            pages += 1
            if not token:
                break

    # Fill details → website/domain; drop .gov and third-party-only
    filled: List[Dict[str, Any]] = []
    for item in out:
        try:
            det = _places.place_details(item["place_id"])
            website = (det.get("website") or "").strip() or None
            dom = _normalize_domain(website)
            if dom and _is_gov_domain(dom):
                continue
            if website and any(tp in website.lower() for tp in _THIRD_PARTY_DOMAINS):
                # Keep place but don't set third-party site as first-class domain
                item["website"] = None
                item["domain"] = None
            else:
                item["website"] = website
                item["domain"] = dom
            filled.append(item)
        except Exception:
            filled.append(item)
    return filled


def _dedupe_by_domain(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for c in cands:
        key = c.get("domain") or _normalize_domain(c.get("website")) or (c.get("name") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def discover_places(query: str, city: str, limit: int = 10) -> List[Dict[str, Any]]:
    target_pool = max(limit * 4, 40)
    places = _places_search_until(query, city, target_pool) if getattr(_places, "PLACES_KEY", None) else []
    pse    = _pse_search_until(query, city, target_pool)
    combined = _dedupe_by_domain([*places, *pse])
    combined = [c for c in combined if not _is_gov_domain(c.get("domain") or _normalize_domain(c.get("website")))]
    return combined


# -------------------- Fetching HTML + contact pages --------------------

def fetch_homepage(domain_or_url: Optional[str]) -> str:
    for base in _base_candidates(domain_or_url):
        try:
            with _client() as c:
                r = c.get(base, follow_redirects=True)
                if r.status_code < 400 and (r.text or "").strip():
                    return r.text
        except Exception:
            continue
    return ""


def _discover_contact_like_links(home_html: str, base: str) -> List[str]:
    out: List[str] = []
    if not home_html:
        return out
    try:
        soup = BeautifulSoup(home_html, "html.parser")
    except Exception:
        return out
    hints = re.compile(
        r"\b(contact|contact-us|contactus|get-in-touch|find-us|visit|where|location|locations|about|team|support|help|enquiries?|privacy|menu|functions|events|catering|faq|faqs)\b",
        re.I,
    )
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        text = (a.get_text(" ", strip=True) or "")
        if not href:
            continue
        if not (hints.search(href) or hints.search(text)):
            continue
        url = urljoin(base + "/", href)
        try:
            u = urlparse(url)
            b = urlparse(base)
            # only internal
            if u.netloc and b.netloc and _normalize_domain(u.netloc) != _normalize_domain(b.netloc):
                continue
        except Exception:
            continue
        if url not in seen:
            seen.add(url); out.append(url)

    # plus classic fallbacks
    for path in _COMMON_CONTACT_PATHS:
        url = urljoin(base + "/", path.lstrip("/"))
        if url not in seen:
            seen.add(url); out.append(url)
    return out[:24]


def _collect_pdf_links(html: str, base: str) -> List[str]:
    urls: List[str] = []
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            if ".pdf" in href.lower():
                urls.append(urljoin(base + "/", href))
    except Exception:
        pass
    # preserve order, dedupe
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out[:10]


def _extract_emails_from_pdf_bytes(raw: bytes) -> List[str]:
    emails: List[str] = []
    try:
        from PyPDF2 import PdfReader  # optional dependency
        reader = PdfReader(io.BytesIO(raw))
        text_chunks = []
        for page in reader.pages:
            try:
                text_chunks.append(page.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(text_chunks)
        emails = _TEXT_EMAIL_RE.findall(text or "")
    except Exception:
        pass
    # dedupe preserve order
    seen, out = set(), []
    for e in emails:
        el = e.lower().strip()
        if el and "@" in el and el not in seen:
            seen.add(el); out.append(e.strip())
    return out


def fetch_common_contact_pages(domain_or_url: Optional[str]) -> List[Tuple[str, str]]:
    pages: List[Tuple[str, str]] = []
    bases = _base_candidates(domain_or_url)
    if not bases:
        return pages
    with _client() as c:
        for base in bases:
            home_html = ""
            try:
                r = c.get(base, follow_redirects=True)
                if r.status_code < 400 and (r.text or "").strip():
                    home_html = r.text
                    pages.append((base, home_html))
                    # PDFs on homepage
                    pdfs = _collect_pdf_links(home_html, base)
                    for purl in pdfs:
                        try:
                            rr = c.get(purl, follow_redirects=True)
                            if rr.status_code < 400 and rr.headers.get("content-type","").lower().startswith("application/pdf"):
                                emails = _extract_emails_from_pdf_bytes(rr.content)
                                if emails:
                                    # store as synthetic "page" content (newline-joined emails)
                                    pages.append((purl, "\n".join(emails)))
                        except Exception:
                            pass
            except Exception:
                pass

            # 1) follow *actual* links
            discovered = _discover_contact_like_links(home_html, base) if home_html else []

            # 1b) sitemap candidates (when thin sites)
            if not discovered:
                discovered = _sitemap_candidates(base) or []

            for url in discovered[:24]:
                try:
                    # Skip third-party booking/social
                    if any(tp in url.lower() for tp in _THIRD_PARTY_DOMAINS):
                        continue
                    r = c.get(url, follow_redirects=True)
                    if r.status_code < 400 and (r.text or "").strip():
                        pages.append((url, r.text))
                        # PDFs on contact-like pages
                        pdfs = _collect_pdf_links(r.text, base)
                        for purl in pdfs:
                            try:
                                rr = c.get(purl, follow_redirects=True)
                                if rr.status_code < 400 and rr.headers.get("content-type","").lower().startswith("application/pdf"):
                                    emails = _extract_emails_from_pdf_bytes(rr.content)
                                    if emails:
                                        pages.append((purl, "\n".join(emails)))
                            except Exception:
                                pass
                except Exception:
                    continue

            # 2) fallback to classic paths only if still empty
            if not discovered:
                for path in _COMMON_CONTACT_PATHS:
                    url = urljoin(base + "/", path.lstrip("/"))
                    try:
                        r = c.get(url, follow_redirects=True)
                        if r.status_code < 400 and (r.text or "").strip():
                            pages.append((url, r.text))
                    except Exception:
                        continue
            if pages:
                break

    # dedupe
    seen, out = set(), []
    for u, h in pages:
        if u not in seen:
            seen.add(u); out.append((u, h))
    return out


def _sitemap_candidates(base: str) -> List[str]:
    urls: List[str] = []
    for path in ("/sitemap.xml", "/sitemap_index.xml"):
        try:
            with _client() as c:
                r = c.get(urljoin(base, path), follow_redirects=True)
                if r.status_code >= 400 or not (r.text or "").strip():
                    continue
                txt = r.text
            try:
                root = ET.fromstring(txt)
                ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                locs = [e.text for e in root.findall(".//sm:url/sm:loc", ns)] or [e.text for e in root.findall(".//loc")]
                for u in locs or []:
                    if not isinstance(u, str): continue
                    if _normalize_domain(u) and _normalize_domain(u) == _normalize_domain(base):
                        urls.append(u.strip())
                if not urls:
                    submaps = [e.text for e in root.findall(".//sm:sitemap/sm:loc", ns)] or [e.text for e in root.findall(".//loc")]
                    for smu in (submaps or [])[:5]:
                        try:
                            with _client() as c:
                                rr = c.get(smu, follow_redirects=True)
                                if rr.status_code >= 400 or not (rr.text or "").strip(): continue
                                subroot = ET.fromstring(rr.text)
                                sublocs = [e.text for e in subroot.findall(".//sm:url/sm:loc", ns)] or [e.text for e in subroot.findall(".//loc")]
                                for u in sublocs or []:
                                    if not isinstance(u, str): continue
                                    if _normalize_domain(u) and _normalize_domain(u) == _normalize_domain(base):
                                        urls.append(u.strip())
                        except Exception:
                            continue
            except Exception:
                pass
        except Exception:
            continue
    hints = [u for u in urls if re.search(r"/(contact|about|team|find-us|get-in-touch|privacy|menu|functions|events|catering|faq|faqs)\b", u, re.I)]
    others = [u for u in urls if u not in hints]
    return (hints + others)[:24]


# -------------------- Email harvesting --------------------

_EMAIL_RE_MAILTO = re.compile(r"mailto:([^\"'<>\\)\s]+)", re.I)
_TEXT_EMAIL_RE   = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)

# JS-assembled mailtos like: "mailto:" + "info" + "@" + "example.com"
_JS_MAILTO_JOIN_RE = re.compile(r"mailto:\s*['\"]?((?:[^'\"+]|(?:\s*\+\s*['\"][^'\"]+['\"]))+)", re.I)
# JS string-joining that results in a raw email "in" DOM text
_JS_STR_JOIN_EMAIL = re.compile(
    r"(?:['\"][a-z0-9._%+\-]+['\"]\s*\+\s*)+['\"][a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}['\"]",
    re.I
)

_BAD_CONTEXT = re.compile(r"(privacy|terms|cookies|policy|legal|your-privacy-rights|terms-of-use)", re.I)

def _deobfuscate_cfemail(hexstr: str) -> Optional[str]:
    """
    Cloudflare __cf_email__ decoder.
    """
    try:
        data = bytes.fromhex(hexstr)
        key = data[0]
        out = bytes([b ^ key for b in data[1:]]).decode("utf-8")
        if "@" in out and "." in out:
            return out
    except Exception:
        return None
    return None


def _deobfuscate_text_email(txt: str) -> List[str]:
    """
    Handle 'info [at] example [dot] com' & common variants.
    """
    t = txt
    # common bracketed/word obfuscations
    t = re.sub(r"\s*\[?\s*(?:at|@)\s*\]?\s*", "@", t, flags=re.I)
    t = re.sub(r"\s*\[?\s*(?:dot|\.|\(dot\))\s*\]?\s*", ".", t, flags=re.I)
    # Remove stray spaces around @ and .
    t = re.sub(r"\s*@\s*", "@", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    return _TEXT_EMAIL_RE.findall(t)


def _extract_js_mailtos(html: str) -> List[str]:
    out: List[str] = []
    for m in _JS_MAILTO_JOIN_RE.findall(html or ""):
        try:
            parts = re.findall(r"['\"]([^'\"]+)['\"]", m)
            if parts:
                e = "".join(parts)
                if "@" in e:
                    out.append(e)
        except Exception:
            continue
    return out


def _extract_more_js_emails(html: str) -> List[str]:
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


def _filter_false_positives(emails: List[str]) -> List[str]:
    """
    Drop obviously broken tokens like 'reserv@ion' and keep RFC-ish shape.
    """
    out = []
    for e in emails:
        e = e.strip()
        if re.search(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", e, re.I):
            out.append(e)
    # dedupe keeping order
    seen, ordered = set(), []
    for e in out:
        el = e.lower()
        if el not in seen:
            seen.add(el); ordered.append(e)
    return ordered


def harvest_emails_from_html(html: str) -> List[str]:
    out: List[str] = []
    if not html:
        return out

    # 1) direct mailtos
    for m in _EMAIL_RE_MAILTO.findall(html):
        e = (m or "").strip()
        if e and "@" in e:
            out.append(e)

    # 2) text emails
    out += _TEXT_EMAIL_RE.findall(html or "")

    # 3) Cloudflare obfuscation + obvious obfuscated text variants
    try:
        soup = BeautifulSoup(html, "html.parser")
        for span in soup.select("span.__cf_email__"):
            hx = span.get("data-cfemail")
            if hx:
                decoded = _deobfuscate_cfemail(hx)
                if decoded:
                    out.append(decoded)
        out += _deobfuscate_text_email(soup.get_text(" ", strip=True) or "")
    except Exception:
        pass

    # 4) JS-built mailtos & joined strings
    out += _extract_js_mailtos(html)
    out += _extract_more_js_emails(html)

    # dedupe preserve order + filter junk
    seen, ordered = set(), []
    for e in out:
        el = e.lower().strip()
        if el and "@" in el and el not in seen:
            seen.add(el); ordered.append(e.strip())
    ordered = _filter_false_positives(ordered)
    return ordered


def _cse_email_harvest_for_domain(domain: str, max_pages: int = 3) -> List[str]:
    """
    When site pages fail, use CSE to find inner pages (and PDFs) with emails.
    """
    if not (GOOGLE_API_KEY and GOOGLE_PSE_CX):
        return []
    queries = [
        f'site:{domain} "@"',
        f'site:{domain} contact',
        f'site:{domain} email',
        f'site:{domain} menu filetype:pdf',
    ]
    emails: List[str] = []
    seen_urls: set[str] = set()
    with _client() as c:
        for q in queries:
            start = 1
            for _ in range(max_pages):
                try:
                    r = c.get(
                        "https://www.googleapis.com/customsearch/v1",
                        params={"key": GOOGLE_API_KEY, "cx": GOOGLE_PSE_CX, "q": q, "num": 10, "start": start},
                    )
                    r.raise_for_status()
                    data = r.json()
                    for it in (data.get("items") or []):
                        link = it.get("link")
                        if not link or link in seen_urls:
                            continue
                        seen_urls.add(link)
                        # skip third-parties
                        if any(tp in (link or "").lower() for tp in _THIRD_PARTY_DOMAINS):
                            continue
                        try:
                            rr = c.get(link, follow_redirects=True)
                            if rr.status_code >= 400 or not (rr.text or rr.content):
                                continue
                            ct = rr.headers.get("content-type","").lower()
                            if ".pdf" in (link.lower()) or ct.startswith("application/pdf"):
                                emails += _extract_emails_from_pdf_bytes(rr.content)
                            else:
                                emails += harvest_emails_from_html(rr.text)
                        except Exception:
                            continue
                except Exception:
                    break
                start += 10
    # final dedupe + filter
    uniq, out = set(), []
    for e in emails:
        el = e.strip().lower()
        if "@" in el and el not in uniq:
            uniq.add(el); out.append(e.strip())
    out = _filter_false_positives(out)
    return out


def _guess_generics_for_domain(domain: Optional[str]) -> List[str]:
    d = (domain or "").strip().lower()
    if not d:
        return []
    return [g + d for g in _GENERIC_INBOXES]


def harvest_emails_for_domain(domain_or_url: Optional[str]) -> List[str]:
    emails: List[str] = []

    # Browser path (Playwright/Puppeteer adapter expected in .browser_fetch)
    if USE_BROWSER and domain_or_url:
        try:
            from .browser_fetch import fetch_emails_with_browser  # type: ignore
            _d(f"[harvest] using BROWSER path for {domain_or_url} (max_pages={BROWSER_MAX_PAGES})")
            emails = fetch_emails_with_browser(
                domain_or_url,
                max_pages=BROWSER_MAX_PAGES,
                timeout_ms=BROWSER_TIMEOUT_MS,
            ) or []
            _d(f"[harvest] browser found {len(emails)} emails for {domain_or_url}: {emails}")
        except Exception as e:
            _d(f"[harvest] browser error for {domain_or_url}: {repr(e)}")
            emails = []

    # Fallback httpx pages
    if not emails:
        _d(f"[harvest] using HTTPX fallback for {domain_or_url}")
        pages = fetch_common_contact_pages(domain_or_url)
        _d(f"[harvest] fallback fetched {len(pages)} pages for {domain_or_url}: {[u for u,_ in pages]}")
        blobs = [h for _, h in pages]
        for html in blobs:
            emails += harvest_emails_from_html(html)
        _d(f"[harvest] fallback found {len(emails)} emails for {domain_or_url}: {emails}")

    # CSE domain-deep search when nothing yet
    if not emails:
        dom = _normalize_domain(domain_or_url)
        if dom:
            _d(f"[harvest] CSE fallback for {dom}")
            emails = _cse_email_harvest_for_domain(dom)

    # Optional: seed with generic inbox guesses when absolutely nothing
    if not emails and GUESS_GENERIC:
        dom = _normalize_domain(domain_or_url)
        if dom:
            emails = _guess_generics_for_domain(dom)
            _d(f"[harvest] guessing generics for {dom}: {emails}")

    # final dedupe/filter
    seen, out = set(), []
    for e in emails:
        el = e.strip().lower()
        if "@" in el and el not in seen:
            seen.add(el); out.append(e.strip())
    out = _filter_false_positives(out)
    return out
