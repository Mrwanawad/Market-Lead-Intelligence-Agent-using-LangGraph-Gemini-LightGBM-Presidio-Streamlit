"""Page fetching: httpx + trafilatura first, Playwright (Chromium) only when the page is JS-heavy."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx
import trafilatura

from ..config import ROOT, get_settings
from ..schemas import Page, SearchHit

log = logging.getLogger(__name__)
FIXTURE_PAGES = ROOT / "fixtures" / "pages"
FIXTURE_HOST = "fixtures.mlia.local"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MLIA-research-bot/0.1 (+public market research)"
MIN_TEXT = 400


def _clean(html: str, url: str) -> tuple[str, str]:
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True, favor_recall=True) or ""
    meta = trafilatura.extract_metadata(html)
    return text.strip(), (meta.title if meta and meta.title else "")


def fixture_path(url: str) -> Path:
    return FIXTURE_PAGES / (urlparse(url).path.strip("/").replace("/", "_") + ".html")


def fetch_http(url: str) -> tuple[str, str]:
    s = get_settings()
    with httpx.Client(follow_redirects=True, timeout=s.http_timeout, headers={"User-Agent": UA}) as c:
        r = c.get(url)
        r.raise_for_status()
        if "html" not in r.headers.get("content-type", "html"):
            raise ValueError(f"non-HTML content: {r.headers.get('content-type')}")
        return _clean(r.text, url)


def fetch_browser(url: str) -> tuple[str, str]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=UA)
            page.goto(url, wait_until="domcontentloaded", timeout=int(get_settings().http_timeout * 1000))
            page.wait_for_timeout(1500)
            return _clean(page.content(), url)
        finally:
            browser.close()


def fetch(hit: SearchHit) -> Page:
    """Fetch a search hit. Raises on failure; the researcher records the error in the run log."""
    url = hit.url
    if urlparse(url).netloc == FIXTURE_HOST:
        html = fixture_path(url).read_text(encoding="utf-8")
        text, title = _clean(html, url)
        return Page(url=url, title=title or hit.title, text=text, fetched_with="fixture",
                    query=hit.query, signal_type=hit.signal_type)

    text, title, how = "", "", "http"
    try:
        text, title = fetch_http(url)
    except Exception as e:  # noqa: BLE001 - fall through to browser / snippet
        log.info("http fetch failed for %s: %s", url, e)
    if len(text) < MIN_TEXT and get_settings().use_browser:
        try:
            text, title = fetch_browser(url)
            how = "browser"
        except Exception as e:  # noqa: BLE001
            log.info("browser fetch failed for %s: %s", url, e)
    if len(text) < MIN_TEXT and hit.snippet:
        # Keep the search snippet so the signal can still be grounded (verifier checks against it).
        text, how = f"{hit.title}\n{hit.snippet}", "snippet"
    if not text:
        raise RuntimeError("no text extracted")
    return Page(url=url, title=title or hit.title, text=text[:20000], fetched_with=how,
                query=hit.query, signal_type=hit.signal_type)
