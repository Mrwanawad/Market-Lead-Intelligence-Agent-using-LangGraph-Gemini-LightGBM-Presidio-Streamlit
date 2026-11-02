"""Researcher agent: run searches, pick URLs, fetch pages (search APIs first, browser only when needed)."""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from ..schemas import Page, SearchHit, SearchPlan
from ..web.fetch import fetch
from ..web.search import SearchProvider, is_allowed

log = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:16]


def research(
    plan: SearchPlan,
    provider: SearchProvider,
    results_per_query: int,
    max_pages: int,
    seen_hashes: set[str] | None = None,
) -> tuple[list[Page], list[dict]]:
    """Returns (new pages, source log). Pages whose content hash was already processed are skipped."""
    seen_hashes = seen_hashes or set()
    source_log: list[dict] = []

    def _search(q):
        t0 = time.perf_counter()
        try:
            hits = provider.search(q, results_per_query)
            source_log.append({"kind": "search", "target": q.query, "provider": provider.name, "ok": True,
                               "n": len(hits), "ms": int((time.perf_counter() - t0) * 1000)})
            return hits
        except Exception as e:  # noqa: BLE001
            source_log.append({"kind": "search", "target": q.query, "provider": provider.name, "ok": False,
                               "error": str(e)[:300], "ms": int((time.perf_counter() - t0) * 1000)})
            return []

    with ThreadPoolExecutor(max_workers=4) as ex:
        all_hits = [h for hits in ex.map(_search, plan.queries) for h in hits]

    unique: dict[str, SearchHit] = {}
    for h in all_hits:
        url = h.url.split("#")[0]
        if not is_allowed(url):
            source_log.append({"kind": "skip", "target": url, "ok": False, "error": "blocked domain (privacy policy)"})
            continue
        unique.setdefault(url, h)
    hits = list(unique.values())[:max_pages]

    def _fetch(h: SearchHit):
        t0 = time.perf_counter()
        try:
            p = fetch(h)
            source_log.append({"kind": "fetch", "target": h.url, "provider": p.fetched_with, "ok": True,
                               "n": len(p.text), "ms": int((time.perf_counter() - t0) * 1000)})
            return p
        except Exception as e:  # noqa: BLE001
            source_log.append({"kind": "fetch", "target": h.url, "ok": False, "error": str(e)[:300],
                               "ms": int((time.perf_counter() - t0) * 1000)})
            return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        pages = [p for p in ex.map(_fetch, hits) if p]

    fresh = []
    for p in pages:
        h = content_hash(p.text)
        if h in seen_hashes:
            source_log.append({"kind": "unchanged", "target": p.url, "ok": True})
            continue
        seen_hashes.add(h)
        fresh.append(p)
    return fresh, source_log
