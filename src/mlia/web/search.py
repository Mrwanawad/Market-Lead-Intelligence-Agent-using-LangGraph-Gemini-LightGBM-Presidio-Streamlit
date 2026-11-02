"""Search providers: Tavily / Exa (keyed APIs), DuckDuckGo (free), fixtures (offline)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

from ..config import ROOT, get_settings
from ..schemas import SearchHit, SearchQuery

log = logging.getLogger(__name__)
FIXTURES = ROOT / "fixtures"

# Domains we never fetch: private social profiles / login-walled personal data (PDPL + platform terms).
BLOCKED_DOMAINS = (
    "facebook.com", "instagram.com", "linkedin.com/in/", "tiktok.com", "x.com", "twitter.com",
    "snapchat.com", "wa.me", "t.me",
    # reference / dictionary pages: never a dated market signal, only burn LLM quota
    "wikipedia.org", "merriam-webster.com", "dictionary.com", "cambridge.org/dictionary", "britannica.com",
)


def is_allowed(url: str) -> bool:
    return not any(b in url for b in BLOCKED_DOMAINS)


class SearchProvider:
    name = "base"

    def search(self, q: SearchQuery, k: int) -> list[SearchHit]:  # pragma: no cover - interface
        raise NotImplementedError


class TavilySearch(SearchProvider):
    name = "tavily"

    def search(self, q: SearchQuery, k: int) -> list[SearchHit]:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": os.environ["TAVILY_API_KEY"], "query": q.query, "max_results": k,
                  "search_depth": "basic", "topic": "news"},
            timeout=get_settings().http_timeout,
        )
        r.raise_for_status()
        return [SearchHit(url=x["url"], title=x.get("title", ""), snippet=x.get("content", ""),
                          query=q.query, signal_type=q.signal_type, provider=self.name)
                for x in r.json().get("results", [])]


class ExaSearch(SearchProvider):
    name = "exa"

    def search(self, q: SearchQuery, k: int) -> list[SearchHit]:
        r = httpx.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": os.environ["EXA_API_KEY"]},
            json={"query": q.query, "numResults": k, "contents": {"text": {"maxCharacters": 800}}},
            timeout=get_settings().http_timeout,
        )
        r.raise_for_status()
        return [SearchHit(url=x["url"], title=x.get("title") or "", snippet=(x.get("text") or "")[:800],
                          query=q.query, signal_type=q.signal_type, provider=self.name)
                for x in r.json().get("results", [])]


class DDGSearch(SearchProvider):
    name = "ddg"

    def search(self, q: SearchQuery, k: int) -> list[SearchHit]:
        from ddgs import DDGS

        rows: list[dict] = []
        with DDGS() as ddgs:
            for timelimit in ("m", "y"):  # prefer last month; widen to last year when empty
                try:
                    rows = list(ddgs.text(q.query, region="xa-en", max_results=k, timelimit=timelimit))
                except Exception as e:  # ddgs raises on "No results found"
                    if "no results" not in str(e).lower():
                        raise
                if rows:
                    break
        return [SearchHit(url=x["href"], title=x.get("title", ""), snippet=x.get("body", ""),
                          query=q.query, signal_type=q.signal_type, provider=self.name)
                for x in rows if x.get("href")]


class FixtureSearch(SearchProvider):
    """Offline search over fixtures/search_index.json (keyword overlap ranking)."""

    name = "fixtures"

    def __init__(self, index_path: Path | None = None):
        path = index_path or FIXTURES / "search_index.json"
        self.docs = json.loads(path.read_text(encoding="utf-8"))

    def search(self, q: SearchQuery, k: int) -> list[SearchHit]:
        terms = {t.lower() for t in q.query.split() if len(t) > 2}
        scored = []
        for d in self.docs:
            hay = f"{d['title']} {d['snippet']} {' '.join(d.get('tags', []))}".lower()
            score = sum(t in hay for t in terms) + (3 if d.get("signal_type") == q.signal_type else 0)
            if score > 2:
                scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        return [SearchHit(url=d["url"], title=d["title"], snippet=d["snippet"], query=q.query,
                          signal_type=q.signal_type, provider=self.name) for _, d in scored[:k]]


def get_provider(name: str | None = None) -> SearchProvider:
    name = name or get_settings().search_provider
    if name == "auto":
        if os.getenv("TAVILY_API_KEY"):
            return TavilySearch()
        if os.getenv("EXA_API_KEY"):
            return ExaSearch()
        return DDGSearch()
    return {"tavily": TavilySearch, "exa": ExaSearch, "ddg": DDGSearch, "fixtures": FixtureSearch}[name]()
