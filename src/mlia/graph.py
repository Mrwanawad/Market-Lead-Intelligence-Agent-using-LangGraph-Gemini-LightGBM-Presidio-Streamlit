"""LangGraph research graph: planner -> researcher -> extractor -> verifier."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import extractor, planner, researcher, verifier
from .config import Brief, Settings
from .geo import find_area
from .llm import LLM
from .schemas import Page, SearchPlan, VerifiedSignal
from .web.search import SearchProvider

log = logging.getLogger(__name__)


_EGYPT = ("egypt", "cairo", "egp", "مصر", "القاهرة")


def is_relevant(page: Page, brief: Brief) -> bool:
    """Cheap pre-LLM gate: page must mention Egypt or a target area."""
    t = f"{page.title} {page.text[:6000]}".lower()
    return any(k in t for k in _EGYPT) or find_area(t) is not None


class ResearchState(TypedDict, total=False):
    brief: Brief
    recent_queries: list[str]
    seen_hashes: set[str]
    plan: SearchPlan
    pages: list[Page]
    source_log: list[dict]
    candidates: list[tuple[dict, Page]]
    verified: list[VerifiedSignal]
    rejected: list[VerifiedSignal]


def build_graph(llm: LLM, provider: SearchProvider, settings: Settings):
    def plan_node(state: ResearchState) -> ResearchState:
        p = planner.plan(state["brief"], llm, settings.max_queries, state.get("recent_queries"))
        log.info("planner: %d queries", len(p.queries))
        return {"plan": p}

    def research_node(state: ResearchState) -> ResearchState:
        pages, slog = researcher.research(state["plan"], provider, settings.results_per_query,
                                          settings.max_pages, state.get("seen_hashes"))
        log.info("researcher: %d new pages", len(pages))
        return {"pages": pages, "source_log": slog}

    def extract_node(state: ResearchState) -> ResearchState:
        def _one(page: Page):
            return [(s.model_dump(), page) for s in extractor.extract(page, llm).signals]

        pages = [p for p in state.get("pages", []) if is_relevant(p, state["brief"])]
        skipped = len(state.get("pages", [])) - len(pages)
        with ThreadPoolExecutor(max_workers=max(1, settings.llm_concurrency)) as ex:
            cands = [c for batch in ex.map(_one, pages) for c in batch]
        log.info("extractor: %d candidate signals (%d off-topic pages skipped)", len(cands), skipped)
        slog = state.get("source_log", []) + [{"kind": "irrelevant", "target": p.url, "ok": False,
                                               "error": "no Egypt/target-area mention"}
                                              for p in state.get("pages", []) if p not in pages]
        return {"candidates": cands, "source_log": slog}

    def verify_node(state: ResearchState) -> ResearchState:
        from .schemas import ExtractedSignal

        ok, bad = [], []
        for raw, page in state.get("candidates", []):
            v = verifier.verify(ExtractedSignal(**raw), page, settings.verifier_min_confidence)
            (ok if v.verified else bad).append(v)
        log.info("verifier: %d kept, %d rejected", len(ok), len(bad))
        return {"verified": ok, "rejected": bad}

    def route_after_research(state: ResearchState) -> str:
        return "extractor" if state.get("pages") else END

    g = StateGraph(ResearchState)
    g.add_node("planner", plan_node)
    g.add_node("researcher", research_node)
    g.add_node("extractor", extract_node)
    g.add_node("verifier", verify_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_conditional_edges("researcher", route_after_research, {"extractor": "extractor", END: END})
    g.add_edge("extractor", "verifier")
    g.add_edge("verifier", END)
    return g.compile()
