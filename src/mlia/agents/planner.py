"""Planner agent: research brief -> search queries per signal type."""

from __future__ import annotations

import logging
from datetime import date
from itertools import cycle

from ..config import Brief
from ..llm import LLM, LLMUnavailable
from ..schemas import SearchPlan, SearchQuery, SignalType

log = logging.getLogger(__name__)

SYSTEM = """You plan web research for an Egyptian real-estate brokerage.
Turn the research brief into precise web search queries that surface PUBLIC, RECENT business signals:
new project launches, price / payment-plan changes, competitor moves, companies opening offices
(corporate housing demand) and relocation firms. Never target private individuals or social profiles.
Spread queries across signal types and areas; prefer news-style phrasing; include the year.
Use only these signal_type values: new_launch, price_change, competitor_move, office_opening, relocation_firm."""


def template_plan(brief: Brief, max_queries: int) -> SearchPlan:
    """Deterministic plan: expand templates over areas/competitors, round-robin across signal types."""
    per_type: dict[str, list[SearchQuery]] = {}
    for spec in brief.signal_types:
        qs: list[SearchQuery] = []
        for tpl in spec.query_templates:
            if "{competitor}" in tpl:
                qs += [SearchQuery(query=tpl.format(competitor=c), signal_type=spec.name, rationale="template")
                       for c in brief.competitors]
            elif "{area}" in tpl:
                qs += [SearchQuery(query=tpl.format(area=a), signal_type=spec.name, area=a, rationale="template")
                       for a in brief.target_areas]
            else:
                qs.append(SearchQuery(query=tpl, signal_type=spec.name, rationale="template"))
        per_type[spec.name] = qs
    out: list[SearchQuery] = []
    iters = {k: iter(v) for k, v in per_type.items()}
    for k in cycle(list(iters)):
        if len(out) >= max_queries or not iters:
            break
        if k not in iters:
            continue
        nxt = next(iters[k], None)
        if nxt is None:
            iters.pop(k)
            continue
        out.append(nxt)
    return SearchPlan(queries=out)


def plan(brief: Brief, llm: LLM, max_queries: int, recent_queries: list[str] | None = None) -> SearchPlan:
    prompt = (
        f"Today: {date.today().isoformat()}\nBrief (JSON):\n{brief.model_dump_json(indent=1)}\n\n"
        f"Queries used in recent runs (avoid exact repeats, vary angle):\n{(recent_queries or [])[:30]}\n\n"
        f"Return at most {max_queries} queries."
    )
    try:
        p = llm.structured(SYSTEM, prompt, SearchPlan, agent="planner")
        valid = {s.value for s in SignalType} - {"inbound_lead"}
        p.queries = [q for q in p.queries if q.signal_type in valid][:max_queries]
        if p.queries:
            return p
    except LLMUnavailable:
        pass
    except Exception as e:  # noqa: BLE001
        log.warning("planner LLM failed (%s); using template plan", e)
    return template_plan(brief, max_queries)
