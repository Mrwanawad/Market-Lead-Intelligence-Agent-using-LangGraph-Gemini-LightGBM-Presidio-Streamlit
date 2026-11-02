"""Extractor agent: page -> list[ExtractedSignal] (Pydantic structured output, evidence-grounded)."""

from __future__ import annotations

import logging
import re

from ..geo import budget_band, find_area, parse_money
from ..llm import LLM, LLMUnavailable
from ..schemas import ExtractedSignal, Extraction, Page, SignalType

log = logging.getLogger(__name__)

SYSTEM = """You extract real-estate market signals for an Egyptian brokerage from ONE web page.
Rules:
- Only extract signals actually stated on the page. No guessing, no outside knowledge.
- evidence_quote MUST be copied verbatim (exact characters) from the page text: one or two full sentences.
- Business information only. Never extract private-life data about individuals (family, health,
  salary guesses, home address, live location). People may appear only as company spokespersons.
- signal_type: new_launch | price_change | competitor_move | office_opening | relocation_firm.
  office_opening = a private company opening/expanding offices or HQ in Egypt (corporate housing demand);
  not government relocations or encyclopedia background.
  competitor_move = a property portal, brokerage or proptech launching a product, partnership or campaign;
  NOT infrastructure, transport, shipping, industrial zones or general economic news (skip those).
- Skip evergreen/background pages (encyclopedias, market-size reports) unless they state a dated new event.
- area: Egyptian city/area (e.g. New Cairo, New Capital, Sheikh Zayed, 6th of October, North Coast, Ain Sokhna).
- budget_band: from unit prices if stated (<3M EGP, 3-6M EGP, 6-10M EGP, 10-20M EGP, >20M EGP) else unknown.
- confidence: 0-1, how clearly the page states the signal.
- Return an empty list if the page has no relevant signal (ads, unrelated, older than ~6 months)."""

KEYWORDS: dict[SignalType, list[str]] = {
    SignalType.new_launch: ["launch", "launches", "launched", "unveil", "new phase", "new project", "new compound", "kicks off sales"],
    SignalType.price_change: ["price increase", "raised prices", "price hike", "increase prices", "cut prices",
                              "payment plan", "down payment", "installment", "per square meter", "price per meter"],
    SignalType.competitor_move: ["partnership", "partners with", "rolls out", "introduces", "new feature", "campaign", "acquires"],
    SignalType.office_opening: ["opens office", "opened its", "new office", "new headquarters", "regional hub",
                                "expands its presence", "inaugurat", "opens its", "new hq", "regional headquarters"],
    SignalType.relocation_firm: ["relocat", "mobility", "expatriate", "assignees", "moving employees"],
}
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_ORG_SUFFIX = re.compile(
    r"((?:[A-Z][\w&'.-]*\s){1,4}(?:Developments?|Group|Properties|Holding|Egypt|Bank|Pharma|Technologies|Tech|"
    r"Software|Corp|Inc|Company|Realty|Finder|Consulting|Relocations?|Mobility|Urban|Capital|Partners))\b")
_CAP_RUN = re.compile(r"^((?:[A-Z][\w&'.-]+)(?:\s+[A-Z][\w&'.-]+){0,3})")
_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")
NOT_ENTITY = {
    "the", "he", "she", "we", "they", "it", "this", "that", "these", "those", "from", "since", "firstly", "instead",
    "meanwhile", "interested", "established", "reserve", "apartment", "new", "our", "your", "in", "on", "at", "as",
    "for", "with", "according", "remember", "press", "however", "also", "moreover", "additionally", "while", "after",
    "before", "during", "egypt", "cairo", "prices", "price", "units", "unit", "buyers", "brokers", "developers",
}
NOT_ENTITY_UPPER = {"EGP", "USD", "LE", "CEO", "CFO", "HQ", "PDPL", "NAC", "UAE", "KSA", "AI", "IT", "EOI", "BR"}


def find_entity(sent: str) -> str | None:
    """Company name from a sentence: corporate-suffix match, else a sentence-initial proper-noun run, else acronym."""
    m = _ORG_SUFFIX.search(sent)
    if m and m.group(1).split()[0].lower() not in NOT_ENTITY:
        return m.group(1).strip()
    m = _CAP_RUN.match(sent)
    if m:
        words = m.group(1).split()
        while words and words[0].lower().strip("-'") in NOT_ENTITY:
            words = words[1:]
        if words and (len(words) >= 2 or words[0].isupper()) and words[0] not in NOT_ENTITY_UPPER:
            return " ".join(words)
    for a in _ACRONYM.findall(sent):
        if a not in NOT_ENTITY_UPPER:
            return a
    return None


_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_PLAN = re.compile(r"(\d+(?:\.\d+)?\s*%\s*down[^.;]{0,40}?(?:over|across|in)?\s*\d+\s*years?)", re.I)


def heuristic_extract(page: Page) -> Extraction:
    """Rule-based fallback (offline mode / LLM failure). Lower confidence than the LLM path."""
    sigs: list[ExtractedSignal] = []
    for sent in _SENT.split(" ".join(page.text.split())):
        low = sent.lower()
        for st, kws in KEYWORDS.items():
            hits = sum(k in low for k in kws)
            if not hits:
                continue
            ent = find_entity(sent)
            if not ent:
                continue
            money = [m for m in parse_money(sent) if m >= 300_000]
            pct = _PCT.search(sent) if st == SignalType.price_change else None
            plan = _PLAN.search(sent)
            sigs.append(ExtractedSignal(
                signal_type=st,
                entity=ent,
                area=find_area(sent) or find_area(page.title) or find_area(page.text[:1500]),
                summary=sent[:220],
                budget_band=budget_band(min(money)) if money else "unknown",
                price_change_pct=float(pct.group(1)) if pct else None,
                payment_plan=plan.group(1) if plan else None,
                evidence_quote=sent,
                confidence=min(0.9, 0.5 + 0.15 * hits),
            ))
            break
    # keep strongest signal per (type, entity)
    best: dict[tuple, ExtractedSignal] = {}
    for s in sigs:
        k = (s.signal_type, s.entity.lower())
        if k not in best or s.confidence > best[k].confidence:
            best[k] = s
    return Extraction(signals=list(best.values())[:5])


def extract(page: Page, llm: LLM, max_chars: int = 12000) -> Extraction:
    prompt = (f"URL: {page.url}\nTITLE: {page.title}\nSEARCH INTENT: {page.signal_type}\n\n"
              f"PAGE TEXT:\n<<<\n{page.text[:max_chars]}\n>>>")
    try:
        return llm.structured(SYSTEM, prompt, Extraction, agent="extractor")
    except LLMUnavailable:
        return heuristic_extract(page)
    except Exception as e:  # noqa: BLE001
        log.warning("extractor LLM failed on %s (%s); heuristic fallback", page.url, e)
        return heuristic_extract(page)
