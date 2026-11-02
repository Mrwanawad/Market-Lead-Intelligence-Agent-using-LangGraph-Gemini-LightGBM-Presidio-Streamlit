"""Verifier agent: every row must cite a quote that actually exists on the page."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from ..geo import normalize_area
from ..schemas import ExtractedSignal, Page, VerifiedSignal
from .extractor import NOT_ENTITY

QUOTE_MIN_MATCH = 90.0
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    return _WS.sub(" ", s).strip().lower()


def quote_match(quote: str, text: str) -> float:
    q, t = _norm(quote), _norm(text)
    if not q:
        return 0.0
    if q in t:
        return 100.0
    if len(q) < 25:
        return 0.0
    return float(fuzz.partial_ratio(q, t))


def verify(sig: ExtractedSignal, page: Page, min_confidence: float) -> VerifiedSignal:
    v = VerifiedSignal(**sig.model_dump(), source_url=page.url, source_title=page.title)
    v.area = normalize_area(v.area)
    v.quote_match = round(quote_match(sig.evidence_quote, page.text), 1)
    reasons = []
    if v.quote_match < QUOTE_MIN_MATCH:
        reasons.append(f"evidence quote not found on page (match {v.quote_match})")
    if sig.confidence < min_confidence:
        reasons.append(f"low confidence {sig.confidence:.2f}")
    if not sig.entity or sig.entity.strip().lower() in {"unknown", "n/a", "none"} | NOT_ENTITY:
        reasons.append("no entity")
    v.verified = not reasons
    v.rejection_reason = "; ".join(reasons) or None
    return v
