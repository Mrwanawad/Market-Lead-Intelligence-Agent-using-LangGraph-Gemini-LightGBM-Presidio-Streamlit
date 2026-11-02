"""Offline evaluation: extraction field accuracy on labeled pages, verifier behaviour, broker precision."""

from __future__ import annotations

import json
from pathlib import Path

from .agents.extractor import extract
from .agents.verifier import verify
from .config import get_settings
from .dedup import same_entity
from .feedback import precision
from .geo import normalize_area
from .llm import LLM
from .pii import has_private_life
from .schemas import SearchHit
from .web.fetch import fetch


def run_eval(labels_path: Path, llm: LLM | None = None) -> dict:
    llm = llm or LLM()
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    s = get_settings()
    n_pos = hits = type_ok = ent_ok = area_ok = plan_ok = plan_n = 0
    false_pos = negatives = 0
    per_page = []
    for lab in labels:
        page = fetch(SearchHit(url=lab["url"]))
        sigs = [verify(x, page, s.verifier_min_confidence) for x in extract(page, llm).signals]
        kept = [v for v in sigs if v.verified and not has_private_life(v.evidence_quote)]
        if lab.get("signal_type") is None:
            negatives += 1
            false_pos += bool(kept)
            per_page.append({"url": lab["url"], "expected": None, "kept": len(kept)})
            continue
        n_pos += 1
        best = next((v for v in kept if v.signal_type == lab["signal_type"]), kept[0] if kept else None)
        if best is None:
            per_page.append({"url": lab["url"], "expected": lab["signal_type"], "kept": 0})
            continue
        hits += 1
        type_ok += best.signal_type == lab["signal_type"]
        ent_ok += same_entity(best.entity, lab["entity"], 80)
        area_ok += normalize_area(best.area) == lab.get("area")
        if lab.get("payment_plan"):
            plan_n += 1
            plan_ok += bool(best.payment_plan)
        per_page.append({"url": lab["url"], "expected": lab["signal_type"], "got": best.signal_type.value,
                         "entity": best.entity, "area": best.area, "quote_match": best.quote_match})
    return {
        "llm": f"{llm.provider}/{llm.model}",
        "recall_pages": round(hits / n_pos, 3) if n_pos else None,
        "field_accuracy": {
            "signal_type": round(type_ok / hits, 3) if hits else None,
            "entity": round(ent_ok / hits, 3) if hits else None,
            "area": round(area_ok / hits, 3) if hits else None,
            "payment_plan": round(plan_ok / plan_n, 3) if plan_n else None,
        },
        "negative_pages_false_positive_rate": round(false_pos / negatives, 3) if negatives else None,
        "cost_usd": llm.usage.cost_usd,
        "broker_feedback": precision(),
        "pages": per_page,
    }
