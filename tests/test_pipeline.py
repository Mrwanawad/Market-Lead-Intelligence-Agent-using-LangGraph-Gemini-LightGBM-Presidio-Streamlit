"""End-to-end offline run: fixtures search + heuristic extractor + synthetic leads/inventory."""

from __future__ import annotations

import openpyxl

from mlia import db
from mlia.llm import LLM
from mlia.pipeline import run, seed
from mlia.scoring import train
from mlia.synthetic import training_frame


def test_model_trains_and_explains(tmp_path):
    m = train(training_frame(n=1500))
    assert m.metrics["auc"] > 0.65
    scores, reasons = m.score([{"signal_type": "inbound_lead", "recency_days": 0.2, "budget_fit": 0.9, "area_demand": 6,
                                "has_budget": 1, "channel": "referral", "pages_viewed": 9, "reply_minutes": 2,
                                "previous_contacts": 1, "intent_keywords": 3}])
    assert 0 <= scores[0] <= 100 and reasons[0]


def test_full_run_is_idempotent(settings):
    eng = db.engine(settings.database_url)
    seed(eng, n_units=150, n_leads=20)
    llm = LLM(provider="fake")

    r1 = run(settings=settings, llm=llm)
    st = r1["stats"]
    assert st["pages_fetched"] >= 6
    assert st["new_signals"] >= 4
    assert st["leads_scored"] > 0 and st["leads_skipped_no_consent"] >= 0
    types = {s.signal_type.value for s in r1["signals"]}
    assert {"new_launch", "price_change", "office_opening"} <= types
    # private-life gossip page must never reach output
    assert not any("wife" in (s.evidence_quote or "").lower() for s in r1["signals"])
    # every signal row cites a verbatim quote and a source
    assert all(s.evidence_quote and s.source_url and s.quote_match >= 90 for s in r1["signals"])
    # Palm Crest launch reported by two sites -> one row
    horizon = [s for s in r1["signals"] if s.signal_type.value == "new_launch" and "horizon" in s.entity.lower()]
    assert len(horizon) == 1

    wb = openpyxl.load_workbook(r1["excel"])
    assert {"CRM Rows", "Market Digest", "Run Log", "Sources", "Rejected"} <= set(wb.sheetnames)
    header = [c.value for c in wb["CRM Rows"][1]]
    for col in ["lead_or_company", "source_url", "signal_type", "area", "budget_band", "intent_score", "score_reasons",
                "matched_units", "assigned_broker", "next_action", "collected_at"]:
        assert col in header
    # exported sheet never holds raw phone numbers / emails
    raw = " ".join(str(c.value) for row in wb["CRM Rows"].iter_rows() for c in row)
    assert "+2010" not in raw and "@example.com" not in raw.replace("***@example.com", "")

    # second run: same pages -> nothing new
    r2 = run(settings=settings, llm=LLM(provider="fake"))
    assert r2["stats"]["new_signals"] == 0
    assert r2["stats"]["rows_new"] == 0
