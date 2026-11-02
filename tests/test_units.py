from __future__ import annotations

from mlia.agents.extractor import heuristic_extract
from mlia.agents.planner import template_plan
from mlia.agents.verifier import quote_match, verify
from mlia.config import load_brief
from mlia.dedup import dedup, fingerprint
from mlia.geo import budget_band, normalize_area, parse_money
from mlia.matcher import BrokerRouter, match_units
from mlia.pii import has_private_life, mask_email, mask_phone, redact
from mlia.schemas import ExtractedSignal, Page, SignalType, VerifiedSignal
from mlia.synthetic import make_units
from mlia.web.search import is_allowed


def _vs(**kw) -> VerifiedSignal:
    base = dict(signal_type="new_launch", entity="Horizon Developments", area="New Cairo", summary="s",
                evidence_quote="q", confidence=0.9, source_url="https://x", verified=True)
    return VerifiedSignal(**{**base, **kw})


def test_area_normalization():
    assert normalize_area("Fifth Settlement") == "New Cairo"
    assert normalize_area("New Administrative Capital") == "New Capital"
    assert normalize_area("sahel") == "North Coast"
    assert normalize_area("Zayed") == "Sheikh Zayed"


def test_money_and_bands():
    assert parse_money("from EGP 7.8 million") == [7_800_000]
    assert budget_band(7.8e6).value == "6-10M EGP"
    assert budget_band(None).value == "unknown"


def test_planner_template_round_robin():
    brief = load_brief()
    plan = template_plan(brief, 10)
    assert len(plan.queries) == 10
    assert len({q.signal_type for q in plan.queries}) == len(brief.signal_types)


def test_verifier_rejects_hallucinated_quote():
    page = Page(url="u", text="Horizon Developments launched Phase 3 of Palm Crest in New Cairo.")
    good = ExtractedSignal(signal_type="new_launch", entity="Horizon Developments", summary="x",
                           evidence_quote="Horizon Developments launched Phase 3 of Palm Crest in New Cairo.", confidence=0.9)
    bad = good.model_copy(update={"evidence_quote": "Horizon cut prices by 30% in Sheikh Zayed last year."})
    assert verify(good, page, 0.55).verified
    assert not verify(bad, page, 0.55).verified
    assert quote_match("launched  Phase 3", page.text) == 0.0 or quote_match("launched  Phase 3", page.text) == 100.0


def test_verifier_low_confidence():
    page = Page(url="u", text="Delta Pharma inaugurated new offices in Sheikh Zayed.")
    s = ExtractedSignal(signal_type="office_opening", entity="Delta Pharma", summary="x",
                        evidence_quote="Delta Pharma inaugurated new offices in Sheikh Zayed.", confidence=0.3)
    v = verify(s, page, 0.55)
    assert not v.verified and "low confidence" in v.rejection_reason


def test_dedup_fuzzy_and_history():
    a = _vs(entity="Horizon Developments")
    b = _vs(entity="Horizon Development", confidence=0.8)
    new, dups, stats = dedup([a, b], [], 90)
    assert len(new) == 1 and len(dups) == 1 and stats["duplicate_rate"] == 0.5
    hist = [{"fingerprint": fingerprint(a), "entity": a.entity, "signal_type": "new_launch", "area": "New Cairo"}]
    new2, _, _ = dedup([a], hist, 90)
    assert new2 == []


def test_dedup_keeps_new_price_change():
    old = _vs(signal_type="price_change", entity="Nile Crest Group", price_change_pct=10.0)
    hist = [{"fingerprint": fingerprint(old), "entity": old.entity, "signal_type": "price_change", "area": "New Cairo"}]
    new, _, _ = dedup([_vs(signal_type="price_change", entity="Nile Crest Group", price_change_pct=12.0)], hist, 90)
    assert len(new) == 1


def test_pii():
    assert has_private_life("His wife and children moved to a villa")
    assert not has_private_life("The company opened a new office")
    r = redact("Call 01012345678 or mail ahmed@example.com, ID 29001011234567")
    assert "01012345678" not in r and "ahmed@example.com" not in r and "29001011234567" not in r
    assert mask_phone("+201012345678") == "***678"
    assert mask_email("ahmed@example.com") == "a***@example.com"


def test_blocked_domains():
    assert not is_allowed("https://www.facebook.com/someone")
    assert not is_allowed("https://www.linkedin.com/in/someone")
    assert is_allowed("https://www.linkedin.com/company/acme")


def test_matcher_and_router():
    units = make_units(300)
    m = match_units(units, "New Cairo", 8e6)
    assert m and all(u.area == "New Cairo" and 6.4e6 <= u.price_egp <= 8.8e6 for u in m)
    assert match_units(units, None, 8e6) == []
    r = BrokerRouter(load_brief())
    assert r.assign("North Coast") == "Broker C"


def test_heuristic_extractor_finds_launch():
    page = Page(url="u", title="t", text="Horizon Developments launched Phase 3 of its Palm Crest compound in New Cairo on Sunday. "
                                         "Prices start from EGP 7.8 million, with a payment plan of 10% down over 8 years.")
    ex = heuristic_extract(page)
    types = {s.signal_type for s in ex.signals}
    assert SignalType.new_launch in types
    launch = next(s for s in ex.signals if s.signal_type == SignalType.new_launch)
    assert launch.area == "New Cairo"
