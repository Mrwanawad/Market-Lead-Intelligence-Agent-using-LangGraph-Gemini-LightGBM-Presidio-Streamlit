"""Unit matcher, broker assignment and next-action rules."""

from __future__ import annotations

from collections import Counter

from .config import Brief
from .geo import BAND_RANGES
from .schemas import BudgetBand, Unit


def match_units(units: list[Unit], area: str | None, budget_egp: float | None = None,
                band: str | None = None, unit_type: str | None = None, k: int = 3) -> list[Unit]:
    if not area:
        return []
    pool = [u for u in units if u.available and u.area == area]
    if unit_type:
        typed = [u for u in pool if u.unit_type.lower() == unit_type.lower()]
        pool = typed or pool
    if budget_egp:
        lo, hi, target = budget_egp * 0.8, budget_egp * 1.1, budget_egp
    elif band and band != BudgetBand.unknown and band in {b.value for b in BAND_RANGES}:
        lo, hi = BAND_RANGES[BudgetBand(band)]
        target = (lo + min(hi, 40e6)) / 2
    else:
        return sorted(pool, key=lambda u: u.price_egp)[:k]
    within = [u for u in pool if lo <= u.price_egp <= hi]
    return sorted(within, key=lambda u: abs(u.price_egp - target))[:k]


def fmt_units(us: list[Unit]) -> str:
    return " | ".join(f"{u.unit_id} {u.project} {u.bedrooms}BR {u.size_m2:.0f}m² {u.price_egp / 1e6:.1f}M "
                      f"({u.down_payment_pct:.0f}%/{u.installment_years}y)" for u in us)


class BrokerRouter:
    """Area owner first; ties and uncovered areas go to the least-loaded broker this run."""

    def __init__(self, brief: Brief):
        self.brokers = brief.brokers or [{"name": "Unassigned", "areas": []}]
        self.load: Counter = Counter()

    def assign(self, area: str | None) -> str:
        owners = [b["name"] for b in self.brokers if area and area in b.get("areas", [])]
        cands = owners or [b["name"] for b in self.brokers]
        name = min(cands, key=lambda n: self.load[n])
        self.load[name] += 1
        return name


def next_action(kind: str, signal_type: str, score: float, has_units: bool) -> str:
    if kind == "lead":
        if score >= 70:
            return "Call within 1 hour; send matched units on WhatsApp; propose site visit this week"
        if score >= 40:
            return "Call today; qualify budget/timeline; share 2-3 matched units"
        return "Add to nurture sequence; follow up in 7 days"
    return {
        "office_opening": "Contact company HR/admin for corporate housing & relocation package"
                          + ("; prepare shortlist of matched units" if has_units else ""),
        "relocation_firm": "Pitch partnership to relocation firm (referral agreement for incoming staff)",
        "new_launch": "Request launch price sheet & EOI terms from developer; brief brokers on area",
        "price_change": "Notify pipeline leads in this area about the price/payment-plan change before it takes effect",
        "competitor_move": "Share with growth team; review our offer/listing coverage vs. competitor",
    }.get(signal_type, "Review")
