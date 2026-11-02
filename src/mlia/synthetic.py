"""Synthetic (clearly fake) inventory, consented inbound leads and bootstrap training data.

Real deployments replace these with the inventory DB, the lead intake (forms / WhatsApp / ad forms)
and broker-confirmed outcomes. All names, phones and emails below are generated and fictional.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from .schemas import InboundLead, SignalType, Unit

AREAS = {  # area: (price per m2 EGP range, typical projects)
    "New Cairo": ((55_000, 95_000), ["Palm Crest", "Azure Heights", "Cedar Walk"]),
    "New Capital": ((40_000, 70_000), ["Capital Gardens", "Iconic Residence", "Green River"]),
    "Mostakbal City": ((38_000, 60_000), ["Future Park", "Bloom Fields"]),
    "Sheikh Zayed": ((50_000, 90_000), ["Zayed Dunes", "Olive Court", "Westside Lofts"]),
    "6th of October": ((30_000, 55_000), ["October Hills", "Garden Mile"]),
    "North Coast": ((80_000, 160_000), ["Sea Breeze Bay", "Hekma Shores"]),
    "Ain Sokhna": ((45_000, 85_000), ["Galala Cliffs", "Sokhna Marina"]),
}
DEVELOPERS = ["Horizon Developments", "Nile Crest Group", "Sahara Urban", "Delta Living", "Pyramid Gate Properties"]
TYPES = [("Apartment", (1, 3), (75, 180)), ("Duplex", (3, 4), (180, 280)), ("Townhouse", (3, 4), (200, 300)),
         ("Villa", (4, 6), (280, 500)), ("Chalet", (1, 3), (70, 140)), ("Office", (0, 0), (40, 200))]
FIRST = ["Ahmed", "Mona", "Omar", "Sara", "Youssef", "Nour", "Karim", "Salma", "Hassan", "Laila", "Tarek", "Dina",
         "Mostafa", "Heba", "Amr", "Rana", "Khaled", "Yasmin", "Ali", "Farida"]
LAST = ["Fahmy", "Saleh", "Mansour", "Ezzat", "Rashad", "Naguib", "Hamdy", "Lotfy", "Shaker", "Gaber"]
MESSAGES = [
    "Looking for a {t} in {a}, budget around {b}M, ready to visit this month",
    "Interested in {a}. Can you send prices and payment plans for {t}s?",
    "Need a {t} for my family in {a}, cash buyer, when can I see units?",
    "Just browsing {a} projects, maybe next year",
    "Company relocating 5 staff to {a}, need furnished apartments ASAP",
    "What is the down payment for {t}s in {a}?",
    "Price list please",
    "Want to book a meeting to discuss a {t} in {a}, budget {b}M",
]


def make_units(n: int = 240, seed: int = 11) -> list[Unit]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        area = rng.choice(list(AREAS))
        (plo, phi), projects = AREAS[area]
        t, (bmin, bmax), (smin, smax) = TYPES[rng.integers(len(TYPES))]
        if area == "North Coast" and t == "Office":
            t, (bmin, bmax), (smin, smax) = TYPES[4]
        size = float(rng.integers(smin, smax + 1))
        price = round(size * rng.uniform(plo, phi), -4)
        out.append(Unit(
            unit_id=f"U{i + 1:04d}", project=str(rng.choice(projects)), developer=str(rng.choice(DEVELOPERS)),
            area=str(area), unit_type=t, bedrooms=int(rng.integers(bmin, bmax + 1)), size_m2=size, price_egp=price,
            down_payment_pct=float(rng.choice([0, 5, 10, 15, 20])), installment_years=int(rng.choice([5, 7, 8, 10, 12])),
            delivery_year=int(rng.choice([2026, 2027, 2028, 2029, 2030])), available=bool(rng.random() > 0.1),
        ))
    return out


def make_leads(n: int = 40, seed: int = 5, now: datetime | None = None) -> list[InboundLead]:
    rng = np.random.default_rng(seed)
    now = now or datetime.now(UTC)
    out = []
    for i in range(n):
        area = str(rng.choice(list(AREAS)))
        t = str(rng.choice(["Apartment", "Villa", "Townhouse", "Duplex", "Chalet"]))
        budget = float(rng.choice([2.5, 4, 5.5, 7, 9, 12, 18, 25]))
        msg = str(rng.choice(MESSAGES)).format(t=t.lower(), a=area, b=budget)
        stated = budget * 1e6 if "{b}" in msg or "budget" in msg.lower() else (budget * 1e6 if rng.random() > 0.5 else None)
        first, last = rng.choice(FIRST), rng.choice(LAST)
        consent = bool(rng.random() > 0.12)
        out.append(InboundLead(
            lead_id=f"L{i + 1:04d}", name=f"{first} {last}",
            channel=str(rng.choice(["website_form", "whatsapp", "ad_form", "referral"], p=[0.4, 0.3, 0.2, 0.1])),
            phone=f"+2010{rng.integers(10_000_000, 99_999_999)}", email=f"{first.lower()}.{last.lower()}{i}@example.com",
            message=msg, area_interest=area, stated_budget_egp=stated, unit_type=t,
            consent_marketing=consent, consent_source="web form checkbox v2" if consent else "",
            created_at=now - timedelta(hours=float(rng.uniform(0.5, 24 * 10))),
            pages_viewed=int(rng.poisson(4)), reply_minutes=[2.0, 5.0, 15.0, 60.0, 240.0, None][int(rng.integers(6))],
            previous_contacts=int(rng.poisson(0.6)),
        ))
    return out


def training_frame(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    """Bootstrap labels from a hidden logistic 'truth' so the model has realistic structure.
    Replaced over time by broker feedback (output_rows.broker_feedback)."""
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["lead", "signal"], size=n, p=[0.6, 0.4])
    types = [s.value for s in SignalType if s != SignalType.inbound_lead]
    rows = []
    for k in kinds:
        if k == "lead":
            r = dict(signal_type="inbound_lead", recency_days=rng.exponential(4), budget_fit=rng.beta(2, 2),
                     area_demand=rng.poisson(5), has_budget=float(rng.random() > 0.4), confidence=np.nan,
                     channel=str(rng.choice(["website_form", "whatsapp", "ad_form", "referral"])),
                     pages_viewed=rng.poisson(4), reply_minutes=rng.choice([2, 5, 15, 60, 240, 1440]),
                     previous_contacts=rng.poisson(0.6), intent_keywords=rng.poisson(1.0))
            z = (-1.2 - 0.25 * r["recency_days"] + 1.8 * r["budget_fit"] + 0.08 * r["area_demand"]
                 + 0.8 * r["has_budget"] + 0.12 * min(r["pages_viewed"], 12) - 0.0015 * r["reply_minutes"]
                 + 0.3 * r["previous_contacts"] + 0.6 * min(r["intent_keywords"], 3)
                 + {"referral": 0.9, "whatsapp": 0.4, "website_form": 0.1, "ad_form": -0.3}[r["channel"]])
        else:
            st = str(rng.choice(types))
            r = dict(signal_type=st, recency_days=rng.exponential(10), budget_fit=rng.beta(2, 3),
                     area_demand=rng.poisson(5), has_budget=float(rng.random() > 0.6), confidence=rng.uniform(0.5, 1),
                     channel=np.nan, pages_viewed=np.nan, reply_minutes=np.nan, previous_contacts=np.nan,
                     intent_keywords=np.nan)
            z = (-1.6 - 0.08 * r["recency_days"] + 1.2 * r["budget_fit"] + 0.1 * r["area_demand"]
                 + 1.5 * (r["confidence"] - 0.5) + 0.3 * r["has_budget"]
                 + {"office_opening": 1.3, "relocation_firm": 0.9, "new_launch": 0.5, "price_change": 0.3,
                    "competitor_move": -0.8}[st])
        r["label"] = int(rng.random() < 1 / (1 + np.exp(-z)))
        rows.append(r)
    return pd.DataFrame(rows)
