"""Pydantic contracts passed between agents and written to storage."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class SignalType(StrEnum):
    new_launch = "new_launch"
    price_change = "price_change"
    competitor_move = "competitor_move"
    office_opening = "office_opening"
    relocation_firm = "relocation_firm"
    inbound_lead = "inbound_lead"


class BudgetBand(StrEnum):
    unknown = "unknown"
    under_3m = "<3M EGP"
    b3_6m = "3-6M EGP"
    b6_10m = "6-10M EGP"
    b10_20m = "10-20M EGP"
    over_20m = ">20M EGP"


# ---------- planner ----------
class SearchQuery(BaseModel):
    query: str
    signal_type: SignalType
    area: str | None = None
    rationale: str = ""


class SearchPlan(BaseModel):
    queries: list[SearchQuery]


# ---------- researcher ----------
class SearchHit(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    query: str = ""
    signal_type: SignalType | None = None
    provider: str = ""


class Page(BaseModel):
    url: str
    title: str = ""
    text: str
    fetched_with: str = "http"  # http | browser | fixture | snippet
    query: str = ""
    signal_type: SignalType | None = None
    fetched_at: datetime = Field(default_factory=utcnow)


# ---------- extractor ----------
class ExtractedSignal(BaseModel):
    """One market signal grounded in a verbatim quote from the page."""

    signal_type: SignalType
    entity: str = Field(description="Company / developer / project name the signal is about")
    project: str | None = Field(default=None, description="Project or compound name if mentioned")
    area: str | None = Field(default=None, description="Egyptian area / city, normalized if possible")
    summary: str = Field(description="One sentence, factual, no speculation")
    budget_band: BudgetBand = BudgetBand.unknown
    price_egp_per_m2: float | None = None
    price_change_pct: float | None = None
    payment_plan: str | None = Field(default=None, description="e.g. '10% down, 8 years'")
    event_date: str | None = Field(default=None, description="ISO date if stated on page")
    evidence_quote: str = Field(description="Exact verbatim sentence copied from the page that supports the signal")
    confidence: float = Field(ge=0, le=1)


class Extraction(BaseModel):
    signals: list[ExtractedSignal] = Field(default_factory=list)


# ---------- verifier / downstream ----------
class VerifiedSignal(ExtractedSignal):
    source_url: str
    source_title: str = ""
    quote_match: float = 0.0  # fuzzy match of evidence quote vs page text (0-100)
    verified: bool = False
    rejection_reason: str | None = None
    collected_at: datetime = Field(default_factory=utcnow)


class InboundLead(BaseModel):
    lead_id: str
    name: str
    channel: str  # website_form | whatsapp | ad_form | referral
    phone: str | None = None
    email: str | None = None
    message: str = ""
    area_interest: str | None = None
    stated_budget_egp: float | None = None
    unit_type: str | None = None
    consent_marketing: bool = False
    consent_source: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    pages_viewed: int = 0
    reply_minutes: float | None = None
    previous_contacts: int = 0


class Unit(BaseModel):
    unit_id: str
    project: str
    developer: str
    area: str
    unit_type: str
    bedrooms: int
    size_m2: float
    price_egp: float
    down_payment_pct: float
    installment_years: int
    delivery_year: int
    available: bool = True


class OutputRow(BaseModel):
    row_id: str
    row_kind: str  # signal | lead
    lead_or_company: str
    source_url: str
    signal_type: str
    area: str | None
    budget_band: str
    intent_score: float
    score_reasons: str
    matched_units: str
    assigned_broker: str
    next_action: str
    summary: str
    evidence_quote: str
    consent: str
    collected_at: datetime
    is_new: bool = True
