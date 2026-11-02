"""Storage (SQLite locally, Postgres via MLIA_DATABASE_URL): run history, signals, leads, inventory."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

from .config import get_settings

metadata = MetaData()

runs = Table(
    "runs", metadata,
    Column("run_id", String(40), primary_key=True),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Column("status", String(20)),
    Column("llm_provider", String(20)),
    Column("llm_model", String(60)),
    Column("search_provider", String(20)),
    Column("stats", JSON),
    Column("cost_usd", Float),
    Column("excel_path", Text),
    Column("errors", JSON),
)

signals = Table(
    "signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("fingerprint", String(64), index=True),
    Column("run_id", String(40), index=True),
    Column("signal_type", String(30)),
    Column("entity", String(200)),
    Column("project", String(200)),
    Column("area", String(80)),
    Column("summary", Text),
    Column("budget_band", String(20)),
    Column("price_change_pct", Float),
    Column("payment_plan", String(200)),
    Column("evidence_quote", Text),
    Column("source_url", Text),
    Column("confidence", Float),
    Column("quote_match", Float),
    Column("verified", Boolean),
    Column("rejection_reason", Text),
    Column("collected_at", DateTime(timezone=True)),
)

leads = Table(
    "leads", metadata,
    Column("lead_id", String(40), primary_key=True),
    Column("payload", JSON),  # InboundLead (PII redacted fields only for export)
    Column("consent_marketing", Boolean),
    Column("consent_source", String(80)),
    Column("created_at", DateTime(timezone=True)),
    Column("label_meeting", Boolean, nullable=True),  # broker feedback: did it become a meeting?
)

units = Table(
    "units", metadata,
    Column("unit_id", String(40), primary_key=True),
    Column("payload", JSON),
    Column("area", String(80), index=True),
    Column("price_egp", Float),
    Column("available", Boolean),
)

output_rows = Table(
    "output_rows", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), index=True),
    Column("row_id", String(64), index=True),
    Column("payload", JSON),
    Column("intent_score", Float),
    Column("broker_feedback", String(20), nullable=True),  # qualified | not_qualified | meeting
    Column("collected_at", DateTime(timezone=True)),
)

page_hashes = Table(
    "page_hashes", metadata,
    Column("hash", String(32), primary_key=True),
    Column("url", Text),
    Column("first_seen", DateTime(timezone=True)),
)

source_log = Table(
    "source_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), index=True),
    Column("kind", String(20)),
    Column("target", Text),
    Column("provider", String(20)),
    Column("ok", Boolean),
    Column("n", Integer),
    Column("ms", Integer),
    Column("error", Text),
)

_engine: Engine | None = None


def engine(url: str | None = None) -> Engine:
    global _engine
    if url:
        return _make(url)
    if _engine is None:
        _engine = _make(get_settings().database_url)
    return _engine


def _make(url: str) -> Engine:
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("sqlite:///"):
        Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, future=True, pool_pre_ping=True)
    metadata.create_all(eng)
    return eng


# ------------------------------------------------------------------ helpers
def json_safe(v):
    """Postgres JSON rejects NaN/Infinity; SQLite accepts them. Normalize to None everywhere."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: json_safe(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [json_safe(x) for x in v]
    return v


# ------------------------------------------------------------------ queries
def known_fingerprints(eng: Engine) -> set[str]:
    with eng.connect() as c:
        return {r[0] for r in c.execute(select(signals.c.fingerprint).where(signals.c.verified.is_(True)))}


def known_signals(eng: Engine) -> list[dict]:
    with eng.connect() as c:
        rows = c.execute(select(signals.c.fingerprint, signals.c.entity, signals.c.signal_type, signals.c.area,
                                signals.c.project).where(signals.c.verified.is_(True))).mappings().all()
    return [dict(r) for r in rows]


def seen_page_hashes(eng: Engine) -> set[str]:
    with eng.connect() as c:
        return {r[0] for r in c.execute(select(page_hashes.c.hash))}


def recent_queries(eng: Engine, limit: int = 3) -> list[str]:
    with eng.connect() as c:
        rs = c.execute(select(source_log.c.target).where(source_log.c.kind == "search")
                       .order_by(source_log.c.id.desc()).limit(limit * 15)).all()
    return list(dict.fromkeys(r[0] for r in rs))


def insert_many(eng: Engine, table: Table, rows: list[dict]) -> None:
    if rows:
        with eng.begin() as c:
            c.execute(insert(table), [json_safe(r) for r in rows])


def upsert_by_key(eng: Engine, table: Table, key: str, rows: list[dict]) -> None:
    with eng.begin() as c:
        for r in map(json_safe, rows):
            exists = c.execute(select(table.c[key]).where(table.c[key] == r[key])).first()
            if exists:
                c.execute(update(table).where(table.c[key] == r[key]).values(**r))
            else:
                c.execute(insert(table).values(**r))


def apply_retention(eng: Engine, days: int) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    with eng.begin() as c:
        n = c.execute(delete(signals).where(signals.c.collected_at < cutoff)).rowcount
        n += c.execute(delete(output_rows).where(output_rows.c.collected_at < cutoff)).rowcount
        n += c.execute(delete(leads).where(leads.c.created_at < cutoff)).rowcount
    return n
