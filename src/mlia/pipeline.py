"""One full run: research graph -> dedup -> PII filter -> scoring -> matcher -> outputs -> history."""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from . import db
from .agents.researcher import content_hash
from .config import Brief, Settings, get_settings, load_brief
from .dedup import dedup, fingerprint
from .geo import budget_band
from .graph import build_graph
from .llm import LLM
from .matcher import BrokerRouter, fmt_units, match_units, next_action
from .observability import Tracer, source_success_rate
from .outputs import digest_markdown, push_crm, write_excel
from .pii import has_private_life, mask_email, mask_phone, redact
from .schemas import InboundLead, OutputRow, Unit, VerifiedSignal
from .scoring import IntentModel, intent_keyword_count, recency_days
from .web.search import SearchProvider, get_provider

log = logging.getLogger(__name__)


def load_inventory(eng) -> list[Unit]:
    with eng.connect() as c:
        return [Unit(**r[0]) for r in c.execute(select(db.units.c.payload))]


def load_leads(eng, since_hours: float | None = None) -> list[InboundLead]:
    with eng.connect() as c:
        leads = [InboundLead(**r[0]) for r in c.execute(select(db.leads.c.payload))]
    if since_hours is not None:
        leads = [lead for lead in leads if recency_days(lead.created_at) * 24 <= since_hours]
    return leads


def _budget_fit(units: list[Unit], area: str | None, budget: float | None, band: str | None) -> float:
    if not area:
        return float("nan")
    pool = [u for u in units if u.available and u.area == area]
    if not pool:
        return 0.0
    m = match_units(units, area, budget, band, k=50)
    return round(min(1.0, len(m) / 10), 3) if (budget or (band and band != "unknown")) else 0.3


def run(
    settings: Settings | None = None,
    brief: Brief | None = None,
    llm: LLM | None = None,
    provider: SearchProvider | None = None,
    model: IntentModel | None = None,
    lead_window_hours: float | None = None,
) -> dict:
    s = settings or get_settings()
    brief = brief or load_brief(s.brief_path)
    llm = llm or LLM()
    provider = provider or get_provider(s.search_provider)
    eng = db.engine(s.database_url)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    t0 = time.perf_counter()
    tracer = Tracer(run_id)
    errors: list[str] = []
    started = datetime.now(UTC)
    db.insert_many(eng, db.runs, [{"run_id": run_id, "started_at": started, "status": "running",
                                   "llm_provider": llm.provider, "llm_model": llm.model,
                                   "search_provider": provider.name}])
    log.info("run %s: llm=%s/%s search=%s", run_id, llm.provider, llm.model, provider.name)

    # 1) research graph ---------------------------------------------------------
    with tracer.span("research_graph"):
        graph = build_graph(llm, provider, s)
        state = graph.invoke({"brief": brief, "recent_queries": db.recent_queries(eng),
                              "seen_hashes": db.seen_page_hashes(eng)})
    pages = state.get("pages", [])
    verified: list[VerifiedSignal] = state.get("verified", [])
    rejected: list[VerifiedSignal] = state.get("rejected", [])
    slog = state.get("source_log", [])

    # 2) PII filter on signals: drop private-life sentences, redact identifiers --------
    clean: list[VerifiedSignal] = []
    for v in verified:
        if has_private_life(v.evidence_quote) or has_private_life(v.summary):
            v.verified, v.rejection_reason = False, "private-life content (PDPL policy)"
            rejected.append(v)
            continue
        v.evidence_quote, v.summary = redact(v.evidence_quote), redact(v.summary)
        clean.append(v)

    # 3) dedup vs history + within run ----------------------------------------------
    new_sigs, dup_sigs, dstats = dedup(clean, db.known_signals(eng), s.dedup_threshold)

    # 4) leads (first-party, consented only) -------------------------------------
    units = load_inventory(eng)
    all_leads = load_leads(eng, lead_window_hours)
    with eng.connect() as c:
        exported = {r[0] for r in c.execute(select(db.output_rows.c.row_id).where(db.output_rows.c.row_id.like("lead:%")))}
    leads = [ld for ld in all_leads if ld.consent_marketing]
    no_consent = len(all_leads) - len(leads)

    # 5) features + scoring ------------------------------------------------------
    area_counts = Counter([x.area for x in new_sigs if x.area] + [ld.area_interest for ld in leads if ld.area_interest])
    feats, items = [], []
    for v in new_sigs:
        feats.append(dict(signal_type=v.signal_type.value, recency_days=recency_days(v.event_date) if v.event_date else 0.5,
                          budget_fit=_budget_fit(units, v.area, None, v.budget_band),
                          area_demand=area_counts.get(v.area, 0), has_budget=float(v.budget_band != "unknown"),
                          confidence=v.confidence))
        items.append(("signal", v))
    for ld in leads:
        feats.append(dict(signal_type="inbound_lead", recency_days=recency_days(ld.created_at),
                          budget_fit=_budget_fit(units, ld.area_interest, ld.stated_budget_egp, None),
                          area_demand=area_counts.get(ld.area_interest, 0),
                          has_budget=float(ld.stated_budget_egp is not None), channel=ld.channel,
                          pages_viewed=ld.pages_viewed, reply_minutes=ld.reply_minutes,
                          previous_contacts=ld.previous_contacts, intent_keywords=intent_keyword_count(ld.message)))
        items.append(("lead", ld))
    with tracer.span("scoring", n=len(feats)):
        model = model or IntentModel.load(s.model_path)
        scores, reasons = model.score(feats)

    # 6) matcher + rows --------------------------------------------------------------
    router = BrokerRouter(brief)
    rows: list[OutputRow] = []
    now = datetime.now(UTC)
    for (kind, obj), score, why in zip(items, scores, reasons, strict=True):
        if kind == "signal":
            v: VerifiedSignal = obj
            mu = match_units(units, v.area, None, v.budget_band) if v.signal_type.value in {
                "office_opening", "relocation_firm", "new_launch", "price_change"} else []
            rows.append(OutputRow(
                row_id=f"sig:{fingerprint(v)}", row_kind="signal",
                lead_or_company=v.entity + (f" / {v.project}" if v.project else ""), source_url=v.source_url,
                signal_type=v.signal_type.value, area=v.area, budget_band=str(v.budget_band.value),
                intent_score=float(score), score_reasons=why, matched_units=fmt_units(mu),
                assigned_broker=router.assign(v.area), next_action=next_action("signal", v.signal_type.value, score, bool(mu)),
                summary=v.summary, evidence_quote=v.evidence_quote, consent="n/a (public business source)",
                collected_at=v.collected_at))
        else:
            ld: InboundLead = obj
            mu = match_units(units, ld.area_interest, ld.stated_budget_egp, None, ld.unit_type)
            rid = f"lead:{ld.lead_id}"
            rows.append(OutputRow(
                row_id=rid, row_kind="lead",
                lead_or_company=f"{ld.name} ({mask_phone(ld.phone)}, {mask_email(ld.email)})",
                source_url=f"crm://leads/{ld.lead_id} ({ld.channel})", signal_type="inbound_lead",
                area=ld.area_interest, budget_band=budget_band(ld.stated_budget_egp).value,
                intent_score=float(score), score_reasons=why, matched_units=fmt_units(mu),
                assigned_broker=router.assign(ld.area_interest), next_action=next_action("lead", "", score, bool(mu)),
                summary=redact(ld.message) or "", evidence_quote="", consent=f"yes - {ld.consent_source}",
                collected_at=now, is_new=rid not in exported))

    # 7) persist + outputs ---------------------------------------------------------
    for p in pages:
        h = content_hash(p.text)
        with eng.begin() as c:
            if not c.execute(select(db.page_hashes.c.hash).where(db.page_hashes.c.hash == h)).first():
                c.execute(db.page_hashes.insert().values(hash=h, url=p.url, first_seen=now))

    def _sig_row(v: VerifiedSignal) -> dict:
        d = {k: getattr(v, k) for k in ("entity", "project", "area", "summary", "price_change_pct", "payment_plan",
                                         "evidence_quote", "source_url", "confidence", "quote_match", "verified",
                                         "rejection_reason", "collected_at")}
        return {**d, "fingerprint": fingerprint(v), "run_id": run_id, "signal_type": v.signal_type.value,
                "budget_band": v.budget_band.value}

    db.insert_many(eng, db.signals, [_sig_row(v) for v in new_sigs + rejected])
    db.insert_many(eng, db.output_rows, [{"run_id": run_id, "row_id": r.row_id, "payload": r.model_dump(mode="json"),
                                          "intent_score": r.intent_score, "collected_at": r.collected_at} for r in rows])
    db.insert_many(eng, db.source_log, [{"run_id": run_id, "kind": x.get("kind"), "target": x.get("target"),
                                         "provider": x.get("provider"), "ok": x.get("ok"), "n": x.get("n"),
                                         "ms": x.get("ms"), "error": x.get("error")} for x in slog])
    crm = push_crm(s.crm_webhook_url, rows, s.http_timeout)
    purged = db.apply_retention(eng, s.retention_days)

    n_candidates = len(verified) + len([r for r in rejected if r.rejection_reason and "PDPL" not in r.rejection_reason])
    stats = {
        "queries": len(state["plan"].queries) if state.get("plan") else 0,
        "pages_fetched": len(pages),
        "fetch_success_rate": source_success_rate(slog),
        "browser_fetches": sum(1 for x in slog if x.get("provider") == "browser"),
        "candidates": n_candidates,
        "verified": len(verified),
        "verifier_rejection_rate": round(len(rejected) / n_candidates, 3) if n_candidates else 0.0,
        "new_signals": len(new_sigs),
        **dstats,
        "leads_scored": len(leads),
        "leads_skipped_no_consent": no_consent,
        "rows": len(rows),
        "rows_new": sum(r.is_new for r in rows),
        "hot_rows_ge_70": sum(r.intent_score >= 70 for r in rows),
        "llm_calls": llm.usage.calls,
        "llm_errors": llm.usage.errors,
        "input_tokens": llm.usage.input_tokens,
        "output_tokens": llm.usage.output_tokens,
        "cost_by_agent": llm.usage.by_agent,
        "crm": crm,
        "retention_purged": purged,
        "duration_s": round(time.perf_counter() - t0, 1),
    }
    if not new_sigs and not leads:
        errors.append("EMPTY_RUN: no new signals and no consented leads")
    if llm.auth_failed:
        errors.append("LLM_AUTH_FAILED: API key rejected; heuristic extraction used")
    failed = [x for x in slog if x.get("kind") in ("search", "fetch") and not x.get("ok")]
    if failed:
        errors.append(f"{len(failed)} failed sources")

    out_dir = Path(s.output_dir) / run_id
    run_log = {"run_id": run_id, "llm": f"{llm.provider}/{llm.model}", "search": provider.name,
               "cost_usd": llm.usage.cost_usd, **{k: v for k, v in stats.items() if k != "cost_by_agent"},
               "errors": errors}
    xlsx = write_excel(out_dir / f"leads_{run_id}.xlsx", rows, new_sigs, rejected + dup_sigs, run_log, slog)
    (out_dir / "market_digest.md").write_text(digest_markdown(new_sigs, run_id), encoding="utf-8")

    with eng.begin() as c:
        c.execute(db.runs.update().where(db.runs.c.run_id == run_id).values(
            finished_at=datetime.now(UTC), status="ok" if new_sigs or leads else "empty", stats=stats,
            cost_usd=llm.usage.cost_usd, excel_path=str(xlsx), errors=errors))
    tracer.flush()
    log.info("run %s done: %s rows (%s new signals, %s leads), cost $%.4f, %ss -> %s", run_id, len(rows),
             len(new_sigs), len(leads), llm.usage.cost_usd, stats["duration_s"], xlsx)
    return {"run_id": run_id, "excel": str(xlsx), "digest": str(out_dir / "market_digest.md"), "stats": stats,
            "cost_usd": llm.usage.cost_usd, "errors": errors, "rows": rows, "signals": new_sigs}


def seed(eng=None, n_units: int = 240, n_leads: int = 40) -> dict:
    """Load synthetic inventory + consented inbound leads (idempotent upsert)."""
    from .synthetic import make_leads, make_units

    eng = eng or db.engine()
    us, ls = make_units(n_units), make_leads(n_leads)
    db.upsert_by_key(eng, db.units, "unit_id", [{"unit_id": u.unit_id, "payload": u.model_dump(mode="json"),
                                                  "area": u.area, "price_egp": u.price_egp, "available": u.available}
                                                 for u in us])
    db.upsert_by_key(eng, db.leads, "lead_id", [{"lead_id": ld.lead_id, "payload": ld.model_dump(mode="json"),
                                                  "consent_marketing": ld.consent_marketing,
                                                  "consent_source": ld.consent_source, "created_at": ld.created_at}
                                                 for ld in ls])
    return {"units": len(us), "leads": len(ls)}

