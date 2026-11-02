"""Outputs: Excel workbook (CRM rows, market digest, run log), CRM webhook push, markdown digest."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pandas as pd

from .schemas import OutputRow, VerifiedSignal

log = logging.getLogger(__name__)

CRM_COLUMNS = ["lead_or_company", "row_kind", "signal_type", "area", "budget_band", "intent_score", "score_reasons",
               "matched_units", "assigned_broker", "next_action", "summary", "evidence_quote", "source_url",
               "consent", "collected_at", "is_new", "row_id"]
DIGEST_TITLES = {
    "new_launch": "New launches", "price_change": "Price & payment-plan changes",
    "competitor_move": "Competitor moves", "office_opening": "Companies opening offices (corporate housing)",
    "relocation_firm": "Relocation firms",
}


def rows_frame(rows: list[OutputRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows], columns=CRM_COLUMNS)
    if not df.empty:
        df["collected_at"] = pd.to_datetime(df["collected_at"]).dt.tz_localize(None)
        df = df.sort_values(["intent_score"], ascending=False)
    return df


def digest_markdown(signals: list[VerifiedSignal], run_id: str) -> str:
    lines = [f"# Market digest - run {run_id}", ""]
    if not signals:
        return "\n".join(lines + ["No new verified market signals this run."])
    for st, title in DIGEST_TITLES.items():
        items = [s for s in signals if s.signal_type == st]
        if not items:
            continue
        lines += [f"## {title} ({len(items)})", ""]
        for s in items:
            extra = ", ".join(x for x in [s.area, s.payment_plan,
                                           f"{s.price_change_pct:+.0f}%" if s.price_change_pct else None] if x)
            lines.append(f"- **{s.entity}**{f' ({extra})' if extra else ''}: {s.summary} [source]({s.source_url})")
        lines.append("")
    return "\n".join(lines)


def write_excel(path: Path, rows: list[OutputRow], signals: list[VerifiedSignal], rejected: list[VerifiedSignal],
                run_log: dict, source_log: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    crm = rows_frame(rows)
    digest = pd.DataFrame([{
        "section": DIGEST_TITLES.get(s.signal_type, s.signal_type), "entity": s.entity, "project": s.project,
        "area": s.area, "summary": s.summary, "price_change_pct": s.price_change_pct, "payment_plan": s.payment_plan,
        "budget_band": s.budget_band, "evidence_quote": s.evidence_quote, "source_url": s.source_url,
    } for s in signals])
    rej = pd.DataFrame([{"entity": s.entity, "signal_type": s.signal_type, "reason": s.rejection_reason,
                         "quote_match": s.quote_match, "confidence": s.confidence, "source_url": s.source_url}
                        for s in rejected])
    srcs = pd.DataFrame(source_log)
    summary = pd.DataFrame([{"metric": k, "value": str(v)} for k, v in run_log.items()])

    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        wb = xw.book
        hdr = wb.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white", "border": 1,
                             "text_wrap": True, "valign": "top"})
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        for name, df, widths in [
            ("CRM Rows", crm, {"lead_or_company": 26, "summary": 50, "evidence_quote": 50, "score_reasons": 40,
                               "matched_units": 55, "next_action": 45, "source_url": 40}),
            ("Market Digest", digest, {"summary": 60, "evidence_quote": 60, "source_url": 40, "entity": 26}),
            ("Rejected", rej, {"reason": 50, "source_url": 50, "entity": 26}),
            ("Sources", srcs, {"target": 70, "error": 50}),
            ("Run Log", summary, {"metric": 28, "value": 80}),
        ]:
            df = df if not df.empty else pd.DataFrame({"info": ["(none)"]})
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for j, col in enumerate(df.columns):
                ws.write(0, j, col, hdr)
                ws.set_column(j, j, widths.get(col, 16), wrap)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)
            if name == "CRM Rows" and "intent_score" in df:
                j = list(df.columns).index("intent_score")
                ws.conditional_format(1, j, len(df), j, {"type": "3_color_scale", "min_color": "#F8696B",
                                                          "mid_color": "#FFEB84", "max_color": "#63BE7B"})
    return path


def push_crm(webhook_url: str | None, rows: list[OutputRow], timeout: float = 20.0) -> dict:
    """POST rows to a CRM webhook (HubSpot/Salesforce/Zoho via Zapier/Make or an internal endpoint)."""
    if not webhook_url:
        return {"pushed": 0, "skipped": "no MLIA_CRM_WEBHOOK_URL"}
    payload = [r.model_dump(mode="json") for r in rows if r.is_new]
    try:
        r = httpx.post(webhook_url, json={"rows": payload}, timeout=timeout)
        r.raise_for_status()
        return {"pushed": len(payload), "status": r.status_code}
    except Exception as e:  # noqa: BLE001
        log.error("CRM push failed: %s", e)
        return {"pushed": 0, "error": str(e)[:300]}
