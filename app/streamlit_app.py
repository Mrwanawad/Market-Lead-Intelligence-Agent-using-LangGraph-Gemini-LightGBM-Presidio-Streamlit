"""Streamlit viewer: browse runs, filter rows, download Excel, give broker feedback."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlia import db  # noqa: E402
from mlia.feedback import LABELS, precision, record  # noqa: E402

st.set_page_config(page_title="Market & Lead Intelligence", layout="wide")
eng = db.engine()


@st.cache_data(ttl=60)
def load_runs() -> pd.DataFrame:
    with eng.connect() as c:
        rs = c.execute(select(db.runs).order_by(db.runs.c.started_at.desc())).mappings().all()
    return pd.DataFrame([dict(r) for r in rs])


@st.cache_data(ttl=60)
def load_rows(run_id: str) -> pd.DataFrame:
    with eng.connect() as c:
        rs = c.execute(select(db.output_rows.c.payload, db.output_rows.c.broker_feedback)
                       .where(db.output_rows.c.run_id == run_id)).all()
    return pd.DataFrame([{**p, "broker_feedback": fb} for p, fb in rs])


runs = load_runs()
st.title("Market & Lead Intelligence")
if runs.empty:
    st.info("No runs yet. Run `mlia seed` then `mlia run`.")
    st.stop()

# ---- health strip
ok = runs[runs.status.isin(["ok", "empty"])]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Runs", len(runs))
c2.metric("Last run", str(runs.iloc[0].started_at)[:16])
c3.metric("Avg cost / run", f"${ok.cost_usd.fillna(0).mean():.4f}" if len(ok) else "-")
c4.metric("Empty runs", int((runs.status == "empty").sum()))
p = precision()
c5.metric("Precision@70 (broker)", p["precision_at_threshold"] if p["precision_at_threshold"] is not None else "-")

tab_rows, tab_runs = st.tabs(["Rows", "Run history & monitoring"])

with tab_rows:
    run_id = st.selectbox("Run", runs.run_id.tolist())
    rows = load_rows(run_id)
    if rows.empty:
        st.warning("No rows in this run.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        kinds = f1.multiselect("Kind", sorted(rows.row_kind.unique()), default=sorted(rows.row_kind.unique()))
        types = f2.multiselect("Signal type", sorted(rows.signal_type.unique()), default=sorted(rows.signal_type.unique()))
        areas = f3.multiselect("Area", sorted(rows.area.dropna().unique()))
        min_score = f4.slider("Min intent score", 0, 100, 0)
        view = rows[rows.row_kind.isin(kinds) & rows.signal_type.isin(types) & (rows.intent_score >= min_score)]
        if areas:
            view = view[view.area.isin(areas)]
        st.dataframe(
            view[["lead_or_company", "row_kind", "signal_type", "area", "budget_band", "intent_score", "score_reasons",
                  "assigned_broker", "next_action", "matched_units", "summary", "source_url", "broker_feedback", "row_id"]],
            width="stretch", hide_index=True,
            column_config={"intent_score": st.column_config.ProgressColumn("intent", min_value=0, max_value=100),
                           "source_url": st.column_config.LinkColumn("source")},
        )
        xlsx = runs.loc[runs.run_id == run_id, "excel_path"].iloc[0]
        if xlsx and Path(xlsx).exists():
            st.download_button("Download Excel", Path(xlsx).read_bytes(), file_name=Path(xlsx).name)
        digest = Path(xlsx).parent / "market_digest.md" if xlsx else None
        if digest and digest.exists():
            with st.expander("Market digest"):
                st.markdown(digest.read_text(encoding="utf-8"))

        st.subheader("Broker feedback")
        fc1, fc2, fc3 = st.columns([3, 2, 1])
        rid = fc1.selectbox("Row", view.row_id.tolist(),
                            format_func=lambda r: f"{rows.set_index('row_id').loc[r, 'lead_or_company']} ({r[:14]})")
        lab = fc2.selectbox("Label", sorted(LABELS))
        if fc3.button("Save"):
            record(rid, lab)
            st.cache_data.clear()
            st.success("saved")

with tab_runs:
    stats = pd.json_normalize(runs.stats.dropna().tolist()) if runs.stats.notna().any() else pd.DataFrame()
    hist = pd.concat([runs[["run_id", "started_at", "status", "llm_model", "search_provider", "cost_usd"]].reset_index(drop=True),
                      stats.reset_index(drop=True)], axis=1)
    st.dataframe(hist, width="stretch", hide_index=True)
    if len(hist) > 1:
        chart = hist.sort_values("started_at").set_index("started_at")
        cols = [c for c in ["cost_usd", "fetch_success_rate", "verifier_rejection_rate", "duplicate_rate"] if c in chart]
        st.line_chart(chart[cols])
    with eng.connect() as c:
        dist = pd.DataFrame(c.execute(select(db.output_rows.c.run_id, db.output_rows.c.intent_score)).all(),
                            columns=["run_id", "intent_score"])
    if not dist.empty:
        st.caption("Score distribution per run (drift monitoring)")
        st.dataframe(dist.groupby("run_id").intent_score.describe()[["count", "mean", "50%", "std"]],
                     width="stretch")
