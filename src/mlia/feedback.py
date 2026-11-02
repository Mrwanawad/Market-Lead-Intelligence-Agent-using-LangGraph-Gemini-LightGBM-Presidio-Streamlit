"""Broker feedback loop: label output rows, compute precision, export training rows."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select, update

from . import db

LABELS = {"qualified", "not_qualified", "meeting"}


def record(row_id: str, label: str) -> dict:
    if label not in LABELS:
        raise ValueError(f"label must be one of {sorted(LABELS)}")
    eng = db.engine()
    with eng.begin() as c:
        n = c.execute(update(db.output_rows).where(db.output_rows.c.row_id == row_id)
                      .values(broker_feedback=label)).rowcount
    return {"row_id": row_id, "label": label, "updated": n}


def precision(threshold: float = 70.0) -> dict:
    """Precision of qualified leads among rows scored >= threshold that brokers reviewed."""
    eng = db.engine()
    with eng.connect() as c:
        rs = c.execute(select(db.output_rows.c.intent_score, db.output_rows.c.broker_feedback)
                       .where(db.output_rows.c.broker_feedback.is_not(None))).all()
    hot = [fb for sc, fb in rs if sc >= threshold]
    good = sum(fb in ("qualified", "meeting") for fb in hot)
    meetings = sum(fb == "meeting" for _, fb in rs)
    return {"reviewed": len(rs), "reviewed_hot": len(hot),
            "precision_at_threshold": round(good / len(hot), 3) if hot else None,
            "lead_to_meeting": round(meetings / len(rs), 3) if rs else None}


def feedback_frame() -> pd.DataFrame:
    """Rows with broker feedback -> training rows (features are recomputed from the stored payload)."""
    from .scoring import intent_keyword_count

    eng = db.engine()
    with eng.connect() as c:
        rs = c.execute(select(db.output_rows.c.payload, db.output_rows.c.broker_feedback)
                       .where(db.output_rows.c.broker_feedback.is_not(None))).all()
    rows = []
    for payload, fb in rs:
        rows.append(dict(signal_type=payload["signal_type"], recency_days=0.5,
                         has_budget=float(payload["budget_band"] != "unknown"),
                         intent_keywords=intent_keyword_count(payload.get("summary")),
                         budget_fit=0.5 if payload.get("matched_units") else 0.0,
                         label=int(fb in ("qualified", "meeting"))))
    return pd.DataFrame(rows)
