"""Intent scoring: LightGBM model over signal + lead features, SHAP reasons per row.

Features (shared by market signals and inbound leads; NaN where not applicable):
  signal_type, recency_days, budget_fit, area_demand, has_budget, confidence,
  channel, pages_viewed, reply_minutes, previous_contacts, intent_keywords
Label: broker-confirmed meeting (synthetic bootstrap + broker feedback from the DB when available).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import get_settings
from .schemas import SignalType

log = logging.getLogger(__name__)

FEATURES = ["signal_type", "recency_days", "budget_fit", "area_demand", "has_budget", "confidence",
            "channel", "pages_viewed", "reply_minutes", "previous_contacts", "intent_keywords"]
SIGNAL_CODES = {s.value: i for i, s in enumerate(SignalType)}
CHANNEL_CODES = {"website_form": 0, "whatsapp": 1, "ad_form": 2, "referral": 3}
INTENT_WORDS = re.compile(r"\b(ready|cash|visit|this month|asap|urgent|down ?payment|book|reserve|meeting|"
                          r"viewing|when can|available now|move in|relocat|family of|budget)\b", re.I)

REASON_TEXT = {
    "signal_type": "signal type",
    "recency_days": "recency",
    "budget_fit": "budget fits available inventory",
    "area_demand": "area demand",
    "has_budget": "stated budget",
    "confidence": "evidence confidence",
    "channel": "inbound channel",
    "pages_viewed": "engagement (pages viewed)",
    "reply_minutes": "reply speed",
    "previous_contacts": "previous contacts",
    "intent_keywords": "intent words in message",
}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=20,
              feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, verbose=-1, seed=7)


def intent_keyword_count(text: str | None) -> int:
    return len(INTENT_WORDS.findall(text or ""))


def recency_days(ts: datetime | str | None, now: datetime | None = None) -> float:
    if ts is None:
        return np.nan
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return np.nan
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return max(0.0, ((now or datetime.now(UTC)) - ts).total_seconds() / 86400)


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for f in FEATURES:
        if f not in df:
            df[f] = np.nan
    df["signal_type"] = df["signal_type"].map(lambda v: SIGNAL_CODES.get(v, np.nan) if isinstance(v, str) else v)
    df["channel"] = df["channel"].map(lambda v: CHANNEL_CODES.get(v, np.nan) if isinstance(v, str) else v)
    return df[FEATURES].astype(float)


class IntentModel:
    def __init__(self, booster: lgb.Booster):
        self.booster = booster
        self._explainer = None

    # --------------------------------------------------------------- io
    @classmethod
    def load(cls, path: Path | None = None) -> IntentModel:
        path = path or get_settings().model_path
        if not path.exists():
            from .synthetic import training_frame

            log.info("no intent model at %s; training on synthetic bootstrap data", path)
            m = train(training_frame())
            m.save(path)
            return m
        return cls(lgb.Booster(model_file=str(path)))

    def save(self, path: Path | None = None) -> Path:
        path = path or get_settings().model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        return path

    # --------------------------------------------------------------- predict
    def score(self, rows: list[dict]) -> tuple[np.ndarray, list[str]]:
        if not rows:
            return np.array([]), []
        X = to_frame(rows)
        proba = self.booster.predict(X)
        return np.round(proba * 100, 1), self._reasons(X)

    def _reasons(self, X: pd.DataFrame, top: int = 3) -> list[str]:
        try:
            import shap

            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.booster)
            sv = self._explainer.shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
        except Exception as e:  # noqa: BLE001 - fall back to LightGBM's built-in contributions
            log.debug("shap failed (%s); using pred_contrib", e)
            sv = self.booster.predict(X, pred_contrib=True)[:, :-1]
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(sv[i]))
            parts = []
            for j in order:
                if len(parts) == top:
                    break
                if np.isnan(X.iloc[i, j]) or abs(sv[i][j]) < 0.02:
                    continue
                parts.append(f"{'+' if sv[i][j] > 0 else '-'} {REASON_TEXT[FEATURES[j]]}")
            out.append("; ".join(parts) or "baseline")
        return out


def train(df: pd.DataFrame, rounds: int = 300) -> IntentModel:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = to_frame(df.to_dict("records")), df["label"].astype(int).values
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    booster = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr), num_boost_round=rounds,
                        valid_sets=[lgb.Dataset(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    auc = roc_auc_score(yva, booster.predict(Xva))
    log.info("intent model trained: n=%d auc=%.3f iters=%d", len(df), auc, booster.best_iteration)
    m = IntentModel(booster)
    m.metrics = {"auc": round(float(auc), 4), "n": len(df), "positives": int(y.sum())}
    return m
