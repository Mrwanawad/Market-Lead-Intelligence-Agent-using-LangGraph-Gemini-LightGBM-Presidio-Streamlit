"""Enrichment & dedup: rapidfuzz entity matching vs. this run and previous runs."""

from __future__ import annotations

import hashlib
import re

from rapidfuzz import fuzz

from .schemas import VerifiedSignal

_STOP = re.compile(r"\b(developments?|development|group|properties|holding|company|co|egypt|real estate|"
                   r"inc|corp|llc|s\.a\.e|sae|the)\b", re.I)


def canon_entity(name: str) -> str:
    s = _STOP.sub(" ", name.lower())
    return " ".join(re.sub(r"[^a-z0-9؀-ۿ ]+", " ", s).split())


def fingerprint(sig: VerifiedSignal) -> str:
    """Stable key: type + canonical entity + area + project + the numbers that define a *change*."""
    parts = [sig.signal_type, canon_entity(sig.entity), (sig.area or "").lower(), canon_entity(sig.project or ""),
             f"{sig.price_change_pct or ''}", (sig.payment_plan or "").lower().replace(" ", "")]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def same_entity(a: str, b: str, threshold: int) -> bool:
    ca, cb = canon_entity(a), canon_entity(b)
    if not ca or not cb:
        return False
    return fuzz.token_set_ratio(ca, cb) >= threshold


def dedup(
    sigs: list[VerifiedSignal], history: list[dict], threshold: int = 90
) -> tuple[list[VerifiedSignal], list[VerifiedSignal], dict]:
    """Returns (new, duplicates, stats). A signal is a duplicate if its fingerprint was seen, or if the
    same type+area already has a fuzzy-matching entity (in history or earlier in this batch) with no new
    numeric change."""
    seen_fp = {h["fingerprint"] for h in history}
    hist_by_key: dict[tuple, list[str]] = {}
    for h in history:
        hist_by_key.setdefault((h["signal_type"], h.get("area")), []).append(h["entity"])

    new: list[VerifiedSignal] = []
    dups: list[VerifiedSignal] = []
    batch: list[VerifiedSignal] = []
    for s in sorted(sigs, key=lambda x: -x.confidence):
        fp = fingerprint(s)
        key = (s.signal_type.value if hasattr(s.signal_type, "value") else s.signal_type, s.area)
        is_dup = fp in seen_fp
        if not is_dup and s.price_change_pct is None and not s.payment_plan:
            is_dup = any(same_entity(s.entity, e, threshold) for e in hist_by_key.get(key, []))
        if not is_dup:
            for b in batch:
                if b.signal_type == s.signal_type and b.area == s.area and same_entity(b.entity, s.entity, threshold):
                    is_dup = True
                    break
        (dups if is_dup else new).append(s)
        if not is_dup:
            batch.append(s)
            seen_fp.add(fp)
    total = len(sigs) or 1
    return new, dups, {"duplicates": len(dups), "duplicate_rate": round(len(dups) / total, 3)}
