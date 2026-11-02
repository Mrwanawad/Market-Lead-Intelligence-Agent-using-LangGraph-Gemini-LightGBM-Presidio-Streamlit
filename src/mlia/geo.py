"""Area normalization and budget-band helpers (Egypt)."""

from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from .schemas import BudgetBand

AREA_ALIASES: dict[str, list[str]] = {
    "New Cairo": ["new cairo", "fifth settlement", "5th settlement", "tagamoa", "el tagamoa", "القاهرة الجديدة", "التجمع"],
    "New Capital": ["new capital", "new administrative capital", "nac", "administrative capital", "العاصمة الإدارية"],
    "Mostakbal City": ["mostakbal city", "mostakbal", "future city", "المستقبل"],
    "Sheikh Zayed": ["sheikh zayed", "zayed", "الشيخ زايد"],
    "6th of October": ["6th of october", "6 october", "october city", "sixth of october", "أكتوبر"],
    "North Coast": ["north coast", "sahel", "ras el hekma", "ras el-hekma", "new alamein", "alamein", "الساحل الشمالي"],
    "Ain Sokhna": ["ain sokhna", "sokhna", "galala", "العين السخنة"],
    "Maadi": ["maadi", "المعادي"],
    "Smart Village": ["smart village", "القرية الذكية"],
    "Downtown Cairo": ["downtown cairo", "garden city", "zamalek"],
}
_ALIAS_TO_AREA = {alias: area for area, aliases in AREA_ALIASES.items() for alias in aliases}


def normalize_area(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    if t in _ALIAS_TO_AREA:
        return _ALIAS_TO_AREA[t]
    for alias, area in sorted(_ALIAS_TO_AREA.items(), key=lambda x: -len(x[0])):
        if len(alias) > 3 and alias in t:
            return area
    m = process.extractOne(t, list(_ALIAS_TO_AREA), scorer=fuzz.ratio, score_cutoff=85)
    return _ALIAS_TO_AREA[m[0]] if m else text.strip().title()


def find_area(text: str) -> str | None:
    t = text.lower()
    for alias, area in sorted(_ALIAS_TO_AREA.items(), key=lambda x: -len(x[0])):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", t):
            return area
    return None


def budget_band(egp: float | None) -> BudgetBand:
    if not egp or egp <= 0:
        return BudgetBand.unknown
    m = egp / 1_000_000
    if m < 3:
        return BudgetBand.under_3m
    if m < 6:
        return BudgetBand.b3_6m
    if m < 10:
        return BudgetBand.b6_10m
    if m < 20:
        return BudgetBand.b10_20m
    return BudgetBand.over_20m


BAND_RANGES = {
    BudgetBand.under_3m: (0, 3e6), BudgetBand.b3_6m: (3e6, 6e6), BudgetBand.b6_10m: (6e6, 10e6),
    BudgetBand.b10_20m: (10e6, 20e6), BudgetBand.over_20m: (20e6, 1e12),
}

_MONEY = re.compile(
    r"(?:EGP|LE|E£)\s*([\d.,]+)\s*(million|mn|m|bn|billion|k|thousand)?|([\d.,]+)\s*(million|mn|bn|billion|k)?\s*(?:EGP|LE|pounds)",
    re.I,
)


def parse_money(text: str) -> list[float]:
    out = []
    for m in _MONEY.finditer(text):
        num, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        unit = (unit or "").lower()
        v *= {"million": 1e6, "mn": 1e6, "m": 1e6, "bn": 1e9, "billion": 1e9, "k": 1e3, "thousand": 1e3}.get(unit, 1)
        out.append(v)
    return out
