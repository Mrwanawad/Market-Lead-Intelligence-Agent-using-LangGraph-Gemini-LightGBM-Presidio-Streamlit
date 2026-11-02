"""PII filter (Presidio + regex fallback).

Policy (Egypt PDPL, Law 151/2020):
- Market signals: business info only. Personal identifiers in quotes/summaries are redacted; sentences
  about private life (family, health, religion, salary, home address, live location) are dropped.
- Inbound leads: consented first-party data. Contact fields stay in the internal CRM push, but the
  exported sheet masks phone/email and keeps only stated business info (budget, area, unit type).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

PRIVATE_LIFE = re.compile(
    r"\b(wife|husband|children|kids|son|daughter|divorc|pregnan|illness|disease|diagnos|religio|"
    r"salary of|earns|net worth|home address|lives (?:at|in)|live location|whereabouts)\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?:\+?20|0)1[0125][\s-]?\d{3,4}[\s-]?\d{4}|\+?\d[\d\s-]{8,}\d")
_NATID = re.compile(r"\b[23]\d{13}\b")  # Egyptian national ID: 14 digits


@lru_cache
def _analyzer():
    try:
        import spacy.util
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        for model in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
            if not spacy.util.is_package(model):  # Presidio would otherwise auto-download it
                continue
            try:
                provider = NlpEngineProvider(nlp_configuration={
                    "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}]})
                return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
            except Exception:  # noqa: BLE001, S112
                continue
    except ImportError:
        pass
    log.warning("Presidio/spaCy model unavailable; using regex PII filter only")
    return None


def redact(text: str | None) -> str | None:
    """Mask emails, phones, national IDs (+ Presidio detections). Company/person-as-spokesperson names kept."""
    if not text:
        return text
    out = _NATID.sub("<NATIONAL_ID>", _EMAIL.sub("<EMAIL>", text))
    out = _PHONE.sub("<PHONE>", out)
    an = _analyzer()
    if an is not None:
        try:
            res = an.analyze(text=out, language="en", entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                                                                 "IBAN_CODE", "IP_ADDRESS"])
            for r in sorted(res, key=lambda r: -r.start):
                if r.score >= 0.5:
                    out = out[:r.start] + f"<{r.entity_type}>" + out[r.end:]
        except Exception as e:  # noqa: BLE001
            log.debug("presidio failed: %s", e)
    return out


def has_private_life(text: str | None) -> bool:
    return bool(text and PRIVATE_LIFE.search(text))


def mask_phone(p: str | None) -> str:
    if not p:
        return ""
    d = re.sub(r"\D", "", p)
    return f"***{d[-3:]}" if len(d) >= 3 else "***"


def mask_email(e: str | None) -> str:
    if not e or "@" not in e:
        return ""
    user, dom = e.split("@", 1)
    return f"{user[:1]}***@{dom}"
