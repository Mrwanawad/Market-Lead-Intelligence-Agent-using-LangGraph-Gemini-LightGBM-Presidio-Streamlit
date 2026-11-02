"""Provider-agnostic structured-output LLM client with token/cost accounting.

Every agent calls ``LLM.structured(system, prompt, Schema)`` and gets back a validated
Pydantic instance. Providers: Gemini (default), OpenAI, Anthropic, and ``fake`` (no network;
agents fall back to their rule-based paths).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import get_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5-5",
}

# USD per 1M tokens (input, output). Models not listed report cost as unknown;
# set MLIA_PRICE_IN / MLIA_PRICE_OUT to price them.
PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMUnavailable(RuntimeError):
    """Raised when the provider is 'fake' or not configured; callers use heuristics."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0
    by_agent: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, agent: str, tin: int, tout: int, cost: float) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += tin
            self.output_tokens += tout
            a = self.by_agent.setdefault(agent, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            a["calls"] += 1
            a["input_tokens"] += tin
            a["output_tokens"] += tout
            a["cost_usd"] += cost

    @property
    def cost_usd(self) -> float:
        return round(sum(a["cost_usd"] for a in self.by_agent.values()), 6)


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None):
        s = get_settings()
        self.provider = provider or s.llm_provider
        self.model = model or s.llm_model or DEFAULT_MODELS.get(self.provider, "")
        self.temperature = s.llm_temperature
        self.usage = Usage()
        self._client = None
        self.auth_failed = False
        self.not_configured = False
        if self.provider != "fake" and not self._has_key():
            log.warning("No API key for provider %s; running in heuristic (fake) mode", self.provider)
            self.not_configured = True
            self.provider = "fake"

    # ---------------------------------------------------------------- helpers
    def _has_key(self) -> bool:
        keys = {
            "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            "openai": ("OPENAI_API_KEY",),
            "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        }[self.provider]
        return any(os.getenv(k) for k in keys)

    @property
    def available(self) -> bool:
        return self.provider != "fake"

    @property
    def price_known(self) -> bool:
        return bool(os.getenv("MLIA_PRICE_IN") and os.getenv("MLIA_PRICE_OUT")) or self.model in PRICES

    def _price(self) -> tuple[float, float]:
        if os.getenv("MLIA_PRICE_IN") and os.getenv("MLIA_PRICE_OUT"):
            return float(os.environ["MLIA_PRICE_IN"]), float(os.environ["MLIA_PRICE_OUT"])
        return PRICES.get(self.model, (0.0, 0.0))

    def _record(self, agent: str, tin: int, tout: int) -> None:
        pin, pout = self._price()
        self.usage.add(agent, tin, tout, (tin * pin + tout * pout) / 1_000_000)

    # ---------------------------------------------------------------- public
    def structured(self, system: str, prompt: str, schema: type[T], agent: str = "llm") -> T:
        if not self.available:
            raise LLMUnavailable("LLM provider is 'fake'")
        try:
            return self._call(system, prompt, schema, agent)
        except Exception as e:
            self.usage.errors += 1
            if _is_auth_error(e):
                # Circuit breaker: a bad key fails every call, so stop calling and let agents use heuristics.
                log.error("LLM auth failed for %s (%s); switching to heuristic mode for this run",
                          self.provider, str(e)[:160])
                self.auth_failed = True
                self.provider = "fake"
                raise LLMUnavailable(str(e)) from e
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20), reraise=True,
           retry=retry_if_exception(lambda e: not _is_auth_error(e)))
    def _call(self, system: str, prompt: str, schema: type[T], agent: str) -> T:
        fn = {"gemini": self._gemini, "openai": self._openai, "anthropic": self._anthropic}[self.provider]
        return fn(system, prompt, schema, agent)

    # ---------------------------------------------------------------- providers
    def _gemini(self, system: str, prompt: str, schema: type[T], agent: str) -> T:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
            ),
        )
        um = resp.usage_metadata
        tin = (um.prompt_token_count or 0) if um else 0
        tout = ((um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0
        self._record(agent, tin, tout)
        return _parse(resp.text, schema)

    def _openai(self, system: str, prompt: str, schema: type[T], agent: str) -> T:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system + "\n\nReturn ONLY JSON matching this JSON Schema:\n"
                 + json.dumps(schema.model_json_schema())},
                {"role": "user", "content": prompt},
            ],
        )
        self._record(agent, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return _parse(resp.choices[0].message.content, schema)

    def _anthropic(self, system: str, prompt: str, schema: type[T], agent: str) -> T:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic()
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        self._record(agent, resp.usage.input_tokens, resp.usage.output_tokens)
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise ValueError(f"Claude returned no parsed output (stop_reason={resp.stop_reason})")
        return resp.parsed_output


def _is_auth_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("api key not valid", "api_key_invalid", "incorrect api key", "invalid x-api-key",
                                  "authentication", "401", "permission_denied"))


def _parse(text: str | None, schema: type[T]) -> T:
    if not text:
        raise ValueError("empty LLM response")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        start, end = text.find("{"), text.rfind("}")
        return schema.model_validate(json.loads(text[start:end + 1]))
