"""Runtime settings (env / .env) and the research brief (YAML)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="MLIA_", extra="ignore")

    # LLM
    llm_provider: Literal["gemini", "openai", "anthropic", "fake"] = "gemini"
    llm_model: str | None = None  # provider default when empty
    llm_temperature: float = 0.0

    # Search: auto picks tavily > exa > ddg, falling back to fixtures when offline
    search_provider: Literal["auto", "tavily", "exa", "ddg", "fixtures"] = "auto"
    max_queries: int = 12
    results_per_query: int = 4
    max_pages: int = 30
    use_browser: bool = True  # Playwright only when plain HTTP returns too little text
    http_timeout: float = 20.0

    # Storage
    database_url: str = f"sqlite:///{(ROOT / 'data' / 'mlia.db').as_posix()}"
    output_dir: Path = ROOT / "outputs"
    model_path: Path = ROOT / "models" / "intent_lgbm.txt"

    # Pipeline
    verifier_min_confidence: float = 0.55
    dedup_threshold: int = 90
    retention_days: int = 180
    brief_path: Path = ROOT / "config" / "brief.yaml"

    # CRM push (optional): POST rows as JSON to this webhook
    crm_webhook_url: str | None = None

    # Scheduler
    schedule_hours: int = 6


class SignalTypeSpec(BaseModel):
    name: str
    description: str
    query_templates: list[str]


class Brief(BaseModel):
    company: str = "Brokerage"
    target_areas: list[str]
    segments: list[str]
    competitors: list[str] = Field(default_factory=list)
    signal_types: list[SignalTypeSpec]
    language_hints: list[str] = Field(default_factory=lambda: ["en"])
    brokers: list[dict] = Field(default_factory=list)  # {name, areas: [...]}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_brief(path: Path | None = None) -> Brief:
    path = path or get_settings().brief_path
    with open(path, encoding="utf-8") as f:
        return Brief.model_validate(yaml.safe_load(f))
