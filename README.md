# Market & Lead Intelligence Agent

A scheduled multi-agent pipeline for a real-estate brokerage. Every 6 hours it researches public market signals (new launches, price and payment-plan changes, competitor moves, companies opening offices that will need corporate housing, relocation firms). It also scores the company's own consented inbound leads and writes a broker-ready Excel/CRM sheet with matched units and next actions.

The full design is in [architecture.md](architecture.md).

```
Scheduler (GitHub Actions cron / APScheduler, every 6h)
  -> LangGraph: Planner -> Researcher -> Extractor -> Verifier
  -> Dedup (rapidfuzz vs. this run + history) -> PII filter (Presidio + PDPL rules)
  -> LightGBM intent score + SHAP reasons -> Unit matcher + broker routing
  -> Excel (xlsxwriter) + CRM webhook + Postgres/SQLite history + Streamlit viewer
  -> Run log: sources, cost per agent, errors, alerts (Langfuse optional)
```

## What each stage does

| Stage | Module | Notes |
|---|---|---|
| Planner | `agents/planner.py` | LLM turns `config/brief.yaml` into queries per signal type and avoids repeating recent queries. Falls back to round-robin templates. |
| Researcher | `agents/researcher.py`, `web/` | Tavily > Exa > DuckDuckGo (free) > fixtures. httpx + trafilatura first; Playwright Chromium only when a page returns too little text. Social and personal-profile domains are blocked. Pages whose content hash is unchanged are skipped. |
| Extractor | `agents/extractor.py` | Pydantic structured output (`ExtractedSignal`), and every signal carries a verbatim `evidence_quote`. A rule-based extractor runs when the LLM is unavailable. |
| Verifier | `agents/verifier.py` | Drops a signal when its quote is not on the page (fuzzy match < 90), its confidence is low, or it has no entity. |
| Dedup | `dedup.py` | Stable fingerprints + fuzzy entity matching. A new price change or payment plan for a known entity still counts as new. |
| PII | `pii.py` | Drops private-life sentences (family, health, salary, home address). Redacts phones, emails and national IDs. Lead contacts are masked in the exported sheet. |
| Scoring | `scoring.py` | LightGBM on shared features (signal type, recency, budget fit vs. inventory, area demand, channel, engagement, intent words). SHAP gives the top-3 reasons for each row. |
| Matcher | `matcher.py` | Matches inventory by area and budget, assigns the area-owner broker (least loaded on ties), and applies rule-based next actions. |
| Outputs | `outputs.py`, `pipeline.py` | `outputs/<run_id>/leads_<run_id>.xlsx` has these sheets: CRM Rows, Market Digest, Rejected, Sources, Run Log. Each run also writes `market_digest.md` and pushes to the CRM webhook. |
| Monitoring | `observability.py`, Streamlit | Cost per agent, source success rate, verifier rejection rate, duplicate rate, empty-run and LLM-auth alerts, and score distribution per run (drift). |

Output columns: lead/company, source URL, signal type, area, budget band, intent score + reasons, matched units, assigned broker, next action, summary, evidence quote, consent, collected_at.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows (source .venv/bin/activate on Linux/macOS)
pip install -e ".[browser,ui,dev]"
python -m spacy download en_core_web_sm               # Presidio NER (falls back to regex if missing)
python -m playwright install chromium
cp .env.example .env                                  # add GEMINI_API_KEY (or OpenAI / Anthropic)

mlia seed                                  # synthetic inventory + consented leads
mlia train                                 # LightGBM intent model (synthetic bootstrap + broker feedback)
mlia run                                   # one full run (live web)
mlia run --provider fixtures --llm fake    # fully offline demo
mlia eval                                  # extraction field accuracy on labeled pages
mlia feedback "lead:L0007" meeting         # broker feedback -> precision + retraining
mlia schedule                              # APScheduler loop, every 6h
streamlit run app/streamlit_app.py         # viewer
```

Postgres: run `docker compose up -d`, then set `MLIA_DATABASE_URL=postgresql://mlia:mlia@localhost:5432/mlia` (works the same for Neon or Render).

## Configuration

All settings are environment variables with the `MLIA_` prefix (see `.env.example` and `src/mlia/config.py`). The research brief (areas, segments, competitors, signal types, brokers) lives in `config/brief.yaml`.

| Setting | Default | |
|---|---|---|
| `MLIA_LLM_PROVIDER` | `gemini` | `gemini` / `openai` / `anthropic` / `fake` |
| `MLIA_LLM_MODEL` | provider default | `gemini-3.1-flash-lite`, `gpt-4o-mini`, `claude-opus-5-5` |
| `MLIA_SEARCH_PROVIDER` | `auto` | Tavily > Exa > DuckDuckGo; `fixtures` for offline |
| `MLIA_MAX_QUERIES` / `MLIA_MAX_PAGES` | 12 / 30 | cost guardrails per run |
| `MLIA_DATABASE_URL` | SQLite in `data/` | Postgres URL in production |
| `MLIA_CRM_WEBHOOK_URL` | unset | JSON POST of new rows |

If the API key is missing or rejected, the run does not fail. The circuit breaker switches to heuristic extraction and logs an `LLM_AUTH_FAILED` alert in the run log.

## Deployment

- **Scheduled runs:** `.github/workflows/scheduled-run.yml` runs at minute 17 every 6 hours. It needs the `MLIA_DATABASE_URL` and `GEMINI_API_KEY` Actions secrets, and skips itself until they are set. Each run uploads its Excel and digest as a workflow artifact.
- **History:** Postgres (Neon or Render). Retention is `MLIA_RETENTION_DAYS` (default 180).
- **UI:** Streamlit Community Cloud. Point it at `app/streamlit_app.py` and add `MLIA_DATABASE_URL` as a secret.
- **CI:** ruff + pytest on every push (`.github/workflows/ci.yml`).

## Evaluation

- `mlia eval` measures extraction field accuracy (signal type, entity, area, payment plan), page recall and the false-positive rate on negative pages (gossip and off-topic), using `eval/labeled_pages.json`.
- Broker-confirmed precision@70 and lead-to-meeting conversion come from feedback stored on output rows.
- Every run's log records duplicate rate, verifier rejection rate, cost and duration per run, and fetch success rate.

## Compliance

This design follows Egypt's PDPL (Law 151/2020):
- It reads only public business sources and the company's own consented inbound leads. Leads without marketing consent are skipped and counted.
- Private social profiles are blocked, and private-life content is dropped.
- Each row logs its source and consent.
- History older than the retention window is purged.

## Data

The inventory, leads and training labels are **synthetic** (`src/mlia/synthetic.py`). The fixture pages describe fictional companies at the fictional host `fixtures.mlia.local`. Replace these with the real inventory DB, the lead intake, and broker outcomes from `mlia feedback`.

## Tests

```bash
pytest          # 13 tests: units + end-to-end offline run (idempotency, PII, dedup, Excel schema)
```
