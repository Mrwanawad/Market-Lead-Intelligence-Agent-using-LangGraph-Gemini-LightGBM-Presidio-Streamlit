# Market & Lead Intelligence Agent — Architecture

## Scenario
Brokers waste hours on manual market research and cold lists. A scheduled agent runs every 6 hours, researches public market signals (new launches, price and payment-plan changes, competitor listings, companies opening offices → corporate housing demand, relocation firms), enriches and scores the company's own consented inbound leads, and delivers a structured Excel/CRM sheet with next actions.

## Input
- Research brief: target areas, segments, signal types, competitors.
- Public web sources (search APIs, developer sites, news, listing portals within their terms).
- First-party inbound leads (website forms, WhatsApp, ad forms) with consent flags.
- Inventory database (for unit matching).

## Output
- Excel / CRM rows: lead or company, source URL, signal type, area, stated budget band, intent score + reasons, matched units, assigned broker, next action, collected_at.
- Market digest: new launches, price changes, competitor moves.
- Run log: sources visited, cost, errors.

## Architecture

```
 [Scheduler]  GitHub Actions cron / APScheduler (every 6 h)
        |
        v
 [Planner agent]  brief -> search queries per signal type
        |
        v
 [Researcher agent]  Tavily / Exa search -> Playwright browser for pages
        |
        v
 [Extractor agent]  page -> Pydantic schema (signal, entity, area, budget, evidence quote)
        |
        v
 [Verifier agent]  checks evidence quote exists on page, drops low-confidence rows
        |
        v
 [Enrichment & dedup]  rapidfuzz entity matching vs. existing CRM / previous runs
        |
        v
 [PII filter]  Presidio: strip private-life data; keep business / stated info only
        |
        v
 [Scoring]  LightGBM intent model (features: signal type, recency, budget fit, area demand)
        |
        v
 [Matcher]  units from inventory by area/budget
        |
        v
 [Outputs]  xlsxwriter Excel + CRM push + Postgres history + Streamlit viewer
        |
        v
 [Observability]  Langfuse traces, cost per run, source success rate
```

## Tools Used
- Agents: Claude tool use, LangGraph (planner → researcher → extractor → verifier)
- Web: Tavily / Exa APIs, Playwright (Chrome), httpx, trafilatura (clean text)
- Extraction: Pydantic structured output
- Data: pandas, polars, rapidfuzz, PostgreSQL
- Privacy: Microsoft Presidio
- ML: LightGBM, scikit-learn, SHAP
- Output: xlsxwriter / openpyxl, CRM API
- Scheduling: GitHub Actions cron or APScheduler
- UI: Streamlit
- Observability: Langfuse; quality: pytest, ruff

## Tools Not Used (and why)
- Scraping private social profiles or personal-life data: PDPL (Law 151/2020) and platform terms.
- Headless browser for everything: search APIs first, browser only when needed (cost, speed).
- Vector DB: rows are structured; dedup with fuzzy matching is enough.

## Methodologies
- Multi-agent pipeline with evidence-grounded extraction (every row cites a source quote).
- Idempotent runs: dedup against history; only new/changed signals reported.
- Consent and source logged per row; retention policy.

## Evaluation Metrics
- Precision of qualified leads (broker-confirmed), extraction field accuracy on labeled pages.
- Duplicate rate, verifier rejection rate.
- Cost and duration per run.
- Business: lead-to-meeting conversion, broker time saved.

## Production Monitoring
- Failed sources / blocked pages, empty-run alerts.
- Cost per run trend, token usage.
- Score distribution drift; broker feedback loop for retraining.

## Deployment Plan
- Scheduled runs: GitHub Actions cron (secrets in Actions secrets).
- Results: Postgres (Render or Neon) + Excel artifact per run.
- UI: Streamlit Community Cloud (browse runs, filter, download Excel).
