"""CLI: mlia seed | train | run | schedule | feedback | eval."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()
app = typer.Typer(add_completion=False, help="Market & Lead Intelligence Agent")


def _logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "trafilatura", "primp", "ddgs"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*shap values output has changed.*")


@app.command()
def seed(units: int = 240, leads: int = 40) -> None:
    """Load synthetic inventory + consented inbound leads into the DB."""
    from .pipeline import seed as _seed

    _logging(False)
    typer.echo(json.dumps(_seed(n_units=units, n_leads=leads)))


@app.command()
def train(use_feedback: bool = True) -> None:
    """Train the LightGBM intent model (synthetic bootstrap + broker feedback from the DB)."""
    import pandas as pd

    from .feedback import feedback_frame
    from .scoring import train as _train
    from .synthetic import training_frame

    _logging(False)
    df = training_frame()
    if use_feedback:
        fb = feedback_frame()
        if len(fb):
            typer.echo(f"adding {len(fb)} broker-feedback rows (weighted x5)")
            df = pd.concat([df] + [fb] * 5, ignore_index=True)
    m = _train(df)
    path = m.save()
    typer.echo(json.dumps({"model": str(path), **m.metrics}))


@app.command()
def run(
    provider: str = typer.Option(None, help="search provider: auto|tavily|exa|ddg|fixtures"),
    llm: str = typer.Option(None, help="LLM provider: gemini|openai|anthropic|fake"),
    max_queries: int = typer.Option(None),
    max_pages: int = typer.Option(None),
    verbose: bool = False,
) -> None:
    """Run the full pipeline once."""
    from .config import get_settings
    from .llm import LLM
    from .pipeline import run as _run

    _logging(verbose)
    s = get_settings().model_copy()
    if provider:
        s.search_provider = provider
    if max_queries:
        s.max_queries = max_queries
    if max_pages:
        s.max_pages = max_pages
    res = _run(settings=s, llm=LLM(provider=llm) if llm else None)
    typer.echo(json.dumps({k: res[k] for k in ("run_id", "excel", "digest", "cost_usd", "errors")}, indent=1))
    typer.echo(json.dumps({k: v for k, v in res["stats"].items() if k != "cost_by_agent"}, default=str))


@app.command()
def schedule(hours: int = typer.Option(None), run_now: bool = True) -> None:
    """Run every N hours with APScheduler (blocking). Use GitHub Actions cron for serverless."""
    from .scheduler import start

    _logging(False)
    start(hours, run_now)


@app.command()
def feedback(row_id: str, label: str = typer.Argument(..., help="qualified | not_qualified | meeting")) -> None:
    """Record broker feedback on an output row (feeds precision metrics + retraining)."""
    from .feedback import record

    _logging(False)
    typer.echo(json.dumps(record(row_id, label)))


@app.command("eval")
def evaluate(labels: Path = typer.Option(Path("eval/labeled_pages.json"))) -> None:
    """Extraction field accuracy on labeled fixture pages + broker-confirmed precision."""
    from .evaluation import run_eval

    _logging(False)
    typer.echo(json.dumps(run_eval(labels), indent=1))


if __name__ == "__main__":
    app()
