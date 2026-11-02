"""APScheduler loop (alternative to the GitHub Actions cron in .github/workflows/scheduled-run.yml)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from .config import get_settings

log = logging.getLogger(__name__)


def _job() -> None:
    from .pipeline import run

    try:
        res = run()
        if res["errors"]:
            log.warning("run %s finished with alerts: %s", res["run_id"], res["errors"])
    except Exception:  # noqa: BLE001 - keep the scheduler alive
        log.exception("scheduled run failed")


def start(hours: int | None = None, run_now: bool = True) -> None:
    hours = hours or get_settings().schedule_hours
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(_job, "interval", hours=hours, id="mlia", max_instances=1, coalesce=True,
                  next_run_time=datetime.now(UTC) if run_now else None)
    log.info("scheduler started: every %sh", hours)
    sched.start()
