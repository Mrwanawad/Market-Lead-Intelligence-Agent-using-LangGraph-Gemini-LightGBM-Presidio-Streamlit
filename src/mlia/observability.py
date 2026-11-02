"""Langfuse tracing (optional: enabled when LANGFUSE_PUBLIC_KEY/SECRET_KEY are set) + run-level metrics."""

from __future__ import annotations

import contextlib
import logging
import os

log = logging.getLogger(__name__)


class Tracer:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.lf = None
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse

                self.lf = Langfuse()
            except Exception as e:  # noqa: BLE001
                log.warning("langfuse init failed: %s", e)

    @property
    def enabled(self) -> bool:
        return self.lf is not None

    @contextlib.contextmanager
    def span(self, name: str, **meta):
        if not self.lf:
            yield None
            return
        cm = None
        try:
            cm = self.lf.start_as_current_span(name=name, metadata={"run_id": self.run_id, **meta})
            span = cm.__enter__()
        except Exception as e:  # noqa: BLE001 - tracing must never break a run
            log.debug("langfuse span failed: %s", e)
            cm, span = None, None
        try:
            yield span
        finally:
            if cm is not None:
                with contextlib.suppress(Exception):
                    cm.__exit__(None, None, None)

    def flush(self) -> None:
        if self.lf:
            with contextlib.suppress(Exception):
                self.lf.flush()


def source_success_rate(source_log: list[dict]) -> float:
    fetches = [s for s in source_log if s.get("kind") == "fetch"]
    return round(sum(s["ok"] for s in fetches) / len(fetches), 3) if fetches else 0.0
