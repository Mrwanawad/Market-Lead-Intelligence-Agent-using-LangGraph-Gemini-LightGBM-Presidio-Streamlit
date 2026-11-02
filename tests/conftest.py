from __future__ import annotations

import os

import pytest

from mlia.config import Settings


@pytest.fixture(params=["sqlite"] + (["postgres"] if os.getenv("MLIA_TEST_POSTGRES_URL") else []))
def settings(request, tmp_path) -> Settings:
    if request.param == "postgres":
        from sqlalchemy import text

        from mlia import db

        url = os.environ["MLIA_TEST_POSTGRES_URL"]
        with db.engine(url).begin() as c:  # clean slate per test
            c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    else:
        url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    return Settings(
        llm_provider="fake",
        search_provider="fixtures",
        database_url=url,
        output_dir=tmp_path / "outputs",
        model_path=tmp_path / "model.txt",
        use_browser=False,
        max_queries=20,
    )
