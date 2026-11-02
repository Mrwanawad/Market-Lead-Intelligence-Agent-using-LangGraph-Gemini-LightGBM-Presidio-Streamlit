from __future__ import annotations

import pytest

from mlia.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="fake",
        search_provider="fixtures",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        output_dir=tmp_path / "outputs",
        model_path=tmp_path / "model.txt",
        use_browser=False,
        max_queries=20,
    )
