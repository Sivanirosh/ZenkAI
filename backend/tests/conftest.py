"""Shared pytest fixtures: in-memory DuckDB swapped for the singleton."""

from __future__ import annotations

import pytest

from backend.models import db as db_module
from backend.services import runtime_config


@pytest.fixture(autouse=True)
def _in_memory_db():
    db_module.close_connection()
    conn = db_module.reset_for_tests()
    yield conn
    db_module.close_connection()


@pytest.fixture(autouse=True)
def _reset_runtime_config(tmp_path, monkeypatch):
    """Isolate runtime LLM options so tests don't read/write the real file."""
    runtime_config.reset_for_tests()
    monkeypatch.setattr(
        runtime_config,
        "_CONFIG_PATH",
        tmp_path / "runtime_config.json",
    )
    yield
    runtime_config.reset_for_tests()
