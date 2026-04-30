"""Shared pytest fixtures: in-memory DuckDB + isolated runtime/memory state."""

from __future__ import annotations

import pytest

from backend.memory import manager as memory_manager
from backend.memory import store_vector as vector_store
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


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path):
    """Point the MemoryManager singleton at a tmp dir so JSONL never leaks."""
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    # Reset vector index after memory; tests that need it fresh re-call
    # ``vector_store.reset_for_tests(embedder=...)`` to inject a fake.
    vector_store._INDEX = None  # noqa: SLF001
    yield
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    vector_store._INDEX = None  # noqa: SLF001
