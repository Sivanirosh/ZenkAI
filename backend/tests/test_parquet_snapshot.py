"""Atomic Parquet snapshot tests (PIVOT_ROADMAP §A.12).

Covers the three things we promise about ``store_parquet``:

- ``snapshot_mastery`` round-trips: write, read back, equal data.
- An empty mastery table writes nothing (no zero-row files).
- A crash mid-write (raised inside ``pyarrow.parquet.write_table``)
  leaves the destination unchanged and no ``*.tmp`` debris behind.
- ``append_session_index_row`` accumulates rows across calls.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from backend.mastery import model as mastery_model
from backend.memory import store_parquet


def test_snapshot_returns_none_when_table_is_empty(tmp_path: Path) -> None:
    dest = tmp_path / "mastery.parquet"
    out = store_parquet.snapshot_mastery(dest)
    assert out is None
    assert not dest.exists()


def test_snapshot_roundtrips(tmp_path: Path) -> None:
    mastery_model.update("verbs.modal", 0.8, source="drill")
    mastery_model.update("daily.greetings", 0.3, source="drill")

    dest = tmp_path / "mastery.parquet"
    out = store_parquet.snapshot_mastery(dest)
    assert out == dest
    assert dest.exists()

    table = pq.read_table(dest)
    by_id = {row["competency_id"]: row for row in table.to_pylist()}
    assert {"verbs.modal", "daily.greetings"} <= set(by_id)
    assert 0.0 < by_id["verbs.modal"]["mu"] <= 1.0
    assert by_id["verbs.modal"]["evidence_count"] >= 1


def test_snapshot_crash_leaves_dest_untouched(tmp_path: Path, monkeypatch) -> None:
    mastery_model.update("verbs.modal", 0.8, source="drill")

    dest = tmp_path / "mastery.parquet"

    # First, write a clean snapshot so the dest exists with known content.
    store_parquet.snapshot_mastery(dest)
    original_bytes = dest.read_bytes()

    # Now make pyarrow.parquet.write_table raise during the next call.
    import pyarrow.parquet as pq_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(pq_mod, "write_table", _boom)

    with pytest.raises(RuntimeError):
        store_parquet.snapshot_mastery(dest)

    assert dest.read_bytes() == original_bytes
    assert not (tmp_path / "mastery.parquet.tmp").exists()


def test_session_index_accumulates_rows(tmp_path: Path) -> None:
    dest = tmp_path / "index.parquet"
    store_parquet.append_session_index_row(
        dest, {"session_id": "a", "closed_at": "2026-04-30T08:00:00Z", "files": ["a.0001.jsonl"]}
    )
    store_parquet.append_session_index_row(
        dest, {"session_id": "b", "closed_at": "2026-04-30T09:00:00Z", "files": ["b.0001.jsonl"]}
    )
    rows = pq.read_table(dest).to_pylist()
    assert [r["session_id"] for r in rows] == ["a", "b"]


def test_close_session_writes_mastery_snapshot(tmp_path: Path) -> None:
    from backend.memory import manager as memory_manager
    from backend.memory.manager import Event, MemoryManager

    mm = MemoryManager(root=tmp_path / "mem", init_git=False)
    memory_manager._singleton = mm

    mastery_model.update("verbs.modal", 0.8, source="drill")
    mm.write_event(Event(session_id="sess-x", kind="turn_open"))
    mm.close_session("sess-x")

    snapshot = tmp_path / "mem" / "learner" / "mastery.parquet"
    assert snapshot.exists()
    rows = pq.read_table(snapshot).to_pylist()
    assert any(r["competency_id"] == "verbs.modal" for r in rows)

    index = tmp_path / "mem" / "sessions" / "index.parquet"
    assert index.exists()
