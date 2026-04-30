"""Compactor tests (PIVOT_ROADMAP §B.16).

Coverage:

- empty memory: noop + ``last_run.json`` written.
- closed session without gist: gist written, second run is a no-op.
- high-mastery competency with 30-day-old evidence: zst archive
  contains the rows, hot ledger no longer contains them, mastery row
  is left untouched.
- lock acquisition prevents a second run; stale lock is reaped.
- crash mid-archive: the dest archive is left intact (atomic replace).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import zstandard as zstd

from backend.mastery import model as mastery_model
from backend.memory import compactor as compactor_module
from backend.memory.compactor import (
    CompactionLockedError,
    Compactor,
    run_for_manager,
)
from backend.memory.manager import Event, MemoryManager


def _make_manager(tmp_path: Path) -> MemoryManager:
    return MemoryManager(root=tmp_path / "mem", init_git=False)


# ─── Empty memory ───────────────────────────────────────────────────────


def test_empty_memory_runs_clean(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    report = run_for_manager(mm, deadline_ms=5000)
    assert report.gists_written == 0
    assert report.archived_evidence_rows == 0
    assert "summarise_missing_sessions" in report.phases_completed
    state_path = mm._root / "compaction" / "last_run.json"  # noqa: SLF001
    assert state_path.exists()
    body = json.loads(state_path.read_text(encoding="utf-8"))
    assert body["duration_ms"] >= 0


# ─── Gist phase ─────────────────────────────────────────────────────────


def test_gist_is_created_for_unsummarised_session(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    sid = "s-gisted"
    for kind, payload in (
        ("turn_open", {}),
        ("tool_intent", {"name": "recommend_text"}),
        ("tool_result", {"name": "recommend_text", "result": {"paragraph_id": "p1"}}),
        ("turn_close", {"turn_ms": 120}),
    ):
        mm.write_event(Event(session_id=sid, kind=kind, payload=payload))
    mm.close_turn(sid)

    report = run_for_manager(mm, deadline_ms=5000)
    assert report.gists_written == 1
    gist = mm._root / "sessions" / f"{sid}.gist.md"  # noqa: SLF001
    summ = mm._root / "sessions" / f"{sid}.summ.md"  # noqa: SLF001
    assert gist.exists()
    # B.12 — summariser now emits both gist + summ; LLM path is unreachable
    # in tests so the mechanical fallback runs (still a deterministic body).
    assert summ.exists() or "_summariser_unavailable" in str(report.notes)
    body = gist.read_text(encoding="utf-8")
    assert sid[:8] in body


def test_gist_is_idempotent_on_second_run(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    mm.write_event(Event(session_id="s-once", kind="turn_open"))
    mm.close_turn("s-once")

    first = run_for_manager(mm, deadline_ms=5000)
    second = run_for_manager(mm, deadline_ms=5000)
    assert first.gists_written == 1
    assert second.gists_written == 0
    assert second.sessions_already_gisted >= 1


# ─── Archive phase ──────────────────────────────────────────────────────


def _seed_archived_evidence(mm: MemoryManager) -> None:
    """High-mastery competency + a couple of stale rows in the hot ledger."""
    cur = mastery_model.db.cursor()
    cur.execute(
        """
        INSERT INTO mastery (competency_id, mu, sigma, last_evidence, evidence_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        ["verbs.modal", 0.95, 0.04, datetime(2026, 1, 1, tzinfo=timezone.utc), 50],
    )
    ledger = mm._root / "evidence" / "ledger.jsonl"  # noqa: SLF001
    ledger.parent.mkdir(parents=True, exist_ok=True)
    old_iso = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    fresh_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    rows = [
        {
            "id": "e1",
            "competency_id": "verbs.modal",
            "occurred_at": old_iso,
            "quality": 1.0,
            "source": "drill",
        },
        {
            "id": "e2",
            "competency_id": "verbs.modal",
            "occurred_at": old_iso,
            "quality": 1.0,
            "source": "drill",
        },
        {
            "id": "e3",
            "competency_id": "verbs.modal",
            "occurred_at": fresh_iso,
            "quality": 1.0,
            "source": "drill",
        },
        {
            "id": "e4",
            "competency_id": "daily.greetings",
            "occurred_at": old_iso,
            "quality": 0.5,
            "source": "drill",
        },
    ]
    ledger.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _read_archive_rows(path: Path) -> list[dict]:
    with path.open("rb") as f:
        decompressed = zstd.ZstdDecompressor().decompress(f.read())
    return [
        json.loads(line)
        for line in decompressed.decode("utf-8").splitlines()
        if line.strip()
    ]


def test_archive_inverse_mastery_moves_old_rows_only(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    _seed_archived_evidence(mm)

    report = run_for_manager(mm, deadline_ms=5000)
    assert "verbs.modal" in report.archived_competencies
    assert report.archived_evidence_rows == 2  # e1 + e2
    assert report.archive_files

    archive = Path(report.archive_files[0])
    rows = _read_archive_rows(archive)
    assert {r["id"] for r in rows} == {"e1", "e2"}

    ledger = mm._root / "evidence" / "ledger.jsonl"  # noqa: SLF001
    surviving = [
        json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line
    ]
    assert {r["id"] for r in surviving} == {"e3", "e4"}

    # Mastery row untouched.
    row = mastery_model.db.cursor().execute(
        "SELECT mu, sigma, evidence_count FROM mastery WHERE competency_id = ?",
        ["verbs.modal"],
    ).fetchone()
    assert row[2] == 50


def test_archive_phase_is_idempotent(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    _seed_archived_evidence(mm)
    first = run_for_manager(mm, deadline_ms=5000)
    second = run_for_manager(mm, deadline_ms=5000)
    assert first.archived_evidence_rows == 2
    assert second.archived_evidence_rows == 0


def test_archive_appends_to_existing_quarter_archive(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    _seed_archived_evidence(mm)
    run_for_manager(mm, deadline_ms=5000)

    # Add another stale row on the same competency.
    ledger = mm._root / "evidence" / "ledger.jsonl"  # noqa: SLF001
    extra = {
        "id": "e5",
        "competency_id": "verbs.modal",
        "occurred_at": (datetime.now(timezone.utc) - timedelta(days=70)).isoformat(),
        "quality": 1.0,
        "source": "drill",
    }
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(extra) + "\n")

    second = run_for_manager(mm, deadline_ms=5000)
    assert second.archived_evidence_rows == 1
    archive = Path(second.archive_files[0])
    archived_ids = {r["id"] for r in _read_archive_rows(archive)}
    assert {"e1", "e2", "e5"} <= archived_ids


# ─── Lock semantics ─────────────────────────────────────────────────────


def test_lock_prevents_concurrent_runs(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    holder = Compactor(mm)
    holder.acquire_lock(owner="holder")
    try:
        with pytest.raises(CompactionLockedError):
            run_for_manager(mm, deadline_ms=5000)
    finally:
        holder.release_lock()


def test_stale_lock_is_reaped(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path)
    lock_dir = mm._root / "compaction" / "locks"  # noqa: SLF001
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / "compactor.lock"
    lock.write_text("{}", encoding="utf-8")
    # Backdate the lock to look stale (>10 min old).
    old = time.time() - 4000
    os.utime(lock, (old, old))

    report = run_for_manager(mm, deadline_ms=5000)
    assert "summarise_missing_sessions" in report.phases_completed


# ─── Crash safety ───────────────────────────────────────────────────────


def test_archive_write_failure_leaves_dest_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    """If the second archive batch fails, the first archive must remain."""
    mm = _make_manager(tmp_path)
    _seed_archived_evidence(mm)

    # Run a clean first compaction so the existing archive holds {e1, e2}.
    run_for_manager(mm, deadline_ms=5000)
    archive = mm._root / "evidence" / "archive"  # noqa: SLF001
    files = list(archive.glob("*.jsonl.zst"))
    assert files, "first run must produce an archive"
    archive_path = files[0]
    pre_bytes = archive_path.read_bytes()

    # Add another stale row; force the next archive write to blow up.
    ledger = mm._root / "evidence" / "ledger.jsonl"  # noqa: SLF001
    extra = {
        "id": "e_boom",
        "competency_id": "verbs.modal",
        "occurred_at": (datetime.now(timezone.utc) - timedelta(days=70)).isoformat(),
        "quality": 1.0,
        "source": "drill",
    }
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(extra) + "\n")

    def boom(self, path, rows):  # type: ignore[no-untyped-def]
        raise OSError("simulated disk error")

    monkeypatch.setattr(Compactor, "_append_zst", boom)

    report = run_for_manager(mm, deadline_ms=5000)
    assert "archive_inverse_mastery" in report.phases_skipped
    assert any("archive_write_failed" in n for n in report.notes)

    # The existing archive must be byte-identical (atomic semantics).
    assert archive_path.read_bytes() == pre_bytes
    # And the hot ledger must NOT have been mutated yet.
    surviving_ids = {
        json.loads(line)["id"]
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "e_boom" in surviving_ids


# ─── Admin endpoint ─────────────────────────────────────────────────────


def test_admin_compact_endpoint_returns_report(tmp_path: Path) -> None:
    """End-to-end through MemoryManager.compact() via the API."""
    from fastapi.testclient import TestClient

    from backend.main import app
    from backend.memory import manager as memory_manager

    memory_manager.reset_for_tests(root=tmp_path / "live_mem")
    client = TestClient(app)
    res = client.post("/api/v1/admin/memory/compact", json={"deadline_ms": 2000})
    assert res.status_code == 200
    body = res.json()
    assert "phases_completed" in body
    assert "summarise_missing_sessions" in body["phases_completed"]
    state_path = tmp_path / "live_mem" / "compaction" / "last_run.json"
    assert state_path.exists()
