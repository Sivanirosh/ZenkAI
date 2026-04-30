"""JSONL rotation tests (PIVOT_ROADMAP §A.11).

Asserts:

- A fresh session writes to ``<sid>.0001.jsonl``.
- Once the active file (plus the pending buffer) crosses
  ``rotation_bytes``, the next flush goes to ``<sid>.0002.jsonl``.
- ``read_session_events`` concatenates the rotated files in order.
"""

from __future__ import annotations

from pathlib import Path

from backend.memory import manager as memory_manager
from backend.memory.manager import Event, MemoryManager


def _make_manager(tmp_path: Path, *, rotation_bytes: int) -> MemoryManager:
    return MemoryManager(
        root=tmp_path / "mem",
        rotation_bytes=rotation_bytes,
        flush_interval_ms=0,        # flush after every write
        buffer_high_water=1,        # don't sit on lines
        init_git=False,
    )


def test_first_file_is_seq_one(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path, rotation_bytes=1024 * 1024)
    mm.write_event(Event(session_id="s1", kind="turn_open"))
    files = mm.list_session_files("s1")
    assert len(files) == 1
    assert files[0].name == "s1.0001.jsonl"


def test_rotation_at_threshold_creates_second_file(tmp_path: Path) -> None:
    # Tiny threshold so we don't have to write 10 MB.
    mm = _make_manager(tmp_path, rotation_bytes=2048)
    bigpayload = "x" * 1500
    for i in range(4):
        mm.write_event(
            Event(session_id="s1", kind="probe", payload={"i": i, "blob": bigpayload})
        )
    files = mm.list_session_files("s1")
    assert len(files) >= 2, f"expected rotation, got: {[f.name for f in files]}"
    assert files[0].name == "s1.0001.jsonl"
    assert files[1].name == "s1.0002.jsonl"


def test_read_session_events_concatenates_in_order(tmp_path: Path) -> None:
    mm = _make_manager(tmp_path, rotation_bytes=512)
    blob = "y" * 400
    for i in range(6):
        mm.write_event(Event(session_id="s2", kind="evt", payload={"i": i, "b": blob}))
    rows = mm.read_session_events("s2")
    assert len(rows) == 6
    assert [r["payload"]["i"] for r in rows] == [0, 1, 2, 3, 4, 5]


def test_real_world_twelve_megabyte_session_rotates(tmp_path: Path) -> None:
    # The roadmap-mandated assertion: a 12 MB session ends up in two
    # files when the rotation threshold is the production 10 MB.
    mm = _make_manager(tmp_path, rotation_bytes=10 * 1024 * 1024)
    blob = "z" * (1024 * 1024 - 256)  # ~1 MB per event
    for i in range(13):
        mm.write_event(Event(session_id="big", kind="evt", payload={"i": i, "b": blob}))
    files = mm.list_session_files("big")
    assert len(files) >= 2
    total_bytes = sum(f.stat().st_size for f in files)
    assert total_bytes >= 12 * 1024 * 1024


def test_singleton_reset_for_tests_passes_rotation_bytes(tmp_path: Path) -> None:
    mm = memory_manager.reset_for_tests(
        root=tmp_path / "mem", rotation_bytes=4096
    )
    assert mm._rotation_bytes == 4096  # noqa: SLF001 — internal probe is fine here


# ─── Audit-trail bootstrap (PIVOT_ROADMAP §A.10) ─────────────────────────


def test_audit_trail_creates_standard_subtree(tmp_path: Path) -> None:
    root = tmp_path / "mem"
    MemoryManager(root=root, init_git=False)
    for sub in ("learner", "evidence", "sessions", "konversation",
                "captures", "facts", "compaction"):
        assert (root / sub).is_dir(), f"missing audit dir {sub}"
    assert (root / ".gitignore").exists()
    body = (root / ".gitignore").read_text(encoding="utf-8")
    assert "captures/img/" in body


def test_audit_trail_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "mem"
    MemoryManager(root=root, init_git=False)
    (root / ".gitignore").write_text("custom user content\n", encoding="utf-8")
    MemoryManager(root=root, init_git=False)
    # Bootstrap must not overwrite a customised .gitignore.
    assert (root / ".gitignore").read_text(encoding="utf-8") == "custom user content\n"
