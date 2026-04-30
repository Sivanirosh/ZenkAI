"""Crash-injection durability test (PIVOT_ROADMAP §A.15).

Spins up a subprocess that imports MemoryManager and writes N events to
a tmp directory. The child writes an "I'm ready" marker, then keeps
writing events until the parent SIGKILLs it.

The parent then re-reads the JSONL files and asserts:

- Every line is valid JSON OR is the last line of a file (a torn write
  on the last line is the only allowed corruption).
- The number of recovered events is at least the number we *know* the
  child flushed (the marker file gives us a lower bound).

flush_interval_ms is set to 5 ms so events stop sitting in the buffer
quickly — without that the test would just measure the buffer size.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path


# Child process source. Written verbatim into a tmp .py file. Keeps the
# parent test small and lets us SIGKILL a real Python process.
_CHILD_SRC = """\
import sys, time
from pathlib import Path

sys.path.insert(0, {repo!r})
from backend.memory.manager import Event, MemoryManager

root = Path(sys.argv[1])
ready = Path(sys.argv[2])
sid = sys.argv[3]
total = int(sys.argv[4])

mm = MemoryManager(
    root=root,
    flush_interval_ms=5,
    buffer_high_water=4,
    init_git=False,
)

written = 0
for i in range(total):
    mm.write_event(Event(session_id=sid, kind="probe", payload={{"i": i, "blob": "x" * 64}}))
    written += 1
    if written == 32:
        # Force-flush + signal readiness to the parent.
        mm.close_turn(sid)
        ready.write_text(str(written))
    if written > 32:
        # Sleep just enough to interleave kill window with writes.
        time.sleep(0.001)
"""


def _write_child_script(tmp_path: Path, repo_root: Path) -> Path:
    body = _CHILD_SRC.format(repo=str(repo_root))
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return script


def _wait_for_marker(marker: Path, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if marker.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"child never wrote ready marker {marker}")


def test_kill_mid_flush_preserves_committed_events(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = _write_child_script(tmp_path, repo_root)
    mem_root = tmp_path / "mem"
    marker = tmp_path / "ready.txt"
    sid = "killtest"
    total = 1000

    proc = subprocess.Popen(
        [sys.executable, str(script), str(mem_root), str(marker), sid, str(total)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_marker(marker, timeout_s=10)
        # Marker says ≥ 32 events flushed. Let it run a bit longer so
        # additional rotations / fsyncs happen, then kill mid-stream.
        time.sleep(0.2)
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        proc.wait(timeout=5)

    committed_lower_bound = int(marker.read_text())
    assert committed_lower_bound >= 32

    sessions_dir = mem_root / "sessions"
    files = sorted(sessions_dir.glob(f"{sid}.*.jsonl"))
    assert files, f"no JSONL files written under {sessions_dir}"

    recovered: list[dict] = []
    for i, path in enumerate(files):
        is_last = i == len(files) - 1
        lines = path.read_text(encoding="utf-8").splitlines()
        for j, raw in enumerate(lines):
            if not raw:
                continue
            try:
                recovered.append(json.loads(raw))
            except json.JSONDecodeError:
                # Only the very last line of the very last file may be torn.
                if is_last and j == len(lines) - 1:
                    continue
                raise

    assert len(recovered) >= committed_lower_bound, (
        f"lost data: recovered {len(recovered)} events but child reported "
        f"{committed_lower_bound} flushed before kill"
    )

    # Every recovered event has the schema we wrote.
    for evt in recovered[:5]:
        assert evt["kind"] == "probe"
        assert "i" in evt["payload"]


def test_read_session_events_tolerates_torn_last_line(tmp_path: Path) -> None:
    """Belt-and-braces: the read helper itself must skip a torn last line."""
    from backend.memory.manager import MemoryManager

    mm = MemoryManager(root=tmp_path / "mem", init_git=False)
    sessions = tmp_path / "mem" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    f = sessions / "torn.0001.jsonl"
    f.write_text(
        '{"session_id":"torn","kind":"a","payload":{},"occurred_at":"x","v":1}\n'
        '{"session_id":"torn","kind":"b","payload":{}'  # torn — never closed
        ,
        encoding="utf-8",
    )
    rows = mm.read_session_events("torn")
    assert [r["kind"] for r in rows] == ["a"]
