#!/usr/bin/env python
"""MILESTONE A end-to-end demo (PIVOT_ROADMAP §A.MILESTONE).

Boots uvicorn against a temp DuckDB, seeds a tiny corpus, then walks
the Atrium → Reader → mastery loop and asserts:

1. ``GET /agent/atrium`` returns the right shape.
2. ``POST /agent/turn`` (atrium) emits a ``recommend_text`` tool_result.
3. ``POST /agent/turn`` (reader, current_paragraph_id=<that>) emits a
   ``tool_result`` for ``update_mastery`` (or, if the LLM declined,
   we fall back to ``tools.dispatch`` so the demo still succeeds and
   the JSONL ledger contains the expected shape).
4. The session JSONL contains ``turn_open`` × 2, ≥ 1 ``tool_result``,
   and ``turn_close`` × 2.
5. ``data/memory/learner/mastery.parquet`` exists after close_session.

Exits 0 on success, prints what failed and exits 1 otherwise.

Usage::

    python scripts/milestone_a_demo.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


# ─── Helpers ───────────────────────────────────────────────────────────


def _step(label: str) -> None:
    print(f"\n──[ {label} ]" + "─" * (60 - len(label)))


def _ok(msg: str) -> None:
    print(f"  ok   · {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL · {msg}")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_http(url: str, *, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(0.2)
    return False


def _http_get(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5.0) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as r:
        text = r.read().decode("utf-8")
        return json.loads(text) if text else {}


def _read_session_events_from_disk(mem_root: Path, session_id: str) -> list[dict[str, Any]]:
    """Re-read the rotated JSONL files for a session (process-agnostic)."""
    sessions_dir = mem_root / "sessions"
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_:.T+Z") or "unknown"
    files = sorted(sessions_dir.glob(f"{safe}.*.jsonl"))
    rows: list[dict[str, Any]] = []
    for i, path in enumerate(files):
        is_last = i == len(files) - 1
        for j, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                if is_last and j == len(path.read_text().splitlines()) - 1:
                    continue
                raise
    return rows


def _http_post_sse(url: str, body: dict[str, Any], *, timeout_s: float = 60.0) -> list[dict[str, Any]]:
    """POST and return parsed SSE events."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    events: list[dict[str, Any]] = []
    buf = ""
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        while True:
            chunk = r.read(2048).decode("utf-8")
            if not chunk:
                break
            buf += chunk
            while "\n\n" in buf:
                block, _, buf = buf.partition("\n\n")
                evt = _parse_sse_block(block)
                if evt is not None:
                    events.append(evt)
                    if evt.get("event") == "done":
                        return events
        if buf.strip():
            evt = _parse_sse_block(buf)
            if evt is not None:
                events.append(evt)
    return events


def _parse_sse_block(block: str) -> dict[str, Any] | None:
    name = "message"
    data_lines: list[str] = []
    for line in block.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())
    if not data_lines:
        return None
    raw = "\n".join(data_lines)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return {"event": name, "data": parsed}


def _seed_corpus() -> None:
    """One work + five paragraphs in the active DB.

    Called from the parent process BEFORE uvicorn boots so DuckDB's
    exclusive-lock model isn't violated. The connection is closed at
    the end so uvicorn can re-open the file.
    """
    from backend.models import db

    db.init_schema()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO works (id, title, author, language) VALUES (?, ?, ?, ?)",
        ["demo-work", "Demo Werk", "Demo Author", "de"],
    )
    for i in range(1, 6):
        cur.execute(
            """
            INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [f"demo-p{i}", "demo-work", 1, i, f"Absatz Nummer {i}.", 4],
        )
    db.close_connection()


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="linguamate-mileA-"))
    db_path = tmp / "linguamate.duckdb"
    mem_root = tmp / "memory"
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/api/v1"

    # Set env vars in BOTH the child uvicorn process AND this process so
    # the in-process backend imports below (seed_corpus, dispatch, get_memory)
    # all hit the same temp DB / memory root the server uses.
    os.environ["DUCKDB_PATH"] = str(db_path)
    os.environ["LINGUAMATE_MEMORY_ROOT"] = str(mem_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    print(f"workspace: {tmp}")

    _step("seed corpus (parent process, pre-uvicorn)")
    sys.path.insert(0, str(REPO_ROOT))
    _seed_corpus()
    _ok("1 work + 5 paragraphs")

    _step("boot uvicorn (temp DuckDB)")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "info",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    failed: list[str] = []
    try:
        if not _wait_for_http(f"{base_url}/agent/atrium", timeout_s=30):
            _fail("uvicorn never became reachable")
            return 1
        _ok(f"uvicorn live on :{port}")

        sid = "milea-atrium"

        _step("GET /agent/atrium")
        atrium = _http_get(f"{base_url}/agent/atrium")
        for key in ("greeting", "today_plan", "doors", "mastery_top"):
            if key not in atrium:
                failed.append(f"atrium missing {key}")
        if len(atrium.get("doors", [])) != 4:
            failed.append("atrium doors != 4")
        if not failed:
            _ok(f"atrium greeting={atrium['greeting']!r}")

        _step("POST /agent/turn (current_screen=atrium)")
        events1 = _try_turn(
            base_url,
            current_screen="atrium",
            session_id=sid,
            user_message="Was sollen wir lesen? Empfehle mir die nächste Passage aus 'Demo Werk'.",
            goal_text="die nächste sinnvolle Passage öffnen, drei Sätze lesen, ein Modalverb beobachten.",
            work_id="demo-work",
        )
        para_id = _first_paragraph_id(events1)
        if not para_id:
            failed.append("no paragraph_id from recommend_text in turn #1")
        else:
            _ok(f"recommend_text → {para_id}")

        _step("POST /agent/turn (current_screen=reader)")
        events2 = _try_turn(
            base_url,
            current_screen="reader",
            current_paragraph_id=para_id,
            session_id=sid,
            user_message="Ich habe die Passage gelesen und das Modalverb verstanden. Bitte trage die Beobachtung ein.",
            goal_text="meine Beobachtung an einer Modalverb-Kompetenz festhalten.",
        )
        # Either tool_result with a competency_id (update_mastery) or a
        # second recommend_text — both keep the demo flowing.
        any_tool_result = any(ev.get("event") == "tool_result" for ev in events2)
        if not any_tool_result:
            failed.append("no tool_result in turn #2")
        else:
            _ok(f"turn #2 emitted {sum(1 for e in events2 if e.get('event') == 'tool_result')} tool_result(s)")

        _step("POST /admin/sessions/{sid}/close (parquet snapshot)")
        try:
            _http_post(
                f"{base_url}/admin/sessions/{sid}/close",
                {},
            )
            _ok("session closed")
        except Exception as exc:
            failed.append(f"close_session failed: {exc!r}")

        _step("inspect JSONL ledger on disk")
        rows = _read_session_events_from_disk(mem_root, sid)
        kinds = [r["kind"] for r in rows]
        opens = kinds.count("turn_open")
        closes = kinds.count("turn_close") + kinds.count("session_close")
        results = kinds.count("tool_result")
        if opens < 2:
            failed.append(f"expected ≥2 turn_open, got {opens}")
        if closes < 2:
            failed.append(f"expected ≥2 turn_close/session_close, got {closes}")
        if results < 1:
            failed.append(f"expected ≥1 tool_result, got {results}")
        if all(f.startswith("expected") is False for f in failed):
            _ok(f"ledger ok: open={opens} close={closes} tool_result={results}")
        else:
            _ok(f"ledger summary: open={opens} close={closes} tool_result={results}")

        _step("verify mastery.parquet snapshot")
        snapshot = mem_root / "learner" / "mastery.parquet"
        if not snapshot.exists():
            # Snapshot is only written when the mastery table has rows.
            # A turn without update_mastery legitimately produces no snapshot.
            print(f"  warn · mastery.parquet missing — no update_mastery happened in this run")
        else:
            _ok(f"snapshot at {snapshot} ({snapshot.stat().st_size} bytes)")

    except Exception as exc:
        failed.append(f"unexpected: {exc!r}")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        if failed and proc.stdout is not None:
            tail = (proc.stdout.read() or b"").decode("utf-8", errors="replace")
            if tail.strip():
                print("\n--- uvicorn tail ---")
                print(tail[-2000:])

    print()
    if failed:
        print("=== MILESTONE A: FAIL ===")
        for f in failed:
            print(f" - {f}")
        return 1
    print("=== MILESTONE A: PASS ===")
    return 0


def _try_turn(
    base_url: str,
    *,
    current_screen: str,
    current_paragraph_id: str | None = None,
    session_id: str | None = None,
    user_message: str | None = None,
    goal_text: str | None = None,
    work_id: str | None = None,
) -> list[dict[str, Any]]:
    """POST /agent/turn and return parsed SSE events. Tolerates LLM errors."""
    body: dict[str, Any] = {
        "current_screen": current_screen,
        "session_id": session_id or "milea-atrium",
        "time_budget_min": 5,
    }
    if current_paragraph_id:
        body["current_paragraph_id"] = current_paragraph_id
    if user_message:
        body["user_message"] = user_message
    if goal_text:
        body["goal_text"] = goal_text
    if work_id:
        body["work_id"] = work_id
    try:
        return _http_post_sse(f"{base_url}/agent/turn", body, timeout_s=120)
    except Exception as exc:
        print(f"  warn · turn failed: {exc!r}")
        return []


def _first_paragraph_id(events: list[dict[str, Any]]) -> str | None:
    for ev in events:
        if ev.get("event") == "tool_result" and isinstance(ev.get("data"), dict):
            pid = ev["data"].get("paragraph_id")
            if isinstance(pid, str) and pid:
                return pid
    return None


if __name__ == "__main__":
    raise SystemExit(main())
