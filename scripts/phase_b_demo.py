#!/usr/bin/env python
"""Phase B end-to-end demo (PIVOT_ROADMAP §B.1, §B.2, §B.4, §B.16).

Wires Onboarding → Planner → Atlas → Agent turn → Compactor in one
process so the whole slice can be exercised without a browser:

1. boot uvicorn against a temp DuckDB + memory root,
2. POST ``/api/v1/onboarding/goal`` with a real medical-FSP goal,
3. assert ``goals`` row + ``atlas_plans`` row created and the
   parsed payload looks reasonable (≥ 4 weeks, ≥ 3 districts),
4. GET ``/api/v1/agent/atlas`` and assert it returns districts +
   edges + a non-empty ``current_route``,
5. POST ``/api/v1/agent/turn`` once with ``current_screen=atlas`` and
   assert the SSE stream completes with at least one event,
6. POST ``/api/v1/admin/memory/compact`` and assert HTTP 200 plus
   the existence of ``compaction/last_run.json``.

Exits 0 on success, prints what failed and exits 1 otherwise.

Usage::

    python scripts/phase_b_demo.py
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
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


# ─── Helpers ────────────────────────────────────────────────────────────


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


def _http_post(url: str, body: dict[str, Any], *, timeout_s: float = 30.0) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        text = r.read().decode("utf-8")
        return json.loads(text) if text else {}


def _http_post_sse(
    url: str, body: dict[str, Any], *, timeout_s: float = 60.0
) -> list[dict[str, Any]]:
    """POST and return parsed SSE events."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
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


def _parse_sse_block(block: str) -> Optional[dict[str, Any]]:
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


# ─── Main ───────────────────────────────────────────────────────────────


GOAL_TEXT = (
    "Ich bin Ärztin aus Brasilien und möchte in 90 Tagen die Fachsprachprüfung "
    "(FSP) bestehen. Anamnesegespräche, körperliche Untersuchungen und "
    "Aufklärungen müssen sicher klappen."
)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="linguamate-phaseB-"))
    db_path = tmp / "linguamate.duckdb"
    mem_root = tmp / "memory"
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/api/v1"

    os.environ["DUCKDB_PATH"] = str(db_path)
    os.environ["LINGUAMATE_MEMORY_ROOT"] = str(mem_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    print(f"workspace: {tmp}")

    _step("boot uvicorn (temp DuckDB + memory root)")
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

        # ─── 1. Onboarding ──────────────────────────────────────
        _step("POST /onboarding/goal (FSP medical goal)")
        try:
            onb = _http_post(
                f"{base_url}/onboarding/goal",
                {"raw_text": GOAL_TEXT, "horizon": "30d"},
                timeout_s=120,
            )
        except Exception as exc:
            failed.append(f"onboarding/goal failed: {exc!r}")
            return _summary(failed, proc)

        for key in ("goal_id", "parsed", "plan_id", "plan", "horizon"):
            if key not in onb:
                failed.append(f"onboarding response missing {key}")
        plan = onb.get("plan", {}) or {}
        weeks = plan.get("weeks") or []
        districts = plan.get("districts") or []
        if len(weeks) < 4:
            failed.append(f"expected ≥4 weeks, got {len(weeks)}")
        if len(districts) < 3:
            failed.append(f"expected ≥3 districts, got {len(districts)}")
        if not failed:
            _ok(
                f"goal_id={onb['goal_id'][:18]}… plan_id={onb['plan_id'][:18]}… "
                f"weeks={len(weeks)} districts={len(districts)}"
            )

        # ─── 2. Onboarding state reflects the new goal ─────────
        _step("GET /onboarding/state")
        try:
            state = _http_get(f"{base_url}/onboarding/state")
        except Exception as exc:
            failed.append(f"onboarding/state failed: {exc!r}")
            state = {}
        if not state.get("has_goal"):
            failed.append("onboarding/state has_goal=False after submit")
        if state.get("latest_goal_id") != onb.get("goal_id"):
            failed.append("onboarding/state latest_goal_id mismatch")
        if state.get("latest_plan_id") != onb.get("plan_id"):
            failed.append("onboarding/state latest_plan_id mismatch")
        if not failed:
            _ok("state reflects the new goal + plan")

        # ─── 3. Atlas payload ──────────────────────────────────
        _step("GET /agent/atlas")
        try:
            atlas = _http_get(f"{base_url}/agent/atlas")
        except urllib.error.HTTPError as exc:
            failed.append(f"atlas HTTP {exc.code}")
            atlas = {}
        except Exception as exc:
            failed.append(f"atlas failed: {exc!r}")
            atlas = {}

        atlas_districts = atlas.get("districts") or []
        atlas_edges = atlas.get("edges") or []
        current_route = atlas.get("current_route") or []
        if not atlas_districts:
            failed.append("atlas has no districts")
        if not atlas_edges:
            failed.append("atlas has no edges")
        if not current_route:
            failed.append("atlas has no current_route")
        if not failed:
            _ok(
                f"atlas: districts={len(atlas_districts)} edges={len(atlas_edges)} "
                f"current_route={current_route[:3]}"
            )

        # ─── 4. Atlas-screen agent turn ────────────────────────
        _step("POST /agent/turn (current_screen=atlas)")
        try:
            events = _http_post_sse(
                f"{base_url}/agent/turn",
                {
                    "current_screen": "atlas",
                    "session_id": "phaseB-atlas",
                    "time_budget_min": 3,
                    "user_message": "Was sollen wir diese Woche zuerst angehen?",
                },
                timeout_s=120,
            )
        except Exception as exc:
            failed.append(f"agent turn failed: {exc!r}")
            events = []
        kinds = [ev.get("event") for ev in events]
        if "done" not in kinds:
            failed.append("agent turn did not emit 'done'")
        if len(events) <= 1:
            failed.append("agent turn produced ≤1 event")
        if not failed:
            _ok(f"turn emitted {len(events)} events ({sorted(set(kinds))})")

        # ─── 5. Compactor ──────────────────────────────────────
        _step("POST /admin/memory/compact")
        try:
            report = _http_post(
                f"{base_url}/admin/memory/compact",
                {"deadline_ms": 4000},
                timeout_s=15,
            )
        except Exception as exc:
            failed.append(f"compact failed: {exc!r}")
            report = {}
        if "phases_completed" not in report:
            failed.append("compact report missing phases_completed")
        last_run = mem_root / "compaction" / "last_run.json"
        if not last_run.exists():
            failed.append(f"compaction/last_run.json missing at {last_run}")
        if not failed:
            _ok(
                f"compactor done: phases={report['phases_completed']} "
                f"duration_ms={report.get('duration_ms')}"
            )

    except Exception as exc:
        failed.append(f"unexpected: {exc!r}")

    return _summary(failed, proc)


def _summary(failed: list[str], proc: subprocess.Popen) -> int:
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
        print("=== PHASE B DEMO: FAIL ===")
        for f in failed:
            print(f" - {f}")
        return 1
    print("=== PHASE B DEMO: PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
