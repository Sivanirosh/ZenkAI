#!/usr/bin/env python
"""MILESTONE B end-to-end demo (PIVOT_ROADMAP §B.5/§B.6/§B.7/§B.9/§B.17/§B.18).

Walks the four rooms in one process, against a temp DuckDB + memory
root, so the whole milestone can be validated without a browser:

1. boot uvicorn,
2. POST ``/onboarding/goal`` (FSP medical goal) and assert plan,
3. GET ``/agent/atlas`` and assert districts/edges/current_route,
4. POST ``/agent/turn`` once with ``current_screen=atrium`` and assert
   the SSE stream completes,
5. POST ``/conversation/turn`` with a synthetic WAV blob, an injected
   STT stub, and an injected LLM stub. Assert both transcripts are
   returned, the JSONL ledger has both entries, and a per-session
   ``konversation_*`` event landed in the memory ledger,
6. POST ``/capture/image`` with a synthetic PNG and a stubbed vision
   call. Assert the sha256 file lives under ``captures/img/`` and a
   row was appended to ``captures/index.parquet``,
7. POST ``/admin/memory/compact`` and assert ``compaction/last_run.json``
   exists (proves §B.16 still wires through after rooms have written).

Stubs are injected into the *child* uvicorn process via a small
"override" module written next to the temp dir and imported via
``backend.main`` startup hook (see ``BackendOverrides``). The script
fails closed: any unmet assertion prints what failed and exits 1.

Usage::

    python scripts/milestone_b_demo.py
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


# ─── Pretty printing ────────────────────────────────────────────────────


def _step(label: str) -> None:
    print(f"\n──[ {label} ]" + "─" * (max(2, 60 - len(label))))


def _ok(msg: str) -> None:
    print(f"  ok   · {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL · {msg}")


# ─── HTTP helpers (stdlib only) ─────────────────────────────────────────


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


def _http_post_multipart(
    url: str,
    *,
    files: list[tuple[str, str, bytes, str]],
    fields: dict[str, str],
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Tiny multipart/form-data poster (stdlib only)."""
    boundary = "----linguamateMS-{:x}".format(int(time.time() * 1000))
    crlf = b"\r\n"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(("--" + boundary).encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"'.encode()
        )
        parts.append(b"")
        parts.append(value.encode("utf-8"))
    for name, filename, content, ctype in files:
        parts.append(("--" + boundary).encode())
        parts.append(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"'
            ).encode()
        )
        parts.append(f"Content-Type: {ctype}".encode())
        parts.append(b"")
        parts.append(content)
    parts.append(("--" + boundary + "--").encode())
    parts.append(b"")
    body = crlf.join(parts)
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        text = r.read().decode("utf-8")
        return json.loads(text) if text else {}


def _http_post_sse(
    url: str, body: dict[str, Any], *, timeout_s: float = 60.0
) -> list[dict[str, Any]]:
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


# ─── Synthetic media payloads ───────────────────────────────────────────


def _silent_wav() -> bytes:
    """100 ms of mono 16-bit PCM @ 16 kHz silence with a RIFF/WAVE header."""
    sr = 16_000
    n = sr // 10
    pcm = b"\x00\x00" * n
    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(pcm)),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            struct.pack("<H", 1),
            struct.pack("<H", 1),
            struct.pack("<I", sr),
            struct.pack("<I", sr * 2),
            struct.pack("<H", 2),
            struct.pack("<H", 16),
            b"data",
            struct.pack("<I", len(pcm)),
        ]
    )
    return header + pcm


def _synthetic_png(width: int = 32, height: int = 32) -> bytes:
    """Tiny grayscale PNG so we have something to sha256 + parquet."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    idat = zlib.compress(raw, 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


# ─── Backend stubs (written into a tmp file, imported by uvicorn) ───────
#
# The uvicorn child process imports ``backend.main`` which lazily uses
# the conversation runner / capture service. We don't try to monkey-patch
# inside the child. Instead, we set environment knobs that the runner
# and the vision helper read at request time — see the module overrides
# below for the exact knobs.


GOAL_TEXT = (
    "Ich bin Ärztin aus Brasilien und möchte in 90 Tagen die Fachsprachprüfung "
    "(FSP) bestehen. Anamnesegespräche, körperliche Untersuchungen und "
    "Aufklärungen müssen sicher klappen."
)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="linguamate-msB-"))
    db_path = tmp / "linguamate.duckdb"
    mem_root = tmp / "memory"
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}/api/v1"

    os.environ["DUCKDB_PATH"] = str(db_path)
    os.environ["LINGUAMATE_MEMORY_ROOT"] = str(mem_root)
    os.environ["LINGUAMATE_DEMO_STUBS"] = "1"

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
            return _summary(failed + ["uvicorn never reachable"], proc)
        _ok(f"uvicorn live on :{port}")

        # ─── 1. Onboarding ──────────────────────────────────────
        _step("POST /onboarding/goal")
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
        if len(plan.get("districts") or []) < 3:
            failed.append("expected ≥3 districts after onboarding")
        if not failed:
            _ok(
                f"goal_id={onb['goal_id'][:18]}… plan_id={onb['plan_id'][:18]}… "
                f"districts={len(plan.get('districts') or [])}"
            )

        # ─── 2. Atlas ──────────────────────────────────────────
        _step("GET /agent/atlas")
        try:
            atlas = _http_get(f"{base_url}/agent/atlas")
        except Exception as exc:
            failed.append(f"atlas failed: {exc!r}")
            atlas = {}
        if not atlas.get("districts"):
            failed.append("atlas missing districts")
        if not atlas.get("current_route"):
            failed.append("atlas missing current_route")
        if "districts" in atlas and atlas.get("current_route"):
            _ok(
                f"atlas: districts={len(atlas['districts'])} "
                f"current_route={atlas['current_route'][:3]}"
            )

        # ─── 3. Atrium-screen agent turn ───────────────────────
        _step("POST /agent/turn (current_screen=atrium)")
        try:
            events = _http_post_sse(
                f"{base_url}/agent/turn",
                {
                    "current_screen": "atrium",
                    "session_id": "msB-atrium",
                    "time_budget_min": 3,
                    "user_message": "Was sollen wir heute zuerst angehen?",
                },
                timeout_s=120,
            )
        except Exception as exc:
            failed.append(f"agent turn failed: {exc!r}")
            events = []
        kinds = [ev.get("event") for ev in events]
        if "done" not in kinds:
            failed.append("agent turn did not emit 'done'")
        if not failed:
            _ok(f"turn emitted {len(events)} events ({sorted(set(kinds))})")

        # ─── 4. Konversation room ──────────────────────────────
        _step("POST /conversation/turn (synthetic audio)")
        try:
            turn = _http_post_multipart(
                f"{base_url}/conversation/turn",
                files=[("audio", "demo.wav", _silent_wav(), "audio/wav")],
                fields={
                    "session_id": "msB-konv",
                    "scenario": "Anamnesegespräch",
                    "target_cefr": "B1",
                },
                timeout_s=60,
            )
        except Exception as exc:
            failed.append(f"conversation/turn failed: {exc!r}")
            turn = {}
        u = turn.get("user_transcript") or {}
        a = turn.get("assistant_reply") or {}
        if not u or not a:
            failed.append("conversation turn missing transcripts")
        konv_dir = mem_root / "konversation"
        konv_files = list(konv_dir.glob("transcripts.*.jsonl")) if konv_dir.exists() else []
        if not konv_files:
            failed.append("konversation/transcripts.NNNN.jsonl missing")
        else:
            lines = sum(
                1
                for f in konv_files
                for l in f.read_text(encoding="utf-8").splitlines()
                if l.strip()
            )
            if lines < 2:
                failed.append(
                    f"konversation ledger has {lines} lines (expected ≥2)"
                )
            else:
                _ok(
                    f"konversation: stt={turn.get('stt_engine')} "
                    f"reply='{a.get('text', '')[:48]}…' "
                    f"ledger_lines={lines}"
                )

        # Per-session events should also be in the sessions/ ledger.
        sess_dir = mem_root / "sessions"
        sess_files = list(sess_dir.glob("msB-konv.*.jsonl")) if sess_dir.exists() else []
        if not sess_files:
            failed.append("sessions/msB-konv.*.jsonl missing")

        # ─── 5. Capture room ───────────────────────────────────
        _step("POST /capture/image (synthetic PNG)")
        png = _synthetic_png()
        try:
            cap = _http_post_multipart(
                f"{base_url}/capture/image",
                files=[("image", "demo.png", png, "image/png")],
                fields={
                    "session_id": "msB-cap",
                    "surface_kind": "menu",
                    "target_cefr": "B1",
                    "note": "Café in München",
                },
                timeout_s=60,
            )
        except Exception as exc:
            failed.append(f"capture/image failed: {exc!r}")
            cap = {}
        sha = cap.get("sha256") or ""
        if not sha:
            failed.append("capture response missing sha256")
        img_path = mem_root / "captures" / "img" / f"{sha}.png" if sha else None
        if img_path and not img_path.exists():
            failed.append(f"capture image not on disk: {img_path}")
        parquet = mem_root / "captures" / "index.parquet"
        if not parquet.exists():
            failed.append("captures/index.parquet missing")
        else:
            try:
                import pyarrow.parquet as pq

                rows = pq.read_table(parquet).to_pylist()
            except Exception as exc:
                failed.append(f"captures/index.parquet read failed: {exc!r}")
                rows = []
            if len(rows) != 1:
                failed.append(
                    f"captures/index.parquet has {len(rows)} rows (expected 1)"
                )
            else:
                _ok(
                    f"capture: sha={sha[:12]}… engine={cap.get('engine')} "
                    f"transcript='{(cap.get('transcript') or '')[:48]}…'"
                )

        # ─── 6. Pronunciation diff (B.8) ──────────────────────
        _step("POST /conversation/pronounce (synthetic WAV)")
        try:
            pron = _http_post_multipart(
                f"{base_url}/conversation/pronounce",
                files=[("audio", "learner.wav", _silent_wav(), "audio/wav")],
                fields={
                    "reference_text": "Guten Morgen.",
                    "target_cefr": "B1",
                },
                timeout_s=30,
            )
        except Exception as exc:
            failed.append(f"conversation/pronounce failed: {exc!r}")
            pron = {}
        engine = pron.get("engine") or ""
        if engine not in {"librosa", "skeleton"}:
            failed.append(f"pronounce engine='{engine}' (expected librosa|skeleton)")
        else:
            _ok(
                f"pronounce: engine={engine} overall={pron.get('overall')} "
                f"segments={len(pron.get('segments') or [])}"
            )

        # ─── 7. Evidence + factsheet + vector (B.13/B.14) ─────
        _step("POST /admin/evidence (drives factsheet + vector index)")
        try:
            ev_resp = _http_post(
                f"{base_url}/admin/evidence",
                {
                    "competency_id": "grammar.cases",
                    "quality": 0.85,
                    "surface_form": "Ich gehe in die Schule.",
                    "source": "agent_post",
                },
                timeout_s=10,
            )
        except Exception as exc:
            failed.append(f"admin/evidence failed: {exc!r}")
            ev_resp = {}
        if not ev_resp.get("evidence_count"):
            failed.append("admin/evidence missing evidence_count")
        sheet_path = mem_root / "facts" / "grammar.cases.md"
        if not sheet_path.exists():
            failed.append(f"factsheet missing: {sheet_path}")
        else:
            body = sheet_path.read_text(encoding="utf-8")
            if "## Confidence:" not in body or "Productive examples" not in body:
                failed.append("factsheet missing required sections")
            else:
                _ok(f"factsheet ok: {len(body)} bytes")

        # ─── 8. Close the konversation session → summaries ────
        _step("POST /admin/sessions/msB-konv/close (writes gist + summ)")
        try:
            _http_post(
                f"{base_url}/admin/sessions/msB-konv/close",
                {},
                timeout_s=15,
            )
        except Exception as exc:
            failed.append(f"close session failed: {exc!r}")
        gist_files = list((mem_root / "sessions").glob("msB-konv.gist.md"))
        summ_files = list((mem_root / "sessions").glob("msB-konv.summ.md"))
        if not gist_files:
            failed.append("sessions/msB-konv.gist.md missing")
        if not summ_files:
            failed.append("sessions/msB-konv.summ.md missing")
        if gist_files and summ_files:
            _ok(
                f"summaries ok: gist={gist_files[0].name} "
                f"summ={summ_files[0].name}"
            )

        # ─── 9. Compactor ──────────────────────────────────────
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
        if not (mem_root / "compaction" / "last_run.json").exists():
            failed.append("compaction/last_run.json missing")
        else:
            _ok(
                f"compactor done: phases={report.get('phases_completed')} "
                f"duration_ms={report.get('duration_ms')}"
            )

        # ─── 10. Vector index (B.14) ───────────────────────────
        _step("inspect data/memory/embeddings/")
        embed_dir = mem_root / "embeddings"
        if not embed_dir.exists():
            _ok("embeddings dir absent (fastembed not installed — Null fallback)")
        else:
            parquets = list(embed_dir.glob("*.parquet"))
            if parquets:
                _ok(
                    "embeddings: "
                    + ", ".join(f"{p.name} ({p.stat().st_size}B)" for p in parquets)
                )
            else:
                _ok("embeddings dir present but empty (Null backend)")

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
            print(tail[-2500:])

    print()
    if failed:
        print("=== MILESTONE B DEMO: FAIL ===")
        for f in failed:
            print(f" - {f}")
        return 1
    print("=== MILESTONE B DEMO: PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
