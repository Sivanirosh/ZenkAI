"""Capture service + router tests (PIVOT_ROADMAP §B.6/§B.18).

Covers:
- Image is persisted under ``captures/img/<sha256>.<ext>``.
- ``captures/index.parquet`` gets one row per call.
- ``captures/meta.jsonl`` mirror is written for the Compactor.
- A ``capture_done`` event lands in the per-session JSONL.
- ``/capture/image`` route returns the result + records the capture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.capture import service as capture_service
from backend.main import create_app
from backend.memory import manager as memory_manager
from backend.multimodal import vision_capture


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _stub_describe_capture(transcript: str = "Hallo Welt"):
    async def _fake(
        image_bytes,
        *,
        target_cefr="B1",
        surface_kind="text",
        note="",
        timeout=30.0,
    ):
        return vision_capture.VisionCaptureResult(
            transcript=transcript,
            surface_kind=surface_kind,
            words=[
                vision_capture.CaptureWord(
                    surface="Welt",
                    lemma="Welt",
                    pos="noun",
                    gender="die",
                    definition_de="Erde",
                    definition_en="world",
                )
            ],
            advice="Versuch das Wort laut auszusprechen.",
            engine="gemma_vision",
            model="gemma4:e4b",
        )

    return _fake


@pytest.mark.asyncio
async def test_process_capture_persists_image_and_index(
    monkeypatch, tmp_path: Path
):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    monkeypatch.setattr(
        capture_service, "describe_capture", _stub_describe_capture()
    )

    req = capture_service.CaptureRequest(
        image_bytes=PNG_BYTES,
        content_type="image/png",
        session_id="sess-cap",
        surface_kind="menu",
        target_competency_id="vocab.food",
    )
    result = await capture_service.process_capture(req)

    expected_sha = hashlib.sha256(PNG_BYTES).hexdigest()
    assert result.sha256 == expected_sha

    img_path = (
        tmp_path / "mira_memory" / "captures" / "img" / f"{expected_sha}.png"
    )
    assert img_path.exists() and img_path.read_bytes() == PNG_BYTES

    parquet_path = tmp_path / "mira_memory" / "captures" / "index.parquet"
    assert parquet_path.exists()
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).to_pylist()
    assert len(rows) == 1
    assert rows[0]["sha256"] == expected_sha
    assert rows[0]["surface_kind"] == "menu"
    assert rows[0]["target_competency_id"] == "vocab.food"

    meta_path = tmp_path / "mira_memory" / "captures" / "meta.jsonl"
    assert meta_path.exists()
    assert json.loads(meta_path.read_text(encoding="utf-8").splitlines()[0])[
        "sha256"
    ] == expected_sha

    memory_manager.get_memory().close_turn("sess-cap")
    sessions_dir = tmp_path / "mira_memory" / "sessions"
    jsonls = list(sessions_dir.glob("sess-cap.*.jsonl"))
    assert jsonls, "MemoryManager must have written the per-session ledger"
    assert any(
        json.loads(line).get("kind") == "capture_done"
        for f in jsonls
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


@pytest.mark.asyncio
async def test_process_capture_dedups_image_by_sha(
    monkeypatch, tmp_path: Path
):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    monkeypatch.setattr(
        capture_service, "describe_capture", _stub_describe_capture()
    )

    req = capture_service.CaptureRequest(
        image_bytes=PNG_BYTES,
        content_type="image/png",
        session_id="sess-dup",
        surface_kind="text",
    )
    r1 = await capture_service.process_capture(req)
    r2 = await capture_service.process_capture(req)
    assert r1.sha256 == r2.sha256

    img_dir = tmp_path / "mira_memory" / "captures" / "img"
    files = list(img_dir.glob("*"))
    assert len(files) == 1, "image dedup must not write the same sha twice"

    parquet_path = tmp_path / "mira_memory" / "captures" / "index.parquet"
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).to_pylist()
    assert len(rows) == 2, "every call appends a row even when sha is reused"


@pytest.mark.asyncio
async def test_process_capture_rejects_oversize_payload(
    monkeypatch, tmp_path: Path
):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    monkeypatch.setattr(
        capture_service, "describe_capture", _stub_describe_capture()
    )
    huge = PNG_BYTES + b"\x00" * (4 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        await capture_service.process_capture(
            capture_service.CaptureRequest(
                image_bytes=huge,
                content_type="image/png",
                session_id="sess-too-big",
            )
        )


@pytest.mark.asyncio
async def test_process_capture_rejects_non_image_content_type(
    monkeypatch, tmp_path: Path
):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    monkeypatch.setattr(
        capture_service, "describe_capture", _stub_describe_capture()
    )
    with pytest.raises(ValueError):
        await capture_service.process_capture(
            capture_service.CaptureRequest(
                image_bytes=PNG_BYTES,
                content_type="application/pdf",
                session_id="sess-pdf",
            )
        )


def test_capture_route_round_trip(monkeypatch, tmp_path: Path):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    monkeypatch.setattr(
        capture_service, "describe_capture", _stub_describe_capture("Speisekarte")
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/capture/image",
        files={"image": ("frame.png", PNG_BYTES, "image/png")},
        data={
            "session_id": "sess-route",
            "surface_kind": "menu",
            "target_cefr": "B1",
            "note": "Café",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "Speisekarte"
    assert body["surface_kind"] == "menu"
    assert body["sha256"] == hashlib.sha256(PNG_BYTES).hexdigest()
    assert body["engine"] == "gemma_vision"

    recent = client.get("/api/v1/capture/recent").json()
    assert recent["count"] >= 1
    assert recent["captures"][0]["sha256"] == body["sha256"]
