"""Tests for backend/multimodal/audio_capture.py (PIVOT_ROADMAP §B.8)."""

from __future__ import annotations

import asyncio
import io
import math
import struct
import wave
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.memory import manager as memory_manager
from backend.multimodal import audio_capture


# ─── Helpers: synthetic audio ───────────────────────────────────────────


def _silent_wav(duration_s: float = 0.5, sample_rate: int = 22_050) -> bytes:
    n = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def _sine_wav(
    freq_hz: float = 220.0, duration_s: float = 0.5, sample_rate: int = 22_050
) -> bytes:
    n = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            sample = int(0.4 * 32_767 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


# ─── Skeleton path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_skeleton_when_audio_too_short(monkeypatch):
    async def fake_ref(text: str):
        return _sine_wav(duration_s=0.5), Path("/tmp/ref.wav")

    monkeypatch.setattr(audio_capture, "_ensure_reference_wav", fake_ref)

    score = await audio_capture.score_pronunciation(
        "Guten Morgen.", _silent_wav(duration_s=0.05)
    )
    assert score.engine == "skeleton"
    assert score.overall is None
    assert score.segments == []


@pytest.mark.asyncio
async def test_returns_skeleton_when_piper_unavailable(monkeypatch):
    async def piper_fail(text: str):
        raise RuntimeError("piper missing")

    monkeypatch.setattr(audio_capture, "_ensure_reference_wav", piper_fail)
    score = await audio_capture.score_pronunciation(
        "Guten Morgen.", _sine_wav(duration_s=0.5)
    )
    assert score.engine == "skeleton"
    assert score.reference_audio_path is None


@pytest.mark.asyncio
async def test_returns_skeleton_when_librosa_missing(monkeypatch):
    async def fake_ref(text: str):
        return _sine_wav(), Path("/tmp/ref.wav")

    monkeypatch.setattr(audio_capture, "_ensure_reference_wav", fake_ref)
    monkeypatch.setattr(audio_capture, "_try_import_librosa", lambda: None)

    score = await audio_capture.score_pronunciation(
        "Guten Morgen.", _sine_wav(duration_s=0.5)
    )
    assert score.engine == "skeleton"


# ─── Happy path with stubbed librosa ────────────────────────────────────


class _FakeLibrosaSequence:
    @staticmethod
    def dtw(*, X, Y, metric="cosine", subseq=False):
        import numpy as np

        n = max(X.shape[1], Y.shape[1])
        wp = [(min(i, X.shape[1] - 1), min(i, Y.shape[1] - 1)) for i in range(n)]
        return np.zeros((X.shape[1], Y.shape[1])), wp


class _FakeLibrosaFeature:
    @staticmethod
    def mfcc(y, sr, n_mfcc=13, hop_length=512):
        import numpy as np

        # Deterministic 13 x 8 matrix derived from the audio length so
        # ref vs learner ranking still varies if pcm differs.
        cols = max(8, int(len(y) / max(hop_length, 1)))
        rng = np.random.default_rng(int(abs(y.sum()) * 100) % 10_000)
        return rng.standard_normal((n_mfcc, cols)).astype("float32")


class _FakeLibrosa:
    feature = _FakeLibrosaFeature()
    sequence = _FakeLibrosaSequence()

    @staticmethod
    def load(buf, sr=22_050, mono=True, res_type="kaiser_fast"):
        import numpy as np

        # We always come through _decode_to_mono's wave path for our
        # generated WAVs; this is just a safety net.
        return np.zeros(int(sr * 0.5), dtype="float32"), sr

    @staticmethod
    def resample(y, *, orig_sr, target_sr):
        return y


@pytest.mark.asyncio
async def test_happy_path_returns_score_and_segments(monkeypatch):
    async def fake_ref(text: str):
        return _sine_wav(freq_hz=220.0, duration_s=1.0), Path("/tmp/ref.wav")

    monkeypatch.setattr(audio_capture, "_ensure_reference_wav", fake_ref)
    monkeypatch.setattr(audio_capture, "_try_import_librosa", lambda: _FakeLibrosa)

    score = await audio_capture.score_pronunciation(
        "Guten Morgen Mira.", _sine_wav(freq_hz=220.0, duration_s=1.0)
    )
    assert score.engine == "librosa"
    assert score.overall is not None
    assert 0.0 <= score.overall <= 1.0
    assert [s.word for s in score.segments] == ["Guten", "Morgen", "Mira"]
    for seg in score.segments:
        assert 0.0 <= seg.score <= 1.0


# ─── Reference cache ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reference_cache_avoids_second_piper_call(monkeypatch, tmp_path):
    memory_manager.reset_for_tests(root=tmp_path / "audio_root")

    calls: list[str] = []

    async def fake_piper(text: str) -> bytes:
        calls.append(text)
        return _sine_wav(duration_s=0.5)

    monkeypatch.setattr(audio_capture, "_run_piper", fake_piper)

    wav1, path1 = await audio_capture._ensure_reference_wav("Guten Morgen.")
    wav2, path2 = await audio_capture._ensure_reference_wav("Guten Morgen.")
    assert calls == ["Guten Morgen."]
    assert path1 == path2
    assert wav1 == wav2
    assert path1.exists()


# ─── Route round-trip ───────────────────────────────────────────────────


def test_pronounce_route_round_trip(monkeypatch, tmp_path):
    memory_manager.reset_for_tests(root=tmp_path / "audio_root")

    async def fake_ref(text: str):
        return _sine_wav(duration_s=0.5), Path("/tmp/ref.wav")

    monkeypatch.setattr(audio_capture, "_ensure_reference_wav", fake_ref)
    monkeypatch.setattr(audio_capture, "_try_import_librosa", lambda: _FakeLibrosa)

    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/api/v1/conversation/pronounce",
        files={"audio": ("learner.wav", _sine_wav(duration_s=0.5), "audio/wav")},
        data={"reference_text": "Guten Morgen.", "target_cefr": "B1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] in {"librosa", "skeleton"}
    assert body["reference_text"] == "Guten Morgen."
    assert isinstance(body["segments"], list)


def test_pronounce_route_rejects_empty_audio(tmp_path):
    memory_manager.reset_for_tests(root=tmp_path / "audio_root")
    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/api/v1/conversation/pronounce",
        files={"audio": ("learner.wav", b"", "audio/wav")},
        data={"reference_text": "Hallo."},
    )
    assert r.status_code == 400


def test_pronounce_route_rejects_empty_reference(tmp_path):
    memory_manager.reset_for_tests(root=tmp_path / "audio_root")
    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/api/v1/conversation/pronounce",
        files={"audio": ("learner.wav", _silent_wav(duration_s=0.2), "audio/wav")},
        data={"reference_text": "  "},
    )
    assert r.status_code == 400
