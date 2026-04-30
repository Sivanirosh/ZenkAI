"""Piper TTS wrapper tests with a fake subprocess."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from backend.services import tts_service


class _FakeProc:
    def __init__(self, pcm: bytes, returncode: int = 0) -> None:
        self._pcm = pcm
        self._return = returncode
        self.returncode = returncode

    async def communicate(self, input: bytes | None = None):  # noqa: ARG002
        return self._pcm, b""

    async def wait(self):
        return self._return

    def kill(self) -> None:
        self.returncode = -9


@pytest.fixture(autouse=True)
def _clear_cache():
    tts_service.clear_cache()
    yield
    tts_service.clear_cache()


@pytest.fixture
def fake_piper(monkeypatch, tmp_path: Path):
    binary = tmp_path / "piper"
    model = tmp_path / "voice.onnx"
    binary.write_text("#!/bin/sh\n")
    model.write_text("voice")

    from backend.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "piper_binary", binary)
    monkeypatch.setattr(settings, "piper_model", model)
    return binary, model


@pytest.mark.asyncio
async def test_synthesise_wraps_pcm_with_wav_header(monkeypatch, fake_piper):
    pcm = b"\x01\x02\x03\x04" * 32

    async def fake_exec(*args, **kwargs):  # noqa: ARG001
        return _FakeProc(pcm, returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    chunks: list[bytes] = []
    async for chunk in tts_service.synthesise("Hallo", speed=1.0):
        chunks.append(chunk)

    wav = b"".join(chunks)
    assert wav.startswith(b"RIFF")
    assert b"WAVE" in wav[:12]
    assert wav.endswith(pcm)


@pytest.mark.asyncio
async def test_synthesise_uses_lru_cache(monkeypatch, fake_piper):
    pcm = b"\x10\x20" * 16
    call_count = {"n": 0}

    async def fake_exec(*args, **kwargs):  # noqa: ARG001
        call_count["n"] += 1
        return _FakeProc(pcm)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def drain():
        async for _ in tts_service.synthesise("Guten Tag", speed=1.0):
            pass

    await drain()
    await drain()
    await drain()

    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_synthesise_raises_when_binary_missing(monkeypatch, tmp_path: Path):
    from backend.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "piper_binary", tmp_path / "nope")
    monkeypatch.setattr(settings, "piper_model", tmp_path / "nope.onnx")

    with pytest.raises(tts_service.PiperUnavailable):
        async for _ in tts_service.synthesise("Hallo"):
            pass


@pytest.mark.asyncio
async def test_synthesise_raises_on_nonzero_exit(monkeypatch, fake_piper):
    async def fake_exec(*args, **kwargs):  # noqa: ARG001
        return _FakeProc(b"", returncode=7)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(tts_service.PiperUnavailable):
        async for _ in tts_service.synthesise("Hallo"):
            pass


@pytest.mark.asyncio
async def test_synthesise_noop_on_empty_text(fake_piper):
    collected: list[bytes] = []
    async for chunk in tts_service.synthesise("   "):
        collected.append(chunk)
    assert collected == []
