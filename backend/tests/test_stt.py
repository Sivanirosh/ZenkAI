"""faster-whisper wrapper tests (validation + mock transcription)."""

from __future__ import annotations

import pytest

from backend.services import stt_service


class _FakeSegment:
    def __init__(self, text: str, avg_logprob: float) -> None:
        self.text = text
        self.avg_logprob = avg_logprob


class _FakeInfo:
    def __init__(self) -> None:
        self.language = "de"
        self.duration = 1.25


class _FakeModel:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def transcribe(self, path, language=None, **_kwargs):  # noqa: ARG002
        return (
            iter(
                [
                    _FakeSegment("Warum benutzt Kafka", -0.2),
                    _FakeSegment("das Wort Ungeziefer?", -0.1),
                ]
            ),
            _FakeInfo(),
        )


@pytest.fixture(autouse=True)
def _reset_model():
    stt_service._model_cache = None
    yield
    stt_service._model_cache = None


@pytest.mark.asyncio
async def test_transcribe_returns_joined_text(monkeypatch):
    monkeypatch.setattr(stt_service, "_load_model", lambda: _FakeModel())

    result = await stt_service.transcribe(
        b"\x00" * 128, content_type="audio/webm"
    )

    assert "Kafka" in result.transcript
    assert "Ungeziefer" in result.transcript
    assert result.language == "de"
    assert 0.0 <= result.confidence <= 1.0
    assert result.duration_s == pytest.approx(1.25, rel=1e-3)


@pytest.mark.asyncio
async def test_transcribe_rejects_oversize_payload():
    with pytest.raises(ValueError):
        await stt_service.transcribe(
            b"\x00" * (10 * 1024 * 1024 + 1), content_type="audio/webm"
        )


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_payload():
    with pytest.raises(ValueError):
        await stt_service.transcribe(b"", content_type="audio/wav")


@pytest.mark.asyncio
async def test_transcribe_rejects_unknown_content_type():
    with pytest.raises(ValueError):
        await stt_service.transcribe(
            b"\x00" * 32, content_type="application/octet-stream"
        )


def test_load_model_raises_when_faster_whisper_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def bad_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", bad_import)

    with pytest.raises(stt_service.WhisperUnavailable):
        stt_service._load_model()
