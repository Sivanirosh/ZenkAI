"""Konversation runner + router tests (PIVOT_ROADMAP §B.5/§B.17).

Covers:
- ``run_turn`` persists the user transcript BEFORE the LLM call (B.17).
- Transcripts go to ``data/memory/konversation/transcripts.NNNN.jsonl``
  with rotation honoured.
- ``stt_service.transcribe_with_fallback`` defaults to faster-whisper,
  falls back to Gemma 4 audio only when ``LINGUAMATE_GEMMA_AUDIO=1`` is
  explicitly set (revised default, 2026-04-30 — see stt_service header),
  and finally returns an empty stub (engine='unavailable') when both fail.
- The ``/conversation/turn`` route returns user + assistant transcripts
  and a duration_ms.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.conversation import runner
from backend.llm import ollama as ollama_module
from backend.main import create_app
from backend.memory import manager as memory_manager
from backend.services import stt_service


# ─── stt_service.transcribe_with_fallback ───────────────────────────────


def _patch_provider(monkeypatch, provider) -> None:
    """Patch every ``get_provider`` binding the runtime might call."""
    import backend.llm as llm_pkg

    monkeypatch.setattr(llm_pkg, "get_provider", lambda: provider)
    monkeypatch.setattr(ollama_module, "get_provider", lambda: provider)


@pytest.mark.asyncio
async def test_fallback_prefers_whisper_by_default(monkeypatch):
    """Default order (post-2026-04-30): faster-whisper is tried first and
    Gemma audio is NOT consulted unless ``LINGUAMATE_GEMMA_AUDIO=1`` is set,
    so a refusal from a text-only Ollama model can no longer poison the
    transcript pipeline."""
    monkeypatch.delenv("LINGUAMATE_GEMMA_AUDIO", raising=False)

    gemma_called = {"hit": False}

    class _Provider:
        name = "fake"

        async def transcribe_audio(
            self, audio_bytes, *, content_type, language, timeout
        ):
            gemma_called["hit"] = True
            return "should not be called"

    _patch_provider(monkeypatch, _Provider())

    async def _fake_transcribe(audio_bytes, content_type, *, language="de"):
        return stt_service.STTResult(
            transcript="hello",
            confidence=0.7,
            language=language,
            duration_s=0.5,
            engine="whisper",
        )

    monkeypatch.setattr(stt_service, "transcribe", _fake_transcribe)

    res = await stt_service.transcribe_with_fallback(
        b"\x00" * 64, "audio/wav", language="de"
    )
    assert res.engine == "whisper"
    assert res.transcript == "hello"
    assert gemma_called["hit"] is False


@pytest.mark.asyncio
async def test_fallback_uses_gemma_audio_when_opted_in_and_whisper_missing(
    monkeypatch,
):
    """With the opt-in flag set AND Whisper unavailable, the chain falls
    through to Gemma 4 audio multimodal."""
    monkeypatch.setenv("LINGUAMATE_GEMMA_AUDIO", "1")

    class _Provider:
        name = "fake"

        async def transcribe_audio(
            self, audio_bytes, *, content_type, language, timeout
        ):
            assert content_type == "audio/wav"
            assert language == "de"
            return "Guten Morgen."

    _patch_provider(monkeypatch, _Provider())

    async def _whisper_missing(audio_bytes, content_type, *, language="de"):
        raise stt_service.WhisperUnavailable(
            detail="missing", install_hint="install"
        )

    monkeypatch.setattr(stt_service, "transcribe", _whisper_missing)

    res = await stt_service.transcribe_with_fallback(
        b"\x00" * 64, "audio/wav", language="de"
    )
    assert res.engine == "gemma_audio"
    assert "Guten Morgen" in res.transcript


@pytest.mark.asyncio
async def test_fallback_returns_unavailable_when_both_fail(monkeypatch):
    monkeypatch.delenv("LINGUAMATE_GEMMA_AUDIO", raising=False)

    class _NoAudioProvider:
        name = "fake"

    _patch_provider(monkeypatch, _NoAudioProvider())

    async def _whisper_missing(audio_bytes, content_type, *, language="de"):
        raise stt_service.WhisperUnavailable(
            detail="missing", install_hint="install"
        )

    monkeypatch.setattr(stt_service, "transcribe", _whisper_missing)

    res = await stt_service.transcribe_with_fallback(
        b"\x00" * 64, "audio/wav", language="de"
    )
    assert res.engine == "unavailable"
    assert res.transcript == ""


# ─── runner.run_turn (B.17 — persist BEFORE LLM) ────────────────────────


@pytest.mark.asyncio
async def test_run_turn_persists_user_transcript_before_llm_runs(
    monkeypatch, tmp_path: Path
):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")

    async def _fake_stt(audio_bytes, content_type, *, language="de"):
        return stt_service.STTResult(
            transcript="Ich möchte Brot kaufen.",
            confidence=0.9,
            language=language,
            duration_s=1.4,
            engine="gemma_audio",
        )

    monkeypatch.setattr(stt_service, "transcribe_with_fallback", _fake_stt)

    seen_files_at_llm_call = {}

    class _Provider:
        async def generate(self, prompt, *, options, timeout=20.0):
            konv_dir = tmp_path / "mira_memory" / "konversation"
            seen_files_at_llm_call["files"] = sorted(
                str(p.name) for p in konv_dir.glob("*.jsonl")
            )
            seen_files_at_llm_call["lines"] = sum(
                1
                for p in konv_dir.glob("*.jsonl")
                for _ in p.read_text(encoding="utf-8").splitlines()
            )
            return "Klar, sag mir bitte, wie viele Brötchen."

    monkeypatch.setattr(runner, "get_provider", lambda: _Provider())

    req = runner.ConversationTurnRequest(
        audio_bytes=b"\x00" * 64,
        content_type="audio/wav",
        session_id="sess-abc",
        scenario="Bäckerei",
    )
    result = await runner.run_turn(req)

    assert result.user_transcript.text == "Ich möchte Brot kaufen."
    assert "Brötchen" in result.assistant_reply.text
    # B.17 guarantee: a transcript existed on disk *before* the LLM ran.
    assert seen_files_at_llm_call.get("lines") == 1

    konv_dir = tmp_path / "mira_memory" / "konversation"
    transcript_files = list(konv_dir.glob("*.jsonl"))
    assert transcript_files, "transcripts.NNNN.jsonl must exist"
    body = transcript_files[0].read_text(encoding="utf-8").splitlines()
    assert len(body) == 2
    roles = [json.loads(l)["role"] for l in body]
    assert roles == ["user", "assistant"]


@pytest.mark.asyncio
async def test_run_turn_falls_back_when_llm_returns_empty(
    monkeypatch, tmp_path: Path
):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")

    async def _fake_stt(audio_bytes, content_type, *, language="de"):
        return stt_service.STTResult(
            transcript="",
            confidence=0.0,
            language=language,
            duration_s=0.0,
            engine="unavailable",
        )

    monkeypatch.setattr(stt_service, "transcribe_with_fallback", _fake_stt)

    class _Provider:
        async def generate(self, prompt, *, options, timeout=20.0):
            return None

    monkeypatch.setattr(runner, "get_provider", lambda: _Provider())

    result = await runner.run_turn(
        runner.ConversationTurnRequest(
            audio_bytes=b"\x00" * 64,
            content_type="audio/wav",
            session_id="sess-fb",
        )
    )
    assert result.user_transcript.text == ""
    assert "noch einmal" in result.assistant_reply.text


# ─── /conversation/turn router ──────────────────────────────────────────


def test_conversation_turn_route_returns_both_transcripts(
    monkeypatch, tmp_path: Path
):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")

    async def _fake_stt(audio_bytes, content_type, *, language="de"):
        return stt_service.STTResult(
            transcript="Wie spät ist es?",
            confidence=0.95,
            language=language,
            duration_s=0.8,
            engine="gemma_audio",
        )

    monkeypatch.setattr(stt_service, "transcribe_with_fallback", _fake_stt)

    class _Provider:
        async def generate(self, prompt, *, options, timeout=20.0):
            return "Es ist kurz nach drei."

    monkeypatch.setattr(runner, "get_provider", lambda: _Provider())

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/conversation/turn",
        files={"audio": ("clip.wav", b"\x00" * 64, "audio/wav")},
        data={"session_id": "sess-route", "scenario": "Smalltalk"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user_transcript"]["text"] == "Wie spät ist es?"
    assert "drei" in body["assistant_reply"]["text"]
    assert body["duration_ms"] >= 0
    assert body["stt_engine"] == "gemma_audio"


def test_conversation_turn_rejects_empty_audio():
    client = TestClient(create_app())
    response = client.post(
        "/api/v1/conversation/turn",
        files={"audio": ("clip.wav", b"", "audio/wav")},
        data={"session_id": "sess-empty"},
    )
    assert response.status_code == 400


def test_conversation_recent_returns_persisted_transcripts(tmp_path: Path):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    konv_dir = tmp_path / "mira_memory" / "konversation"
    konv_dir.mkdir(parents=True, exist_ok=True)
    (konv_dir / "transcripts.0001.jsonl").write_text(
        json.dumps(
            {
                "id": "u1",
                "session_id": "s1",
                "role": "user",
                "text": "Hallo",
                "occurred_at": "2026-04-30T10:00:00.000",
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "a1",
                "session_id": "s1",
                "role": "assistant",
                "text": "Hallo, schön dich zu sehen.",
                "occurred_at": "2026-04-30T10:00:01.000",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app())
    response = client.get("/api/v1/conversation/recent?limit=10")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    texts = [t["text"] for t in body["transcripts"]]
    assert "Hallo" in texts


# ─── Rotation ───────────────────────────────────────────────────────────


def test_transcript_rotation_creates_new_file_above_threshold(
    monkeypatch, tmp_path: Path
):
    memory_manager.reset_for_tests(root=tmp_path / "mira_memory")
    monkeypatch.setattr(runner, "_TRANSCRIPT_ROTATION_BYTES", 200)

    for i in range(40):
        runner._append_transcript(
            runner.TranscriptEntry(
                id=f"u-{i}",
                session_id="s-rot",
                role="user",
                text="x" * 80,
                occurred_at=f"2026-04-30T10:00:{i:02d}.000",
            )
        )
    konv_dir = tmp_path / "mira_memory" / "konversation"
    files = sorted(konv_dir.glob("transcripts.*.jsonl"))
    assert len(files) >= 2, "rotation must create at least one new file"
