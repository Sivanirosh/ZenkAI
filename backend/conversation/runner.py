"""Konversation runner — the speech-loop chokepoint (PIVOT_ROADMAP §B.5/B.17).

One ``run_turn`` call:

1. Validates the audio blob.
2. Transcribes via the Gemma-audio path if available, otherwise
   ``faster-whisper`` (see ``stt_service.transcribe_with_fallback``).
3. **Persists the transcript event BEFORE calling the LLM** (B.17).
   That way a crash mid-reply still preserves what the learner said —
   the most expensive thing in the loop to recover.
4. Asks Gemma 4 for a short, level-appropriate reply via the standard
   ``OllamaProvider.generate`` (text). TTS is a separate streaming
   endpoint; we only return the reply text + metadata here so the
   frontend can fire the TTS stream in parallel with showing the text.
5. Persists the assistant reply event after the LLM returns.

We do NOT call Piper inside ``run_turn``. That belongs in
``/conversation/tts`` (= the existing ``/voice/tts`` route) so the
client can stream audio progressively while we move on.

All transcripts go to ``data/memory/konversation/transcripts.jsonl``
(rotates at 5 MB) so B.11's ``start_conversation`` RecallPolicy has
real Tier-2 source data starting on day one.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from backend.llm import get_provider
from backend.memory.manager import Event, get_memory
from backend.services import stt_service
from backend.services.runtime_config import get_llm_options

logger = logging.getLogger(__name__)


_TRANSCRIPT_ROTATION_BYTES = 5 * 1024 * 1024  # 5 MB
_TRANSCRIPT_NAME_PREFIX = "transcripts"
_LOCKS: dict[str, threading.Lock] = {}
_LOCK_TABLE = threading.Lock()


# ─── Public dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ConversationTurnRequest:
    """Inputs for one turn of the Konversation loop."""

    audio_bytes: bytes
    content_type: str
    session_id: str
    competency_id: Optional[str] = None
    scenario: Optional[str] = None
    target_cefr: Optional[str] = None
    language: str = "de"


@dataclass
class TranscriptEntry:
    """One line in ``konversation/transcripts.jsonl``."""

    id: str
    session_id: str
    role: str               # "user" | "assistant" | "system"
    text: str
    occurred_at: str
    competency_id: Optional[str] = None
    scenario: Optional[str] = None
    audio_seconds: Optional[float] = None
    stt_engine: Optional[str] = None
    confidence: Optional[float] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False) + "\n"


@dataclass
class ConversationTurnResult:
    """Returned to the router after one full turn."""

    user_transcript: TranscriptEntry
    assistant_reply: TranscriptEntry
    duration_ms: int
    stt_engine: str
    llm_model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_transcript": self.user_transcript.__dict__,
            "assistant_reply": self.assistant_reply.__dict__,
            "duration_ms": self.duration_ms,
            "stt_engine": self.stt_engine,
            "llm_model": self.llm_model,
        }


# ─── Public entry point ─────────────────────────────────────────────────


async def run_turn(req: ConversationTurnRequest) -> ConversationTurnResult:
    """Execute one Konversation turn end-to-end.

    Persists transcripts to disk *before* and *after* the LLM call so a
    mid-flight crash never loses the user's audio (B.17).
    """
    started = datetime.now(timezone.utc)

    # 1. transcribe — Gemma audio first, faster-whisper fallback.
    stt = await stt_service.transcribe_with_fallback(
        req.audio_bytes,
        req.content_type,
        language=req.language,
    )
    user_transcript = TranscriptEntry(
        id=f"utx-{uuid.uuid4().hex[:12]}",
        session_id=req.session_id,
        role="user",
        text=stt.transcript,
        occurred_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        competency_id=req.competency_id,
        scenario=req.scenario,
        audio_seconds=stt.duration_s,
        stt_engine=stt.engine,
        confidence=stt.confidence,
    )

    # 2. PERSIST FIRST (B.17) — this is the durability guarantee.
    _append_transcript(user_transcript)
    mm = get_memory()
    mm.write_event(
        Event(
            session_id=req.session_id,
            kind="conversation_user",
            payload={
                "transcript_id": user_transcript.id,
                "text": user_transcript.text,
                "competency_id": user_transcript.competency_id,
                "stt_engine": stt.engine,
                "confidence": stt.confidence,
            },
        )
    )

    # 3. Ask Gemma 4 for a reply.
    options = get_llm_options()
    prompt = _build_reply_prompt(req, user_transcript)
    try:
        raw_reply = await get_provider().generate(
            prompt, options=options, timeout=30.0
        )
    except Exception as exc:  # pragma: no cover — provider already swallows
        logger.warning("Konversation LLM call failed: %s", exc)
        raw_reply = None

    reply_text = (raw_reply or "").strip()
    if not reply_text:
        reply_text = _fallback_reply(req, user_transcript)

    assistant = TranscriptEntry(
        id=f"atx-{uuid.uuid4().hex[:12]}",
        session_id=req.session_id,
        role="assistant",
        text=reply_text,
        occurred_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        competency_id=req.competency_id,
        scenario=req.scenario,
        stt_engine=None,
    )
    _append_transcript(assistant)
    mm.write_event(
        Event(
            session_id=req.session_id,
            kind="conversation_assistant",
            payload={
                "transcript_id": assistant.id,
                "text": assistant.text,
                "model": options.model,
                "in_reply_to": user_transcript.id,
            },
        )
    )

    # 4. B.14 — fire-and-forget index both transcripts so future
    # ``start_conversation`` calls can semantically retrieve them.
    try:
        from backend.memory import store_vector as _store_vector  # noqa: WPS433

        rows: list[_store_vector.VectorRow] = []
        for entry in (user_transcript, assistant):
            if not entry.text.strip():
                continue
            rows.append(
                _store_vector.VectorRow(
                    id=entry.id,
                    text=entry.text,
                    source="transcripts",
                    metadata={
                        "role": entry.role,
                        "session_id": entry.session_id,
                        "scenario": entry.scenario or "",
                        "competency_id": entry.competency_id or "",
                    },
                    ts=entry.occurred_at,
                )
            )
        if rows:
            _store_vector.get_index().add(rows)
    except Exception as exc:  # pragma: no cover — never block the turn
        logger.debug("vector index add (transcripts) failed: %s", exc)

    finished = datetime.now(timezone.utc)
    duration_ms = int((finished - started).total_seconds() * 1000)

    return ConversationTurnResult(
        user_transcript=user_transcript,
        assistant_reply=assistant,
        duration_ms=duration_ms,
        stt_engine=stt.engine,
        llm_model=options.model,
    )


async def stream_turn(req: ConversationTurnRequest) -> AsyncIterator[str]:
    """Streaming variant of run_turn — yields SSE blocks.

    Event sequence emitted on the wire:

        event: transcribed
        data: {"id":"...", "text":"...", "stt_engine":"whisper", ...}

        event: token
        data: {"delta":"Guten "}

        event: token
        data: {"delta":"Tag!"}

        event: done
        data: {"assistant_id":"...", "assistant_text":"...",
               "stt_engine":"whisper", "llm_model":"gemma4:e4b",
               "duration_ms": 1234}

    The frontend shows the user bubble immediately on ``transcribed``,
    builds the assistant bubble token-by-token, then plays TTS on the
    complete text from ``done``.
    """
    started = datetime.now(timezone.utc)

    # 1. STT — same path as run_turn.
    stt = await stt_service.transcribe_with_fallback(
        req.audio_bytes,
        req.content_type,
        language=req.language,
    )
    user_transcript = TranscriptEntry(
        id=f"utx-{uuid.uuid4().hex[:12]}",
        session_id=req.session_id,
        role="user",
        text=stt.transcript,
        occurred_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        competency_id=req.competency_id,
        scenario=req.scenario,
        audio_seconds=stt.duration_s,
        stt_engine=stt.engine,
        confidence=stt.confidence,
    )

    # 2. Persist user turn FIRST (B.17), then emit to client immediately.
    _append_transcript(user_transcript)
    mm = get_memory()
    mm.write_event(
        Event(
            session_id=req.session_id,
            kind="conversation_user",
            payload={
                "transcript_id": user_transcript.id,
                "text": user_transcript.text,
                "stt_engine": stt.engine,
                "confidence": stt.confidence,
            },
        )
    )
    yield _sse("transcribed", user_transcript.__dict__)

    # 3. Stream LLM reply token by token.
    options = get_llm_options()
    prompt = _build_reply_prompt(req, user_transcript)
    messages = [
        {"role": "system", "content": (
            "Du bist Mira, eine ruhige Deutschtutorin. Antworte auf Deutsch "
            "in höchstens drei Sätzen. Keine Markdown-Formatierung."
        )},
        {"role": "user", "content": prompt},
    ]
    reply_chunks: list[str] = []
    try:
        stream = get_provider().stream_chat(messages, options=options, timeout=30.0)
        async for chunk in stream:
            if chunk:
                reply_chunks.append(chunk)
                yield _sse("token", {"delta": chunk})
    except Exception as exc:  # pragma: no cover
        logger.warning("Konversation stream failed: %s", exc)

    reply_text = "".join(reply_chunks).strip()
    if not reply_text:
        reply_text = _fallback_reply(req, user_transcript)
        yield _sse("token", {"delta": reply_text})

    # 4. Persist assistant reply.
    assistant = TranscriptEntry(
        id=f"atx-{uuid.uuid4().hex[:12]}",
        session_id=req.session_id,
        role="assistant",
        text=reply_text,
        occurred_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        competency_id=req.competency_id,
        scenario=req.scenario,
        stt_engine=None,
    )
    _append_transcript(assistant)
    mm.write_event(
        Event(
            session_id=req.session_id,
            kind="conversation_assistant",
            payload={
                "transcript_id": assistant.id,
                "text": assistant.text,
                "model": options.model,
                "in_reply_to": user_transcript.id,
            },
        )
    )

    # 5. Best-effort vector index.
    try:
        from backend.memory import store_vector as _store_vector  # noqa: WPS433

        rows = [
            _store_vector.VectorRow(
                id=e.id,
                text=e.text,
                source="transcripts",
                metadata={
                    "role": e.role,
                    "session_id": e.session_id,
                    "scenario": e.scenario or "",
                    "competency_id": e.competency_id or "",
                },
                ts=e.occurred_at,
            )
            for e in (user_transcript, assistant)
            if e.text.strip()
        ]
        if rows:
            _store_vector.get_index().add(rows)
    except Exception as exc:  # pragma: no cover
        logger.debug("vector index add failed: %s", exc)

    finished = datetime.now(timezone.utc)
    duration_ms = int((finished - started).total_seconds() * 1000)
    yield _sse(
        "done",
        {
            "user_transcript": user_transcript.__dict__,
            "assistant_reply": assistant.__dict__,
            "stt_engine": stt.engine,
            "llm_model": options.model,
            "duration_ms": duration_ms,
        },
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─── Transcript helpers (consumed by RecallPolicy + tests) ──────────────


def list_recent_transcripts(
    *, limit: int = 20, root: Optional[Path] = None
) -> list[TranscriptEntry]:
    """Return the most recent transcripts (newest first) for a budget.

    Reads every rotated ``transcripts.<NNNN>.jsonl`` file and yields up
    to ``limit`` entries. Used by the ``start_conversation``
    RecallPolicy and by the demo / inspector.
    """
    files = _list_transcript_files(root)
    entries: list[TranscriptEntry] = []
    for f in files:
        try:
            for raw in f.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                entries.append(_entry_from_dict(obj))
        except OSError:
            continue
    entries.sort(key=lambda e: e.occurred_at, reverse=True)
    return entries[: max(0, int(limit))]


# ─── Private helpers ────────────────────────────────────────────────────


def _build_reply_prompt(
    req: ConversationTurnRequest, user: TranscriptEntry
) -> str:
    cefr = req.target_cefr or "B1"
    scenario = req.scenario or "ein freundliches Alltagsgespräch"
    comp = req.competency_id or "ungebunden"
    return (
        "Du bist Mira, eine ruhige Deutschtutorin. Antworte AUF DEUTSCH in "
        f"höchstens drei Sätzen, CEFR-Niveau {cefr}. Szenario: {scenario}. "
        f"Aktuelle Kompetenz: {comp}. Sprich natürlich, stelle bei Bedarf eine "
        "kurze Rückfrage. Keine Markdown-Formatierung, keine Listen.\n\n"
        f"Lernende: {user.text or '(unverstanden)'}\n\nMira:"
    )


def _fallback_reply(
    req: ConversationTurnRequest, user: TranscriptEntry
) -> str:
    if not user.text:
        return (
            "Ich habe dich gerade nicht gehört. Sag das bitte noch einmal — "
            "etwas lauter ist genug."
        )
    snippet = user.text[:60]
    return (
        "Ich höre dich. Lass uns das Schritt für Schritt durchgehen: "
        f"Was meinst du mit „{snippet}\u201c?"
    )


def _append_transcript(entry: TranscriptEntry) -> None:
    """Append one transcript line to the active rotated JSONL file."""
    line = entry.to_json_line()
    line_bytes = len(line.encode("utf-8"))
    path = _current_transcript_path(additional_bytes=line_bytes)
    lock = _lock_for(str(path))
    with lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:  # pragma: no cover
                    pass
        except OSError as exc:  # pragma: no cover
            logger.warning("transcript append failed at %s: %s", path, exc)


def _entry_from_dict(obj: dict[str, Any]) -> TranscriptEntry:
    return TranscriptEntry(
        id=str(obj.get("id") or ""),
        session_id=str(obj.get("session_id") or ""),
        role=str(obj.get("role") or "user"),
        text=str(obj.get("text") or ""),
        occurred_at=str(obj.get("occurred_at") or ""),
        competency_id=obj.get("competency_id"),
        scenario=obj.get("scenario"),
        audio_seconds=obj.get("audio_seconds"),
        stt_engine=obj.get("stt_engine"),
        confidence=obj.get("confidence"),
        extras=dict(obj.get("extras") or {}),
    )


def _konversation_dir(root: Optional[Path] = None) -> Path:
    mm = get_memory() if root is None else None
    base = root or mm._root  # type: ignore[union-attr]
    return Path(base) / "konversation"


def _list_transcript_files(root: Optional[Path] = None) -> list[Path]:
    d = _konversation_dir(root)
    if not d.exists():
        return []
    return sorted(d.glob(f"{_TRANSCRIPT_NAME_PREFIX}.*.jsonl"))


def _current_transcript_path(*, additional_bytes: int = 0) -> Path:
    d = _konversation_dir()
    files = sorted(d.glob(f"{_TRANSCRIPT_NAME_PREFIX}.*.jsonl"))
    if not files:
        return d / f"{_TRANSCRIPT_NAME_PREFIX}.0001.jsonl"
    last = files[-1]
    try:
        size = last.stat().st_size
    except OSError:
        size = 0
    if size + additional_bytes > _TRANSCRIPT_ROTATION_BYTES and size > 0:
        try:
            seq = int(last.stem.rsplit(".", 1)[1]) + 1
        except (ValueError, IndexError):
            seq = len(files) + 1
        return d / f"{_TRANSCRIPT_NAME_PREFIX}.{seq:04d}.jsonl"
    return last


def _lock_for(key: str) -> threading.Lock:
    with _LOCK_TABLE:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
    return lock
