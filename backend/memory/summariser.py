"""LLM session summariser (PIVOT_ROADMAP §B.12 / §7.6 pattern 2).

For every closed session we want two artefacts on disk:

- ``<sid>.gist.md``  — ≤ 30 tokens, one sentence (Atrium "letztes Mal: ...")
- ``<sid>.summ.md``  — ≤ 200 tokens, two-paragraph debrief

This module produces both. Two paths:

1. **LLM path** (preferred when reachable): build a compact prompt from
   the JSONL ledger (event count, tool histogram, last 10 textual
   events), ask Gemma 4 with ``temperature=0.1`` for strict-JSON
   ``{"gist": "...", "summ": "..."}``, validate word-counts.
2. **Mechanical fallback**: re-uses the deterministic template that
   already lives in ``compactor._mechanical_gist`` so we never produce
   an empty file; a session always gets *some* gist.

Idempotence: ``write_summary_files`` writes atomically (tmp+rename) and
leaves an existing file alone if the new payload is identical, so the
compactor can re-run without churning git.

This module is best-effort. It must not raise back to its callers (the
``MemoryManager.close_session`` and the ``compactor`` phase pickup) —
errors degrade to mechanical and are logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_GIST_WORD_LIMIT = 25     # ≈ 30 tokens after subwording
_SUMM_WORD_LIMIT = 170    # ≈ 200 tokens after subwording
_MAX_EVENTS_FOR_LLM = 10
_LLM_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class SessionSummary:
    """Result of a single summarisation pass."""

    session_id: str
    gist: str          # one short sentence
    summ: str          # two short paragraphs
    source: str        # "llm" or "mechanical"


# ─── Public API ─────────────────────────────────────────────────────────


def summarise_session(
    session_id: str,
    files: list[Path],
    *,
    provider: Any = None,
    timeout: float = _LLM_TIMEOUT_SECONDS,
) -> SessionSummary:
    """Build a SessionSummary from one session's JSONL files.

    ``provider`` is an optional ``LLMProvider`` (lets tests inject a
    fake). When ``None`` we lazily resolve via ``backend.llm.get_provider``.
    Failures fall back to the mechanical writer; we never raise.
    """
    facts = _extract_facts(session_id, files)
    if not files:
        return _mechanical(session_id, facts)

    if provider is None:
        try:
            from backend.llm import get_provider as _get_provider  # noqa: WPS433

            provider = _get_provider()
        except Exception as exc:  # pragma: no cover
            logger.debug("summariser: provider unavailable (%s)", exc)
            return _mechanical(session_id, facts)

    try:
        result = _try_llm(session_id, facts, provider, timeout=timeout)
    except Exception as exc:  # pragma: no cover
        logger.warning("summariser: LLM path raised (%s)", exc)
        result = None

    if result is None:
        return _mechanical(session_id, facts)
    return result


def write_summary_files(
    sessions_dir: Path,
    summary: SessionSummary,
) -> tuple[Path, Path]:
    """Atomically write ``<sid>.gist.md`` and ``<sid>.summ.md``.

    Returns the two destination paths. Files are only rewritten when the
    payload actually changes (idempotent re-run).
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    gist_path = sessions_dir / f"{summary.session_id}.gist.md"
    summ_path = sessions_dir / f"{summary.session_id}.summ.md"

    gist_body = _wrap_gist(summary)
    summ_body = _wrap_summ(summary)

    _atomic_write_if_changed(gist_path, gist_body)
    _atomic_write_if_changed(summ_path, summ_body)
    return gist_path, summ_path


def has_summary(sessions_dir: Path, session_id: str) -> bool:
    """True iff both gist + summ files exist for ``session_id``."""
    return (
        (sessions_dir / f"{session_id}.gist.md").exists()
        and (sessions_dir / f"{session_id}.summ.md").exists()
    )


# ─── Fact extraction ────────────────────────────────────────────────────


@dataclass
class _SessionFacts:
    session_id: str
    files: list[Path]
    event_count: int
    tools_used: dict[str, int]
    first_ts: Optional[str]
    last_ts: Optional[str]
    text_events: list[str]   # ["role: text", ...] last N
    duration_s: Optional[float]


def _extract_facts(session_id: str, files: list[Path]) -> _SessionFacts:
    event_count = 0
    tools_used: dict[str, int] = {}
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    text_events: list[str] = []

    for f in files:
        try:
            raw = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_count += 1
            ts = obj.get("occurred_at")
            if ts:
                first_ts = first_ts or ts
                last_ts = ts
            kind = obj.get("kind") or ""
            payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
            name = payload.get("name") if isinstance(payload, dict) else None
            if kind == "tool_intent" and isinstance(name, str):
                tools_used[name] = tools_used.get(name, 0) + 1
            text = ""
            if kind in {"user_message", "assistant_message", "transcript"}:
                text = str(payload.get("text") or "")
            elif kind == "tool_result" and isinstance(name, str):
                text = f"tool_result:{name}"
            if text:
                role = payload.get("role") or kind
                text_events.append(f"{role}: {text[:120]}")

    duration_s: Optional[float] = None
    if first_ts and last_ts and first_ts != last_ts:
        try:
            duration_s = (
                datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                - datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            ).total_seconds()
        except ValueError:
            duration_s = None

    return _SessionFacts(
        session_id=session_id,
        files=files,
        event_count=event_count,
        tools_used=tools_used,
        first_ts=first_ts,
        last_ts=last_ts,
        text_events=text_events[-_MAX_EVENTS_FOR_LLM:],
        duration_s=duration_s,
    )


# ─── Mechanical fallback ────────────────────────────────────────────────


def _mechanical(session_id: str, facts: _SessionFacts) -> SessionSummary:
    """Deterministic gist+summ — never empty, never raises."""
    tool_str = ", ".join(f"{k}×{v}" for k, v in sorted(facts.tools_used.items()))
    duration_part = (
        f" ({facts.duration_s:.0f}s)" if facts.duration_s is not None else ""
    )
    gist = (
        f"Sitzung {session_id[:8]}: {facts.event_count} Ereignisse{duration_part}."
    ) if facts.event_count else (
        f"Sitzung {session_id[:8]}: keine Ereignisse aufgezeichnet."
    )

    paragraphs: list[str] = []
    paragraphs.append(
        f"Diese Sitzung umfasste {facts.event_count} Ereignisse"
        + (f" über {facts.duration_s:.0f}s" if facts.duration_s is not None else "")
        + (f"; verwendete Tools: {tool_str}." if tool_str else ".")
    )
    if facts.text_events:
        sample = "; ".join(facts.text_events[-3:])
        paragraphs.append(f"Letzte Ausschnitte: {sample}")
    else:
        paragraphs.append("Keine textuellen Ausschnitte vorhanden.")
    summ = "\n\n".join(paragraphs)

    return SessionSummary(
        session_id=session_id,
        gist=_truncate_words(gist, _GIST_WORD_LIMIT),
        summ=_truncate_words(summ, _SUMM_WORD_LIMIT),
        source="mechanical",
    )


# ─── LLM path ───────────────────────────────────────────────────────────


_PROMPT_TEMPLATE = """Du bist Mira, eine Sprachlern-Tutorin. Schreibe eine kompakte Zusammenfassung dieser Sitzung.

Rohstatistik:
- Sitzungs-ID: {sid}
- Ereignisse: {events}
- Dauer (s): {duration}
- Tools: {tools}

Letzte Ausschnitte (chronologisch):
{snippets}

Antworte ausschließlich mit gültigem JSON in genau diesem Schema:
{{"gist": "<eine deutsche Sätze, max 25 Wörter>", "summ": "<zwei kurze Absätze, max 170 Wörter, durch Leerzeile getrennt>"}}

Ohne Markdown, ohne Erklärung — nur das JSON-Objekt."""


def _try_llm(
    session_id: str,
    facts: _SessionFacts,
    provider: Any,
    *,
    timeout: float,
) -> Optional[SessionSummary]:
    snippets = "\n".join(f"- {line}" for line in facts.text_events) or "- (keine)"
    tools = ", ".join(f"{k}×{v}" for k, v in sorted(facts.tools_used.items())) or "(keine)"
    prompt = _PROMPT_TEMPLATE.format(
        sid=session_id,
        events=facts.event_count,
        duration=f"{facts.duration_s:.1f}" if facts.duration_s is not None else "?",
        tools=tools,
        snippets=snippets,
    )

    try:
        from backend.services.runtime_config import (  # noqa: WPS433
            LlmOptions,
            get_llm_options,
        )

        base = get_llm_options()
        options = LlmOptions(model=base.model, think=False, temperature=0.1)
    except Exception:  # pragma: no cover
        return None

    try:
        raw = _run_async(provider.generate(prompt, options=options, timeout=timeout))
    except Exception as exc:  # pragma: no cover
        logger.warning("summariser: provider.generate raised (%s)", exc)
        return None
    if not raw:
        return None

    parsed = _parse_strict_json(raw)
    if not parsed:
        # one re-prompt attempt with a stricter wrapper
        retry = (
            "Deine vorherige Antwort war kein gültiges JSON. "
            "Antworte erneut, NUR mit dem Objekt — keine Markdown-Backticks.\n\n"
            + prompt
        )
        try:
            raw2 = _run_async(provider.generate(retry, options=options, timeout=timeout))
        except Exception:  # pragma: no cover
            raw2 = None
        parsed = _parse_strict_json(raw2 or "")

    if not parsed:
        return None

    gist = _truncate_words(str(parsed.get("gist", "")).strip(), _GIST_WORD_LIMIT)
    summ = _truncate_words(str(parsed.get("summ", "")).strip(), _SUMM_WORD_LIMIT)
    if not gist or not summ:
        return None

    return SessionSummary(
        session_id=session_id,
        gist=gist,
        summ=summ,
        source="llm",
    )


def _run_async(coro: Any) -> Any:
    """Run an awaitable to completion from sync code.

    ``MemoryManager.close_session`` is called from sync FastAPI handlers
    AND from the compactor — neither has an event loop in scope. We
    spin one up just for the call (cheap; this is best-effort and rare).
    """
    if coro is None:
        return None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Edge case (we're inside an async context already): use a
            # thread-pool runner so we don't deadlock.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_strict_json(text: str) -> Optional[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    # First try the whole thing.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Then any embedded JSON-looking blob.
    for match in _JSON_BLOCK_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and ("gist" in obj or "summ" in obj):
                return obj
        except json.JSONDecodeError:
            continue
    return None


# ─── Render helpers ─────────────────────────────────────────────────────


def _wrap_gist(summary: SessionSummary) -> str:
    return (
        f"# session {summary.session_id} — gist\n"
        f"\n"
        f"{summary.gist}\n"
        f"\n"
        f"_source: {summary.source}_\n"
    )


def _wrap_summ(summary: SessionSummary) -> str:
    return (
        f"# session {summary.session_id} — summary\n"
        f"\n"
        f"{summary.summ}\n"
        f"\n"
        f"_source: {summary.source}_\n"
    )


def _truncate_words(text: str, limit: int) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(",;:") + "…"


def _atomic_write_if_changed(path: Path, body: str) -> None:
    try:
        if path.exists() and path.read_text(encoding="utf-8") == body:
            return
    except OSError:  # pragma: no cover
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(body)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:  # pragma: no cover
                pass
        os.replace(tmp, path)
    except Exception:  # pragma: no cover
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


__all__ = [
    "SessionSummary",
    "has_summary",
    "summarise_session",
    "write_summary_files",
]
