"""LLM prompts + the legacy facade over the new ``backend.llm`` provider layer.

All prompt templates live here as named constants (AGENT.md rule). No other
module may inline a prompt. When modifying a prompt, add a comment with the
date and rationale immediately above the constant.

This module used to talk to Ollama directly; it now delegates to
``backend.llm.OllamaProvider``. The public API (``annotate_word``,
``stream_answer``, ``probe_ollama``) is preserved exactly so the existing
routers, tests, and runtime configuration keep working unchanged. The
private ``_ThinkStripper`` / ``_strip_thinking`` symbols are re-exported
because the legacy think-stripper tests import them by path.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, Optional

import httpx  # noqa: F401 — kept so monkeypatch.setattr(httpx, "AsyncClient", ...) in tests still patches the same module

from backend.config import get_settings
from backend.llm.ollama import (
    ThinkStripper as _ThinkStripper,
    detect_missing_model as _detect_missing_model,
    get_provider,
    strip_thinking as _strip_thinking,
)
from backend.llm.provider import ProbeReport
from backend.models import db
from backend.models.pydantic_models import ChatTurn, WordAnnotation
from backend.services import corpus_service, rag_service
from backend.services.runtime_config import get_llm_options

logger = logging.getLogger(__name__)


# Re-exports for legacy callers / tests that import these names from this
# module path. The actual implementations live in backend.llm.ollama.
__all__ = [
    "LITERARY_QA_PROMPT",
    "WORD_ANNOTATION_PROMPT",
    "GRAMMAR_PROMPT",
    "annotate_word",
    "stream_answer",
    "probe_ollama",
    "_ThinkStripper",
    "_strip_thinking",
]


# ─── Prompt constants (AGENT.md: only here, never inlined) ───────────────


# 2025-04-21 — initial prompt, copied verbatim from CLAUDE.md.
LITERARY_QA_PROMPT = """
Du bist ein einfühlsamer Literaturtutor für Deutschlernende auf B1–B2-Niveau.
Der Lernende liest gerade die folgende Passage:

PASSAGE:
{paragraph}

KONTEXT (benachbarte Absätze):
{rag_context}

WERK: {work_title} von {author} ({year})
EPOCHE: {epoch}

Beantworte die Frage des Lernenden.
- Auf Deutsch, aber erkläre schwierige Wörter kurz auf Englisch in Klammern.
- Halte dich kurz (max. 4 Sätze), es sei denn, die Frage erfordert mehr.
- Bei grammatischen Fragen: zeige die Struktur mit einem klaren Beispiel.
- Bei literarischen Fragen: verbinde den Text mit dem historischen Kontext.

FRAGE: {question}
""".strip()


# 2025-04-21 — initial prompt, copied verbatim from CLAUDE.md.
WORD_ANNOTATION_PROMPT = """
Erkläre das deutsche Wort „{word}" für einen B1-Lernenden.
Es erscheint in diesem Satz: „{sentence}"
Grammatikalische Rolle: {grammatical_role}, Kasus: {case_label}

Antworte NUR als JSON ohne Markdown-Formatierung:
{{
  "definition_de": "kurze Definition auf einfachem Deutsch",
  "definition_en": "English translation",
  "literary_note": "one sentence on why Kafka/Goethe/etc. chose this word",
  "etymology": "brief etymology note if interesting, else null",
  "related_words": ["word1", "word2", "word3"]
}}
""".strip()


# 2025-04-21 — initial prompt for sentence-level grammar analysis.
GRAMMAR_PROMPT = """
Analysiere diesen deutschen Satz für einen B1-Lernenden:

SATZ: „{sentence}"

Gib die Analyse als JSON zurück:
{{
  "subject": "...",
  "verb": "...",
  "objects": ["..."],
  "subordinate_clauses": ["..."],
  "tense": "...",
  "mood": "Indikativ / Konjunktiv / Imperativ",
  "gloss_en": "short English paraphrase"
}}
""".strip()


# ─── Annotation: provider.generate + DuckDB-backed mock fallback ─────────


def _parse_annotation_json(raw: str) -> Optional[dict]:
    """Extract the first JSON object from an LLM response."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _lookup_word(lemma_or_surface: str) -> Optional[dict]:
    """Try several keys to find a stored word record for offline mocking."""
    key_lower = lemma_or_surface.lower()
    row = db.cursor().execute(
        """
        SELECT id, lemma, pos, gender, plural_form, etymology,
               definition_de, definition_en
        FROM words
        WHERE id = ? OR LOWER(lemma) = ? OR LOWER(lemma) = ?
        LIMIT 1
        """,
        [key_lower, key_lower, key_lower.rstrip("s")],
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0],
        "lemma": row[1],
        "pos": row[2],
        "gender": row[3],
        "plural_form": row[4],
        "etymology": row[5],
        "definition_de": row[6],
        "definition_en": row[7],
    }


def _mock_annotation(word: str, sentence: str) -> WordAnnotation:
    """Deterministic fallback used when Ollama is unreachable."""
    stored = _lookup_word(word)
    if stored is not None:
        lemma = stored["lemma"]
        definition_de = (
            stored.get("definition_de")
            or f"{lemma} — ein Wort aus dem literarischen Kontext."
        )
        definition_en = stored.get("definition_en") or f"(no English gloss for {lemma})"
        etymology = stored.get("etymology")
    else:
        lemma = word
        definition_de = (
            f"\u201e{word}\u201c \u2014 Bedeutung offline nicht verfuegbar. "
            "Starte Ollama fuer eine AI-generierte Erklaerung."
        )
        definition_en = (
            f"'{word}' \u2014 offline definition unavailable. "
            "Start Ollama for an AI-generated explanation."
        )
        etymology = None

    return WordAnnotation(
        definition_de=definition_de,
        definition_en=definition_en,
        literary_note=(
            "Kontextuelle Bedeutung: Das Wort erscheint in diesem Satz in einer "
            "spezifischen literarischen Funktion."
        ),
        etymology=etymology,
        related_words=[],
        source="mock",
    )


async def annotate_word(
    word: str,
    sentence: str,
    grammatical_role: Optional[str] = None,
    case_label: Optional[str] = None,
) -> WordAnnotation:
    """Annotate a word via the LLM provider, falling back to a deterministic mock."""
    prompt = WORD_ANNOTATION_PROMPT.format(
        word=word,
        sentence=sentence,
        grammatical_role=grammatical_role or "n/a",
        case_label=case_label or "n/a",
    )
    raw = await get_provider().generate(prompt, options=get_llm_options())
    if raw is None:
        return _mock_annotation(word, sentence)

    parsed = _parse_annotation_json(_strip_thinking(raw))
    if parsed is None:
        logger.warning("Could not parse LLM JSON response for '%s'", word)
        return _mock_annotation(word, sentence)

    stored = _lookup_word(word)
    return WordAnnotation(
        definition_de=parsed.get("definition_de")
        or (stored.get("definition_de") if stored else "")
        or word,
        definition_en=parsed.get("definition_en")
        or (stored.get("definition_en") if stored else "")
        or word,
        literary_note=parsed.get("literary_note"),
        etymology=parsed.get("etymology")
        or (stored.get("etymology") if stored else None),
        related_words=list(parsed.get("related_words") or []),
        source="ollama",
    )


# ─── Streaming literary Q&A (delegates to provider.stream_chat) ──────────


def _build_qa_prompt(
    question: str,
    paragraph_id: Optional[str],
) -> str:
    """Build a LITERARY_QA_PROMPT-filled prompt string."""
    passage = ""
    rag_context = ""
    work_title = ""
    author = ""
    year = ""
    epoch = ""

    if paragraph_id:
        paragraph = corpus_service.get_paragraph(paragraph_id)
        if paragraph is not None:
            passage = paragraph.text
            neighbours = rag_service.retrieve(paragraph_id, top_k=5)
            rag_context = rag_service.format_context(
                [n for n in neighbours if n.id != paragraph_id],
                anchor_id=paragraph_id,
            )
            work = corpus_service.get_work(paragraph.work_id)
            if work is not None:
                work_title = work.title
                author = work.author
                year = str(work.year) if work.year else ""
                epoch = work.epoch or ""

    return LITERARY_QA_PROMPT.format(
        paragraph=passage or "(keine Passage ausgewählt)",
        rag_context=rag_context or "(kein zusätzlicher Kontext verfügbar)",
        work_title=work_title or "(unbekanntes Werk)",
        author=author or "(unbekannt)",
        year=year or "n/a",
        epoch=epoch or "n/a",
        question=question,
    )


def _build_chat_messages(
    question: str,
    paragraph_id: Optional[str],
    history: list[ChatTurn],
) -> list[dict[str, str]]:
    """Prepare an Ollama /api/chat messages list.

    The filled prompt becomes the `system` message so the model keeps the
    literary frame across multi-turn exchanges; prior turns are replayed as
    they arrived (last ~6 to stay under context limits).
    """
    system_prompt = _build_qa_prompt(question, paragraph_id)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in history[-6:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": question})
    return messages


def _offline_answer(
    question: str,
    paragraph_id: Optional[str],
    *,
    missing_model: Optional[str] = None,
    reason: Optional[str] = None,
) -> str:
    """Deterministic fallback used when Ollama is unreachable or misconfigured."""
    hint = "„" + question.strip() + "\""
    context = ""
    if paragraph_id:
        paragraph = corpus_service.get_paragraph(paragraph_id)
        if paragraph is not None:
            snippet = paragraph.text.strip()
            if len(snippet) > 180:
                snippet = snippet[:177] + "…"
            context = f" Die aktuelle Passage lautet: „{snippet}\""

    if missing_model:
        action = (
            f"Das Modell `{missing_model}` ist auf deiner Ollama-Instanz nicht "
            f"installiert. Führe `ollama pull {missing_model}` aus oder passe "
            "`OLLAMA_MODEL` in deiner .env an ein bereits vorhandenes Modell an "
            "(siehe `ollama list`)."
        )
    elif reason:
        action = (
            f"Ollama meldete: {reason}. Prüfe den lokalen Dienst "
            "(z. B. `ollama serve`) und dass das konfigurierte Modell "
            "verfügbar ist."
        )
    else:
        action = (
            "Starte den lokalen Ollama-Dienst (z. B. `ollama serve`), damit die "
            "KI-Antwort generiert werden kann."
        )

    return (
        "Offline-Antwort: Ich kann deine Frage " + hint + " gerade nicht an "
        "Ollama weiterleiten." + context + " " + action
    )


async def stream_answer(
    question: str,
    paragraph_id: Optional[str] = None,
    history: Optional[list[ChatTurn]] = None,
    *,
    timeout: float = 60.0,
) -> AsyncIterator[str]:
    """Yield answer text chunks from the LLM provider's streamed chat.

    On any provider failure (no chunks yielded) we emit a single offline
    fallback string so the caller always gets a user-visible answer. Never
    raises.
    """
    options = get_llm_options()
    messages = _build_chat_messages(question, paragraph_id, history or [])

    chunks_yielded = 0
    try:
        async for chunk in get_provider().stream_chat(
            messages, options=options, timeout=timeout
        ):
            chunks_yielded += 1
            yield chunk
    except Exception as exc:  # pragma: no cover — provider is supposed to swallow
        logger.warning("Provider stream raised unexpectedly: %s", exc)

    if chunks_yielded == 0:
        # Provider was unreachable / model missing / empty response: surface
        # a deterministic offline answer so the chat UI is never silent.
        # We don't have detailed error info from the provider here; rely on
        # the configured-model probe to detect "missing model" specifically.
        missing: Optional[str] = None
        try:
            probe = await get_provider().probe(timeout=2.0)
            if probe.reachable and not probe.configured_available:
                missing = probe.configured
        except Exception:  # pragma: no cover
            pass
        yield _offline_answer(question, paragraph_id, missing_model=missing)


# ─── Health probe (startup diagnostics + admin endpoint) ─────────────────


async def probe_ollama(*, timeout: float = 3.0) -> dict:
    """Return a small availability report used at startup and by admin.

    Returns the legacy dict shape so existing callers (admin router, main
    lifespan log) keep working without modification.
    """
    report: ProbeReport = await get_provider().probe(timeout=timeout)
    return report.to_dict()
