"""LLM prompts + client.

All prompt templates live here as named constants (AGENT.md rule). No other
module may inline a prompt. When modifying a prompt, add a comment with the
date and rationale immediately above the constant.

`annotate_word` tries a local Ollama server; on any failure it returns a
deterministic mock annotation composed from the `words` table and the
surrounding sentence. This keeps the Word Card usable offline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, Optional

import httpx

from backend.config import get_settings
from backend.models import db
from backend.models.pydantic_models import ChatTurn, WordAnnotation
from backend.services import corpus_service, rag_service
from backend.services.runtime_config import get_llm_options

logger = logging.getLogger(__name__)


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


# ─── Ollama client ────────────────────────────────────────────────────────


async def _call_ollama(prompt: str, *, timeout: float = 20.0) -> Optional[str]:
    """Call Ollama's /api/generate with streaming disabled. Returns raw text."""
    settings = get_settings()
    options = get_llm_options()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": options.model,
        "prompt": prompt,
        "stream": False,
        "think": options.think,
        "options": {"temperature": options.temperature},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400:
                body = response.text
                missing = _detect_missing_model(body)
                if missing:
                    logger.warning(
                        "Ollama model '%s' is not installed. "
                        "Run `ollama pull %s` or update OLLAMA_MODEL in .env.",
                        missing,
                        missing,
                    )
                else:
                    logger.warning(
                        "Ollama returned %s: %s",
                        response.status_code,
                        body.strip()[:500],
                    )
                return None
            data = response.json()
            return data.get("response", "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Ollama call failed, falling back to mock: %s", exc)
        return None


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
    """Annotate a word via Ollama, falling back to a deterministic mock."""
    prompt = WORD_ANNOTATION_PROMPT.format(
        word=word,
        sentence=sentence,
        grammatical_role=grammatical_role or "n/a",
        case_label=case_label or "n/a",
    )
    raw = await _call_ollama(prompt)
    if raw is None:
        return _mock_annotation(word, sentence)

    parsed = _parse_annotation_json(_strip_thinking(raw))
    if parsed is None:
        logger.warning("Could not parse Ollama JSON response for '%s'", word)
        return _mock_annotation(word, sentence)

    stored = _lookup_word(word)
    return WordAnnotation(
        definition_de=parsed.get("definition_de")
        or (stored.get("definition_de") if stored else "") or word,
        definition_en=parsed.get("definition_en")
        or (stored.get("definition_en") if stored else "") or word,
        literary_note=parsed.get("literary_note"),
        etymology=parsed.get("etymology")
        or (stored.get("etymology") if stored else None),
        related_words=list(parsed.get("related_words") or []),
        source="ollama",
    )


# ─── Streaming literary Q&A ──────────────────────────────────────────────


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
            neighbours = corpus_service.get_paragraph_neighbours(paragraph_id, window=2)
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


_MODEL_NOT_FOUND_RE = re.compile(
    r"model ['\"]?([^'\"]+?)['\"]? not found|try pulling", re.IGNORECASE
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove any complete `<think>…</think>` blocks from a response."""
    return _THINK_BLOCK_RE.sub("", text).lstrip()


class _ThinkStripper:
    """Incremental <think>…</think> filter for streaming tokens.

    Ollama's `think=false` flag suppresses thinking on models that support it.
    For models that ignore the flag (or older Ollama versions) we still want
    to keep the UI clean, so we maintain a tiny state machine:

    - When we see `<think>` we swallow tokens until `</think>` closes.
    - Short partial matches at a chunk boundary are buffered until we know
      whether they're the start of a thinking tag.
    """

    def __init__(self) -> None:
        self._inside = False
        self._pending = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        out: list[str] = []
        buf = self._pending + chunk
        self._pending = ""

        while buf:
            if self._inside:
                end = buf.find("</think>")
                if end == -1:
                    # Keep a small tail in case the close tag straddles chunks.
                    keep = min(len(buf), 8)
                    self._pending = buf[-keep:]
                    buf = ""
                    break
                buf = buf[end + len("</think>"):]
                self._inside = False
                continue

            start = buf.find("<think>")
            if start == -1:
                # Might be a partial open tag at the tail — keep up to 7 chars.
                tail_keep = 0
                for i in range(min(7, len(buf)), 0, -1):
                    if "<think>".startswith(buf[-i:]):
                        tail_keep = i
                        break
                if tail_keep:
                    out.append(buf[:-tail_keep])
                    self._pending = buf[-tail_keep:]
                else:
                    out.append(buf)
                buf = ""
                break

            out.append(buf[:start])
            buf = buf[start + len("<think>"):]
            self._inside = True

        return "".join(out)

    def flush(self) -> str:
        leftover = self._pending
        self._pending = ""
        if self._inside:
            # Unterminated think block — drop it entirely.
            return ""
        return leftover


def _detect_missing_model(body: str) -> Optional[str]:
    """Return the missing model name if Ollama's error body reports one."""
    if not body:
        return None
    match = _MODEL_NOT_FOUND_RE.search(body)
    if not match:
        return None
    return match.group(1) or get_settings().ollama_model


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
    """Yield answer text chunks from Ollama `/api/chat` (stream=true).

    On any transport/parse error we yield a single offline fallback string so
    the caller always gets a user-visible answer. Model-not-found errors get
    a tailored hint telling the user which `ollama pull` to run. Never raises.
    """
    settings = get_settings()
    options = get_llm_options()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": options.model,
        "stream": True,
        "think": options.think,
        "messages": _build_chat_messages(question, paragraph_id, history or []),
        "options": {"temperature": options.temperature},
    }

    error_body: Optional[str] = None
    reason: Optional[str] = None
    stripper = _ThinkStripper()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    try:
                        raw = await response.aread()
                        error_body = raw.decode("utf-8", errors="replace")
                    except Exception:  # pragma: no cover
                        error_body = ""
                    reason = (
                        f"HTTP {response.status_code} — "
                        f"{(error_body or '').strip()[:200]}"
                    )
                    logger.warning(
                        "Ollama chat returned %s: %s",
                        response.status_code,
                        (error_body or "").strip()[:500],
                    )
                else:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            logger.debug("Skipping non-JSON chat line: %r", line)
                            continue
                        delta = (obj.get("message") or {}).get("content") or ""
                        if delta:
                            clean = stripper.feed(delta)
                            if clean:
                                yield clean
                        if obj.get("done"):
                            tail = stripper.flush()
                            if tail:
                                yield tail
                            return
                    tail = stripper.flush()
                    if tail:
                        yield tail
                    return
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Ollama chat stream failed, using offline fallback: %s", exc)
        reason = str(exc)

    missing = _detect_missing_model(error_body or "") if error_body else None
    yield _offline_answer(
        question,
        paragraph_id,
        missing_model=missing,
        reason=None if missing else reason,
    )


# ─── Health check (startup diagnostics) ──────────────────────────────────


async def probe_ollama(*, timeout: float = 3.0) -> dict:
    """Return a small availability report used at startup.

    Never raises — returns `{reachable: bool, models: list[str], configured: str,
    configured_available: bool, error: str | None}`.
    """
    settings = get_settings()
    active_model = get_llm_options().model
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    report: dict = {
        "reachable": False,
        "models": [],
        "configured": active_model,
        "configured_available": False,
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        report["reachable"] = True
        report["models"] = models
        report["configured_available"] = active_model in models
    except Exception as exc:  # pragma: no cover — best effort
        report["error"] = str(exc)
    return report
