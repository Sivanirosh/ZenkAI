"""Request/response schemas shared between routers and services."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─── Corpus ──────────────────────────────────────────────────────────────


class Work(BaseModel):
    id: str
    title: str
    author: str
    epoch: Optional[str] = None
    year: Optional[int] = None
    gutenberg_id: Optional[int] = None
    language: str = "de"
    spine_color: Optional[str] = None
    epoch_color: Optional[str] = None


class WorkProgress(BaseModel):
    work_id: str
    known_pct: float = 0.0
    learning_pct: float = 0.0
    new_pct: float = 100.0
    total_words: int = 0
    known_words: int = 0


class WorkWithProgress(Work):
    progress: WorkProgress


class Chapter(BaseModel):
    work_id: str
    chapter: int
    paragraph_count: int
    first_paragraph_id: str


class WordToken(BaseModel):
    """One word token rendered in the reader.

    The reader slices ``paragraph.text[char_start:char_end]`` to produce the
    clickable surface. Inter-token characters (punctuation, whitespace) are
    rendered as plain text, so the original typography is preserved without
    ever being transmitted as tokens.
    """

    surface_form: str
    word_id: Optional[str] = None
    lemma: Optional[str] = None
    pos: Optional[str] = None
    case_label: Optional[str] = None
    grammatical_role: Optional[str] = None
    char_start: int
    char_end: int


class Paragraph(BaseModel):
    id: str
    work_id: str
    chapter: int
    position: int
    text: str
    word_count: int


class ParagraphWithTokens(Paragraph):
    tokens: list[WordToken]


class ParagraphBatchRequest(BaseModel):
    """Body for ``POST /corpus/paragraphs/batch``."""

    ids: list[str] = Field(min_length=1, max_length=32)


# ─── Words (dictionary rows served to the Wortkarte) ─────────────────────


class Word(BaseModel):
    id: str
    lemma: str
    pos: Optional[str] = None
    gender: Optional[str] = None
    plural_form: Optional[str] = None
    etymology: Optional[str] = None
    definition_de: Optional[str] = None
    definition_en: Optional[str] = None





VocabStatus = Literal["queued", "known", "exported"]


class VocabEntry(BaseModel):
    """A row in the vocab harvester queue.

    Joined with `words` so the frontend can render a queue entry without a
    second round-trip per word.
    """

    word_id: str
    status: VocabStatus
    lemma: str
    pos: Optional[str] = None
    gender: Optional[str] = None
    definition_de: Optional[str] = None
    definition_en: Optional[str] = None
    source_paragraph: Optional[str] = None
    source_sentence: Optional[str] = None
    added_at: Optional[datetime] = None
    exported_at: Optional[datetime] = None
    question_override: Optional[str] = None
    answer_override: Optional[str] = None
    extra_tags: Optional[str] = None
    needs_annotation: bool = False


class VocabEnqueueRequest(BaseModel):
    word_id: str
    paragraph_id: Optional[str] = None
    sentence: Optional[str] = None


class VocabKnownRequest(BaseModel):
    word_id: str


class VocabOverrideRequest(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    extra_tags: Optional[str] = None


class VocabStatusMap(BaseModel):
    """Status snapshot for a batch of word ids (used to hydrate the reader)."""

    statuses: dict[str, VocabStatus]


class VocabExportResult(BaseModel):
    exported: int
    csv_path: str


# ─── Annotations ──────────────────────────────────────────────────────────


class WordAnnotationRequest(BaseModel):
    word: str
    sentence: str
    grammatical_role: Optional[str] = None
    case_label: Optional[str] = None


class WordAnnotation(BaseModel):
    definition_de: str
    definition_en: str
    literary_note: Optional[str] = None
    etymology: Optional[str] = None
    related_words: list[str] = Field(default_factory=list)
    source: Literal["ollama", "mock"] = "mock"


# ─── Chat / Voice ─────────────────────────────────────────────────────────


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    paragraph_id: Optional[str] = None
    history: list[ChatTurn] = Field(default_factory=list)


class TtsRequest(BaseModel):
    text: str
    speed: float = Field(default=1.0, ge=0.5, le=1.5)


class SttResponse(BaseModel):
    transcript: str
    confidence: float
    language: str
    duration_s: float
