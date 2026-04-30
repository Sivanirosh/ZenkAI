"""Vocabulary harvester — the authoritative reader-state service.

See decision `.agent/decisions/0001-external-srs-pivot.md`. Three statuses:

- ``queued``   — user wants this word on the export list
- ``known``    — user knows it (seed or explicit); never decorated in the reader
- ``exported`` — CSV row has been written to the external SRS; terminal

The legacy ``word_states`` table is deprecated but untouched by this module.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from backend.models import db
from backend.models.pydantic_models import (
    VocabEntry,
    VocabStatus,
    WordAnnotation,
)
from backend.services import llm_service

logger = logging.getLogger(__name__)


# ─── CSV query (decision 0001) ───────────────────────────────────────────
# Columns are fixed: question, answer, tags, card_type, options,
# correct_option_idx, typed_compare_mode. card_type is always 'basic' for v1.
#
# Answer composition (2026-04-24 update): to mirror the Wortkarte's "Bedeutung"
# section, the auto-answer combines the German and English definitions on two
# lines. The override still wins if present. CONCAT_WS drops NULL arguments,
# so a word with only one language produces a single line.

_ANSWER_EXPR = """COALESCE(
    NULLIF(TRIM(v.answer_override), ''),
    NULLIF(
      CONCAT_WS(
        chr(10),
        NULLIF(TRIM(w.definition_de), ''),
        NULLIF(TRIM(w.definition_en), '')
      ),
      ''
    )
  )"""

_EXPORT_SELECT = f"""
SELECT
    COALESCE(v.question_override,
      CASE
        WHEN w.pos = 'NOUN' AND w.gender = 'm' THEN 'der ' || w.lemma || ' (m)'
        WHEN w.pos = 'NOUN' AND w.gender = 'f' THEN 'die ' || w.lemma || ' (f)'
        WHEN w.pos = 'NOUN' AND w.gender = 'n' THEN 'das ' || w.lemma || ' (n)'
        WHEN w.pos = 'VERB'                     THEN w.lemma || ' (v)'
        WHEN w.pos = 'ADJ'                      THEN w.lemma || ' (adj)'
        ELSE w.lemma
      END
    ) AS question,
    {_ANSWER_EXPR} AS answer,
    'de,' || COALESCE(p.work_id, '') || ',' || COALESCE(wk.epoch, '')
          || ',' || COALESCE(w.pos, '')
          || COALESCE(',' || NULLIF(v.extra_tags, ''), '') AS tags,
    'basic' AS card_type,
    ''      AS options,
    0       AS correct_option_idx,
    'ci'    AS typed_compare_mode
FROM vocab_queue v
JOIN words w            ON w.id = v.word_id
LEFT JOIN paragraphs p  ON p.id = v.source_paragraph
LEFT JOIN works wk      ON wk.id = p.work_id
WHERE v.status = 'queued'
  AND {_ANSWER_EXPR} IS NOT NULL
"""

EXPORT_CSV_HEADER = [
    "question",
    "answer",
    "tags",
    "card_type",
    "options",
    "correct_option_idx",
    "typed_compare_mode",
]


# ─── Row mapping ────────────────────────────────────────────────────────


_LIST_QUERY = """
SELECT
    v.word_id, v.status, w.lemma, w.pos, w.gender,
    w.definition_de, w.definition_en,
    v.source_paragraph, v.source_sentence,
    v.added_at, v.exported_at,
    v.question_override, v.answer_override, v.extra_tags
FROM vocab_queue v
JOIN words w ON w.id = v.word_id
"""


def _row_to_entry(row: tuple) -> VocabEntry:
    definition_de = row[5]
    definition_en = row[6]
    answer_override = row[12]
    # The auto-answer mirrors the Wortkarte "Bedeutung" section: German +
    # English on separate lines. A row is exportable as long as at least one
    # of {override, DE, EN} has non-empty content; annotation is still offered
    # for rows missing either language so the external flashcard matches.
    def _nz(value: Optional[str]) -> bool:
        return bool(value and value.strip())

    has_any = _nz(answer_override) or _nz(definition_de) or _nz(definition_en)
    return VocabEntry(
        word_id=row[0],
        status=row[1],
        lemma=row[2],
        pos=row[3],
        gender=row[4],
        definition_de=definition_de,
        definition_en=definition_en,
        source_paragraph=row[7],
        source_sentence=row[8],
        added_at=row[9],
        exported_at=row[10],
        question_override=row[11],
        answer_override=answer_override,
        extra_tags=row[13],
        needs_annotation=not has_any,
    )


def _get(word_id: str) -> Optional[VocabEntry]:
    row = db.cursor().execute(
        _LIST_QUERY + " WHERE v.word_id = ?",
        [word_id],
    ).fetchone()
    return _row_to_entry(row) if row is not None else None


# ─── Mutations ───────────────────────────────────────────────────────────


def enqueue(
    word_id: str,
    paragraph_id: Optional[str] = None,
    sentence: Optional[str] = None,
) -> VocabEntry:
    """Upsert a word into the queue with status='queued'.

    Idempotent: re-queueing updates the source context and resets exported
    rows back to queued so the user can re-export after editing.
    """
    now = datetime.utcnow()
    db.cursor().execute(
        """
        INSERT INTO vocab_queue (word_id, status, source_paragraph,
                                 source_sentence, added_at)
        VALUES (?, 'queued', ?, ?, ?)
        ON CONFLICT (word_id) DO UPDATE SET
            status           = 'queued',
            source_paragraph = COALESCE(excluded.source_paragraph,
                                        vocab_queue.source_paragraph),
            source_sentence  = COALESCE(excluded.source_sentence,
                                        vocab_queue.source_sentence),
            exported_at      = NULL
        """,
        [word_id, paragraph_id, sentence, now],
    )
    entry = _get(word_id)
    assert entry is not None
    return entry


def mark_known(word_id: str) -> VocabEntry:
    """Upsert a word with status='known'. Idempotent."""
    now = datetime.utcnow()
    db.cursor().execute(
        """
        INSERT INTO vocab_queue (word_id, status, added_at)
        VALUES (?, 'known', ?)
        ON CONFLICT (word_id) DO UPDATE SET
            status = 'known'
        """,
        [word_id, now],
    )
    entry = _get(word_id)
    assert entry is not None
    return entry


def remove(word_id: str) -> None:
    """Delete the row (back to implicit 'new')."""
    db.cursor().execute("DELETE FROM vocab_queue WHERE word_id = ?", [word_id])


def update_overrides(
    word_id: str,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    extra_tags: Optional[str] = None,
) -> Optional[VocabEntry]:
    """Set per-row overrides. Pass empty strings to clear a field; None leaves untouched."""

    def _norm(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    db.cursor().execute(
        """
        UPDATE vocab_queue
        SET question_override = COALESCE(?, question_override),
            answer_override   = COALESCE(?, answer_override),
            extra_tags        = COALESCE(?, extra_tags)
        WHERE word_id = ?
        """,
        [_norm(question), _norm(answer), _norm(extra_tags), word_id],
    )
    return _get(word_id)


# ─── Reads ───────────────────────────────────────────────────────────────


def list_queue(
    status: Optional[VocabStatus] = "queued",
    limit: Optional[int] = None,
) -> list[VocabEntry]:
    sql = _LIST_QUERY
    params: list = []
    if status is not None:
        sql += " WHERE v.status = ?"
        params.append(status)
    sql += " ORDER BY v.added_at DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = db.cursor().execute(sql, params).fetchall()
    return [_row_to_entry(r) for r in rows]


def get_status_map(word_ids: Iterable[str]) -> dict[str, VocabStatus]:
    """Return {word_id: status} for the given ids. Missing ids are omitted."""
    ids = [w for w in word_ids if w]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    rows = db.cursor().execute(
        f"SELECT word_id, status FROM vocab_queue WHERE word_id IN ({placeholders})",
        ids,
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def counts_by_status() -> dict[str, int]:
    rows = db.cursor().execute(
        """
        SELECT status, COUNT(*) FROM vocab_queue GROUP BY status
        """
    ).fetchall()
    out = {"queued": 0, "known": 0, "exported": 0}
    for status, count in rows:
        out[status] = int(count or 0)
    return out


# ─── Export ──────────────────────────────────────────────────────────────


def preview_export_rows() -> list[tuple]:
    """Return the exact rows that export_to_csv would write. Used by tests."""
    rows = db.cursor().execute(_EXPORT_SELECT).fetchall()
    return rows


def export_to_csv(out_path: Path) -> int:
    """Write all exportable rows to CSV, then flip them to ``exported``.

    "Exportable" = ``status='queued'`` AND (``answer_override`` OR
    ``words.definition_en``) is non-empty. Returns the number of rows
    written. The CSV write and the status update are not wrapped in a
    BEGIN/COMMIT because DuckDB's ``COPY TO`` statement auto-commits; for
    a single-writer local database this is acceptable.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = db.get_connection()
    rows = conn.cursor().execute(_EXPORT_SELECT).fetchall()
    if not rows:
        return 0

    path_literal = str(out_path).replace("'", "''")
    copy_sql = (
        f"COPY ({_EXPORT_SELECT}) TO '{path_literal}' "
        "(FORMAT CSV, HEADER, QUOTE '\"', ESCAPE '\"')"
    )
    conn.cursor().execute(copy_sql)

    # Flip exactly the rows the SELECT emitted. Reusing _EXPORT_SELECT as an
    # IN subquery is not possible (it projects all CSV columns), so we match
    # by the same WHERE clause expressed over a correlated join.
    conn.cursor().execute(
        f"""
        UPDATE vocab_queue
        SET status = 'exported', exported_at = now()
        WHERE word_id IN (
            SELECT v.word_id
            FROM vocab_queue v
            JOIN words w ON w.id = v.word_id
            WHERE v.status = 'queued'
              AND {_ANSWER_EXPR} IS NOT NULL
        )
        """
    )
    return len(rows)


# ─── Seeding ─────────────────────────────────────────────────────────────


def bulk_seed_known(lemmas: Iterable[str]) -> dict[str, int]:
    """Insert or upsert the given lemmas as status='known'.

    Ensures a matching ``words`` row exists first (minimal: id = lemma lower,
    lemma = provided). Never regresses a queued/exported word.

    Returns {seeded: int, skipped: int, created_words: int}.
    """
    seeded = 0
    skipped = 0
    created_words = 0
    conn = db.get_connection()
    now = datetime.utcnow()

    for raw in lemmas:
        lemma = (raw or "").strip()
        if not lemma:
            continue
        word_id = lemma.lower()

        existing_word = conn.cursor().execute(
            "SELECT id FROM words WHERE id = ? OR LOWER(lemma) = ? LIMIT 1",
            [word_id, word_id],
        ).fetchone()
        if existing_word is None:
            conn.cursor().execute(
                "INSERT INTO words (id, lemma) VALUES (?, ?)",
                [word_id, lemma],
            )
            created_words += 1
        else:
            word_id = existing_word[0]

        existing_queue = conn.cursor().execute(
            "SELECT status FROM vocab_queue WHERE word_id = ?",
            [word_id],
        ).fetchone()
        if existing_queue is None:
            conn.cursor().execute(
                "INSERT INTO vocab_queue (word_id, status, added_at) "
                "VALUES (?, 'known', ?)",
                [word_id, now],
            )
            seeded += 1
        elif existing_queue[0] == "known":
            skipped += 1
        else:
            skipped += 1

    return {"seeded": seeded, "skipped": skipped, "created_words": created_words}


# ─── Just-in-time annotation (idempotent) ────────────────────────────────


async def ensure_annotated(word_id: str) -> Optional[VocabEntry]:
    """Populate ``words.definition_de/_en/etymology`` via Ollama if missing.

    Invoked on-demand from the Export screen, never on queue. Uses the
    cached source sentence from ``vocab_queue`` so the annotation reflects
    the word in its literary context.
    """
    entry = _get(word_id)
    if entry is None:
        return None

    if entry.definition_en and entry.definition_en.strip():
        return entry

    sentence = entry.source_sentence or entry.lemma
    annotation: WordAnnotation = await llm_service.annotate_word(
        entry.lemma,
        sentence,
    )

    db.cursor().execute(
        """
        UPDATE words
        SET definition_de = COALESCE(NULLIF(?, ''), definition_de),
            definition_en = COALESCE(NULLIF(?, ''), definition_en),
            etymology     = COALESCE(NULLIF(?, ''), etymology)
        WHERE id = ?
        """,
        [
            annotation.definition_de or "",
            annotation.definition_en or "",
            annotation.etymology or "",
            word_id,
        ],
    )
    return _get(word_id)
