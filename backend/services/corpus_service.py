"""Corpus queries: works, chapters, paragraphs with annotated tokens.

All DuckDB access for corpus data must go through this module. Routers
never write raw SQL. The public API returns Pydantic models.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.models import db
from backend.models.pydantic_models import (
    Chapter,
    Paragraph,
    ParagraphWithTokens,
    Work,
    WorkProgress,
    WorkWithProgress,
    WordToken,
)

logger = logging.getLogger(__name__)


def list_works() -> list[WorkWithProgress]:
    """Return all works along with their learner progress."""
    rows = db.cursor().execute(
        """
        SELECT id, title, author, epoch, year, gutenberg_id, language,
               spine_color, epoch_color
        FROM works
        ORDER BY COALESCE(year, 9999), title
        """
    ).fetchall()

    result: list[WorkWithProgress] = []
    for row in rows:
        work_id = row[0]
        progress = get_work_progress(work_id)
        result.append(
            WorkWithProgress(
                id=work_id,
                title=row[1],
                author=row[2],
                epoch=row[3],
                year=row[4],
                gutenberg_id=row[5],
                language=row[6] or "de",
                spine_color=row[7],
                epoch_color=row[8],
                progress=progress,
            )
        )
    return result


def get_work(work_id: str) -> Optional[Work]:
    row = db.cursor().execute(
        """
        SELECT id, title, author, epoch, year, gutenberg_id, language,
               spine_color, epoch_color
        FROM works WHERE id = ?
        """,
        [work_id],
    ).fetchone()
    if row is None:
        return None
    return Work(
        id=row[0],
        title=row[1],
        author=row[2],
        epoch=row[3],
        year=row[4],
        gutenberg_id=row[5],
        language=row[6] or "de",
        spine_color=row[7],
        epoch_color=row[8],
    )


def list_chapters(work_id: str) -> list[Chapter]:
    rows = db.cursor().execute(
        """
        SELECT chapter, COUNT(*) AS n, MIN(id) AS first_id
        FROM paragraphs
        WHERE work_id = ?
        GROUP BY chapter
        ORDER BY chapter
        """,
        [work_id],
    ).fetchall()
    return [
        Chapter(
            work_id=work_id,
            chapter=row[0] or 0,
            paragraph_count=row[1],
            first_paragraph_id=row[2],
        )
        for row in rows
    ]


_OCCURRENCE_COLUMNS = """
    o.paragraph_id,
    o.surface_form, o.word_id, w.lemma, w.pos,
    o.case_label, o.grammatical_role,
    COALESCE(s.familiarity, 0) AS familiarity,
    o.char_start, o.char_end
"""


def _row_to_token(row) -> WordToken:
    """Map an occurrence row (skipping the leading paragraph_id) to a WordToken."""
    return WordToken(
        surface_form=row[1],
        word_id=row[2],
        lemma=row[3],
        pos=row[4],
        case_label=row[5],
        grammatical_role=row[6],
        familiarity=int(row[7] or 0),
        char_start=int(row[8] or 0),
        char_end=int(row[9] or 0),
    )


def get_paragraph_with_tokens(paragraph_id: str) -> Optional[ParagraphWithTokens]:
    """Return a paragraph with per-token familiarity state."""
    para_row = db.cursor().execute(
        """
        SELECT id, work_id, chapter, position, text, word_count
        FROM paragraphs WHERE id = ?
        """,
        [paragraph_id],
    ).fetchone()
    if para_row is None:
        return None

    occ_rows = db.cursor().execute(
        f"""
        SELECT {_OCCURRENCE_COLUMNS}
        FROM word_occurrences o
        LEFT JOIN words w ON w.id = o.word_id
        LEFT JOIN word_states s ON s.word_id = o.word_id
        WHERE o.paragraph_id = ?
        ORDER BY o.char_start
        """,
        [paragraph_id],
    ).fetchall()

    tokens = [_row_to_token(row) for row in occ_rows]

    return ParagraphWithTokens(
        id=para_row[0],
        work_id=para_row[1],
        chapter=para_row[2] or 0,
        position=para_row[3] or 0,
        text=para_row[4],
        word_count=para_row[5] or 0,
        tokens=tokens,
    )


def get_paragraphs_with_tokens(
    paragraph_ids: list[str],
) -> list[ParagraphWithTokens]:
    """Batch variant of :func:`get_paragraph_with_tokens`.

    Two DB round-trips regardless of ``len(paragraph_ids)``: one for the
    paragraph rows, one for every occurrence joined against ``words`` and
    ``word_states``. Results preserve the caller's id order; unknown ids
    are silently absent (the router is responsible for 404-ing on misses
    so batch semantics stay deterministic).
    """
    if not paragraph_ids:
        return []

    placeholders = ", ".join("?" for _ in paragraph_ids)

    para_rows = db.cursor().execute(
        f"""
        SELECT id, work_id, chapter, position, text, word_count
        FROM paragraphs
        WHERE id IN ({placeholders})
        """,
        list(paragraph_ids),
    ).fetchall()

    if not para_rows:
        return []

    occ_rows = db.cursor().execute(
        f"""
        SELECT {_OCCURRENCE_COLUMNS}
        FROM word_occurrences o
        LEFT JOIN words w ON w.id = o.word_id
        LEFT JOIN word_states s ON s.word_id = o.word_id
        WHERE o.paragraph_id IN ({placeholders})
        ORDER BY o.paragraph_id, o.char_start
        """,
        list(paragraph_ids),
    ).fetchall()

    tokens_by_pid: dict[str, list[WordToken]] = {}
    for row in occ_rows:
        tokens_by_pid.setdefault(row[0], []).append(_row_to_token(row))

    by_id: dict[str, ParagraphWithTokens] = {
        r[0]: ParagraphWithTokens(
            id=r[0],
            work_id=r[1],
            chapter=r[2] or 0,
            position=r[3] or 0,
            text=r[4],
            word_count=r[5] or 0,
            tokens=tokens_by_pid.get(r[0], []),
        )
        for r in para_rows
    }

    return [by_id[pid] for pid in paragraph_ids if pid in by_id]


def get_paragraph(paragraph_id: str) -> Optional[Paragraph]:
    """Return a bare Paragraph row (no tokens)."""
    row = db.cursor().execute(
        """
        SELECT id, work_id, chapter, position, text, word_count
        FROM paragraphs WHERE id = ?
        """,
        [paragraph_id],
    ).fetchone()
    if row is None:
        return None
    return Paragraph(
        id=row[0],
        work_id=row[1],
        chapter=row[2] or 0,
        position=row[3] or 0,
        text=row[4],
        word_count=row[5] or 0,
    )


def get_paragraph_neighbours(
    paragraph_id: str, window: int = 2
) -> list[Paragraph]:
    """Return the paragraph plus up to `window` surrounding paragraphs.

    The window is ordered by (chapter, position) and kept strictly inside the
    same work — we never mix works in a retrieval window (AGENT.md RAG rule).
    Prefers the same chapter; if the window spills past the chapter edge we
    fall back to preceding / following chapters within the same work.
    """
    if window < 0:
        window = 0

    anchor = db.cursor().execute(
        """
        SELECT work_id, chapter, position
        FROM paragraphs
        WHERE id = ?
        """,
        [paragraph_id],
    ).fetchone()
    if anchor is None:
        return []

    work_id, chapter, position = anchor[0], anchor[1] or 0, anchor[2] or 0

    rows = db.cursor().execute(
        """
        WITH ordered AS (
            SELECT id, work_id, chapter, position, text, word_count,
                   ROW_NUMBER() OVER (ORDER BY chapter, position) AS rn
            FROM paragraphs
            WHERE work_id = ?
        ),
        anchor AS (
            SELECT rn FROM ordered
            WHERE chapter = ? AND position = ?
        )
        SELECT o.id, o.work_id, o.chapter, o.position, o.text, o.word_count
        FROM ordered o, anchor a
        WHERE o.rn BETWEEN a.rn - ? AND a.rn + ?
        ORDER BY o.chapter, o.position
        """,
        [work_id, chapter, position, window, window],
    ).fetchall()

    return [
        Paragraph(
            id=r[0],
            work_id=r[1],
            chapter=r[2] or 0,
            position=r[3] or 0,
            text=r[4],
            word_count=r[5] or 0,
        )
        for r in rows
    ]


def list_paragraphs_for_chapter(work_id: str, chapter: int) -> list[Paragraph]:
    rows = db.cursor().execute(
        """
        SELECT id, work_id, chapter, position, text, word_count
        FROM paragraphs
        WHERE work_id = ? AND chapter = ?
        ORDER BY position
        """,
        [work_id, chapter],
    ).fetchall()
    return [
        Paragraph(
            id=r[0],
            work_id=r[1],
            chapter=r[2] or 0,
            position=r[3] or 0,
            text=r[4],
            word_count=r[5] or 0,
        )
        for r in rows
    ]


def get_work_progress(work_id: str) -> WorkProgress:
    """Aggregate familiarity over all word occurrences in this work."""
    row = db.cursor().execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN COALESCE(s.familiarity, 0) >= 4 THEN 1 ELSE 0 END) AS known,
            SUM(CASE WHEN COALESCE(s.familiarity, 0) BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS learning,
            SUM(CASE WHEN COALESCE(s.familiarity, 0) = 0 THEN 1 ELSE 0 END) AS new_w
        FROM word_occurrences o
        JOIN paragraphs p ON p.id = o.paragraph_id
        LEFT JOIN word_states s ON s.word_id = o.word_id
        WHERE p.work_id = ?
        """,
        [work_id],
    ).fetchone()

    if row is None:
        return WorkProgress(
            work_id=work_id, known_pct=0.0, learning_pct=0.0, new_pct=0.0,
            total_words=0, known_words=0,
        )

    total = int(row[0] or 0)
    known = int(row[1] or 0)
    learning = int(row[2] or 0)
    new_w = int(row[3] or 0)

    if total == 0:
        return WorkProgress(
            work_id=work_id,
            known_pct=0.0,
            learning_pct=0.0,
            new_pct=0.0,
            total_words=0,
            known_words=0,
        )

    return WorkProgress(
        work_id=work_id,
        known_pct=round(100.0 * known / total, 1),
        learning_pct=round(100.0 * learning / total, 1),
        new_pct=round(100.0 * new_w / total, 1),
        total_words=total,
        known_words=known,
    )
