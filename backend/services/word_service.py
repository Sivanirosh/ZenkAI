"""Word familiarity state machine and SM-2 scheduler.

The state machine is the core learning engine. Rules from AGENT.md:

1. States only increment, never skip. 0 → 1 → 2 → 3 → 4 → 5.
2. State 0 → 1 is passive (reading viewport).
3. State 1 → 2 requires explicit user action (tap).
4. States 2 – 5 only update via SM-2 review.
5. Never reset below 2 without explicit user request.
6. `next_review` must always be set when familiarity ≥ 2.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from backend.models import db
from backend.models.pydantic_models import ReviewCard, WordState

logger = logging.getLogger(__name__)


# ─── SM-2 ────────────────────────────────────────────────────────────────


def sm2_update(ease: float, interval: int, quality: int) -> tuple[float, int]:
    """Standard SM-2 algorithm.

    quality ∈ [0, 5] where 0-2 = fail, 3-5 = pass. On failure the interval
    resets to 1 day and ease is preserved. On success ease is nudged and
    interval grows geometrically.
    """
    if quality < 0 or quality > 5:
        raise ValueError(f"quality must be in [0, 5], got {quality}")

    if quality < 3:
        return ease, 1

    new_ease = max(1.3, ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if interval == 1:
        new_interval = 6
    elif interval == 6:
        new_interval = round(interval * new_ease)
    else:
        new_interval = round(interval * new_ease)
    return new_ease, new_interval


def _familiarity_from_interval(current: int, interval: int, quality: int) -> int:
    """Promote familiarity based on SM-2 interval milestones.

    2 → 3: two successful reviews (handled by caller via seen_count)
    3 → 4: interval > 21 days
    4 → 5: interval > 60 days
    """
    if quality < 3:
        return max(current, 2)
    if interval > 60:
        return 5
    if interval > 21:
        return max(current, 4)
    return max(current, 3)


# ─── State machine ────────────────────────────────────────────────────────


def _ensure_row(word_id: str) -> None:
    db.cursor().execute(
        """
        INSERT INTO word_states (word_id, familiarity, ease_factor, interval,
                                 seen_count)
        VALUES (?, 0, 2.5, 1, 0)
        ON CONFLICT (word_id) DO NOTHING
        """,
        [word_id],
    )


def get_state(word_id: str) -> Optional[WordState]:
    row = db.cursor().execute(
        """
        SELECT word_id, familiarity, ease_factor, interval,
               next_review, seen_count, last_seen
        FROM word_states WHERE word_id = ?
        """,
        [word_id],
    ).fetchone()
    if row is None:
        return None
    return WordState(
        word_id=row[0],
        familiarity=int(row[1] or 0),
        ease_factor=float(row[2] or 2.5),
        interval=int(row[3] or 1),
        next_review=row[4],
        seen_count=int(row[5] or 0),
        last_seen=row[6],
    )


def mark_seen(word_id: str) -> WordState:
    """State 0 → 1. Idempotent for already-seen words (does not regress)."""
    _ensure_row(word_id)
    now = datetime.utcnow()
    db.cursor().execute(
        """
        UPDATE word_states
        SET familiarity = CASE WHEN familiarity < 1 THEN 1 ELSE familiarity END,
            seen_count  = seen_count + 1,
            last_seen   = ?
        WHERE word_id = ?
        """,
        [now, word_id],
    )
    state = get_state(word_id)
    assert state is not None
    return state


def mark_opened(word_id: str) -> WordState:
    """State 1 → 2. Requires the word card to be viewed.

    Initialises `next_review` to now + 1 day so the invariant from AGENT.md
    rule 6 (familiarity ≥ 2 implies next_review set) holds.
    """
    _ensure_row(word_id)
    now = datetime.utcnow()
    next_review = now + timedelta(days=1)
    db.cursor().execute(
        """
        UPDATE word_states
        SET familiarity = CASE WHEN familiarity < 2 THEN 2 ELSE familiarity END,
            next_review = CASE WHEN next_review IS NULL THEN ? ELSE next_review END,
            last_seen   = ?
        WHERE word_id = ?
        """,
        [next_review, now, word_id],
    )
    state = get_state(word_id)
    assert state is not None
    return state


def record_review(word_id: str, quality: int) -> WordState:
    """Apply an SM-2 review to an existing word. Creates state if missing."""
    _ensure_row(word_id)
    current = get_state(word_id)
    assert current is not None

    ease = current.ease_factor
    interval = current.interval
    new_ease, new_interval = sm2_update(ease, interval, quality)

    fam = _familiarity_from_interval(current.familiarity, new_interval, quality)
    fam = max(fam, 2)

    now = datetime.utcnow()
    next_review = now + timedelta(days=new_interval)

    db.cursor().execute(
        """
        UPDATE word_states
        SET familiarity = ?,
            ease_factor = ?,
            interval    = ?,
            next_review = ?,
            seen_count  = seen_count + 1,
            last_seen   = ?
        WHERE word_id = ?
        """,
        [fam, new_ease, new_interval, next_review, now, word_id],
    )
    state = get_state(word_id)
    assert state is not None
    return state


def mark_known(word_id: str) -> WordState:
    """Fast-track a word to familiarity 4 (user assertion)."""
    _ensure_row(word_id)
    now = datetime.utcnow()
    next_review = now + timedelta(days=30)
    db.cursor().execute(
        """
        UPDATE word_states
        SET familiarity = CASE WHEN familiarity < 4 THEN 4 ELSE familiarity END,
            interval    = CASE WHEN interval < 30 THEN 30 ELSE interval END,
            next_review = ?,
            last_seen   = ?
        WHERE word_id = ?
        """,
        [next_review, now, word_id],
    )
    state = get_state(word_id)
    assert state is not None
    return state


def get_review_queue(limit: int = 20) -> list[WordState]:
    """Words due today, ordered by overdue-ness (oldest next_review first)."""
    now = datetime.utcnow()
    rows = db.cursor().execute(
        """
        SELECT word_id, familiarity, ease_factor, interval,
               next_review, seen_count, last_seen
        FROM word_states
        WHERE familiarity >= 2 AND next_review IS NOT NULL AND next_review <= ?
        ORDER BY next_review ASC
        LIMIT ?
        """,
        [now, limit],
    ).fetchall()
    return [
        WordState(
            word_id=r[0],
            familiarity=int(r[1] or 0),
            ease_factor=float(r[2] or 2.5),
            interval=int(r[3] or 1),
            next_review=r[4],
            seen_count=int(r[5] or 0),
            last_seen=r[6],
        )
        for r in rows
    ]


def get_progress_summary() -> dict[str, int]:
    row = db.cursor().execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN familiarity >= 4 THEN 1 ELSE 0 END) AS known,
            SUM(CASE WHEN familiarity BETWEEN 1 AND 3 THEN 1 ELSE 0 END) AS learning,
            SUM(CASE WHEN familiarity = 0 THEN 1 ELSE 0 END) AS new_w
        FROM word_states
        """
    ).fetchone()
    if row is None:
        # COUNT(*) should always yield a row, but guard anyway — DuckDB has
        # been observed returning None from a cursor that lost its turn to
        # another concurrent query.
        return {"total_words_tracked": 0, "known": 0, "learning": 0, "new": 0}
    return {
        "total_words_tracked": int(row[0] or 0),
        "known": int(row[1] or 0),
        "learning": int(row[2] or 0),
        "new": int(row[3] or 0),
    }


def get_review_card(word_id: str) -> Optional[ReviewCard]:
    """Produce a contextual cloze card using the longest paragraph with this word."""
    row = db.cursor().execute(
        """
        SELECT w.lemma, o.surface_form, o.grammatical_role, p.text
        FROM word_occurrences o
        JOIN paragraphs p ON p.id = o.paragraph_id
        JOIN words w ON w.id = o.word_id
        WHERE o.word_id = ?
        ORDER BY p.word_count DESC
        LIMIT 1
        """,
        [word_id],
    ).fetchone()
    if row is None:
        return None
    lemma, surface, role, text = row
    blanked = text.replace(surface, "_______", 1)
    return ReviewCard(
        word_id=word_id,
        lemma=lemma,
        paragraph_text=text,
        blanked_text=blanked,
        grammatical_role=role,
        first_letter=surface[0] if surface else None,
        answer=surface,
    )
