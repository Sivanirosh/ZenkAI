"""SM-2 and word state machine tests."""

from __future__ import annotations

import pytest

from backend.models import db
from backend.services import word_service


# ─── SM-2 algorithm ───────────────────────────────────────────────────────


def test_sm2_failure_resets_interval_keeps_ease():
    new_ease, new_interval = word_service.sm2_update(ease=2.5, interval=15, quality=1)
    assert new_interval == 1
    assert new_ease == 2.5


def test_sm2_quality_zero_is_failure():
    _, new_interval = word_service.sm2_update(ease=2.5, interval=6, quality=0)
    assert new_interval == 1


def test_sm2_first_success_jumps_to_six_days():
    _, new_interval = word_service.sm2_update(ease=2.5, interval=1, quality=4)
    assert new_interval == 6


def test_sm2_quality_three_barely_passes():
    new_ease, new_interval = word_service.sm2_update(ease=2.5, interval=6, quality=3)
    assert new_interval == round(6 * new_ease)
    assert new_ease < 2.5


def test_sm2_quality_five_boosts_ease():
    new_ease, _ = word_service.sm2_update(ease=2.5, interval=6, quality=5)
    assert new_ease > 2.5


def test_sm2_ease_floor_is_1_3():
    ease = 1.3
    for _ in range(10):
        ease, _ = word_service.sm2_update(ease=ease, interval=6, quality=3)
    assert ease >= 1.3


def test_sm2_quality_out_of_range_raises():
    with pytest.raises(ValueError):
        word_service.sm2_update(ease=2.5, interval=6, quality=6)


# ─── State machine ────────────────────────────────────────────────────────


def _seed_word(word_id: str = "ungeziefer") -> None:
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO words (id, lemma, pos) VALUES (?, ?, 'NOUN')",
        [word_id, word_id.capitalize()],
    )


def test_mark_seen_promotes_state_zero_to_one():
    _seed_word()
    state = word_service.mark_seen("ungeziefer")
    assert state.familiarity == 1
    assert state.seen_count == 1


def test_mark_seen_is_idempotent_and_does_not_regress():
    _seed_word()
    word_service.mark_seen("ungeziefer")
    word_service.mark_opened("ungeziefer")
    state_after = word_service.mark_seen("ungeziefer")
    assert state_after.familiarity == 2


def test_mark_opened_promotes_to_two_and_sets_next_review():
    _seed_word()
    state = word_service.mark_opened("ungeziefer")
    assert state.familiarity == 2
    assert state.next_review is not None


def test_record_review_success_promotes_to_three():
    _seed_word()
    word_service.mark_opened("ungeziefer")
    state = word_service.record_review("ungeziefer", quality=4)
    assert state.familiarity >= 3
    assert state.interval == 6
    assert state.next_review is not None


def test_record_review_long_interval_promotes_to_four():
    _seed_word()
    word_service.mark_opened("ungeziefer")
    conn = db.get_connection()
    conn.execute(
        "UPDATE word_states SET interval = 22 WHERE word_id = 'ungeziefer'"
    )
    state = word_service.record_review("ungeziefer", quality=5)
    assert state.familiarity == 4


def test_record_review_very_long_interval_promotes_to_five():
    _seed_word()
    word_service.mark_opened("ungeziefer")
    conn = db.get_connection()
    conn.execute(
        "UPDATE word_states SET interval = 61 WHERE word_id = 'ungeziefer'"
    )
    state = word_service.record_review("ungeziefer", quality=5)
    assert state.familiarity == 5


def test_record_review_failure_does_not_demote_below_two():
    _seed_word()
    word_service.mark_opened("ungeziefer")
    state = word_service.record_review("ungeziefer", quality=0)
    assert state.familiarity >= 2
    assert state.interval == 1
    assert state.next_review is not None


def test_mark_known_fast_tracks_to_four():
    _seed_word()
    state = word_service.mark_known("ungeziefer")
    assert state.familiarity == 4
    assert state.next_review is not None


def test_review_queue_only_returns_due_words():
    _seed_word("a")
    _seed_word("b")
    word_service.mark_opened("a")
    word_service.mark_opened("b")
    conn = db.get_connection()
    conn.execute("UPDATE word_states SET next_review = '2099-01-01' WHERE word_id = 'b'")
    conn.execute("UPDATE word_states SET next_review = '2000-01-01' WHERE word_id = 'a'")

    queue = word_service.get_review_queue(limit=10)
    queue_ids = [s.word_id for s in queue]

    assert "a" in queue_ids
    assert "b" not in queue_ids


def test_progress_summary_counts_states():
    for wid in ("a", "b", "c"):
        _seed_word(wid)
    word_service.mark_seen("a")
    word_service.mark_opened("b")
    word_service.mark_known("c")

    summary = word_service.get_progress_summary()

    assert summary["total_words_tracked"] == 3
    assert summary["known"] == 1
    assert summary["learning"] >= 1
