"""Vocabulary harvester tests (decision 0001).

Covers enqueue / mark_known / remove / list_queue / update_overrides /
bulk_seed_known. CSV export is covered in test_vocab_export.py.
"""

from __future__ import annotations

from backend.models import db
from backend.services import vocab_service


def _seed_word(word_id: str = "ungeziefer", lemma: str = "Ungeziefer", pos: str = "NOUN") -> None:
    db.cursor().execute(
        "INSERT INTO words (id, lemma, pos) VALUES (?, ?, ?)",
        [word_id, lemma, pos],
    )


def _seed_paragraph(paragraph_id: str = "para-1") -> None:
    """Create a minimal work + paragraph so the FK on vocab_queue resolves."""
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO works (id, title, author) VALUES "
        "('w-test', 'Test Work', 'Test Author')"
    )
    conn.execute(
        "INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count) "
        "VALUES (?, 'w-test', 1, 0, 'Er fand ein Ungeziefer.', 4)",
        [paragraph_id],
    )


def test_enqueue_creates_queued_row():
    _seed_word()
    _seed_paragraph()
    entry = vocab_service.enqueue("ungeziefer", "para-1", "Er fand ein Ungeziefer.")
    assert entry.status == "queued"
    assert entry.source_paragraph == "para-1"
    assert entry.source_sentence == "Er fand ein Ungeziefer."


def test_enqueue_is_idempotent_and_preserves_context():
    _seed_word()
    _seed_paragraph()
    vocab_service.enqueue("ungeziefer", "para-1", "first sentence")
    again = vocab_service.enqueue("ungeziefer", None, None)
    # Should still have the original context (COALESCE keeps it).
    assert again.status == "queued"
    assert again.source_paragraph == "para-1"
    assert again.source_sentence == "first sentence"


def test_mark_known_promotes_from_new():
    _seed_word()
    entry = vocab_service.mark_known("ungeziefer")
    assert entry.status == "known"


def test_mark_known_after_enqueue_overrides_status():
    _seed_word()
    _seed_paragraph()
    vocab_service.enqueue("ungeziefer", "para-1", "ctx")
    entry = vocab_service.mark_known("ungeziefer")
    assert entry.status == "known"
    # No duplicate row.
    rows = db.cursor().execute(
        "SELECT COUNT(*) FROM vocab_queue WHERE word_id = 'ungeziefer'"
    ).fetchone()
    assert rows[0] == 1


def test_enqueue_after_known_reverts_status_to_queued():
    _seed_word()
    _seed_paragraph()
    vocab_service.mark_known("ungeziefer")
    entry = vocab_service.enqueue("ungeziefer", "para-1", "ctx")
    assert entry.status == "queued"


def test_remove_deletes_row():
    _seed_word()
    vocab_service.enqueue("ungeziefer", None, None)
    vocab_service.remove("ungeziefer")
    rows = db.cursor().execute(
        "SELECT COUNT(*) FROM vocab_queue WHERE word_id = 'ungeziefer'"
    ).fetchone()
    assert rows[0] == 0


def test_list_queue_filters_by_status():
    _seed_word("a", "Haus")
    _seed_word("b", "Katze")
    _seed_word("c", "Baum")
    vocab_service.enqueue("a", None, None)
    vocab_service.enqueue("b", None, None)
    vocab_service.mark_known("c")

    queued = vocab_service.list_queue(status="queued")
    known = vocab_service.list_queue(status="known")
    everything = vocab_service.list_queue(status=None)

    assert {e.word_id for e in queued} == {"a", "b"}
    assert {e.word_id for e in known} == {"c"}
    assert {e.word_id for e in everything} == {"a", "b", "c"}


def test_update_overrides_sets_fields_and_ignores_untouched():
    _seed_word()
    vocab_service.enqueue("ungeziefer", None, None)
    vocab_service.update_overrides(
        "ungeziefer", question="der Ungeziefer (m)", extra_tags="kafka"
    )
    entry = vocab_service.list_queue(status="queued")[0]
    assert entry.question_override == "der Ungeziefer (m)"
    assert entry.extra_tags == "kafka"
    assert entry.answer_override is None

    # Passing None for answer keeps previous value.
    vocab_service.update_overrides("ungeziefer", answer="vermin, pest")
    entry = vocab_service.list_queue(status="queued")[0]
    assert entry.answer_override == "vermin, pest"
    assert entry.question_override == "der Ungeziefer (m)"


def test_needs_annotation_flag_reflects_definition_presence():
    _seed_word()
    vocab_service.enqueue("ungeziefer", None, None)
    [entry] = vocab_service.list_queue(status="queued")
    assert entry.needs_annotation is True

    db.cursor().execute(
        "UPDATE words SET definition_en = 'vermin' WHERE id = 'ungeziefer'"
    )
    [entry] = vocab_service.list_queue(status="queued")
    assert entry.needs_annotation is False


def test_bulk_seed_known_is_idempotent():
    result1 = vocab_service.bulk_seed_known(["Haus", "Katze", "Baum"])
    result2 = vocab_service.bulk_seed_known(["Haus", "Katze", "Baum"])

    assert result1["seeded"] == 3
    assert result2["seeded"] == 0
    assert result2["skipped"] == 3

    rows = db.cursor().execute(
        "SELECT status FROM vocab_queue"
    ).fetchall()
    assert all(r[0] == "known" for r in rows)
    assert len(rows) == 3


def test_bulk_seed_does_not_regress_queued_or_exported():
    _seed_word("haus", "Haus", pos="NOUN")
    _seed_paragraph()
    vocab_service.enqueue("haus", None, None)
    vocab_service.bulk_seed_known(["Haus"])
    [entry] = vocab_service.list_queue(status=None)
    # A previously queued word must stay queued even when seeded.
    assert entry.status == "queued"


def test_counts_by_status_returns_all_keys():
    _seed_word("a", "A")
    _seed_word("b", "B")
    vocab_service.enqueue("a", None, None)
    vocab_service.mark_known("b")

    counts = vocab_service.counts_by_status()
    assert counts == {"queued": 1, "known": 1, "exported": 0}


def test_get_status_map_batch_lookup():
    _seed_word("a", "A")
    _seed_word("b", "B")
    _seed_word("c", "C")
    vocab_service.enqueue("a", None, None)
    vocab_service.mark_known("b")

    out = vocab_service.get_status_map(["a", "b", "c", "nonexistent"])
    assert out == {"a": "queued", "b": "known"}
