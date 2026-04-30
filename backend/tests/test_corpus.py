"""Corpus retrieval tests against an in-memory DuckDB."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.models import db
from backend.services import corpus_service


def _seed_work(conn, work_id: str = "kafka_verwandlung") -> str:
    conn.execute(
        """
        INSERT INTO works (id, title, author, epoch, year, gutenberg_id, language,
                           spine_color, epoch_color)
        VALUES (?, 'Die Verwandlung', 'Franz Kafka', 'Expressionismus', 1915,
                5200, 'de', '#085041', '#9FE1CB')
        """,
        [work_id],
    )
    return work_id


def _seed_paragraph(
    conn, work_id: str, chapter: int, position: int, text: str
) -> str:
    pid = f"{work_id}_c{chapter:02d}_p{position:04d}"
    conn.execute(
        """
        INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [pid, work_id, chapter, position, text, len(text.split())],
    )
    return pid


def _seed_word(conn, word_id: str, lemma: str) -> None:
    conn.execute(
        "INSERT INTO words (id, lemma, pos) VALUES (?, ?, 'NOUN')",
        [word_id, lemma],
    )


def _seed_occurrence(
    conn,
    paragraph_id: str,
    word_id: str,
    surface: str,
    position: int,
    case_label: str = "Nominativ",
    role: str = "Subjekt",
    char_start: int | None = None,
    char_end: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO word_occurrences (id, paragraph_id, word_id, surface_form,
                                      position, case_label, grammatical_role,
                                      char_start, char_end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(uuid.uuid4()),
            paragraph_id,
            word_id,
            surface,
            position,
            case_label,
            role,
            char_start,
            char_end,
        ],
    )


def _seed_paragraph_with_tokens(
    conn,
    work_id: str,
    chapter: int,
    position: int,
    text: str,
    words: list[tuple[str, str]],
) -> str:
    """Seed a paragraph plus its word_occurrences with real char offsets.

    ``words`` is a list of ``(word_id, surface)`` pairs in reading order.
    Each surface is located in ``text`` left-to-right; ``word_id`` doubles
    as the lemma for the ``words`` row (matching the slug-of-lemma
    convention used by the ingest script).
    """
    pid = _seed_paragraph(conn, work_id, chapter, position, text)
    cursor = 0
    for idx, (word_id, surface) in enumerate(words):
        start = text.index(surface, cursor)
        end = start + len(surface)
        if conn.execute(
            "SELECT id FROM words WHERE id = ?", [word_id]
        ).fetchone() is None:
            _seed_word(conn, word_id, word_id)
        _seed_occurrence(
            conn,
            pid,
            word_id,
            surface,
            idx,
            char_start=start,
            char_end=end,
        )
        cursor = end
    return pid


def test_list_works_returns_seeded_work():
    conn = db.get_connection()
    _seed_work(conn)

    works = corpus_service.list_works()

    assert len(works) == 1
    assert works[0].id == "kafka_verwandlung"
    assert works[0].author == "Franz Kafka"
    assert works[0].year == 1915


def test_list_chapters_orders_by_chapter_number():
    conn = db.get_connection()
    wid = _seed_work(conn)
    _seed_paragraph(conn, wid, chapter=2, position=0, text="Chapter two.")
    _seed_paragraph(conn, wid, chapter=1, position=0, text="Chapter one A.")
    _seed_paragraph(conn, wid, chapter=1, position=1, text="Chapter one B.")

    chapters = corpus_service.list_chapters(wid)

    assert [c.chapter for c in chapters] == [1, 2]
    assert chapters[0].paragraph_count == 2
    assert chapters[1].paragraph_count == 1


def test_get_paragraph_with_tokens_returns_familiarity():
    conn = db.get_connection()
    wid = _seed_work(conn)
    text = "Gregor verwandelt sich."
    pid = _seed_paragraph_with_tokens(
        conn, wid, 1, 0, text,
        [("gregor", "Gregor"), ("verwandeln", "verwandelt")],
    )
    conn.execute(
        "INSERT INTO word_states (word_id, familiarity) VALUES ('gregor', 4)"
    )

    paragraph = corpus_service.get_paragraph_with_tokens(pid)

    assert paragraph is not None
    assert paragraph.id == pid
    assert len(paragraph.tokens) == 2
    familiarities = {t.lemma: t.familiarity for t in paragraph.tokens}
    assert familiarities == {"gregor": 4, "verwandeln": 0}
    # Tokens come back in reading order (char_start ascending).
    assert [t.surface_form for t in paragraph.tokens] == ["Gregor", "verwandelt"]


def test_get_work_progress_percentages():
    conn = db.get_connection()
    wid = _seed_work(conn)
    pid = _seed_paragraph(conn, wid, 1, 0, "Ein Satz mit Wörtern.")
    for idx, lemma in enumerate(["wort_a", "wort_b", "wort_c", "wort_d"]):
        _seed_word(conn, lemma, lemma)
        _seed_occurrence(conn, pid, lemma, lemma, idx)

    conn.execute("INSERT INTO word_states (word_id, familiarity) VALUES ('wort_a', 4)")
    conn.execute("INSERT INTO word_states (word_id, familiarity) VALUES ('wort_b', 2)")

    progress = corpus_service.get_work_progress(wid)

    assert progress.total_words == 4
    assert progress.known_words == 1
    assert progress.known_pct == 25.0
    assert progress.learning_pct == 25.0
    assert progress.new_pct == 50.0


def test_get_work_returns_none_for_missing():
    assert corpus_service.get_work("nope") is None


# ─── char offsets + batch fetch ──────────────────────────────────────────


# A German sentence deliberately rich in punctuation: comma, low/high
# quotation marks, em-dash, exclamation point, period. If any of these is
# lost the reader would render „Er sagte Komm doch her" — the exact bug
# that motivated storing offsets.
PUNCTUATION_TEXT = 'Er sagte: „Komm doch her!" — und lachte.'
PUNCTUATION_WORDS = [
    ("er", "Er"),
    ("sagen", "sagte"),
    ("kommen", "Komm"),
    ("doch", "doch"),
    ("her", "her"),
    ("und", "und"),
    ("lachen", "lachte"),
]


def test_tokens_offsets_round_trip_to_surface_form():
    conn = db.get_connection()
    wid = _seed_work(conn)
    pid = _seed_paragraph_with_tokens(
        conn, wid, 1, 0, PUNCTUATION_TEXT, PUNCTUATION_WORDS,
    )

    p = corpus_service.get_paragraph_with_tokens(pid)
    assert p is not None
    for t in p.tokens:
        assert p.text[t.char_start:t.char_end] == t.surface_form


def test_tokens_offsets_are_non_overlapping_and_monotonic():
    conn = db.get_connection()
    wid = _seed_work(conn)
    pid = _seed_paragraph_with_tokens(
        conn, wid, 1, 0, PUNCTUATION_TEXT, PUNCTUATION_WORDS,
    )

    p = corpus_service.get_paragraph_with_tokens(pid)
    assert p is not None
    prev_end = 0
    for t in p.tokens:
        assert t.char_start >= prev_end, "ranges must not overlap"
        assert t.char_end > t.char_start, "range must be non-empty"
        prev_end = t.char_end


def test_inter_token_characters_contain_no_letters():
    """Everything between two word ranges is pure punctuation/whitespace."""
    conn = db.get_connection()
    wid = _seed_work(conn)
    pid = _seed_paragraph_with_tokens(
        conn, wid, 1, 0, PUNCTUATION_TEXT, PUNCTUATION_WORDS,
    )

    p = corpus_service.get_paragraph_with_tokens(pid)
    assert p is not None
    cursor = 0
    for t in p.tokens:
        gap = p.text[cursor:t.char_start]
        assert not any(c.isalpha() for c in gap), (
            f"gap {gap!r} leaks letters between tokens"
        )
        cursor = t.char_end
    tail = p.text[cursor:]
    assert not any(c.isalpha() for c in tail)


def test_get_paragraphs_with_tokens_preserves_order():
    conn = db.get_connection()
    wid = _seed_work(conn)
    pid_a = _seed_paragraph_with_tokens(
        conn, wid, 1, 0, "Alpha eins.", [("alpha", "Alpha"), ("eins", "eins")],
    )
    pid_b = _seed_paragraph_with_tokens(
        conn, wid, 1, 1, "Beta zwei.", [("beta", "Beta"), ("zwei", "zwei")],
    )
    pid_c = _seed_paragraph_with_tokens(
        conn, wid, 1, 2, "Gamma drei.", [("gamma", "Gamma"), ("drei", "drei")],
    )

    out = corpus_service.get_paragraphs_with_tokens([pid_c, pid_a, pid_b])
    assert [p.id for p in out] == [pid_c, pid_a, pid_b]
    assert all(len(p.tokens) == 2 for p in out)


def test_get_paragraphs_with_tokens_empty_input():
    assert corpus_service.get_paragraphs_with_tokens([]) == []


def test_batch_endpoint_returns_paragraphs_in_request_order():
    conn = db.get_connection()
    wid = _seed_work(conn)
    pid_a = _seed_paragraph_with_tokens(
        conn, wid, 1, 0, "Alpha eins.", [("alpha", "Alpha"), ("eins", "eins")],
    )
    pid_b = _seed_paragraph_with_tokens(
        conn, wid, 1, 1, "Beta zwei.", [("beta", "Beta"), ("zwei", "zwei")],
    )

    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/corpus/paragraphs/batch",
        json={"ids": [pid_b, pid_a]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [p["id"] for p in body] == [pid_b, pid_a]
    # Each token carries the new offset fields and slices back to its surface.
    for paragraph in body:
        for tok in paragraph["tokens"]:
            assert paragraph["text"][tok["char_start"]:tok["char_end"]] == tok["surface_form"]


def test_batch_endpoint_404s_when_any_id_is_missing():
    conn = db.get_connection()
    wid = _seed_work(conn)
    pid = _seed_paragraph_with_tokens(
        conn, wid, 1, 0, "Nur eines.", [("nur", "Nur"), ("eins", "eines")],
    )

    client = TestClient(create_app())
    resp = client.post(
        "/api/v1/corpus/paragraphs/batch",
        json={"ids": [pid, "does_not_exist"]},
    )

    assert resp.status_code == 404
    assert "does_not_exist" in resp.json()["detail"]


def test_batch_endpoint_rejects_empty_and_oversized_batches():
    client = TestClient(create_app())
    empty = client.post(
        "/api/v1/corpus/paragraphs/batch", json={"ids": []},
    )
    oversized = client.post(
        "/api/v1/corpus/paragraphs/batch",
        json={"ids": [f"p{i}" for i in range(33)]},
    )

    assert empty.status_code == 422
    assert oversized.status_code == 422
