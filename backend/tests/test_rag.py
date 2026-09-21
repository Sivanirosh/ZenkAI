"""Structural RAG retrieval tests."""

from __future__ import annotations

from backend.models import db
from backend.services import corpus_service, rag_service


def _seed_work(conn, work_id: str) -> str:
    conn.execute(
        """
        INSERT INTO works (id, title, author, epoch, year, gutenberg_id, language,
                           spine_color, epoch_color)
        VALUES (?, 'T', 'A', 'Expressionismus', 1915, 1, 'de', '#000', '#111')
        """,
        [work_id],
    )
    return work_id


def _seed_paragraph(conn, work_id: str, chapter: int, position: int, text: str) -> str:
    pid = f"{work_id}_c{chapter:02d}_p{position:04d}"
    conn.execute(
        """
        INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [pid, work_id, chapter, position, text, len(text.split())],
    )
    return pid


def test_neighbours_returns_window_inside_same_chapter():
    conn = db.get_connection()
    w = _seed_work(conn, "w1")
    ids = [
        _seed_paragraph(conn, w, 1, i, f"par {i}") for i in range(5)
    ]

    window = corpus_service.get_paragraph_neighbours(ids[2], window=1)

    assert [p.id for p in window] == [ids[1], ids[2], ids[3]]


def test_neighbours_spill_across_chapters_inside_same_work():
    conn = db.get_connection()
    w = _seed_work(conn, "w2")
    p1 = _seed_paragraph(conn, w, 1, 0, "ch1 last")
    p2 = _seed_paragraph(conn, w, 2, 0, "ch2 first")
    p3 = _seed_paragraph(conn, w, 2, 1, "ch2 second")

    window = corpus_service.get_paragraph_neighbours(p2, window=1)

    assert [p.id for p in window] == [p1, p2, p3]


def test_neighbours_never_crosses_work_boundary():
    conn = db.get_connection()
    _seed_work(conn, "other")
    _seed_paragraph(conn, "other", 1, 0, "foreign")
    w = _seed_work(conn, "home")
    mine = _seed_paragraph(conn, w, 1, 0, "home para")

    window = corpus_service.get_paragraph_neighbours(mine, window=3)

    assert len(window) == 1
    assert window[0].work_id == "home"


def test_format_context_marks_anchor():
    conn = db.get_connection()
    w = _seed_work(conn, "w4")
    p0 = _seed_paragraph(conn, w, 1, 0, "alpha")
    p1 = _seed_paragraph(conn, w, 1, 1, "beta")

    paragraphs = corpus_service.get_paragraph_neighbours(p1, window=1)
    text = rag_service.format_context(paragraphs, anchor_id=p1)

    assert ">>>" in text
    assert "alpha" in text
    assert "beta" in text
