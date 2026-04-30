"""CSV export conformance tests for the vocab harvester (decision 0001).

Every assertion here is a contract guard for the external SRS importer.
"""

from __future__ import annotations

import csv

from backend.models import db
from backend.services import vocab_service


EXPECTED_COLUMNS = [
    "question",
    "answer",
    "tags",
    "card_type",
    "options",
    "correct_option_idx",
    "typed_compare_mode",
]


def _seed_work_and_paragraph(
    work_id: str = "kafka_verwandlung",
    epoch: str = "Expressionismus",
    paragraph_id: str = "para-1",
) -> None:
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO works (id, title, author, epoch, year) VALUES (?, ?, ?, ?, ?)",
        [work_id, "Die Verwandlung", "Franz Kafka", epoch, 1915],
    )
    conn.execute(
        "INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count) "
        "VALUES (?, ?, 1, 0, 'Er fand ein Ungeziefer.', 4)",
        [paragraph_id, work_id],
    )


def _seed_annotated_word(
    word_id: str,
    lemma: str,
    pos: str = "NOUN",
    gender: str | None = "n",
    definition_en: str | None = "vermin, pest",
    definition_de: str | None = None,
) -> None:
    db.cursor().execute(
        """
        INSERT INTO words (id, lemma, pos, gender, definition_de, definition_en)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [word_id, lemma, pos, gender, definition_de, definition_en],
    )


def test_preview_rows_have_fixed_column_order():
    _seed_work_and_paragraph()
    _seed_annotated_word("ungeziefer", "Ungeziefer")
    vocab_service.enqueue("ungeziefer", "para-1", "Er fand ein Ungeziefer.")

    rows = vocab_service.preview_export_rows()
    assert len(rows) == 1
    # Each row is a tuple matching EXPECTED_COLUMNS positionally.
    assert len(rows[0]) == len(EXPECTED_COLUMNS)


def test_export_to_csv_writes_header_and_single_row(tmp_path):
    _seed_work_and_paragraph()
    _seed_annotated_word("ungeziefer", "Ungeziefer", gender="n")
    vocab_service.enqueue("ungeziefer", "para-1", "Er fand ein Ungeziefer.")

    out = tmp_path / "vocab.csv"
    written = vocab_service.export_to_csv(out)
    assert written == 1
    assert out.exists()

    with out.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header == EXPECTED_COLUMNS

        [row] = list(reader)
        assert row[0] == "das Ungeziefer (n)"
        assert row[1] == "vermin, pest"
        assert "de" in row[2]
        assert "kafka_verwandlung" in row[2]
        assert "Expressionismus" in row[2]
        assert "NOUN" in row[2]
        assert row[3] == "basic"
        assert row[4] == ""
        assert row[5] == "0"
        assert row[6] == "ci"


def test_export_marks_rows_as_exported(tmp_path):
    _seed_work_and_paragraph()
    _seed_annotated_word("ungeziefer", "Ungeziefer")
    vocab_service.enqueue("ungeziefer", "para-1", "Er fand ein Ungeziefer.")

    out = tmp_path / "vocab.csv"
    vocab_service.export_to_csv(out)

    entries = vocab_service.list_queue(status=None)
    assert len(entries) == 1
    assert entries[0].status == "exported"
    assert entries[0].exported_at is not None


def test_export_excludes_rows_without_definition(tmp_path):
    _seed_work_and_paragraph()
    _seed_annotated_word("ungeziefer", "Ungeziefer", definition_en="vermin")
    _seed_annotated_word("foo", "Foo", gender="m", definition_en=None)
    vocab_service.enqueue("ungeziefer", "para-1", "Er fand ein Ungeziefer.")
    vocab_service.enqueue("foo", "para-1", "Das Foo ist da.")

    out = tmp_path / "vocab.csv"
    written = vocab_service.export_to_csv(out)
    assert written == 1

    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 2  # header + 1 data row

    # The annotated word exported, the other stayed queued.
    entries = {e.word_id: e for e in vocab_service.list_queue(status=None)}
    assert entries["ungeziefer"].status == "exported"
    assert entries["foo"].status == "queued"


def test_export_nouns_use_correct_article():
    _seed_work_and_paragraph()
    _seed_annotated_word("hund", "Hund", gender="m", definition_en="dog")
    _seed_annotated_word("katze", "Katze", gender="f", definition_en="cat")
    _seed_annotated_word("buch", "Buch", gender="n", definition_en="book")
    for wid in ("hund", "katze", "buch"):
        vocab_service.enqueue(wid, "para-1", "ctx")

    rows = vocab_service.preview_export_rows()
    questions = [r[0] for r in rows]
    assert "der Hund (m)" in questions
    assert "die Katze (f)" in questions
    assert "das Buch (n)" in questions


def test_export_verb_and_adjective_suffixes():
    _seed_work_and_paragraph()
    _seed_annotated_word("laufen", "laufen", pos="VERB", gender=None, definition_en="to run")
    _seed_annotated_word("schön", "schön", pos="ADJ", gender=None, definition_en="beautiful")
    for wid in ("laufen", "schön"):
        vocab_service.enqueue(wid, "para-1", "ctx")

    rows = vocab_service.preview_export_rows()
    questions = [r[0] for r in rows]
    assert "laufen (v)" in questions
    assert "schön (adj)" in questions


def test_override_question_and_answer_win_over_auto(tmp_path):
    _seed_work_and_paragraph()
    _seed_annotated_word("ungeziefer", "Ungeziefer", definition_en="vermin")
    vocab_service.enqueue("ungeziefer", "para-1", "ctx")
    vocab_service.update_overrides(
        "ungeziefer",
        question="Ungeziefer (custom)",
        answer="a disgusting creature",
    )

    out = tmp_path / "vocab.csv"
    vocab_service.export_to_csv(out)

    with out.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        [row] = list(reader)

    assert row[0] == "Ungeziefer (custom)"
    assert row[1] == "a disgusting creature"


def test_extra_tags_appear_after_base_tags(tmp_path):
    _seed_work_and_paragraph()
    _seed_annotated_word("ungeziefer", "Ungeziefer", definition_en="vermin")
    vocab_service.enqueue("ungeziefer", "para-1", "ctx")
    vocab_service.update_overrides("ungeziefer", extra_tags="entry,alternatives")

    out = tmp_path / "vocab.csv"
    vocab_service.export_to_csv(out)

    with out.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        [row] = list(reader)

    tag_field = row[2]
    parts = tag_field.split(",")
    assert parts[0] == "de"
    assert "entry" in parts
    assert "alternatives" in parts


def test_csv_quoting_handles_commas_and_quotes_in_answer(tmp_path):
    _seed_work_and_paragraph()
    _seed_annotated_word(
        "ungeziefer",
        "Ungeziefer",
        definition_en='vermin, pest; pejorative "creature" term',
    )
    vocab_service.enqueue("ungeziefer", "para-1", "ctx")

    out = tmp_path / "vocab.csv"
    vocab_service.export_to_csv(out)

    with out.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        [row] = list(reader)

    assert row["answer"] == 'vermin, pest; pejorative "creature" term'
    assert row["question"] == "das Ungeziefer (n)"


def test_empty_queue_export_writes_nothing(tmp_path):
    out = tmp_path / "vocab.csv"
    written = vocab_service.export_to_csv(out)
    assert written == 0
    # File is not created when there are zero rows (transaction short-circuits).
    assert not out.exists()


def test_answer_combines_german_and_english_definitions(tmp_path):
    """Mirrors the Wortkarte "Bedeutung" section: DE on first line, EN on second."""
    _seed_work_and_paragraph()
    _seed_annotated_word(
        "ungeziefer",
        "Ungeziefer",
        gender="n",
        definition_de=(
            "Ein Sch\u00e4dling oder unerw\u00fcnschtes Tier, "
            "abwertend f\u00fcr ein st\u00f6rendes Gesch\u00f6pf."
        ),
        definition_en="vermin, pest; pejorative for nuisance creature",
    )
    vocab_service.enqueue("ungeziefer", "para-1", "Er fand ein Ungeziefer.")

    out = tmp_path / "vocab.csv"
    vocab_service.export_to_csv(out)

    with out.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        [row] = list(reader)

    lines = row["answer"].split("\n")
    assert len(lines) == 2, f"expected two-line answer, got: {row['answer']!r}"
    assert lines[0].startswith("Ein Sch")
    assert lines[1].startswith("vermin")


def test_answer_falls_back_to_single_language_when_only_one_present(tmp_path):
    _seed_work_and_paragraph(work_id="w1", paragraph_id="p1")
    _seed_annotated_word(
        "nur_de",
        "NurDE",
        gender="m",
        definition_de="Nur Deutsch vorhanden.",
        definition_en=None,
    )
    _seed_annotated_word(
        "nur_en",
        "NurEN",
        gender="f",
        definition_de=None,
        definition_en="only English present",
    )
    vocab_service.enqueue("nur_de", "p1", "ctx")
    vocab_service.enqueue("nur_en", "p1", "ctx")

    out = tmp_path / "vocab.csv"
    written = vocab_service.export_to_csv(out)
    assert written == 2

    with out.open("r", encoding="utf-8", newline="") as handle:
        rows = {r["question"].split()[-2]: r["answer"] for r in csv.DictReader(handle)}

    assert rows["NurDE"] == "Nur Deutsch vorhanden."
    assert rows["NurEN"] == "only English present"


def test_row_with_only_german_definition_is_exportable(tmp_path):
    """Previously rows without definition_en were excluded; now DE alone is fine."""
    _seed_work_and_paragraph()
    _seed_annotated_word(
        "haus",
        "Haus",
        gender="n",
        definition_de="Ein Geb\u00e4ude zum Wohnen.",
        definition_en=None,
    )
    vocab_service.enqueue("haus", "para-1", "Das Haus ist gross.")

    out = tmp_path / "vocab.csv"
    written = vocab_service.export_to_csv(out)
    assert written == 1

    entries = {e.word_id: e for e in vocab_service.list_queue(status=None)}
    assert entries["haus"].status == "exported"


def test_answer_override_trumps_combined_definitions(tmp_path):
    _seed_work_and_paragraph()
    _seed_annotated_word(
        "ungeziefer",
        "Ungeziefer",
        definition_de="Ein Insekt.",
        definition_en="vermin",
    )
    vocab_service.enqueue("ungeziefer", "para-1", "ctx")
    vocab_service.update_overrides("ungeziefer", answer="my custom answer")

    out = tmp_path / "vocab.csv"
    vocab_service.export_to_csv(out)

    with out.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        [row] = list(reader)
    assert row["answer"] == "my custom answer"
