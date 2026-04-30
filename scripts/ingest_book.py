"""Ingest a book end-to-end: download → clean → tokenise → DuckDB.

Usage:
    python scripts/ingest_book.py --gutenberg-id 5200
    python scripts/ingest_book.py --epub /path/to/book.epub \
        --work-id custom_id --title "Title" --author "Author"

The full pipeline runs inside a single DuckDB transaction. On any failure
the transaction is rolled back and no partial rows remain (AGENT.md rule).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion import gutenberg as gb  # noqa: E402
from backend.ingestion.epub_parser import parse_epub  # noqa: E402
from backend.ingestion.tokeniser import Token, tokenise_paragraph  # noqa: E402
from backend.models import db  # noqa: E402

logger = logging.getLogger("ingest_book")


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return value or "work"


def derive_work_id(author: str, title: str) -> str:
    return f"{slugify(author.split(',')[0])}_{slugify(title)}"[:80]


def detect_chapters(paragraphs: list[str]) -> list[tuple[int, str]]:
    """Return (chapter, paragraph_text) pairs. Detects simple chapter headings."""
    chapter = 1
    out: list[tuple[int, str]] = []
    heading = re.compile(
        r"^(KAPITEL|KAPITTEL|CAPÍTULO|CHAPTER|I{1,3}V?I{0,3}|"
        r"\d{1,3})[\.\s:]",
        flags=re.IGNORECASE,
    )
    for para in paragraphs:
        stripped = para.strip()
        if len(stripped) < 60 and heading.match(stripped):
            chapter += 1
            continue
        out.append((chapter, para))
    if not out:
        return [(1, p) for p in paragraphs]
    return out


def _insert_work(conn, work_id: str, title: str, author: str, **kwargs) -> None:
    row = conn.execute("SELECT id FROM works WHERE id = ?", [work_id]).fetchone()
    if row is not None:
        raise ValueError(
            f"Work '{work_id}' already ingested. "
            "Remove it from DuckDB first if you want to re-ingest."
        )
    conn.execute(
        """
        INSERT INTO works (id, title, author, epoch, year, gutenberg_id,
                           language, spine_color, epoch_color)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            work_id,
            title,
            author,
            kwargs.get("epoch"),
            kwargs.get("year"),
            kwargs.get("gutenberg_id"),
            kwargs.get("language", "de"),
            kwargs.get("spine_color", "#085041"),
            kwargs.get("epoch_color", "#9FE1CB"),
        ],
    )


def _insert_paragraph(
    conn, work_id: str, chapter: int, position: int, text: str
) -> str:
    paragraph_id = f"{work_id}_c{chapter:02d}_p{position:04d}"
    word_count = sum(1 for t in text.split() if t.strip())
    conn.execute(
        """
        INSERT INTO paragraphs (id, work_id, chapter, position, text, word_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [paragraph_id, work_id, chapter, position, text, word_count],
    )
    return paragraph_id


def _upsert_word(conn, lemma: str, pos: str | None) -> str:
    word_id = slugify(lemma)
    row = conn.execute("SELECT id FROM words WHERE id = ?", [word_id]).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO words (id, lemma, pos) VALUES (?, ?, ?)",
            [word_id, lemma, pos],
        )
        return word_id
    return row[0]


def _insert_occurrence(
    conn, paragraph_id: str, word_id: str, token: Token
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
            token.surface_form,
            token.position,
            token.case_label,
            token.grammatical_role,
            token.char_start,
            token.char_end,
        ],
    )


def run(
    *,
    gutenberg_id: int | None,
    epub_path: str | None,
    work_id: str | None,
    title: str | None,
    author: str | None,
    epoch: str | None,
    year: int | None,
    language: str,
    spine_color: str | None,
    epoch_color: str | None,
) -> int:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")

    if gutenberg_id is not None:
        fetched = gb.fetch_text(gutenberg_id)
        paragraphs_raw = gb.split_into_paragraphs(fetched.body)
        title = title or fetched.raw_title or f"Gutenberg #{gutenberg_id}"
        author = author or fetched.raw_author or "Unbekannt"
        effective_work_id = work_id or derive_work_id(author, title)
        gb_id = gutenberg_id
    elif epub_path is not None:
        paragraphs_raw = parse_epub(epub_path)
        title = title or Path(epub_path).stem
        author = author or "Unbekannt"
        effective_work_id = work_id or derive_work_id(author, title)
        gb_id = None
    else:
        raise SystemExit("Either --gutenberg-id or --epub is required")

    paragraphs_with_chapter = detect_chapters(paragraphs_raw)
    logger.info(
        "Prepared %d paragraphs for '%s' by %s (work_id=%s)",
        len(paragraphs_with_chapter),
        title,
        author,
        effective_work_id,
    )

    conn = db.get_connection()
    db.init_schema(conn)

    existing = conn.execute(
        "SELECT id FROM works WHERE id = ?", [effective_work_id]
    ).fetchone()
    if existing is not None:
        logger.warning("Work '%s' already present — skipping", effective_work_id)
        return 0

    unique_word_ids: set[str] = set()
    new_word_ids: set[str] = set()
    total_paragraphs = 0

    conn.execute("BEGIN")
    try:
        _insert_work(
            conn,
            effective_work_id,
            title,
            author,
            epoch=epoch,
            year=year,
            gutenberg_id=gb_id,
            language=language,
            spine_color=spine_color or "#085041",
            epoch_color=epoch_color or "#9FE1CB",
        )

        chapter_position_counters: dict[int, int] = {}
        for chapter, paragraph_text in paragraphs_with_chapter:
            pos = chapter_position_counters.get(chapter, 0)
            paragraph_id = _insert_paragraph(
                conn, effective_work_id, chapter, pos, paragraph_text
            )
            chapter_position_counters[chapter] = pos + 1
            total_paragraphs += 1

            tokens = tokenise_paragraph(paragraph_text)
            for token in tokens:
                if not token.is_word:
                    continue
                pre = conn.execute(
                    "SELECT id FROM words WHERE id = ?",
                    [slugify(token.lemma)],
                ).fetchone()
                word_id = _upsert_word(conn, token.lemma, token.pos)
                unique_word_ids.add(word_id)
                if pre is None:
                    new_word_ids.add(word_id)
                _insert_occurrence(conn, paragraph_id, word_id, token)

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        logger.exception("Ingestion failed; transaction rolled back")
        raise

    logger.info(
        "Done: %d paragraphs, %d unique words, %d new words (work_id=%s)",
        total_paragraphs,
        len(unique_word_ids),
        len(new_word_ids),
        effective_work_id,
    )
    db.close_connection()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a book into DuckDB")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gutenberg-id", type=int, help="Project Gutenberg book ID")
    source.add_argument("--epub", type=str, help="Path to a local .epub file")
    parser.add_argument("--work-id", type=str, help="Override generated work_id")
    parser.add_argument("--title", type=str)
    parser.add_argument("--author", type=str)
    parser.add_argument("--epoch", type=str, help="e.g. Expressionismus, Romantik")
    parser.add_argument("--year", type=int)
    parser.add_argument("--language", type=str, default="de")
    parser.add_argument("--spine-color", type=str)
    parser.add_argument("--epoch-color", type=str)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(
        run(
            gutenberg_id=args.gutenberg_id,
            epub_path=args.epub,
            work_id=args.work_id,
            title=args.title,
            author=args.author,
            epoch=args.epoch,
            year=args.year,
            language=args.language,
            spine_color=args.spine_color,
            epoch_color=args.epoch_color,
        )
    )
