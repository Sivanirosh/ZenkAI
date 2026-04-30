"""One-shot: populate word_occurrences.char_start / char_end.

Existing databases store word occurrences without character offsets into
the paragraph text. The reader now slices paragraphs.text by those offsets,
so every pre-existing row needs them filled in.

Strategy: re-tokenise each paragraph with the same spaCy model used during
ingestion, take the alphabetic tokens in order, and pair them with the
existing word_occurrences rows in DB order. Pairing must be exact — if any
pair disagrees on ``surface_form`` the whole paragraph (and run) aborts so
we never write silently-drifted offsets.

Run once after applying the schema migration:

    conda activate medvlm-base
    python scripts/backfill_offsets.py            # all paragraphs
    python scripts/backfill_offsets.py --work-id kafka_verwandlung
    python scripts/backfill_offsets.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingestion.tokeniser import tokenise_paragraph  # noqa: E402
from backend.models import db  # noqa: E402

logger = logging.getLogger("backfill_offsets")


class BackfillMismatch(RuntimeError):
    """Raised when a freshly-tokenised word does not match the stored row."""


def _paragraphs_missing_offsets(
    conn, work_id: str | None
) -> list[tuple[str, str]]:
    """Return (paragraph_id, text) for paragraphs with unfilled offsets.

    A paragraph is considered "needs backfill" if any of its occurrences
    still has ``char_start IS NULL`` — re-running on already-backfilled
    paragraphs would be wasted work.
    """
    params: list[object] = []
    where_work = ""
    if work_id is not None:
        where_work = "AND p.work_id = ?"
        params.append(work_id)

    rows = conn.execute(
        f"""
        SELECT DISTINCT p.id, p.text
        FROM paragraphs p
        JOIN word_occurrences o ON o.paragraph_id = p.id
        WHERE o.char_start IS NULL
          {where_work}
        ORDER BY p.work_id, p.chapter, p.position
        """,
        params,
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _backfill_paragraph(conn, paragraph_id: str, text: str) -> int:
    """Fill char_start/char_end for one paragraph. Returns rows updated."""
    occ_rows = conn.execute(
        """
        SELECT id, surface_form
        FROM word_occurrences
        WHERE paragraph_id = ?
        ORDER BY position
        """,
        [paragraph_id],
    ).fetchall()

    word_tokens = [t for t in tokenise_paragraph(text) if t.is_word]

    if len(word_tokens) != len(occ_rows):
        raise BackfillMismatch(
            f"Paragraph {paragraph_id}: DB has {len(occ_rows)} occurrences "
            f"but spaCy produced {len(word_tokens)} word tokens — refusing "
            f"to guess the alignment."
        )

    for (occ_id, stored_surface), tok in zip(occ_rows, word_tokens):
        if stored_surface != tok.surface_form:
            raise BackfillMismatch(
                f"Paragraph {paragraph_id} at position-ordered row {occ_id}: "
                f"stored surface {stored_surface!r} != tokenised "
                f"{tok.surface_form!r}."
            )
        if text[tok.char_start : tok.char_end] != tok.surface_form:
            raise BackfillMismatch(
                f"Paragraph {paragraph_id}: offset invariant broken for "
                f"{tok.surface_form!r} at [{tok.char_start}, {tok.char_end})."
            )
        conn.execute(
            """
            UPDATE word_occurrences
            SET char_start = ?, char_end = ?
            WHERE id = ?
            """,
            [tok.char_start, tok.char_end, occ_id],
        )
    return len(occ_rows)


def run(work_id: str | None, dry_run: bool) -> int:
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")

    conn = db.get_connection()
    db.init_schema(conn)

    pending = _paragraphs_missing_offsets(conn, work_id)
    if not pending:
        logger.info("No paragraphs need backfill.")
        return 0

    logger.info("Backfilling %d paragraphs%s", len(pending),
                f" for work_id={work_id}" if work_id else "")

    if dry_run:
        logger.info("Dry-run: no writes performed.")
        return 0

    conn.execute("BEGIN")
    try:
        total = 0
        for pid, text in pending:
            total += _backfill_paragraph(conn, pid, text)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        logger.exception("Backfill failed; transaction rolled back")
        raise

    logger.info("Updated %d word_occurrences rows across %d paragraphs",
                total, len(pending))
    db.close_connection()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill char_start/char_end on word_occurrences."
    )
    parser.add_argument(
        "--work-id",
        type=str,
        default=None,
        help="Restrict to a single work (default: all works).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without writing.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(run(work_id=args.work_id, dry_run=args.dry_run))
