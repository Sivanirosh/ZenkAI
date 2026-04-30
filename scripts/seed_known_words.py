"""Seed high-frequency German lemmas as ``status='known'`` in vocab_queue.

Idempotent: re-running does not regress queued or exported words.

Usage:
    python scripts/seed_known_words.py
    python scripts/seed_known_words.py --file custom.txt --top 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models import db  # noqa: E402
from backend.services import vocab_service  # noqa: E402


DEFAULT_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "de_frequency_top1000.txt"
)


def _read_lemmas(path: Path, top: int | None) -> Iterator[str]:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            lemma = line.split("#", 1)[0].strip()
            if not lemma:
                continue
            yield lemma
            count += 1
            if top is not None and count >= top:
                return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help=f"Path to the frequency list (default: {DEFAULT_FILE})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only seed the first N lemmas (default: seed the whole file)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO", format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("seed_known_words")

    if not args.file.exists():
        logger.error("Frequency file not found: %s", args.file)
        return 1

    logger.info(
        "Seeding known words from %s%s",
        args.file,
        f" (top {args.top})" if args.top else "",
    )
    db.init_schema()

    lemmas = list(_read_lemmas(args.file, args.top))
    if not lemmas:
        logger.warning("No lemmas to seed (empty file?)")
        return 0

    result = vocab_service.bulk_seed_known(lemmas)
    logger.info(
        "Done: %d seeded, %d skipped (already present), %d new word rows",
        result["seeded"],
        result["skipped"],
        result["created_words"],
    )
    db.close_connection()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
