"""EPUB → ordered paragraph list via ebooklib + BeautifulSoup."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import ebooklib  # type: ignore
    from ebooklib import epub  # type: ignore
except Exception:  # pragma: no cover
    ebooklib = None  # type: ignore
    epub = None  # type: ignore

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_epub(path: str | Path) -> list[str]:
    """Return paragraphs from the EPUB at `path`, in spine order."""
    if ebooklib is None or epub is None:
        raise RuntimeError(
            "ebooklib is not installed. Add it to pyproject.toml dependencies."
        )
    book = epub.read_epub(str(path))
    paragraphs: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for p in soup.find_all(["p", "div"]):
            text = p.get_text(" ", strip=True)
            if len(text) >= 40:
                paragraphs.append(text)
    logger.info("Parsed %d paragraphs from %s", len(paragraphs), path)
    return paragraphs
