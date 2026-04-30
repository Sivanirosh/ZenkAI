"""Download and clean Project Gutenberg texts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_GUTENBERG_URLS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
]

_BOILERPLATE_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    flags=re.IGNORECASE | re.DOTALL,
)
_BOILERPLATE_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass
class GutenbergText:
    gutenberg_id: int
    raw_title: Optional[str]
    raw_author: Optional[str]
    body: str


def _extract_metadata(header: str) -> tuple[Optional[str], Optional[str]]:
    title_match = re.search(r"^Title:\s*(.+)$", header, flags=re.MULTILINE)
    author_match = re.search(r"^Author:\s*(.+)$", header, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else None
    author = author_match.group(1).strip() if author_match else None
    return title, author


def strip_boilerplate(text: str) -> tuple[str, str]:
    """Return (header, body) split at the `*** START …` / `*** END …` markers."""
    start = _BOILERPLATE_START.search(text)
    end = _BOILERPLATE_END.search(text)
    if start is None or end is None:
        logger.warning("Could not find Gutenberg boilerplate markers; using full text")
        return "", text.strip()
    header = text[: start.start()]
    body = text[start.end() : end.start()]
    return header, body.strip()


def fetch_text(gutenberg_id: int, *, timeout: float = 30.0) -> GutenbergText:
    """Download a Gutenberg text by ID, trying the common URL patterns."""
    last_error: Optional[Exception] = None
    for template in _GUTENBERG_URLS:
        url = template.format(id=gutenberg_id)
        try:
            logger.info("Fetching %s", url)
            response = httpx.get(url, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            encoding = response.encoding or "utf-8"
            text = response.content.decode(encoding, errors="replace")
            header, body = strip_boilerplate(text)
            title, author = _extract_metadata(header or text[:4000])
            return GutenbergText(
                gutenberg_id=gutenberg_id,
                raw_title=title,
                raw_author=author,
                body=body,
            )
        except httpx.HTTPError as exc:
            logger.debug("URL %s failed: %s", url, exc)
            last_error = exc
            continue
    raise RuntimeError(
        f"Could not fetch Gutenberg ID {gutenberg_id}: {last_error}"
    )


def split_into_paragraphs(body: str) -> list[str]:
    """Split cleaned body into paragraphs on blank lines, preserving order."""
    raw = re.split(r"\n\s*\n", body)
    paragraphs = []
    for chunk in raw:
        cleaned = re.sub(r"\s+", " ", chunk).strip()
        if len(cleaned) >= 40:
            paragraphs.append(cleaned)
    return paragraphs
