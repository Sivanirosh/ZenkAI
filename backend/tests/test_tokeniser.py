"""Property tests for char-offset guarantees from the tokeniser.

These exercise the reader's single-source-of-truth invariant:
``paragraph.text[char_start:char_end] == surface_form`` for every word
token, and the text between adjacent word ranges contains no alphabetic
characters. The reader relies on both to render punctuation correctly.

Skipped automatically when ``de_core_news_sm`` is not installed so CI on
bare environments still passes.
"""

from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")

try:
    spacy.load("de_core_news_sm")
    _MODEL_AVAILABLE = True
except OSError:  # pragma: no cover — skip branch
    _MODEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="spaCy model 'de_core_news_sm' not installed",
)

from backend.ingestion.tokeniser import tokenise_paragraph  # noqa: E402


SAMPLES = [
    'Er sagte: „Komm doch her!" — und lachte.',
    "Gregor verwandelte sich — über Nacht — in ein Ungeziefer.",
    "Wörter, Sätze und ganz normale Tage … nichts Besonderes!",
    'Sie rief "Halt!" und rannte weiter.',
]


@pytest.mark.parametrize("text", SAMPLES)
def test_offsets_slice_back_to_surface_form(text):
    tokens = tokenise_paragraph(text)
    assert tokens, "tokeniser produced no tokens"
    for t in tokens:
        assert text[t.char_start:t.char_end] == t.surface_form


@pytest.mark.parametrize("text", SAMPLES)
def test_ranges_are_monotonic_and_non_overlapping(text):
    tokens = tokenise_paragraph(text)
    prev_end = 0
    for t in tokens:
        assert t.char_start >= prev_end
        assert t.char_end > t.char_start
        prev_end = t.char_end


@pytest.mark.parametrize("text", SAMPLES)
def test_inter_word_gaps_contain_no_letters(text):
    """Everything between two adjacent word ranges must be punctuation/space."""
    word_tokens = [t for t in tokenise_paragraph(text) if t.is_word]
    cursor = 0
    for t in word_tokens:
        gap = text[cursor:t.char_start]
        assert not any(c.isalpha() for c in gap), (
            f"gap {gap!r} leaks letters between words"
        )
        cursor = t.char_end
    tail = text[cursor:]
    assert not any(c.isalpha() for c in tail)
