"""spaCy-based tokenisation for German text.

Uses `de_core_news_sm`. Every paragraph is tokenised into typed tokens that
capture surface form, lemma, POS, and — where the dep parse gives enough
signal — grammatical role and morphological case.

Each token also carries ``char_start`` / ``char_end`` offsets into the
source paragraph text (Python codepoint indices, matching spaCy's
``Token.idx``). These offsets are the contract the reader relies on:
``paragraph.text[char_start:char_end] == surface_form`` must hold for every
word token. Punctuation is implicit — it's every character between two
adjacent word ranges.

Constraint: offsets are Python codepoint indices. JS ``String.prototype.slice``
operates on UTF-16 code units, which coincide with codepoints for all BMP
characters (covers every European script, including German umlauts / Eszett).
Corpora containing supplementary-plane characters (emoji, rare CJK
extensions) would need UTF-16 offsets or ``Array.from(text)`` on the client.

Install once before running:

    python -m spacy download de_core_news_sm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

_CASE_MAP = {
    "Nom": "Nominativ",
    "Acc": "Akkusativ",
    "Dat": "Dativ",
    "Gen": "Genitiv",
}

_ROLE_MAP = {
    "sb": "Subjekt",
    "oa": "Objekt (Akk.)",
    "da": "Dativ-Objekt",
    "og": "Genitiv-Objekt",
    "op": "Präpositionalobjekt",
    "mo": "Adverbial",
    "nk": "Attribut",
    "ag": "Genitiv-Attribut",
    "pd": "Prädikativ",
    "cj": "Konjunkt",
    "ROOT": "Prädikat",
}


@dataclass
class Token:
    surface_form: str
    lemma: str
    pos: Optional[str]
    is_word: bool
    case_label: Optional[str]
    grammatical_role: Optional[str]
    position: int
    char_start: int
    char_end: int


@lru_cache(maxsize=1)
def _load_nlp() -> Language:
    try:
        return spacy.load("de_core_news_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'de_core_news_sm' is not installed. "
            "Run: python -m spacy download de_core_news_sm"
        ) from exc


def _map_case(token) -> Optional[str]:
    raw = token.morph.get("Case")
    if not raw:
        return None
    return _CASE_MAP.get(raw[0], raw[0])


def _map_role(dep: str) -> Optional[str]:
    return _ROLE_MAP.get(dep)


def tokenise_paragraph(text: str) -> list[Token]:
    """Tokenise a single paragraph, keeping punctuation as non-word tokens."""
    nlp = _load_nlp()
    doc = nlp(text)
    tokens: list[Token] = []
    position = 0
    for spacy_tok in doc:
        if spacy_tok.is_space:
            continue
        is_word = spacy_tok.is_alpha
        start = spacy_tok.idx
        end = start + len(spacy_tok.text)
        tokens.append(
            Token(
                surface_form=spacy_tok.text,
                lemma=spacy_tok.lemma_.lower() if is_word else spacy_tok.text,
                pos=spacy_tok.pos_ if is_word else None,
                is_word=is_word,
                case_label=_map_case(spacy_tok) if is_word else None,
                grammatical_role=_map_role(spacy_tok.dep_) if is_word else None,
                position=position,
                char_start=start,
                char_end=end,
            )
        )
        position += 1
    return tokens
