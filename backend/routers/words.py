"""Dictionary word lookup for the Wortkarte."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models import db
from backend.models.pydantic_models import Word

router = APIRouter()


@router.get("/{word_id}", response_model=Word)
def get_word(word_id: str) -> Word:
    row = db.cursor().execute(
        """
        SELECT id, lemma, pos, gender, plural_form, etymology,
               definition_de, definition_en
        FROM words WHERE id = ?
        """,
        [word_id],
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Word '{word_id}' not found")
    return Word(
        id=row[0],
        lemma=row[1],
        pos=row[2],
        gender=row[3],
        plural_form=row[4],
        etymology=row[5],
        definition_de=row[6],
        definition_en=row[7],
    )
