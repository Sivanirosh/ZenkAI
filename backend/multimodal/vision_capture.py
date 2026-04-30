"""Strict-JSON Gemma 4 vision call for the Capture room (PIVOT_ROADMAP §B.7).

One public function:

- ``describe_capture(image_bytes, *, target_cefr, surface_kind, note)``
  → ``VisionCaptureResult`` with ``transcript``, ``words[*]`` (top-3
  vocab with quick definitions), and ``advice`` (one short sentence).

Order of preference:

1. Gemma 4 image-multimodal call via the active provider's
   ``describe_image`` method (Ollama provider does this today).
2. One reprompt asking the model to "fix the JSON" if the first
   response failed schema validation.
3. Deterministic skeleton — never blocks the Capture room.

We always return a ``VisionCaptureResult`` so the router can present
something coherent even when the LLM is unavailable. The caller
records the surface kind in the captures index regardless of which
path produced the transcript.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ─── Result schema ──────────────────────────────────────────────────────


@dataclass
class CaptureWord:
    """One vocabulary item the model surfaced from the image."""

    surface: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    gender: Optional[str] = None
    definition_de: Optional[str] = None
    definition_en: Optional[str] = None


@dataclass
class VisionCaptureResult:
    """Returned to the router after one ``describe_capture`` call."""

    transcript: str
    surface_kind: str
    words: list[CaptureWord] = field(default_factory=list)
    advice: str = ""
    engine: str = "gemma_vision"  # | "skeleton"
    model: Optional[str] = None
    raw: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "surface_kind": self.surface_kind,
            "words": [asdict(w) for w in self.words],
            "advice": self.advice,
            "engine": self.engine,
            "model": self.model,
        }


# ─── Internal Pydantic schema for parsing the model output ──────────────


class _ParsedWord(BaseModel):
    surface: str = Field(min_length=1, max_length=80)
    lemma: Optional[str] = Field(default=None, max_length=80)
    pos: Optional[str] = Field(default=None, max_length=24)
    gender: Optional[str] = Field(default=None, max_length=8)
    definition_de: Optional[str] = Field(default=None, max_length=200)
    definition_en: Optional[str] = Field(default=None, max_length=200)


class _ParsedResult(BaseModel):
    transcript: str = Field(default="", max_length=2000)
    words: list[_ParsedWord] = Field(default_factory=list)
    advice: str = Field(default="", max_length=240)


# ─── Public entry point ─────────────────────────────────────────────────


_VISION_PROMPT = (
    "You are Mira, a calm German tutor. The learner just photographed "
    "something they saw in the world: kind = '{surface_kind}'. "
    "Write OUTPUT STRICT JSON ONLY (no Markdown, no commentary), "
    "matching exactly:\n\n"
    "{{\n"
    '  "transcript": string,           // verbatim German text in the '
    "image, line breaks preserved with \\n. Empty string if no text.\n"
    '  "words": [                       // 0-3 most useful items for a '
    "{cefr} learner\n"
    "    {{\n"
    '      "surface": string,\n'
    '      "lemma": string|null,\n'
    '      "pos": string|null,         // noun|verb|adj|adv|...\n'
    '      "gender": "der"|"die"|"das"|null,\n'
    '      "definition_de": string|null,\n'
    '      "definition_en": string|null\n'
    "    }}\n"
    "  ],\n"
    '  "advice": string                // one short German sentence the '
    "learner can act on\n"
    "}}\n\n"
    "Note from the learner: {note}"
)


async def describe_capture(
    image_bytes: bytes,
    *,
    target_cefr: str = "B1",
    surface_kind: str = "text",
    note: str = "",
    timeout: float = 30.0,
) -> VisionCaptureResult:
    """Describe a captured image using the active provider's vision call.

    Never raises. Always returns a usable ``VisionCaptureResult`` —
    falls back to a deterministic skeleton when the LLM is unavailable
    or its output cannot be parsed.
    """
    if not image_bytes:
        return _skeleton(surface_kind=surface_kind, note=note)

    try:
        from backend.llm import get_provider  # noqa: WPS433
        from backend.services.runtime_config import get_llm_options  # noqa: WPS433
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("vision_capture imports failed: %s", exc)
        return _skeleton(surface_kind=surface_kind, note=note)

    provider = get_provider()
    describe_fn = getattr(provider, "describe_image", None)
    if describe_fn is None:
        return _skeleton(surface_kind=surface_kind, note=note)

    prompt = _VISION_PROMPT.format(
        surface_kind=surface_kind,
        cefr=target_cefr,
        note=(note or "(none)"),
    )
    options = get_llm_options()

    raw = await _call(describe_fn, image_bytes, prompt, timeout)
    parsed = _parse(raw)
    if parsed is None:
        fix_prompt = (
            prompt
            + "\n\nYour previous output was not valid JSON or did not "
            "match the schema. Re-emit ONLY the JSON object. No prose."
        )
        raw = await _call(describe_fn, image_bytes, fix_prompt, timeout)
        parsed = _parse(raw)

    if parsed is None:
        result = _skeleton(surface_kind=surface_kind, note=note)
        result.raw = raw
        return result

    return VisionCaptureResult(
        transcript=parsed.transcript.strip(),
        surface_kind=surface_kind,
        words=[
            CaptureWord(
                surface=w.surface,
                lemma=w.lemma,
                pos=w.pos,
                gender=w.gender,
                definition_de=w.definition_de,
                definition_en=w.definition_en,
            )
            for w in parsed.words[:3]
        ],
        advice=parsed.advice.strip(),
        engine="gemma_vision",
        model=options.model,
        raw=raw,
    )


# ─── Helpers ────────────────────────────────────────────────────────────


async def _call(
    describe_fn: Any,
    image_bytes: bytes,
    prompt: str,
    timeout: float,
) -> Optional[str]:
    try:
        return await describe_fn(
            image_bytes,
            prompt=prompt,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover — provider already swallows
        logger.info("describe_image raised: %s", exc)
        return None


def _parse(text: Optional[str]) -> Optional[_ParsedResult]:
    obj = _parse_json_object(text)
    if obj is None:
        return None
    try:
        return _ParsedResult.model_validate(obj)
    except ValidationError as exc:
        logger.info("VisionCaptureResult validation failed: %s", exc)
        return None


def _parse_json_object(text: Optional[str]) -> Optional[dict[str, Any]]:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _skeleton(*, surface_kind: str, note: str) -> VisionCaptureResult:
    """Used when the LLM is unavailable — keep the room responsive."""
    advice_for = {
        "menu": "Frag mich nach einem Wort, das du nicht erkennst.",
        "sign": "Ich beschreibe dir das Schild beim nächsten Versuch.",
        "object": "Beim nächsten Versuch nenne ich dir den Gegenstand.",
        "text": "Beim nächsten Versuch lese ich den Text vor.",
    }
    return VisionCaptureResult(
        transcript="",
        surface_kind=surface_kind,
        words=[],
        advice=advice_for.get(surface_kind, "")
        + (f" (Notiz: {note})" if note else ""),
        engine="skeleton",
        model=None,
        raw=None,
    )
