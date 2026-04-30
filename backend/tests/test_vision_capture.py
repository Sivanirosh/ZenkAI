"""Vision pipeline tests (PIVOT_ROADMAP §B.7).

Covers:
- Strict-JSON parsing of a happy Gemma 4 vision response.
- Re-prompt on validation failure (one retry).
- Deterministic skeleton when both attempts fail.
- The skeleton path is also picked when the active provider has no
  ``describe_image`` method.
"""

from __future__ import annotations

import pytest

from backend.llm import ollama as ollama_module
from backend.multimodal import vision_capture


GOOD_JSON = (
    '{"transcript": "Brot 3,50 €",'
    ' "words": [{"surface": "Brot", "lemma": "Brot", "pos": "noun",'
    ' "gender": "das", "definition_de": "Backware", "definition_en": "bread"}],'
    ' "advice": "Frag nach dem Preis pro Stück."}'
)

BAD_JSON = "Sure, here it is:\nThis isn't valid JSON at all."


def _patch_provider(monkeypatch, provider) -> None:
    import backend.llm as llm_pkg

    monkeypatch.setattr(llm_pkg, "get_provider", lambda: provider)
    monkeypatch.setattr(ollama_module, "get_provider", lambda: provider)


@pytest.mark.asyncio
async def test_describe_capture_parses_happy_path(monkeypatch):
    class _Provider:
        async def describe_image(self, image_bytes, *, prompt, timeout):
            assert b"" != image_bytes
            return GOOD_JSON

    _patch_provider(monkeypatch, _Provider())

    result = await vision_capture.describe_capture(
        b"\xff\xd8\xff" + b"\x00" * 32,
        target_cefr="B1",
        surface_kind="menu",
        note="ein Café in München",
    )
    assert result.engine == "gemma_vision"
    assert "Brot" in result.transcript
    assert len(result.words) == 1
    assert result.words[0].lemma == "Brot"
    assert "Frag" in result.advice


@pytest.mark.asyncio
async def test_describe_capture_retries_then_falls_back_to_skeleton(monkeypatch):
    calls = {"n": 0}

    class _Provider:
        async def describe_image(self, image_bytes, *, prompt, timeout):
            calls["n"] += 1
            return BAD_JSON

    _patch_provider(monkeypatch, _Provider())

    result = await vision_capture.describe_capture(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 32,
        surface_kind="sign",
        note="",
    )
    assert calls["n"] == 2, "must retry exactly once before giving up"
    assert result.engine == "skeleton"
    assert result.transcript == ""
    assert result.advice  # skeleton always provides a fallback sentence


@pytest.mark.asyncio
async def test_describe_capture_skeleton_when_provider_has_no_vision(
    monkeypatch,
):
    class _Provider:
        pass

    _patch_provider(monkeypatch, _Provider())

    result = await vision_capture.describe_capture(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
        surface_kind="object",
        note="ein Schraubendreher",
    )
    assert result.engine == "skeleton"
    assert "Schraubendreher" in result.advice


@pytest.mark.asyncio
async def test_describe_capture_handles_extracted_json_from_prose(monkeypatch):
    class _Provider:
        async def describe_image(self, image_bytes, *, prompt, timeout):
            return (
                "Sure! Here's the JSON you asked for:\n\n" + GOOD_JSON + "\n"
            )

    _patch_provider(monkeypatch, _Provider())

    result = await vision_capture.describe_capture(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,
        surface_kind="menu",
        note="",
    )
    assert result.engine == "gemma_vision"
    assert result.transcript.startswith("Brot")
