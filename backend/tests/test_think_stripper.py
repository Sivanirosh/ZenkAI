"""Tests for incremental `<think>` block stripping in LLM streams."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
import pytest

from backend.services import llm_service
from backend.services.llm_service import _ThinkStripper, _strip_thinking


def _feed(stripper: _ThinkStripper, chunks: list[str]) -> str:
    out = "".join(stripper.feed(c) for c in chunks)
    return out + stripper.flush()


def test_strip_thinking_removes_single_block():
    raw = "<think>Internal notes.</think>Hallo Welt."
    assert _strip_thinking(raw) == "Hallo Welt."


def test_strip_thinking_keeps_plain_text():
    assert _strip_thinking("Hallo Welt.") == "Hallo Welt."


def test_thinkstripper_passes_plain_text_through():
    stripper = _ThinkStripper()
    assert _feed(stripper, ["Hallo ", "Welt."]) == "Hallo Welt."


def test_thinkstripper_removes_inline_block():
    stripper = _ThinkStripper()
    assert (
        _feed(stripper, ["Pre <think>hidden</think>Post"]) == "Pre Post"
    )


def test_thinkstripper_handles_block_split_across_chunks():
    stripper = _ThinkStripper()
    assert (
        _feed(
            stripper,
            ["Hel", "lo <thi", "nk>hid", "den tho", "ughts</thi", "nk>World"],
        )
        == "Hello World"
    )


def test_thinkstripper_drops_unterminated_block():
    stripper = _ThinkStripper()
    assert _feed(stripper, ["Start <think>never closes"]) == "Start "


def test_thinkstripper_preserves_angle_brackets_that_are_not_think():
    stripper = _ThinkStripper()
    assert _feed(stripper, ["a <b>c</b> d"]) == "a <b>c</b> d"


@pytest.mark.asyncio
async def test_stream_answer_strips_think_blocks(monkeypatch):
    ndjson = [
        json.dumps({"message": {"content": "<think>plan</think>"}}),
        json.dumps({"message": {"content": "Die Antwort "}}),
        json.dumps({"message": {"content": "lautet A."}, "done": True}),
    ]

    class _Resp:
        def __init__(self) -> None:
            self.status_code = 200

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in ndjson:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Client())

    chunks: list[str] = []
    async for chunk in llm_service.stream_answer("Q", paragraph_id=None):
        chunks.append(chunk)

    assert "".join(chunks) == "Die Antwort lautet A."
    assert "plan" not in "".join(chunks)
