"""LLMProvider Protocol conformance + OllamaProvider chat_with_tools wire format.

Goal: prove that
  1. ``OllamaProvider`` satisfies the ``LLMProvider`` Protocol structurally.
  2. ``chat_with_tools`` translates Ollama's NDJSON message stream into the
     correct sequence of ``ProviderEvent`` (text -> tool_call -> done).
  3. Tool arguments are coerced from both the dict form and the JSON-string
     form Ollama may emit.
  4. HTTP errors yield exactly one error event followed by exactly one done.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
import pytest

from backend.llm.ollama import OllamaProvider
from backend.llm.provider import (
    LLMProvider,
    ProviderEvent,
    ProviderEventKind,
    ToolSchema,
)
from backend.services.runtime_config import LlmOptions


# ─── Stream stubs (mirror what Ollama emits over /api/chat) ──────────────


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeClient:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, _method, _url, **_kwargs):
        return self._response


def _install(monkeypatch, response: _FakeStreamResponse) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _FakeClient(response))


# ─── Tests ───────────────────────────────────────────────────────────────


def test_ollama_provider_satisfies_protocol() -> None:
    provider = OllamaProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.name == "ollama"


@pytest.mark.asyncio
async def test_chat_with_tools_streams_text_then_tool_call_then_done(monkeypatch):
    ndjson = [
        json.dumps({"message": {"content": "Ich pruefe... "}}),
        json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "recommend_text",
                                "arguments": {"reason": "continuation"},
                            },
                        }
                    ],
                }
            }
        ),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]
    _install(monkeypatch, _FakeStreamResponse(ndjson))

    provider = OllamaProvider()
    events: list[ProviderEvent] = []
    async for ev in provider.chat_with_tools(
        messages=[{"role": "user", "content": "go"}],
        tools=[
            ToolSchema(
                name="recommend_text",
                description="next paragraph",
                parameters={"type": "object", "properties": {}},
            )
        ],
        options=LlmOptions(model="gemma4:e4b"),
    ):
        events.append(ev)

    kinds = [e.kind for e in events]
    assert kinds[0] == ProviderEventKind.TEXT
    assert events[0].text and "Ich pruefe" in events[0].text
    tool_evs = [e for e in events if e.kind == ProviderEventKind.TOOL_CALL]
    assert len(tool_evs) == 1
    assert tool_evs[0].tool_name == "recommend_text"
    assert tool_evs[0].tool_args == {"reason": "continuation"}
    assert tool_evs[0].call_id == "call_1"
    assert kinds[-1] == ProviderEventKind.DONE
    # Exactly one done event.
    assert sum(1 for k in kinds if k == ProviderEventKind.DONE) == 1


@pytest.mark.asyncio
async def test_chat_with_tools_parses_json_string_arguments(monkeypatch):
    ndjson = [
        json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "start_drill",
                                "arguments": '{"competency_id":"verbs.modal","count":5}',
                            }
                        }
                    ],
                }
            }
        ),
        json.dumps({"done": True}),
    ]
    _install(monkeypatch, _FakeStreamResponse(ndjson))

    provider = OllamaProvider()
    events = [
        ev
        async for ev in provider.chat_with_tools(
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            options=LlmOptions(model="gemma4:e4b"),
        )
    ]
    tool_evs = [e for e in events if e.kind == ProviderEventKind.TOOL_CALL]
    assert tool_evs[0].tool_args == {"competency_id": "verbs.modal", "count": 5}


@pytest.mark.asyncio
async def test_chat_with_tools_emits_error_then_done_on_http_error(monkeypatch):
    _install(monkeypatch, _FakeStreamResponse([], status_code=503))

    provider = OllamaProvider()
    events = [
        ev
        async for ev in provider.chat_with_tools(
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            options=LlmOptions(model="gemma4:e4b"),
        )
    ]
    kinds = [e.kind for e in events]
    assert kinds == [ProviderEventKind.ERROR, ProviderEventKind.DONE]


@pytest.mark.asyncio
async def test_chat_with_tools_emits_error_then_done_on_connect_error(monkeypatch):
    class _Boom:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            raise httpx.ConnectError("no daemon")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Boom())

    provider = OllamaProvider()
    events = [
        ev
        async for ev in provider.chat_with_tools(
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            options=LlmOptions(model="gemma4:e4b"),
        )
    ]
    kinds = [e.kind for e in events]
    assert ProviderEventKind.ERROR in kinds
    assert kinds[-1] == ProviderEventKind.DONE


@pytest.mark.asyncio
async def test_chat_with_tools_aclose_after_done_does_not_raise(monkeypatch):
    """Regression: previously the generator suspended after ``yield done()``
    (because ``emitted_done = True`` ran AFTER the yield) and then a later
    ``aclose()`` triggered ``RuntimeError: async generator ignored
    GeneratorExit`` because the ``finally`` block tried to yield again.

    Reproduces the exact warning seen in dev (``[api] ERROR:asyncio: Task
    exception was never retrieved ... async generator ignored GeneratorExit``)
    by manually iterating until DONE, then calling ``aclose()``.
    """
    ndjson = [
        json.dumps({"message": {"content": "hi"}}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]
    _install(monkeypatch, _FakeStreamResponse(ndjson))
    provider = OllamaProvider()

    gen = provider.chat_with_tools(
        messages=[{"role": "user", "content": "go"}],
        tools=[],
        options=LlmOptions(model="gemma4:e4b"),
    )
    seen: list[ProviderEventKind] = []
    async for ev in gen:
        seen.append(ev.kind)
        if ev.kind == ProviderEventKind.DONE:
            break  # mimic the controller's break — leaves gen suspended

    assert ProviderEventKind.DONE in seen
    assert sum(1 for k in seen if k == ProviderEventKind.DONE) == 1
    # Must NOT raise RuntimeError("async generator ignored GeneratorExit").
    await gen.aclose()
    # Idempotent.
    await gen.aclose()


@pytest.mark.asyncio
async def test_list_models_returns_empty_when_unreachable(monkeypatch):
    class _BoomClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, _url):
            raise httpx.ConnectError("nope")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _BoomClient())
    provider = OllamaProvider()
    assert await provider.list_models() == []
