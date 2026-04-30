"""Stream-answer tests: mocked Ollama NDJSON feed + offline fallback."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
import pytest

from backend.services import llm_service


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom", request=None, response=None  # type: ignore[arg-type]
            )

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

    def stream(self, method, url, **kwargs):
        return self._response


@pytest.mark.asyncio
async def test_stream_answer_yields_ollama_chunks(monkeypatch):
    ndjson = [
        json.dumps({"message": {"content": "Hallo "}}),
        json.dumps({"message": {"content": "Welt."}}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]
    fake = _FakeClient(_FakeStreamResponse(ndjson))

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **kw: fake
    )

    chunks: list[str] = []
    async for chunk in llm_service.stream_answer("Frage?", paragraph_id=None):
        chunks.append(chunk)

    assert chunks == ["Hallo ", "Welt."]


@pytest.mark.asyncio
async def test_stream_answer_falls_back_on_connect_error(monkeypatch):
    class _BoomClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, *a, **kw):
            raise httpx.ConnectError("no ollama")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _BoomClient())

    chunks: list[str] = []
    async for chunk in llm_service.stream_answer("Warum?", paragraph_id=None):
        chunks.append(chunk)

    joined = "".join(chunks)
    assert "Ollama" in joined
    assert "Offline" in joined


@pytest.mark.asyncio
async def test_stream_answer_stops_on_done(monkeypatch):
    ndjson = [
        json.dumps({"message": {"content": "A"}}),
        json.dumps({"message": {"content": "B"}, "done": True}),
        json.dumps({"message": {"content": "should-not-see"}}),
    ]
    fake = _FakeClient(_FakeStreamResponse(ndjson))

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: fake)

    chunks: list[str] = []
    async for chunk in llm_service.stream_answer("Q", paragraph_id=None):
        chunks.append(chunk)

    assert chunks == ["A", "B"]
