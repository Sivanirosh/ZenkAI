"""LLMProvider Protocol — the contract every backend implements.

A provider is an async, stateless interface over an LLM runtime. We deliberately
stay close to OpenAI's tool-calling shape because it is the lingua franca for
function calling across Ollama, LiteRT bridges, llama.cpp, and Cactus.

Tool schemas use OpenAI's ``{"type": "function", "function": {"name", "description",
"parameters": <JSON Schema>}}`` envelope — Ollama 0.4+ passes this straight through
to the underlying chat template for Gemma / Llama / Qwen tool-calling models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional, Protocol, runtime_checkable

from backend.services.runtime_config import LlmOptions


# ─── Tool descriptor (OpenAI-compatible) ─────────────────────────────────


@dataclass(frozen=True)
class ToolSchema:
    """A single tool descriptor passed to the model.

    ``parameters`` is a JSON-Schema fragment describing the call arguments.
    Keep it small: edge models (Gemma 4 e4b) handle 5-10 tools cleanly but
    degrade with sprawling schemas.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        """Serialise to the OpenAI ``{"type": "function", ...}`` envelope."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ─── Streamed event from chat_with_tools ─────────────────────────────────


class ProviderEventKind(str, Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    DONE = "done"
    ERROR = "error"


@dataclass(frozen=True)
class ProviderEvent:
    """One chunk of streamed output from ``chat_with_tools``.

    Exactly one of ``text`` / ``tool_name`` / ``error`` is populated based on
    ``kind``. ``call_id`` distinguishes parallel tool calls in a single turn
    (most edge models emit one at a time, but the protocol allows many).
    """

    kind: ProviderEventKind
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def text_chunk(cls, content: str) -> "ProviderEvent":
        return cls(kind=ProviderEventKind.TEXT, text=content)

    @classmethod
    def tool_call(
        cls, name: str, args: dict[str, Any], *, call_id: Optional[str] = None
    ) -> "ProviderEvent":
        return cls(
            kind=ProviderEventKind.TOOL_CALL,
            tool_name=name,
            tool_args=args,
            call_id=call_id,
        )

    @classmethod
    def done(cls) -> "ProviderEvent":
        return cls(kind=ProviderEventKind.DONE)

    @classmethod
    def error_event(cls, message: str) -> "ProviderEvent":
        return cls(kind=ProviderEventKind.ERROR, error=message)


# ─── Probe report ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeReport:
    """Cheap availability check used by ``/admin/ollama`` and startup logging."""

    reachable: bool
    models: list[str]
    configured: str
    configured_available: bool
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "models": list(self.models),
            "configured": self.configured,
            "configured_available": self.configured_available,
            "error": self.error,
        }


# ─── The Protocol ────────────────────────────────────────────────────────


@runtime_checkable
class LLMProvider(Protocol):
    """Async, stateless interface every LLM backend implements.

    Implementations must be safe to call from many concurrent FastAPI requests.
    They must not raise on transport / model errors — return None / yield an
    error event so the caller can decide how to recover. Raising is reserved
    for programmer errors (bad arguments, missing config).
    """

    name: str  # short identifier, e.g. "ollama"

    async def generate(
        self,
        prompt: str,
        *,
        options: LlmOptions,
        timeout: float = 20.0,
    ) -> Optional[str]:
        """One-shot completion. Returns the raw text response, or None on failure."""
        ...

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        options: LlmOptions,
        timeout: float = 60.0,
    ) -> AsyncIterator[str]:
        """Stream incremental text chunks from a multi-turn chat.

        Each yielded string is a delta to append; concatenation reconstructs
        the full reply. ``<think>...</think>`` blocks are stripped by the
        provider before yielding (per AGENT.md "no thinking tokens to UI").
        Implementations yield nothing on failure — callers should always have
        a fallback path (see ``llm_service._offline_answer``).
        """
        ...

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSchema],
        *,
        options: LlmOptions,
        timeout: float = 60.0,
    ) -> AsyncIterator[ProviderEvent]:
        """Streamed chat with native function calling.

        Yields ``ProviderEvent.text_chunk(...)`` for free-text chunks,
        ``ProviderEvent.tool_call(name, args)`` whenever the model emits a
        tool call, then ``ProviderEvent.done()`` exactly once. On transport
        failure the provider yields one ``ProviderEvent.error_event(msg)``
        before ``done()``.
        """
        ...

    async def list_models(self, *, timeout: float = 3.0) -> list[str]:
        """Return installed model tags. Empty list if the runtime is unreachable."""
        ...

    async def probe(self, *, timeout: float = 3.0) -> ProbeReport:
        """Return a small availability report; never raises."""
        ...
