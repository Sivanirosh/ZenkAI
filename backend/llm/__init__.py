"""LLM provider abstraction.

The agent loop, the prompt-driven services (annotation, chat), and the
benchmark script all go through one of the providers in this package. The
goal is that swapping the runtime (Ollama -> LiteRT, on-device Cactus, etc.)
in Phase C requires implementing a single Protocol, not editing the agent.

Public surface:

- ``LLMProvider``     — the runtime-agnostic Protocol every provider implements
- ``ProviderEvent``   — the chunk type yielded by ``chat_with_tools``
- ``ToolSchema``      — the tool descriptor we hand to the model
- ``ProbeReport``     — return type for ``LLMProvider.probe()``
- ``get_provider()``  — process-wide singleton accessor (currently Ollama)
"""

from __future__ import annotations

from backend.llm.provider import (
    LLMProvider,
    ProbeReport,
    ProviderEvent,
    ProviderEventKind,
    ToolSchema,
)
from backend.llm.ollama import OllamaProvider, get_provider

__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "ProbeReport",
    "ProviderEvent",
    "ProviderEventKind",
    "ToolSchema",
    "get_provider",
]
