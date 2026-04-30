"""Mira agent — the Observe → Decide → Act loop on top of Gemma 4.

Public surface:

- ``prompts.MIRA_SYSTEM_PROMPT_V1`` — the system prompt (versioned)
- ``tools.TOOL_SCHEMAS``            — schemas handed to the model
- ``tools.dispatch``                 — Pydantic-validated tool runner
- ``controller.turn``                — async generator yielding agent events
- ``policies``                       — turn caps + safety constants
"""

from __future__ import annotations
