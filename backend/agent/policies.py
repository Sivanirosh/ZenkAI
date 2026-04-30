"""Agent runtime guardrails (PIVOT_ROADMAP.md §6.4).

This file is deliberately small in Phase A. The full policy table
(per-tool RecallPolicy, per-source mastery weighting, ramp/decay rates)
lands in Phase B alongside the real Memory layer.
"""

from __future__ import annotations


# Hard cap on tool dispatches per single agent turn. Prevents runaway
# loops on a model that decides to call recommend_text indefinitely.
MAX_TOOL_CALLS_PER_TURN: int = 5

# Wall-clock cap (seconds) for one agent turn. The controller cancels
# the LLM stream and returns a partial result if exceeded.
TURN_TIMEOUT_S: float = 30.0

# Per-tool dispatch timeout (seconds). If a tool takes longer the agent
# gets a tool_error event and continues.
TOOL_TIMEOUT_S: float = 8.0

# Token budget the runtime asks the MemoryManager to fit Recall into.
# The Phase A stub ignores this; Phase B respects it.
RECALL_TOKEN_BUDGET: int = 4000
