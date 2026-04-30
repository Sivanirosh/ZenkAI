"""Memory layer (PIVOT_ROADMAP.md §7) — runtime tutor memory on disk.

This is the **stub** delivered in Phase A first slice. It honours the
contract from §7.4 (recall / write_event / close_turn / close_session /
compact / snapshot_mastery) but the recall path is intentionally empty.
The full implementation (Tier 1/2/3 retrieval, multi-resolution summaries,
fact sheets, vector index, two-stage retriever, background compactor) lands
in Phase A.10–A.16 and Phase B.11–B.18.

The reason we build the stub now: the agent loop (controller.py) calls
``mem.recall(...)`` and ``mem.write_event(...)`` from day one. Locking the
interface here means **zero rewrite** when the real recall arrives — only
one module changes.

Public surface:

- ``Event``           — typed event going into the JSONL ledger
- ``Recall``          — typed bundle returned by ``recall()`` (always empty in v1)
- ``MemoryManager``   — the chokepoint
- ``get_memory()``    — process-wide singleton accessor
"""

from __future__ import annotations

from backend.memory.manager import (
    Event,
    MemoryManager,
    Recall,
    RecallContext,
    Section,
    get_memory,
    reset_for_tests,
)

__all__ = [
    "Event",
    "MemoryManager",
    "Recall",
    "RecallContext",
    "Section",
    "get_memory",
    "reset_for_tests",
]
