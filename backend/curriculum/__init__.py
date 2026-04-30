"""Curriculum planner (PIVOT_ROADMAP §B.2).

Generates a 30/60/90-day learning plan from a parsed goal + the
competency taxonomy + the current mastery snapshot. The plan is the
data the Atlas screen renders as a literal map.

Public surface:

- ``Plan``               — TypedDict locking Decision 0005
- ``Week`` / ``District`` — sub-shapes
- ``generate_plan(...)`` — deterministic skeleton + optional LLM polish
- ``persist_plan(...)``  — INSERT into ``atlas_plans`` (with supersede)
- ``latest_plan_for_goal(goal_id)`` — read helper for the Atlas helper
"""

from __future__ import annotations

from backend.curriculum.planner import (
    District,
    Plan,
    Week,
    generate_plan,
    latest_plan_for_goal,
    persist_plan,
)

__all__ = [
    "District",
    "Plan",
    "Week",
    "generate_plan",
    "latest_plan_for_goal",
    "persist_plan",
]
