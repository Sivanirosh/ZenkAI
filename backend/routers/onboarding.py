"""Onboarding flow endpoints (PIVOT_ROADMAP §B.1).

Two routes:

- ``POST /onboarding/goal``  — submit the free-text answer; we parse it
  via Gemma 4 (with a deterministic fallback), persist a row in
  ``goals``, and trigger the curriculum planner so an Atlas plan is
  ready before the learner ever sees the Atlas tab.

- ``GET /onboarding/state``  — returns ``{has_goal, latest_goal_id,
  latest_plan_id}`` so the app shell can decide whether to render the
  Onboarding screen on first paint.

Never raises after a 200 — the goal parser falls back to a skeleton if
the LLM is offline, so the learner can always continue.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.agent import goals as agent_goals
from backend.curriculum import planner as curriculum_planner
from backend.models import db

logger = logging.getLogger(__name__)
router = APIRouter()


class GoalPayload(BaseModel):
    """Body for ``POST /onboarding/goal``."""

    raw_text: str = Field(min_length=1, max_length=4000)
    horizon: str = Field(default="30d")


@router.post("/goal")
async def submit_goal(payload: GoalPayload) -> dict[str, Any]:
    """Parse + persist the learner's goal, then build the first plan."""
    parsed = await agent_goals.parse_goal_text(payload.raw_text)
    parsed_dict = parsed.model_dump()
    goal_id = f"goal-{uuid.uuid4().hex[:12]}"

    cur = db.cursor()
    cur.execute(
        "INSERT INTO goals (id, raw_text, parsed) VALUES (?, ?, ?)",
        [goal_id, payload.raw_text, json.dumps(parsed_dict, ensure_ascii=False)],
    )

    plan = curriculum_planner.generate_plan(
        parsed_dict,
        horizon=payload.horizon,
        enrich_via_llm=False,  # router is async — keep onboarding cheap
    )
    plan_id = curriculum_planner.persist_plan(
        goal_id, plan, horizon=payload.horizon
    )

    return {
        "goal_id": goal_id,
        "parsed": parsed_dict,
        "plan_id": plan_id,
        "horizon": payload.horizon,
        "plan": plan,
    }


@router.get("/state")
async def onboarding_state() -> dict[str, Any]:
    """Return the latest goal + plan so the shell can route on cold start."""
    goal_row = db.cursor().execute(
        """
        SELECT id, raw_text, parsed, created_at
        FROM goals
        ORDER BY created_at DESC
        LIMIT 1
        """
    ).fetchone()
    if goal_row is None:
        return {
            "has_goal": False,
            "latest_goal_id": None,
            "latest_plan_id": None,
            "parsed": None,
        }
    goal_id, raw_text, parsed_raw, created_at = goal_row
    parsed: Optional[dict[str, Any]] = None
    if parsed_raw is not None:
        if isinstance(parsed_raw, str):
            try:
                parsed = json.loads(parsed_raw)
            except json.JSONDecodeError:
                parsed = None
        elif isinstance(parsed_raw, dict):
            parsed = parsed_raw

    plan_id: Optional[str] = None
    horizon: Optional[str] = None
    plan_row = db.cursor().execute(
        """
        SELECT id, horizon
        FROM atlas_plans
        WHERE goal_id = ? AND superseded_by IS NULL
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [goal_id],
    ).fetchone()
    if plan_row is not None:
        plan_id = plan_row[0]
        horizon = plan_row[1]

    return {
        "has_goal": True,
        "latest_goal_id": goal_id,
        "latest_plan_id": plan_id,
        "horizon": horizon,
        "raw_text": raw_text,
        "parsed": parsed,
        "created_at": str(created_at) if created_at is not None else None,
    }
