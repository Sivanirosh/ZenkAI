"""Mira's tool catalogue: schemas + Pydantic-validated dispatcher.

Eight tools in this slice (B.9 added ``start_conversation`` and
``start_capture`` as navigation intents into the Konversation and
Capture rooms). Each one:

1. Has an OpenAI-compatible JSON Schema (``TOOL_SCHEMAS``) handed to the
   model so it can emit ``tool_call`` events with structured arguments.
2. Has a Pydantic ``BaseModel`` for runtime arg validation — invalid
   arguments produce a structured ``ToolError`` instead of crashing the
   turn.
3. Returns a small JSON-serialisable dict; the controller writes both
   the call and the result to the JSONL ledger.

Adding a new tool is a four-step recipe (do all four or it stays broken):
  - Define the args ``BaseModel``
  - Add the ``ToolSchema`` entry to ``TOOL_SCHEMAS``
  - Implement the ``_handle_<name>`` function
  - Register it in ``_HANDLERS``
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from backend.agent import goals as agent_goals
from backend.agent import policies
from backend.curriculum import planner as curriculum_planner
from backend.llm.provider import ToolSchema
from backend.mastery import model as mastery_model
from backend.models import db
from backend.services import corpus_service

logger = logging.getLogger(__name__)


# ─── Per-tool argument models ────────────────────────────────────────────


class RecommendTextArgs(BaseModel):
    """Args for ``recommend_text``."""

    work_id: Optional[str] = Field(
        default=None,
        description="Restrict the recommendation to a specific work id.",
    )
    current_paragraph_id: Optional[str] = Field(
        default=None,
        description=(
            "Where the learner is right now. If supplied, the next paragraph "
            "in the same work is returned."
        ),
    )
    reason: Optional[str] = Field(
        default=None,
        description="Why this recommendation makes sense (free text).",
    )
    time_budget_min: Optional[int] = Field(
        default=None,
        ge=1,
        le=120,
        description="Minutes the learner has available; informs paragraph length later.",
    )


class StartDrillArgs(BaseModel):
    """Args for ``start_drill``."""

    competency_id: str = Field(min_length=1)
    count: int = Field(ge=1, le=20)
    surface_examples: list[str] = Field(default_factory=list)


_ALLOWED_SOURCES = (
    "conversation",
    "drill",
    "reader",
    "capture",
    "agent_pre",
    "agent_post",
)


class UpdateMasteryArgs(BaseModel):
    """Args for ``update_mastery``."""

    competency_id: str = Field(min_length=1)
    delta: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Quality observed in [-1..1]. We map -1->fail (0.0), +1->success "
            "(1.0), 0->no signal (0.5)."
        ),
    )
    source: str = Field(default="agent_post")
    surface_form: Optional[str] = None
    notes: Optional[str] = None


class ParseGoalArgs(BaseModel):
    """Args for ``parse_goal``."""

    raw_text: str = Field(min_length=1, max_length=4000)


class PlanCurriculumArgs(BaseModel):
    """Args for ``plan_curriculum``."""

    goal_id: str = Field(min_length=1)
    horizon: str = Field(default="30d")


class ReplanAtlasArgs(BaseModel):
    """Args for ``replan_atlas``."""

    goal_id: str = Field(min_length=1)
    horizon: str = Field(default="30d")


class StartConversationArgs(BaseModel):
    """Args for ``start_conversation``."""

    competency_id: Optional[str] = Field(
        default=None,
        description=(
            "Anchor the conversation around one competency. The Konversation "
            "room uses this to seed the opening prompt."
        ),
    )
    scenario: Optional[str] = Field(
        default=None,
        max_length=160,
        description=(
            "Real-world setting (e.g. 'Anamnesegespräch', 'Café bestellen')."
        ),
    )
    minutes: int = Field(default=5, ge=1, le=30)
    target_cefr: Optional[str] = Field(default=None)


class StartCaptureArgs(BaseModel):
    """Args for ``start_capture`` (alias of ``capture_text``)."""

    surface_kind: str = Field(
        default="text",
        description="What we expect to see: 'text', 'sign', 'menu', 'object'.",
    )
    target_competency_id: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None, max_length=160)


# ─── OpenAI-compatible schemas (handed to the model via tools=...) ───────


TOOL_SCHEMAS: list[ToolSchema] = [
    ToolSchema(
        name="recommend_text",
        description=(
            "Pick the next paragraph the learner should read. Pass "
            "current_paragraph_id when continuing, work_id to switch books."
        ),
        parameters={
            "type": "object",
            "properties": {
                "work_id": {"type": "string"},
                "current_paragraph_id": {"type": "string"},
                "reason": {"type": "string"},
                "time_budget_min": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            "required": [],
        },
    ),
    ToolSchema(
        name="start_drill",
        description=(
            "Launch a focused practice round on one competency. Use a small "
            "count (3-7) so the learner isn't overwhelmed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "competency_id": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 20},
                "surface_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["competency_id", "count"],
        },
    ),
    ToolSchema(
        name="update_mastery",
        description=(
            "Record an evidence event for a competency and refresh its "
            "Bayesian posterior. Always supply source='agent_pre' for an "
            "opening probe and 'agent_post' for an outcome you just observed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "competency_id": {"type": "string"},
                "delta": {"type": "number", "minimum": -1, "maximum": 1},
                "source": {"type": "string", "enum": list(_ALLOWED_SOURCES)},
                "surface_form": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["competency_id", "delta"],
        },
    ),
    ToolSchema(
        name="parse_goal",
        description=(
            "Parse a learner's free-text goal answer into structured JSON "
            "(domain, target_cefr, deadline, scenarios). Use when the "
            "learner volunteers a new goal mid-session; for the initial "
            "onboarding the dedicated /onboarding/goal endpoint runs this."
        ),
        parameters={
            "type": "object",
            "properties": {
                "raw_text": {"type": "string"},
            },
            "required": ["raw_text"],
        },
    ),
    ToolSchema(
        name="plan_curriculum",
        description=(
            "Generate a 30/60/90-day Atlas plan for an existing goal. "
            "Persists one row in atlas_plans and returns its id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string"},
                "horizon": {"type": "string", "enum": ["30d", "60d", "90d"]},
            },
            "required": ["goal_id"],
        },
    ),
    ToolSchema(
        name="replan_atlas",
        description=(
            "Re-plan the curriculum after a change in mastery / scenario. "
            "Marks the previous plan as superseded; returns the new plan_id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string"},
                "horizon": {"type": "string", "enum": ["30d", "60d", "90d"]},
            },
            "required": ["goal_id"],
        },
    ),
    ToolSchema(
        name="start_conversation",
        description=(
            "Open the Konversation (voice) room for a short live dialogue. "
            "Use when the learner asks to speak, when a District's 'Sprechen' "
            "door fires, or when an Atrium capability sentence implies "
            "spoken practice. The frontend acts on the returned room_id by "
            "navigating; this tool does NOT actually run the dialogue."
        ),
        parameters={
            "type": "object",
            "properties": {
                "competency_id": {"type": "string"},
                "scenario": {"type": "string"},
                "minutes": {"type": "integer", "minimum": 1, "maximum": 30},
                "target_cefr": {"type": "string", "enum": ["A2", "B1", "B2", "C1"]},
            },
            "required": [],
        },
    ),
    ToolSchema(
        name="start_capture",
        description=(
            "Open the Capture (camera) room for a single multimodal image. "
            "Use when the learner asks 'what does this say?' or when a "
            "District's 'Sehen' door fires. The frontend acts on the "
            "returned room_id by navigating; this tool does NOT take "
            "the photo itself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "surface_kind": {
                    "type": "string",
                    "enum": ["text", "sign", "menu", "object"],
                },
                "target_competency_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": [],
        },
    ),
]


# ─── Result + error containers ───────────────────────────────────────────


@dataclass(frozen=True)
class ToolError(Exception):
    """Raised inside dispatch when a tool fails. Always JSON-serialisable."""

    name: str
    code: str
    message: str
    detail: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "code": self.code,
            "message": self.message,
            "detail": self.detail or {},
        }


# ─── Tool implementations ────────────────────────────────────────────────


def _handle_recommend_text(args: RecommendTextArgs) -> dict[str, Any]:
    paragraph = corpus_service.next_paragraph(
        work_id=args.work_id,
        current_paragraph_id=args.current_paragraph_id,
    )
    if paragraph is None:
        raise ToolError(
            name="recommend_text",
            code="no_paragraph",
            message=(
                "No suitable paragraph found. The corpus may be empty or "
                "the learner has reached the end of this work."
            ),
            detail={
                "work_id": args.work_id,
                "current_paragraph_id": args.current_paragraph_id,
            },
        )
    return {
        "paragraph_id": paragraph.id,
        "work_id": paragraph.work_id,
        "chapter": paragraph.chapter,
        "position": paragraph.position,
        "word_count": paragraph.word_count,
        "preview": paragraph.text[:160],
        "why": args.reason
        or "next paragraph in the current reading",
    }


def _handle_start_drill(args: StartDrillArgs) -> dict[str, Any]:
    # Phase A: stub plan. Phase B replaces with a generator that pulls
    # surface forms from the evidence ledger + competency-anchored examples.
    row = mastery_model.get(args.competency_id)
    return {
        "competency_id": args.competency_id,
        "count": args.count,
        "surface_examples": args.surface_examples,
        "current_confidence": row.confidence,
        "current_variance": row.variance,
        "plan_kind": "stub",
        "next_action": "open the Drill room (Phase B)",
    }


def _handle_update_mastery(args: UpdateMasteryArgs) -> dict[str, Any]:
    if args.source not in _ALLOWED_SOURCES:
        raise ToolError(
            name="update_mastery",
            code="bad_source",
            message=f"source must be one of {_ALLOWED_SOURCES}",
        )
    quality = (args.delta + 1.0) / 2.0
    try:
        new_row = mastery_model.update(
            competency_id=args.competency_id,
            quality=quality,
            source=args.source,
            surface_form=args.surface_form,
            notes=args.notes,
        )
    except Exception as exc:  # pragma: no cover — DB failure
        raise ToolError(
            name="update_mastery",
            code="persist_failed",
            message=str(exc),
        ) from exc
    return {
        "competency_id": new_row.competency_id,
        "confidence": new_row.confidence,
        "variance": new_row.variance,
        "evidence_count": new_row.evidence_count,
    }


def _handle_parse_goal(args: ParseGoalArgs) -> dict[str, Any]:
    parsed = _run_async(agent_goals.parse_goal_text(args.raw_text))
    return {
        "domain": parsed.domain,
        "deadline_iso": parsed.deadline_iso,
        "target_cefr": parsed.target_cefr,
        "scenarios": list(parsed.scenarios),
        "motivations": list(parsed.motivations),
        "language_profile": parsed.language_profile.model_dump(),
    }


def _handle_plan_curriculum(args: PlanCurriculumArgs) -> dict[str, Any]:
    parsed = _read_goal_parsed(args.goal_id)
    if parsed is None:
        raise ToolError(
            name="plan_curriculum",
            code="goal_not_found",
            message=f"goal '{args.goal_id}' has no parsed JSON",
        )
    plan = curriculum_planner.generate_plan(
        parsed,
        horizon=args.horizon,
        enrich_via_llm=False,
    )
    plan_id = curriculum_planner.persist_plan(args.goal_id, plan, horizon=args.horizon)
    return {
        "plan_id": plan_id,
        "horizon": args.horizon,
        "week_count": len(plan["weeks"]),
        "district_count": len(plan["districts"]),
    }


def _handle_replan_atlas(args: ReplanAtlasArgs) -> dict[str, Any]:
    parsed = _read_goal_parsed(args.goal_id)
    if parsed is None:
        raise ToolError(
            name="replan_atlas",
            code="goal_not_found",
            message=f"goal '{args.goal_id}' has no parsed JSON",
        )
    plan = curriculum_planner.generate_plan(
        parsed,
        horizon=args.horizon,
        enrich_via_llm=False,
    )
    plan_id = curriculum_planner.persist_plan(args.goal_id, plan, horizon=args.horizon)
    return {
        "plan_id": plan_id,
        "horizon": args.horizon,
        "week_count": len(plan["weeks"]),
        "district_count": len(plan["districts"]),
        "supersedes_previous": True,
    }


def _handle_start_conversation(args: StartConversationArgs) -> dict[str, Any]:
    """Open the Konversation room. Pure intent — no audio yet.

    The frontend uses the returned ``room_id`` to navigate to the
    ``Konversation`` screen and seeds the opening prompt with the
    ``scenario`` + competency label.
    """
    label: Optional[str] = None
    cefr: Optional[str] = args.target_cefr
    if args.competency_id:
        try:
            from backend.mastery.competencies import SEED  # noqa: WPS433

            for c in SEED:
                if c.id == args.competency_id:
                    label = c.label
                    cefr = cefr or c.cefr
                    break
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("competency lookup failed: %s", exc)
    return {
        "room_id": "konversation",
        "competency_id": args.competency_id,
        "competency_label": label,
        "scenario": args.scenario,
        "minutes": args.minutes,
        "target_cefr": cefr,
        "next_action": "navigate to /konversation and start the room",
    }


def _handle_start_capture(args: StartCaptureArgs) -> dict[str, Any]:
    """Open the Capture room. Pure intent — no image yet.

    The frontend uses the returned ``room_id`` to navigate to the
    ``Capture`` screen, opens ``getUserMedia``, and on shutter posts
    the frame to ``/api/v1/capture/image``.
    """
    return {
        "room_id": "capture",
        "surface_kind": args.surface_kind,
        "target_competency_id": args.target_competency_id,
        "note": args.note,
        "next_action": "navigate to /capture and open the camera",
    }


def _read_goal_parsed(goal_id: str) -> Optional[dict[str, Any]]:
    row = db.cursor().execute(
        "SELECT parsed FROM goals WHERE id = ?",
        [goal_id],
    ).fetchone()
    if row is None:
        return None
    raw = row[0]
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict):
        return raw
    return None


def _run_async(coro):
    """Run an async helper from a synchronous tool handler.

    The agent dispatches tools via ``asyncio.to_thread``, so we are
    always on a worker thread with no running loop. ``asyncio.run`` is
    safe here.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


_HANDLERS: dict[str, tuple[type[BaseModel], Callable[[Any], dict[str, Any]]]] = {
    "recommend_text": (RecommendTextArgs, _handle_recommend_text),  # type: ignore[dict-item]
    "start_drill": (StartDrillArgs, _handle_start_drill),  # type: ignore[dict-item]
    "update_mastery": (UpdateMasteryArgs, _handle_update_mastery),  # type: ignore[dict-item]
    "parse_goal": (ParseGoalArgs, _handle_parse_goal),  # type: ignore[dict-item]
    "plan_curriculum": (PlanCurriculumArgs, _handle_plan_curriculum),  # type: ignore[dict-item]
    "replan_atlas": (ReplanAtlasArgs, _handle_replan_atlas),  # type: ignore[dict-item]
    "start_conversation": (StartConversationArgs, _handle_start_conversation),  # type: ignore[dict-item]
    "start_capture": (StartCaptureArgs, _handle_start_capture),  # type: ignore[dict-item]
}


# ─── Dispatcher ──────────────────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate args + run the named tool. Raises ToolError on any failure.

    The ``policies.TOOL_TIMEOUT_S`` ceiling is the controller's job (asyncio
    wait_for). Tools themselves stay synchronous + fast in Phase A.
    """
    handler_entry = _HANDLERS.get(name)
    if handler_entry is None:
        raise ToolError(
            name=name,
            code="unknown_tool",
            message=f"No tool named '{name}'",
            detail={"available": sorted(_HANDLERS)},
        )
    args_model, handler = handler_entry
    try:
        validated = args_model(**(args or {}))
    except ValidationError as exc:
        raise ToolError(
            name=name,
            code="invalid_args",
            message="Tool arguments failed validation.",
            detail={"errors": exc.errors()},
        ) from exc
    logger.debug("dispatch tool=%s args=%s", name, validated.model_dump())
    return handler(validated)


# Re-export the policy cap so the controller and tests reference one source.
MAX_TOOL_CALLS_PER_TURN = policies.MAX_TOOL_CALLS_PER_TURN
TOOL_TIMEOUT_S = policies.TOOL_TIMEOUT_S
