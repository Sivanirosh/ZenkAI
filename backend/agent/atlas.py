"""Atlas screen helper (PIVOT_ROADMAP §B.4).

Renders the data the literal-map UI needs in one round-trip — no LLM.
Reads the latest non-superseded ``atlas_plans`` row for the active
goal, joins each district's competencies with live mastery (so the
fill colour and the status reflect what the learner has actually done
since the plan was generated).

Status logic per district:

- ``mastered`` — every competency has μ ≥ 70 and σ ≤ 15
- ``current``  — the district sits in the current "study week"
                 (week with the smallest week_index that isn't fully
                 mastered)
- ``queued``   — every prerequisite district is mastered
- ``locked``   — otherwise

Returns 404 from the router when there is no plan yet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from backend.curriculum import planner as curriculum_planner
from backend.mastery import competencies as competencies_seed
from backend.mastery import model as mastery_model
from backend.models import db

logger = logging.getLogger(__name__)


# ─── Public payload shape ────────────────────────────────────────────────


@dataclass
class AtlasGoal:
    id: str
    raw_text: str
    parsed: dict[str, Any] = field(default_factory=dict)


@dataclass
class AtlasCompetency:
    competency_id: str
    label: str
    cefr: Optional[str]
    confidence: int
    variance: int
    evidence_count: int


@dataclass
class AtlasDistrict:
    id: str
    label: str
    week_index: int
    status: str            # mastered | current | queued | locked
    competencies: list[AtlasCompetency] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    confidence: int = 0    # average of competency confidences
    capability_sentence: Optional[str] = None


@dataclass
class AtlasPayload:
    goal: AtlasGoal
    plan_id: str
    horizon: str
    districts: list[AtlasDistrict] = field(default_factory=list)
    edges: list[list[str]] = field(default_factory=list)
    current_route: list[str] = field(default_factory=list)
    rationale: str = ""
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": asdict(self.goal),
            "plan_id": self.plan_id,
            "horizon": self.horizon,
            "districts": [_district_to_dict(d) for d in self.districts],
            "edges": [list(e) for e in self.edges],
            "current_route": list(self.current_route),
            "rationale": self.rationale,
            "generated_at": self.generated_at,
        }


def _district_to_dict(d: AtlasDistrict) -> dict[str, Any]:
    return {
        "id": d.id,
        "label": d.label,
        "week_index": d.week_index,
        "status": d.status,
        "competencies": [asdict(c) for c in d.competencies],
        "prerequisites": list(d.prerequisites),
        "confidence": d.confidence,
        "capability_sentence": d.capability_sentence,
    }


# ─── Public entry point ─────────────────────────────────────────────────


def build_atlas_payload(
    *,
    goal_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Optional[AtlasPayload]:
    """Assemble the Atlas payload. Returns ``None`` when no plan exists."""
    now = now or datetime.now()
    goal_row = _fetch_goal(goal_id)
    if goal_row is None:
        return None
    g_id, raw_text, parsed = goal_row

    plan_tuple = curriculum_planner.latest_plan_for_goal(g_id)
    if plan_tuple is None:
        return None
    plan_id, plan, horizon = plan_tuple

    label_map = _label_lookup()
    cefr_map = _cefr_lookup()

    capability_for: dict[int, str] = {}
    for w in plan.get("weeks") or []:
        idx = int(w.get("week_index") or 0)
        caps = [c for c in (w.get("capabilities") or []) if isinstance(c, str)]
        if idx and caps:
            capability_for.setdefault(idx, caps[0])

    districts: list[AtlasDistrict] = []
    for d in plan.get("districts") or []:
        comps: list[AtlasCompetency] = []
        for cid in d.get("competencies") or []:
            row = mastery_model.get(cid)
            comps.append(
                AtlasCompetency(
                    competency_id=cid,
                    label=label_map.get(cid, cid),
                    cefr=cefr_map.get(cid),
                    confidence=row.confidence,
                    variance=row.variance,
                    evidence_count=row.evidence_count,
                )
            )
        avg_conf = (
            int(sum(c.confidence for c in comps) / len(comps)) if comps else 0
        )
        districts.append(
            AtlasDistrict(
                id=str(d.get("id") or "district"),
                label=str(d.get("label") or "District"),
                week_index=int(d.get("week_index") or 1),
                status="locked",
                competencies=comps,
                prerequisites=list(d.get("prerequisites") or []),
                confidence=avg_conf,
                capability_sentence=capability_for.get(
                    int(d.get("week_index") or 1)
                ),
            )
        )

    _recompute_status(districts)
    current_route = [d.id for d in districts if d.status == "current"]
    edges = [list(e) for e in plan.get("edges") or [] if len(e) == 2]

    return AtlasPayload(
        goal=AtlasGoal(id=g_id, raw_text=raw_text, parsed=parsed),
        plan_id=plan_id,
        horizon=horizon,
        districts=districts,
        edges=edges,
        current_route=current_route,
        rationale=str(plan.get("rationale") or ""),
        generated_at=now.isoformat(),
    )


# ─── Internals ───────────────────────────────────────────────────────────


def _fetch_goal(goal_id: Optional[str]) -> Optional[tuple[str, str, dict[str, Any]]]:
    cur = db.cursor()
    if goal_id is not None:
        row = cur.execute(
            "SELECT id, raw_text, parsed FROM goals WHERE id = ?",
            [goal_id],
        ).fetchone()
    else:
        row = cur.execute(
            """
            SELECT id, raw_text, parsed
            FROM goals
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    parsed: dict[str, Any] = {}
    raw_parsed = row[2]
    if isinstance(raw_parsed, str):
        try:
            parsed = json.loads(raw_parsed)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(raw_parsed, dict):
        parsed = raw_parsed
    return row[0], (row[1] or ""), parsed


def _recompute_status(districts: list[AtlasDistrict]) -> None:
    """Mutate districts in place: mastered / current / queued / locked.

    Algorithm:
    1. Mark each district mastered iff every competency has confidence
       >= 70 AND variance <= 15.
    2. Mark a district queued iff all of its prerequisites (by id) are
       mastered. Districts with no prereqs start queued.
    3. The current week is the smallest week_index that contains at
       least one non-mastered district. All non-mastered districts
       in that week become 'current'.
    4. Anything still undecided is 'locked'.
    """
    mastered_ids: set[str] = set()
    for d in districts:
        if d.competencies and all(
            c.confidence >= 70 and c.variance <= 15 for c in d.competencies
        ):
            d.status = "mastered"
            mastered_ids.add(d.id)

    weeks_with_unfinished = sorted(
        {d.week_index for d in districts if d.id not in mastered_ids}
    )
    current_week = weeks_with_unfinished[0] if weeks_with_unfinished else None

    for d in districts:
        if d.id in mastered_ids:
            continue
        if current_week is not None and d.week_index == current_week:
            d.status = "current"
            continue
        if all(p in mastered_ids for p in d.prerequisites):
            d.status = "queued"
        else:
            d.status = "locked"


def _label_lookup() -> dict[str, str]:
    return {c.id: c.label for c in competencies_seed.SEED}


def _cefr_lookup() -> dict[str, str]:
    return {c.id: c.cefr for c in competencies_seed.SEED if c.cefr}
