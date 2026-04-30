"""Curriculum planner — turns a parsed goal into an Atlas plan.

Responsibilities (PIVOT_ROADMAP §B.2):

1. Pick the competency clusters that match the learner's goal.
2. Group them into 4 weeks (30d) / 8 weeks (60d) / 12 weeks (90d).
3. Lay out districts with prerequisite edges so the Atlas can render
   the literal map.
4. Optionally (if Gemma 4 is reachable) enrich week titles + capability
   sentences. The deterministic skeleton always works — the LLM polish
   is best-effort.

The output JSON schema is locked here (Decision 0005). Anything that
reads ``atlas_plans.plan`` (the Atlas helper, the planner reverse
parser, eventual replan logic) reads it through these TypedDicts.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional, TypedDict

from backend.llm import get_provider
from backend.mastery import competencies as competencies_seed
from backend.mastery.competencies import CompetencyDef
from backend.models import db
from backend.services.runtime_config import get_llm_options

logger = logging.getLogger(__name__)


# ─── Public schema (Decision 0005) ───────────────────────────────────────


class Week(TypedDict):
    week_index: int             # 1-based
    title: str
    competencies: list[str]     # competencies.SEED IDs only
    capabilities: list[str]     # short German „Du kannst ..." sentences
    doors: list[str]            # subset of {"reader","voice","capture","drill"}


class District(TypedDict):
    id: str                     # slug, e.g. "medical_core"
    label: str
    competencies: list[str]
    prerequisites: list[str]    # district ids
    week_index: int
    status: str                 # 'mastered' | 'current' | 'queued' | 'locked'


class Plan(TypedDict):
    horizon: str                # '30d' | '60d' | '90d'
    weeks: list[Week]
    districts: list[District]
    edges: list[list[str]]      # [[from, to], ...] — list-of-list for JSON
    rationale: str


_HORIZON_TO_WEEKS = {"30d": 4, "60d": 8, "90d": 12}


# ─── Domain → competency clusters ────────────────────────────────────────
#
# Each tuple is (district_id, label, comp_ids, prerequisites). Order in
# the list matters: earlier entries land in earlier weeks.


_DOMAIN_CLUSTERS: dict[str, list[tuple[str, str, list[str], list[str]]]] = {
    "medical": [
        (
            "daily_basics",
            "Alltag und Begrüßung",
            [
                "daily.greetings",
                "daily.numbers_time",
                "daily.appointments",
            ],
            [],
        ),
        (
            "anatomy",
            "Anatomie und Körper",
            [
                "medical.body_parts",
            ],
            ["daily_basics"],
        ),
        (
            "patient_dialogue",
            "Anamnese und Symptome",
            [
                "medical.history.questions",
                "medical.symptoms",
                "daily.complaints",
            ],
            ["anatomy"],
        ),
        (
            "diagnostics",
            "Diagnostik und Befund",
            [
                "medical.diagnostics",
                "medical.pharmacology",
            ],
            ["patient_dialogue"],
        ),
        (
            "consent",
            "Aufklärung und Konsens",
            [
                "medical.consent",
                "grammar.subordinate",
                "grammar.konjunktiv2",
            ],
            ["diagnostics"],
        ),
    ],
    "academic": [
        (
            "daily_basics",
            "Alltag und Verwaltung",
            [
                "daily.greetings",
                "daily.numbers_time",
                "academic.email",
            ],
            [],
        ),
        (
            "methods",
            "Methodik und Aufbau",
            [
                "academic.methodology",
                "grammar.subordinate",
            ],
            ["daily_basics"],
        ),
        (
            "results",
            "Ergebnisse präsentieren",
            [
                "academic.results",
                "grammar.passive",
            ],
            ["methods"],
        ),
        (
            "defense",
            "Disputation und Verteidigung",
            [
                "academic.defense",
                "grammar.konjunktiv2",
            ],
            ["results"],
        ),
    ],
    "daily": [
        (
            "first_steps",
            "Erste Schritte",
            [
                "daily.greetings",
                "daily.numbers_time",
                "daily.shopping",
            ],
            [],
        ),
        (
            "in_the_city",
            "In der Stadt",
            [
                "daily.directions",
                "daily.transit",
            ],
            ["first_steps"],
        ),
        (
            "appointments",
            "Termine und Service",
            [
                "daily.appointments",
                "daily.complaints",
            ],
            ["in_the_city"],
        ),
        (
            "everyday_grammar",
            "Alltagsgrammatik",
            [
                "grammar.cases",
                "grammar.modal_verbs",
                "grammar.perfekt",
            ],
            ["first_steps"],
        ),
    ],
    "other": [
        (
            "first_steps",
            "Erste Schritte",
            [
                "daily.greetings",
                "daily.numbers_time",
                "daily.shopping",
            ],
            [],
        ),
        (
            "core_grammar",
            "Grundgrammatik",
            [
                "grammar.cases",
                "grammar.modal_verbs",
                "grammar.perfekt",
            ],
            ["first_steps"],
        ),
        (
            "everyday_practice",
            "Alltagsroutinen",
            [
                "daily.directions",
                "daily.transit",
                "daily.appointments",
            ],
            ["first_steps"],
        ),
        (
            "reading",
            "Lesen aktivieren",
            [
                "reading.news",
                "grammar.subordinate",
            ],
            ["core_grammar"],
        ),
    ],
}


# CEFR uplift: when target_cefr == "B2" we always tack on the B2 grammar
# stack so the plan reflects the learner's stated reach.
_CEFR_BOOSTERS: dict[str, list[str]] = {
    "A2": [],
    "B1": [],
    "B2": ["grammar.passive", "grammar.konjunktiv2"],
    "C1": ["grammar.passive", "grammar.konjunktiv2", "academic.defense"],
}


# ─── Public API ──────────────────────────────────────────────────────────


def generate_plan(
    goal_parsed: dict[str, Any],
    *,
    horizon: str = "30d",
    mastery_rows: Optional[list] = None,
    enrich_via_llm: bool = True,
) -> Plan:
    """Build a curriculum plan from a parsed goal.

    Step 1 — deterministic skeleton from the cluster table.
    Step 2 — optional LLM polish on week titles + capabilities.
    Step 3 — schema validation (whitelist competency IDs, drop empties).

    ``mastery_rows`` is accepted for forward-compatibility but the v1
    skeleton ignores it (the planner runs at onboarding when the table
    is empty). The Atlas helper recomputes per-district status against
    live mastery on every read, so plans don't need to embed it.
    """
    horizon = horizon if horizon in _HORIZON_TO_WEEKS else "30d"
    weeks_total = _HORIZON_TO_WEEKS[horizon]

    domain = (goal_parsed.get("domain") or "other").lower()
    if domain not in _DOMAIN_CLUSTERS:
        domain = "other"
    target_cefr = (goal_parsed.get("target_cefr") or "B1").upper()
    boosters = _CEFR_BOOSTERS.get(target_cefr, [])

    valid_ids = {c.id for c in competencies_seed.SEED}
    cluster_specs = _DOMAIN_CLUSTERS[domain]

    skeleton_districts: list[District] = []
    skeleton_edges: list[list[str]] = []

    for i, (slug, label, comps, prereqs) in enumerate(cluster_specs):
        keep = [c for c in comps if c in valid_ids]
        if not keep:
            continue
        week_idx = _district_week_index(i, len(cluster_specs), weeks_total)
        district: District = {
            "id": slug,
            "label": label,
            "competencies": keep,
            "prerequisites": [p for p in prereqs if any(d["id"] == p for d in skeleton_districts)],
            "week_index": week_idx,
            "status": "current" if week_idx == 1 else (
                "queued" if not prereqs else "locked"
            ),
        }
        skeleton_districts.append(district)
        for prereq in district["prerequisites"]:
            skeleton_edges.append([prereq, slug])

    # Boosters land in the last week as a single CEFR-stretch district.
    booster_keep = [b for b in boosters if b in valid_ids]
    if booster_keep:
        slug = f"cefr_stretch_{target_cefr.lower()}"
        prereq = skeleton_districts[-1]["id"] if skeleton_districts else None
        skeleton_districts.append(
            {
                "id": slug,
                "label": f"CEFR {target_cefr} Endspurt",
                "competencies": booster_keep,
                "prerequisites": [prereq] if prereq else [],
                "week_index": weeks_total,
                "status": "locked",
            }
        )
        if prereq:
            skeleton_edges.append([prereq, slug])

    # Build weeks from districts grouped by week_index.
    weeks_by_index: dict[int, list[District]] = {}
    for d in skeleton_districts:
        weeks_by_index.setdefault(d["week_index"], []).append(d)

    skeleton_weeks: list[Week] = []
    for w in range(1, weeks_total + 1):
        ds = weeks_by_index.get(w, [])
        comps = sorted({c for d in ds for c in d["competencies"]})
        capabilities = [
            f"Du kannst: {_humanise(c)}." for c in comps[:2]
        ]
        skeleton_weeks.append(
            {
                "week_index": w,
                "title": _default_week_title(w, ds),
                "competencies": comps,
                "capabilities": capabilities,
                "doors": _default_doors_for(domain),
            }
        )

    skeleton: Plan = {
        "horizon": horizon,
        "weeks": skeleton_weeks,
        "districts": skeleton_districts,
        "edges": skeleton_edges,
        "rationale": _default_rationale(goal_parsed, horizon),
    }

    if enrich_via_llm:
        polished = _try_enrich_via_llm(skeleton, goal_parsed)
        if polished is not None:
            return _validate(polished, valid_ids, weeks_total, horizon)

    return _validate(skeleton, valid_ids, weeks_total, horizon)


def persist_plan(goal_id: str, plan: Plan, *, horizon: Optional[str] = None) -> str:
    """Insert a plan row, super-seding any previous plan for this goal.

    Returns the new ``atlas_plans.id``. Always writes; the caller can
    decide whether to read it back via ``latest_plan_for_goal``.
    """
    new_id = f"plan-{uuid.uuid4().hex[:12]}"
    horizon = horizon or plan.get("horizon", "30d")
    cur = db.cursor()

    prior_rows = cur.execute(
        """
        SELECT id FROM atlas_plans
        WHERE goal_id = ? AND superseded_by IS NULL
        ORDER BY created_at DESC
        """,
        [goal_id],
    ).fetchall()

    cur.execute(
        """
        INSERT INTO atlas_plans (id, goal_id, horizon, plan)
        VALUES (?, ?, ?, ?)
        """,
        [new_id, goal_id, horizon, json.dumps(plan)],
    )

    for (prev_id,) in prior_rows:
        cur.execute(
            "UPDATE atlas_plans SET superseded_by = ? WHERE id = ?",
            [new_id, prev_id],
        )

    return new_id


def latest_plan_for_goal(goal_id: str) -> Optional[tuple[str, Plan, str]]:
    """Return ``(plan_id, plan_json, horizon)`` for the active plan of a goal."""
    row = db.cursor().execute(
        """
        SELECT id, plan, horizon
        FROM atlas_plans
        WHERE goal_id = ? AND superseded_by IS NULL
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [goal_id],
    ).fetchone()
    if row is None:
        return None
    plan_raw = row[1]
    if isinstance(plan_raw, str):
        try:
            plan = json.loads(plan_raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(plan_raw, dict):
        plan = plan_raw
    else:
        return None
    return row[0], plan, row[2] or "30d"


# ─── LLM enrichment (best-effort) ────────────────────────────────────────


_ENRICH_PROMPT = """\
You are polishing a German learning plan for a learner with this goal.
Goal:
{goal_text}

Plan skeleton (do NOT change competency IDs, district IDs, edges or
the week_index of districts — only `weeks[*].title`, `weeks[*].capabilities`
and `rationale` are editable):

{skeleton_json}

Return STRICT JSON only matching the same top-level shape, with:
- `weeks[*].title`        — short German title, max 60 chars
- `weeks[*].capabilities` — 1 to 3 short German „Du kannst..." sentences
- `rationale`             — one paragraph explaining the plan to the learner

Keep `competencies`, `districts`, `edges`, `horizon` and every
`week_index` exactly as given.
"""


async def _async_enrich(skeleton: Plan, goal_parsed: dict[str, Any]) -> Optional[Plan]:
    """Async helper that does the real LLM work. Returns None on any failure."""
    goal_text = _stringify_goal(goal_parsed)
    prompt = _ENRICH_PROMPT.format(
        goal_text=goal_text,
        skeleton_json=json.dumps(skeleton, ensure_ascii=False, indent=2),
    )
    try:
        raw = await get_provider().generate(
            prompt, options=get_llm_options(), timeout=20.0
        )
    except Exception as exc:  # pragma: no cover
        logger.info("plan-enrich provider call failed: %s", exc)
        return None
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    # Merge: take editable fields from the LLM, lock everything else
    # to the skeleton so the model can't break district shape.
    merged = dict(skeleton)
    if isinstance(obj.get("rationale"), str):
        merged["rationale"] = obj["rationale"].strip()
    llm_weeks = obj.get("weeks") or []
    if isinstance(llm_weeks, list):
        new_weeks: list[Week] = []
        for sk_week, llm_week in zip(skeleton["weeks"], llm_weeks):
            new_week = dict(sk_week)
            if isinstance(llm_week, dict):
                title = llm_week.get("title")
                if isinstance(title, str) and title.strip():
                    new_week["title"] = title.strip()[:80]
                caps = llm_week.get("capabilities")
                if isinstance(caps, list):
                    keep = [c.strip() for c in caps if isinstance(c, str) and c.strip()]
                    if keep:
                        new_week["capabilities"] = keep[:3]
            new_weeks.append(new_week)  # type: ignore[arg-type]
        merged["weeks"] = new_weeks
    return merged  # type: ignore[return-value]


def _try_enrich_via_llm(skeleton: Plan, goal_parsed: dict[str, Any]) -> Optional[Plan]:
    """Synchronous wrapper. Runs the async coroutine on a private loop."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Called from inside an async caller (router); they should
            # await the async path themselves. Skip enrichment here.
            return None
    except RuntimeError:
        pass

    try:
        return asyncio.run(_async_enrich(skeleton, goal_parsed))
    except Exception as exc:  # pragma: no cover — defensive
        logger.info("plan-enrich loop failed: %s", exc)
        return None


# ─── Validators / pure helpers ───────────────────────────────────────────


def _validate(plan: Plan, valid_ids: set[str], weeks_total: int, horizon: str) -> Plan:
    plan_dict = dict(plan)
    plan_dict["horizon"] = horizon

    cleaned_districts: list[District] = []
    for d in plan_dict.get("districts") or []:
        comps = [c for c in (d.get("competencies") or []) if c in valid_ids]
        if not comps:
            continue
        cleaned_districts.append(
            {
                "id": str(d.get("id") or "district"),
                "label": str(d.get("label") or "District"),
                "competencies": comps,
                "prerequisites": list(d.get("prerequisites") or []),
                "week_index": int(d.get("week_index") or 1),
                "status": str(d.get("status") or "queued"),
            }
        )
    plan_dict["districts"] = cleaned_districts

    cleaned_weeks: list[Week] = []
    for w in plan_dict.get("weeks") or []:
        comps = [c for c in (w.get("competencies") or []) if c in valid_ids]
        cleaned_weeks.append(
            {
                "week_index": int(w.get("week_index") or 1),
                "title": str(w.get("title") or "Woche"),
                "competencies": comps,
                "capabilities": [
                    s for s in (w.get("capabilities") or []) if isinstance(s, str)
                ],
                "doors": [
                    s for s in (w.get("doors") or []) if isinstance(s, str)
                ],
            }
        )
    if not cleaned_weeks:
        cleaned_weeks = [
            {
                "week_index": 1,
                "title": "Woche 1",
                "competencies": [],
                "capabilities": [],
                "doors": ["reader"],
            }
        ]
    plan_dict["weeks"] = cleaned_weeks

    cleaned_edges: list[list[str]] = []
    district_ids = {d["id"] for d in cleaned_districts}
    for e in plan_dict.get("edges") or []:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            a, b = str(e[0]), str(e[1])
            if a in district_ids and b in district_ids:
                cleaned_edges.append([a, b])
    plan_dict["edges"] = cleaned_edges

    plan_dict["rationale"] = str(plan_dict.get("rationale") or "")
    return plan_dict  # type: ignore[return-value]


def _district_week_index(i: int, total_clusters: int, weeks_total: int) -> int:
    """Spread N clusters evenly across the available weeks."""
    if total_clusters <= 0:
        return 1
    span = max(1, weeks_total // max(total_clusters, 1))
    return min(weeks_total, 1 + i * span)


def _default_doors_for(domain: str) -> list[str]:
    if domain == "medical":
        return ["reader", "voice"]
    if domain == "academic":
        return ["reader", "drill"]
    if domain == "daily":
        return ["voice", "reader"]
    return ["reader"]


def _default_week_title(week: int, districts: list[District]) -> str:
    if districts:
        return f"Woche {week}: {districts[0]['label']}"
    return f"Woche {week}: Vertiefung"


def _default_rationale(goal_parsed: dict[str, Any], horizon: str) -> str:
    domain = goal_parsed.get("domain") or "other"
    cefr = goal_parsed.get("target_cefr") or "B1"
    return (
        f"Ein {horizon}-Plan für {domain.upper()}-Lernende auf "
        f"{cefr}-Niveau. Wir starten mit Alltag und Grundgrammatik, "
        "danach folgt der domänenspezifische Wortschatz."
    )


def _humanise(competency_id: str) -> str:
    """Map a competency id to a short German verb phrase for capabilities."""
    cmap = {c.id: c for c in competencies_seed.SEED}
    cdef: Optional[CompetencyDef] = cmap.get(competency_id)
    if cdef is not None:
        label = cdef.label.lower()
        return label
    return competency_id


def _stringify_goal(goal_parsed: dict[str, Any]) -> str:
    parts = [f"domain={goal_parsed.get('domain')}", f"target_cefr={goal_parsed.get('target_cefr')}"]
    deadline = goal_parsed.get("deadline_iso")
    if deadline:
        parts.append(f"deadline={deadline}")
    scenarios = goal_parsed.get("scenarios") or []
    if scenarios:
        parts.append("scenarios=" + ", ".join(scenarios[:3]))
    motivations = goal_parsed.get("motivations") or []
    if motivations:
        parts.append("motivations=" + ", ".join(motivations[:2]))
    return " | ".join(parts)
