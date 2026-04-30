"""Per-tool RecallPolicy table (PIVOT_ROADMAP §7.5 + §A.13).

The MemoryManager dispatches every ``recall(tool_hint=...)`` call into
this table. Each policy:

- Decides which slices of long-term memory to load for a given tool.
- Has its own token budget (smaller for "narrow" tools like
  ``update_mastery``, larger for "open the world" tools like
  ``recommend_text``).
- Returns a ``Recall`` made of labelled ``Section`` objects per tier.

Tiers (PIVOT_ROADMAP §7.7):

- Tier 1 (always)  — learner profile, active goal, top-N mastery.
- Tier 2 (recent)  — last few sessions, last few evidence rows for the
                     target competency, last reading paragraph.
- Tier 3 (vector)  — semantic neighbours (Phase B; we leave a stub here
                     so the policy table is shape-complete).

The policy callables are PURE: they take a ``RecallContext`` and return
a ``Recall``. They never write. That's what makes this layer testable
without spinning up the agent.

When a tool_hint isn't in the table (or has a colon-suffixed extension,
e.g. ``"open_turn:reader"``), ``find_policy`` matches by prefix on the
canonical key.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from backend.config import PROJECT_ROOT
from backend.mastery import model as mastery_model
from backend.memory.manager import Recall, RecallContext, Section
from backend.models import db

logger = logging.getLogger(__name__)


# Cheap token estimator. We never pay for accuracy here — the budget
# bookkeeping is for the agent UX (don't blow past the model's window),
# not for billing. If we ever switch to a tokenizer call this is the one
# place to swap.
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _section(label: str, text: str) -> Section:
    text = text.strip()
    return Section(label=label, text=text, tokens=_estimate_tokens(text))


# ─── Tier 1 (always-on) builders ────────────────────────────────────────


def _learner_profile_section() -> Optional[Section]:
    """Read ``data/memory/learner/profile.md`` if it exists."""
    profile = PROJECT_ROOT / "data" / "memory" / "learner" / "profile.md"
    if not profile.exists():
        return None
    try:
        text = profile.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None
    return _section("learner_profile", text)


def _active_goal_section() -> Optional[Section]:
    """Pick the goals row most recently written, if any."""
    try:
        row = db.cursor().execute(
            "SELECT raw_text, parsed FROM goals ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    except Exception:  # pragma: no cover — schema not initialised in unit tests
        return None
    if not row:
        # Fall back to the JSON file (the seed script writes one).
        return _goal_json_section()
    raw_text, parsed = row[0], row[1]
    body = (raw_text or "").strip()
    if parsed:
        body += "\n\nparsed: " + (parsed if isinstance(parsed, str) else json.dumps(parsed))
    if not body:
        return None
    return _section("active_goal", body)


def _goal_json_section() -> Optional[Section]:
    path = PROJECT_ROOT / "data" / "memory" / "learner" / "goal.json"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    return _section("active_goal", text)


def _top_mastery_section(limit: int = 10) -> Optional[Section]:
    rows = mastery_model.list_top(limit=limit)
    if not rows:
        return None
    lines = [
        f"- {r.competency_id}: confidence={r.confidence}/100 "
        f"variance={r.variance}/100 (n={r.evidence_count})"
        for r in rows
    ]
    return _section("top_mastery", "\n".join(lines))


def _gist_of_last_session_section() -> Optional[Section]:
    """Tiny gist of the most-recent session, read from JSONL.

    Phase B replaces this with the proper multi-resolution summariser;
    for Phase A we just take the last 3 events and stringify them.
    """
    sessions_dir = PROJECT_ROOT / "data" / "memory" / "sessions"
    if not sessions_dir.exists():
        return None
    files = sorted(sessions_dir.glob("*.jsonl"))
    if not files:
        return None
    last = files[-1]
    try:
        tail = last.read_text(encoding="utf-8").strip().splitlines()[-3:]
    except OSError:
        return None
    if not tail:
        return None
    return _section("last_session_tail", "\n".join(tail))


# ─── Tier 2 (recent) builders ───────────────────────────────────────────


def _recent_evidence_for(
    competency_id: str,
    *,
    limit: int = 20,
) -> Optional[Section]:
    try:
        rows = db.cursor().execute(
            """
            SELECT occurred_at, quality, source
            FROM evidence
            WHERE competency_id = ?
            ORDER BY occurred_at DESC
            LIMIT ?
            """,
            [competency_id, limit],
        ).fetchall()
    except Exception:  # pragma: no cover
        return None
    if not rows:
        return None
    lines = [
        f"- {r[0]} quality={float(r[1]):.2f} source={r[2]}" for r in rows
    ]
    return _section(
        f"recent_evidence:{competency_id}",
        "\n".join(lines),
    )


def _recent_reading_sessions(limit: int = 5) -> Optional[Section]:
    """Last few reader sessions, read from the session index parquet."""
    index_path = (
        PROJECT_ROOT / "data" / "memory" / "sessions" / "index.parquet"
    )
    if not index_path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(index_path)
        rows = table.to_pylist()
    except Exception:  # pragma: no cover
        return None
    if not rows:
        return None
    rows = rows[-limit:]
    lines = [f"- session {r.get('session_id')} closed_at={r.get('closed_at')}" for r in rows]
    return _section("recent_sessions", "\n".join(lines))


def _query_for_two_stage(ctx: RecallContext) -> str:
    """Pick the best free-text query the policy can offer the retriever.

    Falls back to known arg keys (``query``, ``text``, ``utterance``)
    then to the observation if it's a string, then to an empty string
    so the caller knows to skip Tier 3.
    """
    args = ctx.args or {}
    for key in ("query", "text", "utterance", "user_text", "input"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    obs = ctx.observation
    if isinstance(obs, str) and obs.strip():
        return obs.strip()
    if isinstance(obs, dict):
        for key in ("text", "user_text"):
            val = obs.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _two_stage_section(
    query: str,
    *,
    source: str,
    filter: Optional[dict[str, Any]] = None,
    label_suffix: str = "",
    top_k: int = 3,
    deadline_ms: int = 250,
) -> Optional[Section]:
    """Run B.15's two-stage retriever and pack hits as a Tier-3 Section."""
    if not query or not query.strip():
        return None
    try:
        from backend.memory.retriever import recall_two_stage  # noqa: WPS433

        hits = recall_two_stage(
            query,
            source=source,
            top_k=top_k,
            filter=filter,
            deadline_ms=deadline_ms,
        )
    except Exception:  # pragma: no cover
        return None
    if not hits:
        return None
    lines: list[str] = []
    for h in hits:
        role = h.metadata.get("role") or h.metadata.get("source_kind") or source
        text = (h.text or "").strip().replace("\n", " ")
        lines.append(f"- [{role}] {text[:160]}")
    label = "two_stage_" + source + (f":{label_suffix}" if label_suffix else "")
    return _section(label, "\n".join(lines))


def _factsheet_section(competency_id: str) -> Optional[Section]:
    """Read ``data/memory/facts/<safe_id>.md`` if a sheet exists (B.13)."""
    if not competency_id:
        return None
    try:
        from backend.memory import factsheets as _factsheets  # noqa: WPS433

        body = _factsheets.load_factsheet(competency_id)
    except Exception:  # pragma: no cover
        return None
    if not body or not body.strip():
        return None
    return _section(f"factsheet:{competency_id}", body)


def _target_mastery_row(competency_id: str) -> Optional[Section]:
    row = mastery_model.get(competency_id)
    if row.evidence_count == 0:
        return _section(
            f"mastery:{competency_id}",
            "(no evidence yet — this is the first probe)",
        )
    body = (
        f"competency_id: {row.competency_id}\n"
        f"confidence: {row.confidence}/100\n"
        f"variance:   {row.variance}/100\n"
        f"evidence_count: {row.evidence_count}"
    )
    return _section(f"mastery:{competency_id}", body)


# ─── Per-tool builders ──────────────────────────────────────────────────


def _build_open_turn(ctx: RecallContext) -> Recall:
    tier1: list[Section] = []
    for sec in (
        _learner_profile_section(),
        _active_goal_section(),
        _top_mastery_section(limit=10),
    ):
        if sec is not None:
            tier1.append(sec)
    tier2: list[Section] = []
    last = _gist_of_last_session_section()
    if last is not None:
        tier2.append(last)
    used = sum(s.tokens for s in (*tier1, *tier2))
    return Recall(tier1=tier1, tier2=tier2, used_tokens=used)


def _build_recommend_text(ctx: RecallContext) -> Recall:
    tier1: list[Section] = []
    goal = _active_goal_section()
    if goal:
        tier1.append(goal)
    top = _top_mastery_section(limit=10)
    if top:
        tier1.append(top)
    tier2: list[Section] = []
    sessions = _recent_reading_sessions(limit=5)
    if sessions:
        tier2.append(sessions)
    used = sum(s.tokens for s in (*tier1, *tier2))
    return Recall(tier1=tier1, tier2=tier2, used_tokens=used)


def _build_update_mastery(ctx: RecallContext) -> Recall:
    competency_id = (ctx.args or {}).get("competency_id", "")
    tier1: list[Section] = []
    if competency_id:
        target = _target_mastery_row(competency_id)
        if target:
            tier1.append(target)
        sheet = _factsheet_section(competency_id)
        if sheet:
            tier1.append(sheet)
    tier2: list[Section] = []
    if competency_id:
        ev = _recent_evidence_for(competency_id, limit=20)
        if ev:
            tier2.append(ev)
    used = sum(s.tokens for s in (*tier1, *tier2))
    return Recall(tier1=tier1, tier2=tier2, used_tokens=used)


def _build_start_drill(ctx: RecallContext) -> Recall:
    competency_id = (ctx.args or {}).get("competency_id", "")
    tier1: list[Section] = []
    if competency_id:
        target = _target_mastery_row(competency_id)
        if target:
            tier1.append(target)
        sheet = _factsheet_section(competency_id)
        if sheet:
            tier1.append(sheet)
    top = _top_mastery_section(limit=5)
    if top:
        tier1.append(top)
    tier2: list[Section] = []
    if competency_id:
        ev = _recent_evidence_for(competency_id, limit=10)
        if ev:
            tier2.append(ev)
    used = sum(s.tokens for s in (*tier1, *tier2))
    return Recall(tier1=tier1, tier2=tier2, used_tokens=used)


# ─── B.11 — RecallPolicies for Phase B tools ────────────────────────────


def _build_parse_goal(ctx: RecallContext) -> Recall:
    """parse_goal — only the learner profile (no goal yet by definition)."""
    tier1: list[Section] = []
    profile = _learner_profile_section()
    if profile:
        tier1.append(profile)
    used = sum(s.tokens for s in tier1)
    return Recall(tier1=tier1, used_tokens=used)


def _build_plan_curriculum(ctx: RecallContext) -> Recall:
    """plan_curriculum — profile, active goal, top mastery, last gist."""
    tier1: list[Section] = []
    for sec in (
        _learner_profile_section(),
        _active_goal_section(),
        _top_mastery_section(limit=15),
    ):
        if sec is not None:
            tier1.append(sec)
    tier2: list[Section] = []
    sessions = _recent_reading_sessions(limit=3)
    if sessions:
        tier2.append(sessions)
    used = sum(s.tokens for s in (*tier1, *tier2))
    return Recall(tier1=tier1, tier2=tier2, used_tokens=used)


def _build_replan_atlas(ctx: RecallContext) -> Recall:
    """replan_atlas — profile, goal, full mastery vector, recent activity."""
    tier1: list[Section] = []
    for sec in (
        _learner_profile_section(),
        _active_goal_section(),
        _full_mastery_section(),
    ):
        if sec is not None:
            tier1.append(sec)
    tier2: list[Section] = []
    histogram = _evidence_histogram_section(days=7)
    if histogram:
        tier2.append(histogram)
    sessions = _recent_reading_sessions(limit=7)
    if sessions:
        tier2.append(sessions)
    used = sum(s.tokens for s in (*tier1, *tier2))
    return Recall(tier1=tier1, tier2=tier2, used_tokens=used)


def _build_start_conversation(ctx: RecallContext) -> Recall:
    """start_conversation — profile + goal (T1), recent transcripts (T2),
    two-stage retrieval over older transcripts (T3, B.15)."""
    tier1: list[Section] = []
    for sec in (_learner_profile_section(), _active_goal_section()):
        if sec is not None:
            tier1.append(sec)
    tier2: list[Section] = []
    scenario = (ctx.args or {}).get("scenario") or ""
    transcripts = _recent_transcripts(scenario=scenario, limit=2)
    if transcripts:
        tier2.append(transcripts)
    tier3: list[Section] = []
    query = _query_for_two_stage(ctx)
    if query:
        filt: dict[str, Any] = {}
        if scenario:
            filt["scenario"] = scenario
        sec = _two_stage_section(
            query,
            source="transcripts",
            filter=filt or None,
            label_suffix=scenario,
        )
        if sec:
            tier3.append(sec)
    used = sum(s.tokens for s in (*tier1, *tier2, *tier3))
    return Recall(tier1=tier1, tier2=tier2, tier3=tier3, used_tokens=used)


def _build_capture_text(ctx: RecallContext) -> Recall:
    """capture_text — profile + goal (T1), last 3 captures meta (T2),
    two-stage transcripts retrieval scoped to the surface (T3, B.15).

    Aliased from B.11's ``start_capture`` mention via colon-prefix
    matching in find_policy (see test_recall_policies).
    """
    tier1: list[Section] = []
    for sec in (_learner_profile_section(), _active_goal_section()):
        if sec is not None:
            tier1.append(sec)
    tier2: list[Section] = []
    captures = _recent_captures_meta(limit=3)
    if captures:
        tier2.append(captures)
    tier3: list[Section] = []
    query = _query_for_two_stage(ctx)
    if query:
        surface = (ctx.args or {}).get("surface_kind") or ""
        filt: dict[str, Any] = {}
        if surface:
            filt["scenario"] = surface  # capture writes scenario=surface_kind
        sec = _two_stage_section(
            query,
            source="transcripts",
            filter=filt or None,
            label_suffix=surface,
        )
        if sec:
            tier3.append(sec)
    used = sum(s.tokens for s in (*tier1, *tier2, *tier3))
    return Recall(tier1=tier1, tier2=tier2, tier3=tier3, used_tokens=used)


def _build_explain_grammar(ctx: RecallContext) -> Recall:
    """explain_grammar — competency rows + factsheet (T1), recent errors (T2),
    two-stage retrieval over evidence scoped to the competency (T3, B.15)."""
    competency_id = (ctx.args or {}).get("competency_id", "")
    tier1: list[Section] = []
    if competency_id:
        target = _target_mastery_row(competency_id)
        if target:
            tier1.append(target)
        sheet = _factsheet_section(competency_id)
        if sheet:
            tier1.append(sheet)
    tier2: list[Section] = []
    if competency_id:
        ev = _recent_evidence_for(competency_id, limit=10)
        if ev:
            tier2.append(ev)
    tier3: list[Section] = []
    query = _query_for_two_stage(ctx)
    if query and competency_id:
        sec = _two_stage_section(
            query,
            source="evidence",
            filter={"competency_id": competency_id},
            label_suffix=competency_id,
        )
        if sec:
            tier3.append(sec)
    used = sum(s.tokens for s in (*tier1, *tier2, *tier3))
    return Recall(tier1=tier1, tier2=tier2, tier3=tier3, used_tokens=used)


# Helpers used only by the Phase B builders ------------------------------


def _full_mastery_section() -> Optional[Section]:
    """Every mastery row, ordered by domain and confidence."""
    rows = mastery_model.list_top(limit=200)
    if not rows:
        return None
    lines = [
        f"- {r.competency_id}: confidence={r.confidence}/100 "
        f"variance={r.variance}/100 (n={r.evidence_count})"
        for r in rows
    ]
    return _section("full_mastery", "\n".join(lines))


def _evidence_histogram_section(days: int = 7) -> Optional[Section]:
    """Per-competency evidence count over the last N days."""
    try:
        rows = db.cursor().execute(
            f"""
            SELECT competency_id, COUNT(*) AS n
            FROM evidence
            WHERE occurred_at >= NOW() - INTERVAL {int(days)} DAY
            GROUP BY competency_id
            ORDER BY n DESC
            """
        ).fetchall()
    except Exception:  # pragma: no cover
        return None
    if not rows:
        return None
    lines = [f"- {r[0]}: {int(r[1])} events" for r in rows]
    return _section(f"evidence_histogram_{days}d", "\n".join(lines))


def _konversation_dirs() -> list[Path]:
    """Return every konversation/ dir that actually contains data.

    We look at both the live ``MemoryManager._root`` (which B.5 writes
    to) and the legacy ``PROJECT_ROOT / data / memory`` path (which
    early prototypes and the B.11 stub tests use). A dir is included
    only when it has at least one ``*.jsonl`` file — bare empty
    directories created by the audit-trail bootstrap don't count.
    """
    out: list[Path] = []
    try:
        from backend.memory.manager import get_memory  # noqa: WPS433

        live = Path(get_memory()._root) / "konversation"  # type: ignore[union-attr]
        if live.exists() and any(live.glob("*.jsonl")):
            out.append(live)
    except Exception:  # pragma: no cover — defensive
        pass
    legacy = PROJECT_ROOT / "data" / "memory" / "konversation"
    if legacy.exists() and any(legacy.glob("*.jsonl")) and legacy not in out:
        out.append(legacy)
    return out


def _captures_dirs() -> list[Path]:
    """Return every captures/ dir that actually contains data."""
    out: list[Path] = []
    try:
        from backend.memory.manager import get_memory  # noqa: WPS433

        live = Path(get_memory()._root) / "captures"  # type: ignore[union-attr]
        if live.exists() and (
            (live / "index.parquet").exists()
            or (live / "meta.jsonl").exists()
        ):
            out.append(live)
    except Exception:  # pragma: no cover
        pass
    legacy = PROJECT_ROOT / "data" / "memory" / "captures"
    if legacy.exists() and (
        (legacy / "index.parquet").exists()
        or (legacy / "meta.jsonl").exists()
    ) and legacy not in out:
        out.append(legacy)
    return out


def _recent_transcripts(*, scenario: str = "", limit: int = 2) -> Optional[Section]:
    """Read transcripts.jsonl tail and emit ``role: text`` lines.

    B.5 ships real transcripts via ``backend.conversation.runner``; this
    function reads them. Older session-id-named jsonl files (the B.11
    stub fixture) are still tolerated: we just emit their filenames.
    """
    dirs = _konversation_dirs()
    if not dirs:
        return None

    real_lines: list[str] = []
    fallback_files: list[Path] = []
    for konv_dir in dirs:
        files = sorted(konv_dir.glob("*.jsonl"))
        for f in files[-4:]:
            try:
                raw = f.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            parsed_any = False
            for line in raw:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = str(obj.get("role") or "").strip()
                text = str(obj.get("text") or "").strip()
                if not role or not text:
                    continue
                parsed_any = True
                real_lines.append(f"- {role}: {text[:140]}")
            if not parsed_any:
                fallback_files.append(f)

    label = "recent_transcripts" + (f":{scenario}" if scenario else "")

    if real_lines:
        if scenario:
            filtered = [
                line for line in real_lines if scenario.lower() in line.lower()
            ]
            if filtered:
                real_lines = filtered
        body = "\n".join(real_lines[-2 * max(1, limit):])
        return _section(label, body)

    if not fallback_files:
        return None
    chosen = fallback_files[-limit:]
    body = "\n".join(f"- {f.name}" for f in chosen)
    return _section(label, body)


def _recent_captures_meta(*, limit: int = 3) -> Optional[Section]:
    """Tail ``captures/index.parquet`` OR the legacy ``meta.jsonl``.

    B.6/B.18 writes ``captures/index.parquet``; we prefer that. The
    older B.11 stub honoured ``captures/meta.jsonl`` — still supported
    so existing tests keep passing.
    """
    for captures_dir in _captures_dirs():
        parquet_path = captures_dir / "index.parquet"
        if parquet_path.exists():
            rows = _read_capture_index_tail(parquet_path, limit=limit)
            if rows:
                body = "\n".join(
                    f"- {r.get('id', '?')} kind={r.get('surface_kind', '?')} "
                    f"at={r.get('captured_at', '?')}"
                    for r in rows
                )
                return _section("recent_captures_meta", body)

        meta_path = captures_dir / "meta.jsonl"
        if meta_path.exists():
            try:
                lines = meta_path.read_text(encoding="utf-8").strip().splitlines()
            except OSError:
                continue
            if lines:
                tail = lines[-limit:]
                return _section("recent_captures_meta", "\n".join(tail))
    return None


def _read_capture_index_tail(path: Path, *, limit: int) -> list[dict[str, object]]:
    try:
        import pyarrow.parquet as pq  # noqa: WPS433

        table = pq.read_table(path)
        rows = table.to_pylist()
    except Exception:
        return []
    return list(rows[-limit:])


# ─── Policy table ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecallPolicy:
    """Recipe for what to load when a given tool is about to fire."""

    tool_hint: str
    budget_tokens: int
    build: Callable[[RecallContext], Recall]


POLICIES: dict[str, RecallPolicy] = {
    # Phase A
    "open_turn":         RecallPolicy("open_turn", 1500, _build_open_turn),
    "recommend_text":    RecallPolicy("recommend_text", 2500, _build_recommend_text),
    "update_mastery":    RecallPolicy("update_mastery", 1000, _build_update_mastery),
    "start_drill":       RecallPolicy("start_drill", 1500, _build_start_drill),
    # Phase B (B.11)
    "parse_goal":        RecallPolicy("parse_goal", 800, _build_parse_goal),
    "plan_curriculum":   RecallPolicy("plan_curriculum", 3500, _build_plan_curriculum),
    "replan_atlas":      RecallPolicy("replan_atlas", 4000, _build_replan_atlas),
    "start_conversation": RecallPolicy(
        "start_conversation", 3500, _build_start_conversation
    ),
    "capture_text":      RecallPolicy("capture_text", 1000, _build_capture_text),
    "explain_grammar":   RecallPolicy("explain_grammar", 2000, _build_explain_grammar),
}


# B.11 mentions ``start_capture`` for the camera room while §7.5 names
# the same tool ``capture_text``. Treat them as aliases so an SSE turn
# emitted from the camera UI doesn't fall through to the stub-policy
# branch.
_ALIASES: dict[str, str] = {
    "start_capture": "capture_text",
}


def find_policy(tool_hint: str) -> Optional[RecallPolicy]:
    """Look up a policy by exact match, alias, or colon-prefix.

    ``"open_turn:reader"``  -> ``POLICIES["open_turn"]``.
    ``"start_capture"``     -> ``POLICIES["capture_text"]`` via alias.
    """
    if not tool_hint:
        return None
    canonical = _ALIASES.get(tool_hint, tool_hint)
    if canonical in POLICIES:
        return POLICIES[canonical]
    head = canonical.split(":", 1)[0]
    head = _ALIASES.get(head, head)
    return POLICIES.get(head)
