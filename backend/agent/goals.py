"""Goal parsing for the Onboarding flow (PIVOT_ROADMAP §B.1).

The learner answers a single free-text question — "warum bist du hier?" —
and Gemma 4 turns that answer into a structured ``ParsedGoal`` JSON the
curriculum planner can consume.

Two paths in this module:

- ``parse_goal_text(raw_text)`` is the public entry point. It calls the
  active LLM provider with a JSON-strict prompt, retries once with a
  "fix the JSON" reprompt on parse failure, then falls back to a
  deterministic skeleton so the onboarding screen *never* blocks.
- ``_parse_json_object(...)`` and ``_skeleton(...)`` are the two cheap
  failure-path helpers; both are unit-tested without touching Ollama.

The schema is intentionally small. Phase B.4 (Atlas) consumes it; if
we add fields we keep the same names to avoid a JSON migration.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from backend.llm import get_provider
from backend.services.runtime_config import get_llm_options

logger = logging.getLogger(__name__)


# ─── Public schema ───────────────────────────────────────────────────────


_ALLOWED_DOMAINS = ("medical", "academic", "daily", "other")
_ALLOWED_CEFR = ("A2", "B1", "B2", "C1")


class LanguageProfile(BaseModel):
    """Learner's native + current language background."""

    native: Optional[str] = None
    current_cefr: Optional[str] = None


class ParsedGoal(BaseModel):
    """Structured representation of the learner's goal.

    Persisted as JSON in ``goals.parsed`` (DuckDB). Consumed by
    ``backend.curriculum.planner.generate_plan`` to pick competency
    clusters.
    """

    domain: str = Field(default="other")
    deadline_iso: Optional[str] = None
    target_cefr: str = Field(default="B1")
    scenarios: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    language_profile: LanguageProfile = Field(default_factory=LanguageProfile)

    def normalise(self) -> "ParsedGoal":
        """Clamp to the controlled vocabulary; collapse junk to defaults."""
        domain = self.domain.lower().strip()
        if domain not in _ALLOWED_DOMAINS:
            domain = "other"
        cefr = self.target_cefr.upper().strip()
        if cefr not in _ALLOWED_CEFR:
            cefr = "B1"
        deadline = self.deadline_iso
        if deadline is not None:
            deadline = deadline.strip() or None
            if deadline and not re.match(r"^\d{4}-\d{2}-\d{2}$", deadline):
                deadline = None
        scenarios = [s.strip() for s in self.scenarios if isinstance(s, str) and s.strip()]
        motivations = [
            m.strip() for m in self.motivations if isinstance(m, str) and m.strip()
        ]
        return ParsedGoal(
            domain=domain,
            deadline_iso=deadline,
            target_cefr=cefr,
            scenarios=scenarios[:5],
            motivations=motivations[:3],
            language_profile=self.language_profile,
        )


# ─── Prompt (versioned) ──────────────────────────────────────────────────


# 2026-04-30 — GOAL_PARSER_PROMPT v1. Strict-JSON only; the controlled
# vocabularies are listed inline so the model can't invent new domains.
GOAL_PARSER_PROMPT = """\
You classify a learner's free-text answer to "Why are you learning German?".

Output STRICT JSON only — no Markdown fences, no commentary. The JSON
must conform exactly to this schema (no extra keys):

{{
  "domain":         "medical" | "academic" | "daily" | "other",
  "deadline_iso":   "YYYY-MM-DD" or null,
  "target_cefr":    "A2" | "B1" | "B2" | "C1",
  "scenarios":      [string, ...],     // max 5, each <= 80 chars
  "motivations":    [string, ...],     // max 3, each <= 80 chars
  "language_profile": {{
    "native":       string or null,
    "current_cefr": "A1" | "A2" | "B1" | "B2" | "C1" or null
  }}
}}

Rules
- domain: pick the closest of medical / academic / daily / other. If
  unsure, use "other".
- deadline_iso: ISO date (YYYY-MM-DD) only if the learner mentioned a
  concrete date. Otherwise null. NEVER guess.
- target_cefr: default to "B1" if the learner didn't state a level.
- scenarios: specific real-world situations the learner needs the
  language for (e.g. "patient-history dialogue", "research defence").
- motivations: short reasons in the learner's own words, lightly
  paraphrased.
- language_profile.current_cefr: the learner's CURRENT level if stated.

Learner answer
{raw_text}
"""


# ─── Public entry point ──────────────────────────────────────────────────


async def parse_goal_text(raw_text: str, *, timeout: float = 20.0) -> ParsedGoal:
    """Parse a free-text goal into a structured ``ParsedGoal``.

    Order of preference:

    1. LLM call (Gemma 4 via the active provider) with a JSON-strict
       prompt.
    2. One reprompt asking the model to "fix the JSON" if the first
       response failed schema validation.
    3. Deterministic skeleton — domain heuristics from keyword match,
       target_cefr defaults to B1.

    Never raises. The onboarding flow always gets a usable object.
    """
    text = (raw_text or "").strip()
    if not text:
        return _skeleton("")

    prompt = GOAL_PARSER_PROMPT.format(raw_text=text)
    parsed = await _try_llm_parse(prompt, timeout=timeout)
    if parsed is not None:
        return parsed.normalise()

    fix_prompt = (
        prompt
        + "\n\nYour previous response was not valid JSON or did not "
        "match the schema. Re-emit ONLY the JSON object. No prose."
    )
    parsed = await _try_llm_parse(fix_prompt, timeout=timeout)
    if parsed is not None:
        return parsed.normalise()

    logger.info("parse_goal_text falling back to skeleton (LLM unavailable or invalid)")
    return _skeleton(text)


# ─── Helpers ─────────────────────────────────────────────────────────────


async def _try_llm_parse(prompt: str, *, timeout: float) -> Optional[ParsedGoal]:
    try:
        raw = await get_provider().generate(
            prompt, options=get_llm_options(), timeout=timeout
        )
    except Exception as exc:  # pragma: no cover — provider already swallows
        logger.warning("provider.generate raised: %s", exc)
        return None
    if not raw:
        return None
    obj = _parse_json_object(raw)
    if obj is None:
        return None
    try:
        return ParsedGoal.model_validate(obj)
    except ValidationError as exc:
        logger.info("ParsedGoal validation failed: %s", exc)
        return None


def _parse_json_object(text: str) -> Optional[dict[str, Any]]:
    """Extract the first ``{...}`` JSON object from an LLM response."""
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


_DOMAIN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "medical",
        (
            "fsp", "med", "medizin", "medical", "patient", "anamnese",
            "dentist", "zahnarzt", "krankenhaus", "klinik", "doctor",
            "pflege", "nurse", "approbation",
        ),
    ),
    (
        "academic",
        (
            "phd", "doktor", "promotion", "thesis", "defence", "defense",
            "verteidigung", "research", "forschung", "uni ", "university",
            "postdoc", "publikation", "paper", "konferenz",
        ),
    ),
    (
        "daily",
        (
            "alltag", "leben", "shopping", "einkauf", "u-bahn", "bahn",
            "everyday", "daily", "vacation", "urlaub", "tourist", "freunde",
            "kinder", "schule",
        ),
    ),
)


def _skeleton(raw_text: str) -> ParsedGoal:
    """Best-effort deterministic parse used when the LLM is unavailable."""
    lowered = raw_text.lower()
    domain = "other"
    for candidate, words in _DOMAIN_KEYWORDS:
        if any(w in lowered for w in words):
            domain = candidate
            break

    motivations = []
    if raw_text:
        snippet = raw_text.strip().splitlines()[0][:80]
        if snippet:
            motivations = [snippet]

    return ParsedGoal(
        domain=domain,
        deadline_iso=None,
        target_cefr="B1",
        scenarios=[],
        motivations=motivations,
        language_profile=LanguageProfile(),
    ).normalise()
