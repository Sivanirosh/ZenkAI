"""Per-competency fact sheets (PIVOT_ROADMAP §7.6 pattern 3).

One Markdown file per competency, capped at ~300 tokens (≈ 220 words),
updated incrementally after every evidence event. Lives at
``data/memory/facts/<safe_id>.md`` so:

- Humans can read and edit it (teachers, learners). Mira picks up
  manual edits on the next recall cycle.
- The file is grep-able from a terminal — debug "what does Mira
  think she knows about X?" without spinning up a UI.
- The model loads the raw bytes, no JSON parse step needed.

Sections (rendered in fixed order so manual edits don't drift):

    # <competency_id>

    ## Confidence: μ_int / 100  (σ = σ_int)

    ## Productive examples (last N successful)
    - "..." — YYYY-MM-DD
    ...

    ## Recurring mistakes
    - "..." ×N

    ## Next teaching priority
    <one short line, set by mastery.scheduler or a teacher edit>

Eviction policy when the cap is hit: drop the oldest entry from
"Productive examples" first; if still over, merge the two smallest
counts in "Recurring mistakes". Atomic writes via tmp+rename so a
crash never half-rewrites a sheet.

Update is best-effort: callers wrap us in try/except and never let
an MD failure abort an evidence write.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_TOKEN_BUDGET = 300       # the §7.6 spec
_WORD_BUDGET = 220        # ~1.36 tokens / word for German-ish text
_MAX_EXAMPLES = 5
_MAX_MISTAKES = 8
_FILENAME_RE = re.compile(r"[^a-z0-9._-]+")
_LOCKS: dict[str, threading.Lock] = {}
_LOCK_TABLE = threading.Lock()


# ─── Public dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class FactsheetEvidence:
    """Minimal shape needed to update a sheet from a single event."""

    competency_id: str
    quality: float
    surface_form: Optional[str] = None
    notes: Optional[str] = None
    occurred_at: Optional[str] = None  # ISO8601; defaults to "now"

    @property
    def is_success(self) -> bool:
        return self.quality >= 0.6

    @property
    def is_failure(self) -> bool:
        return self.quality <= 0.35


@dataclass(frozen=True)
class FactsheetMastery:
    """Slim subset of MasteryRow used to render the confidence header."""

    confidence: int   # 0..100
    variance: int     # 0..100


# ─── Public API ─────────────────────────────────────────────────────────


def update_factsheet(
    evidence: FactsheetEvidence,
    mastery: FactsheetMastery,
    *,
    next_priority: Optional[str] = None,
) -> Optional[Path]:
    """Read-modify-write the fact sheet for ``evidence.competency_id``.

    Atomic via tmp+rename. Returns the destination path on success;
    returns ``None`` if the underlying MemoryManager is unavailable
    (tests that don't bootstrap one). Never raises — callers (the
    mastery update path) must keep working when the MD layer breaks.
    """
    facts_dir = _facts_dir()
    if facts_dir is None:
        return None
    try:
        facts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover
        logger.warning("could not create facts dir %s: %s", facts_dir, exc)
        return None

    path = facts_dir / f"{_safe_filename(evidence.competency_id)}.md"
    lock = _lock_for(str(path))
    with lock:
        existing = _read(path)
        sheet = _parse(existing) if existing else _empty_sheet(evidence.competency_id)
        sheet = _apply(sheet, evidence, mastery, next_priority=next_priority)
        sheet = _evict_until_under_budget(sheet)
        rendered = _render(sheet)
        try:
            _atomic_write(path, rendered)
        except OSError as exc:  # pragma: no cover
            logger.warning("factsheet write failed at %s: %s", path, exc)
            return None
    return path


def load_factsheet(competency_id: str) -> Optional[str]:
    """Return the raw MD body for ``competency_id`` if a sheet exists."""
    facts_dir = _facts_dir()
    if facts_dir is None:
        return None
    path = facts_dir / f"{_safe_filename(competency_id)}.md"
    return _read(path)


def list_facts() -> list[Path]:
    """Return every fact MD on disk (alphabetical)."""
    facts_dir = _facts_dir()
    if facts_dir is None or not facts_dir.exists():
        return []
    return sorted(facts_dir.glob("*.md"))


# ─── Internal sheet model ───────────────────────────────────────────────


@dataclass
class _Mistake:
    text: str
    count: int


@dataclass
class _Example:
    text: str
    when: str  # ISO date


@dataclass
class _Sheet:
    competency_id: str
    confidence: int
    variance: int
    examples: list[_Example]
    mistakes: list[_Mistake]
    next_priority: Optional[str]


def _empty_sheet(competency_id: str) -> _Sheet:
    return _Sheet(
        competency_id=competency_id,
        confidence=0,
        variance=0,
        examples=[],
        mistakes=[],
        next_priority=None,
    )


def _apply(
    sheet: _Sheet,
    evidence: FactsheetEvidence,
    mastery: FactsheetMastery,
    *,
    next_priority: Optional[str],
) -> _Sheet:
    sheet.confidence = max(0, min(100, int(mastery.confidence)))
    sheet.variance = max(0, min(100, int(mastery.variance)))
    if next_priority:
        sheet.next_priority = next_priority.strip() or sheet.next_priority

    surface = (evidence.surface_form or "").strip()
    note = (evidence.notes or "").strip()
    when = _short_iso(evidence.occurred_at)

    if evidence.is_success and surface:
        sheet.examples = _prepend_example(sheet.examples, _Example(surface, when))
    elif evidence.is_failure:
        target = (note or surface).strip()
        if target:
            sheet.mistakes = _bump_mistake(sheet.mistakes, target)

    return sheet


def _prepend_example(examples: list[_Example], item: _Example) -> list[_Example]:
    deduped = [e for e in examples if e.text != item.text]
    deduped.insert(0, item)
    return deduped[:_MAX_EXAMPLES]


def _bump_mistake(mistakes: list[_Mistake], text: str) -> list[_Mistake]:
    for m in mistakes:
        if m.text == text:
            m.count += 1
            return mistakes
    mistakes.insert(0, _Mistake(text=text, count=1))
    return mistakes[:_MAX_MISTAKES]


def _evict_until_under_budget(sheet: _Sheet) -> _Sheet:
    """Drop oldest examples first; merge smallest mistakes if still over."""
    while _word_count(_render(sheet)) > _WORD_BUDGET and sheet.examples:
        sheet.examples.pop()
    while _word_count(_render(sheet)) > _WORD_BUDGET and len(sheet.mistakes) >= 2:
        sheet.mistakes.sort(key=lambda m: m.count)
        merged = _Mistake(
            text=f"{sheet.mistakes[0].text} / {sheet.mistakes[1].text}",
            count=sheet.mistakes[0].count + sheet.mistakes[1].count,
        )
        sheet.mistakes = [merged] + sheet.mistakes[2:]
    return sheet


# ─── Render / parse ─────────────────────────────────────────────────────


def _render(sheet: _Sheet) -> str:
    lines: list[str] = []
    lines.append(f"# {sheet.competency_id}")
    lines.append("")
    lines.append(f"## Confidence: {sheet.confidence}/100  (σ = {sheet.variance})")
    lines.append("")
    lines.append("## Productive examples (last 5 successful)")
    if sheet.examples:
        for ex in sheet.examples:
            lines.append(f'- "{ex.text}" — {ex.when}')
    else:
        lines.append("- (none yet)")
    lines.append("")
    lines.append("## Recurring mistakes")
    if sheet.mistakes:
        for m in sheet.mistakes:
            suffix = f" ×{m.count}" if m.count > 1 else ""
            lines.append(f'- "{m.text}"{suffix}')
    else:
        lines.append("- (none yet)")
    lines.append("")
    lines.append("## Next teaching priority")
    lines.append(sheet.next_priority or "(set by scheduler / teacher)")
    lines.append("")
    return "\n".join(lines)


_HEADER_RE = re.compile(
    r"^##\s+Confidence:\s+(\d+)/100\s+\(σ\s*=\s*(\d+)\)", re.MULTILINE
)
_EXAMPLE_RE = re.compile(r'^- "(?P<text>.+?)"\s+—\s+(?P<when>\S+)\s*$')
_MISTAKE_RE = re.compile(r'^- "(?P<text>.+?)"(?:\s+×(?P<count>\d+))?\s*$')
_SECTION_RE = re.compile(r"^##\s+(.+)$")


def _parse(body: str) -> _Sheet:
    competency_id = ""
    confidence = 0
    variance = 0
    examples: list[_Example] = []
    mistakes: list[_Mistake] = []
    next_priority: Optional[str] = None

    section: Optional[str] = None
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.startswith("# ") and not line.startswith("## "):
            competency_id = line[2:].strip()
            continue
        m_header = _HEADER_RE.match(line)
        if m_header:
            confidence = int(m_header.group(1))
            variance = int(m_header.group(2))
            section = "header"
            continue
        m_section = _SECTION_RE.match(line)
        if m_section:
            label = m_section.group(1).lower()
            if label.startswith("productive examples"):
                section = "examples"
            elif label.startswith("recurring mistakes"):
                section = "mistakes"
            elif label.startswith("next teaching priority"):
                section = "priority"
            else:
                section = None
            continue
        if section == "examples":
            m_ex = _EXAMPLE_RE.match(line)
            if m_ex and m_ex.group("text") != "(none yet)":
                examples.append(
                    _Example(text=m_ex.group("text"), when=m_ex.group("when"))
                )
        elif section == "mistakes":
            m_mi = _MISTAKE_RE.match(line)
            if m_mi and m_mi.group("text") != "(none yet)":
                count = int(m_mi.group("count")) if m_mi.group("count") else 1
                mistakes.append(_Mistake(text=m_mi.group("text"), count=count))
        elif section == "priority":
            stripped = line.strip()
            if stripped and not stripped.startswith("(set by"):
                next_priority = (
                    stripped if next_priority is None
                    else f"{next_priority} {stripped}"
                ).strip()

    return _Sheet(
        competency_id=competency_id,
        confidence=confidence,
        variance=variance,
        examples=examples[:_MAX_EXAMPLES],
        mistakes=mistakes[:_MAX_MISTAKES],
        next_priority=next_priority,
    )


# ─── Helpers ────────────────────────────────────────────────────────────


def _facts_dir() -> Optional[Path]:
    try:
        from backend.memory.manager import get_memory  # noqa: WPS433

        return Path(get_memory()._root) / "facts"  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover
        logger.debug("MemoryManager unavailable for factsheets: %s", exc)
        return None


def _safe_filename(competency_id: str) -> str:
    safe = _FILENAME_RE.sub("_", competency_id.lower())
    return safe.strip("_") or "unknown"


def _short_iso(value: Optional[str]) -> str:
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).date().isoformat()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _read(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _atomic_write(path: Path, body: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(body)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:  # pragma: no cover
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:  # pragma: no cover
            pass
        raise


def _lock_for(key: str) -> threading.Lock:
    with _LOCK_TABLE:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
    return lock


def _approx_token_budget() -> int:
    """Exposed for tests so we can assert the renderer respects the cap."""
    return _TOKEN_BUDGET


__all__ = [
    "FactsheetEvidence",
    "FactsheetMastery",
    "list_facts",
    "load_factsheet",
    "update_factsheet",
]
