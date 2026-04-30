"""First-cut idempotent memory Compactor (PIVOT_ROADMAP §B.16).

Run from a cron / admin endpoint / unit test, this:

1. Locks ``data/memory/compaction/locks/compactor.lock`` (atomic
   ``O_EXCL`` create; reaps stale locks older than 10 min so a crashed
   prior run never wedges the system).
2. **summarise_missing_sessions** — for every session in
   ``sessions/<sid>.0001.jsonl`` (and rotation siblings) that has no
   companion ``sessions/<sid>.gist.md``, write a mechanical gist
   (event count, tool names, duration). The LLM-backed summariser is
   B.12 and is deferred.
3. **archive_inverse_mastery** — for every mastery row with
   ``mu >= 0.8 AND sigma <= 0.08 AND last_evidence <= now - 30d``,
   archive that competency's evidence rows older than 30 days from
   ``evidence/ledger.jsonl`` into
   ``evidence/archive/<YYYY>-Q<N>.jsonl.zst``. Hot ledger gets the
   archived rows removed; the mastery row itself is left untouched.
4. **vector_rebuild** — no-op stub; lights up in B.14.
5. Writes ``compaction/last_run.json`` for resumability + UI hint
   („zuletzt komprimiert: 12:42").
6. Releases the lock.

All phases honour a single deadline: each gets ⅓ of the remaining
budget. The Compactor never blocks the agent loop — it always returns
within ``deadline_ms`` (defaults to 30 s).

Wired from ``MemoryManager.compact()`` and exposed via
``POST /admin/memory/compact``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_LOCK_NAME = "compactor.lock"
_LOCK_STALE_SECONDS = 600  # 10 min
_HOT_LEDGER_NAME = "ledger.jsonl"


# ─── Public report shape ────────────────────────────────────────────────


@dataclass
class CompactionReport:
    started_at: str
    finished_at: str
    duration_ms: int
    phases_completed: list[str] = field(default_factory=list)
    phases_skipped: list[str] = field(default_factory=list)
    gists_written: int = 0
    sessions_already_gisted: int = 0
    archived_competencies: list[str] = field(default_factory=list)
    archived_evidence_rows: int = 0
    archive_files: list[str] = field(default_factory=list)
    cursor: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Lock errors ────────────────────────────────────────────────────────


class CompactionLockedError(RuntimeError):
    """Raised when another compactor is currently holding the lock."""


# ─── Compactor ──────────────────────────────────────────────────────────


class Compactor:
    """Background-safe compactor pinned to one ``MemoryManager``.

    The manager hands us its memory root + sessions dir + flush helper.
    We never touch the agent loop directly; all writes go through plain
    file I/O so a crash mid-archive cannot corrupt agent state.
    """

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._root: Path = Path(getattr(manager, "_root"))
        self._sessions_dir: Path = self._root / "sessions"
        self._evidence_dir: Path = self._root / "evidence"
        self._compaction_dir: Path = self._root / "compaction"
        self._lock_dir: Path = self._compaction_dir / "locks"
        self._state_path: Path = self._compaction_dir / "last_run.json"
        for d in (self._compaction_dir, self._lock_dir, self._evidence_dir):
            d.mkdir(parents=True, exist_ok=True)
        (self._evidence_dir / "archive").mkdir(parents=True, exist_ok=True)
        self._lock_path: Optional[Path] = None

    # ── lock ───────────────────────────────────────────────────────────

    def acquire_lock(self, *, owner: Optional[str] = None) -> Path:
        """Atomic O_EXCL create with stale-lock reaping."""
        lock = self._lock_dir / _LOCK_NAME
        owner = owner or f"pid={os.getpid()}-{uuid.uuid4().hex[:6]}"
        if lock.exists():
            try:
                age = time.time() - lock.stat().st_mtime
            except OSError:
                age = 0
            if age >= _LOCK_STALE_SECONDS:
                logger.warning(
                    "compactor: reaping stale lock (age=%.1fs)", age
                )
                try:
                    lock.unlink()
                except OSError:
                    pass
            else:
                raise CompactionLockedError(
                    f"compactor lock held (age={age:.1f}s)"
                )
        # O_EXCL atomic create — fails loudly if another process raced us.
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise CompactionLockedError("compactor lock just acquired by peer") from exc
        try:
            os.write(
                fd,
                json.dumps(
                    {
                        "owner": owner,
                        "acquired_at": datetime.now(timezone.utc).isoformat(),
                    }
                ).encode("utf-8"),
            )
        finally:
            os.close(fd)
        self._lock_path = lock
        return lock

    def release_lock(self) -> None:
        if self._lock_path is None:
            return
        try:
            self._lock_path.unlink()
        except OSError:
            pass
        self._lock_path = None

    # ── orchestrator ───────────────────────────────────────────────────

    def run(self, *, deadline_ms: int = 30_000) -> CompactionReport:
        """Run all phases under one deadline. Lock is acquired + released."""
        started = datetime.now(timezone.utc)
        report = CompactionReport(
            started_at=started.isoformat(),
            finished_at=started.isoformat(),
            duration_ms=0,
        )
        deadline = time.monotonic() + max(0.5, deadline_ms / 1000.0)

        try:
            self.acquire_lock()
        except CompactionLockedError as exc:
            report.notes.append(f"lock_held:{exc}")
            report.finished_at = datetime.now(timezone.utc).isoformat()
            report.duration_ms = int(
                (datetime.now(timezone.utc) - started).total_seconds() * 1000
            )
            self._persist_state(report)
            raise

        try:
            self._phase_summarise_missing_sessions(report, deadline)
            self._phase_archive_inverse_mastery(report, deadline)
            self._phase_vector_rebuild(report, deadline)
        finally:
            finished = datetime.now(timezone.utc)
            report.finished_at = finished.isoformat()
            report.duration_ms = int((finished - started).total_seconds() * 1000)
            self._persist_state(report)
            self.release_lock()

        return report

    # ── phase 1: gist missing sessions ─────────────────────────────────

    def _phase_summarise_missing_sessions(
        self, report: CompactionReport, deadline: float
    ) -> None:
        if time.monotonic() >= deadline:
            report.phases_skipped.append("summarise_missing_sessions")
            return
        if not self._sessions_dir.exists():
            report.phases_completed.append("summarise_missing_sessions")
            return

        # Group rotated files per session id (drop the .NNNN suffix).
        groups: dict[str, list[Path]] = {}
        for path in sorted(self._sessions_dir.glob("*.jsonl")):
            stem = path.stem  # "sid.NNNN"
            if "." not in stem:
                continue
            sid, _, _ = stem.rpartition(".")
            groups.setdefault(sid, []).append(path)

        # B.12 — delegate to the LLM-backed summariser when reachable.
        # Falls back to the mechanical writer cleanly. We honour the
        # remaining deadline per session and stop early if it elapses.
        try:
            from backend.memory import summariser as _summariser  # noqa: WPS433
        except Exception as exc:  # pragma: no cover
            _summariser = None  # type: ignore[assignment]
            report.notes.append(f"summariser_unavailable:{exc}")

        for sid, files in groups.items():
            if time.monotonic() >= deadline:
                report.notes.append("deadline_in_phase_summarise")
                break
            gist_path = self._sessions_dir / f"{sid}.gist.md"
            summ_path = self._sessions_dir / f"{sid}.summ.md"
            if gist_path.exists() and summ_path.exists():
                report.sessions_already_gisted += 1
                continue

            wrote = False
            if _summariser is not None:
                try:
                    # Per-session timeout is whatever budget remains, capped
                    # at the summariser's own ceiling — we never let one
                    # bad session blow the whole compaction window.
                    remaining = max(0.5, deadline - time.monotonic())
                    summary = _summariser.summarise_session(
                        sid,
                        files,
                        timeout=min(remaining, 5.0),
                    )
                    _summariser.write_summary_files(self._sessions_dir, summary)
                    report.gists_written += 1
                    wrote = True
                except Exception as exc:  # pragma: no cover
                    report.notes.append(f"summariser_failed:{sid}:{exc}")
            if not wrote:
                try:
                    gist_path.write_text(
                        self._mechanical_gist(sid, files), encoding="utf-8"
                    )
                    report.gists_written += 1
                except OSError as exc:
                    report.notes.append(f"gist_failed:{sid}:{exc}")
        report.phases_completed.append("summarise_missing_sessions")

    @staticmethod
    def _mechanical_gist(sid: str, files: list[Path]) -> str:
        """Cheap, deterministic summary. Real LLM gist ships in B.12."""
        events = 0
        tools_used: dict[str, int] = {}
        first_ts: Optional[str] = None
        last_ts: Optional[str] = None
        for f in files:
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    events += 1
                    ts = obj.get("occurred_at")
                    if ts:
                        first_ts = first_ts or ts
                        last_ts = ts
                    payload = obj.get("payload") or {}
                    name = payload.get("name") if isinstance(payload, dict) else None
                    kind = obj.get("kind") or ""
                    if kind == "tool_intent" and isinstance(name, str):
                        tools_used[name] = tools_used.get(name, 0) + 1
            except OSError:
                continue
        duration_s: Optional[float] = None
        if first_ts and last_ts and first_ts != last_ts:
            try:
                duration_s = (
                    datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                    - datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                ).total_seconds()
            except ValueError:
                duration_s = None

        lines = [
            f"# session {sid}",
            "",
            f"events: {events}",
            f"files: {[f.name for f in files]}",
            f"tools_used: {tools_used or '{}'}",
        ]
        if duration_s is not None:
            lines.append(f"duration_s: {duration_s:.1f}")
        lines.extend(
            [
                "",
                "_Mechanical gist (B.12 will replace with an LLM summary)._",
            ]
        )
        return "\n".join(lines) + "\n"

    # ── phase 2: archive inverse-mastery evidence ───────────────────────

    def _phase_archive_inverse_mastery(
        self, report: CompactionReport, deadline: float
    ) -> None:
        if time.monotonic() >= deadline:
            report.phases_skipped.append("archive_inverse_mastery")
            return

        # Late import to avoid a top-level cycle (mastery imports db
        # which imports config which we already hold open).
        try:
            from backend.models import db
        except Exception as exc:
            report.notes.append(f"db_unavailable:{exc}")
            report.phases_completed.append("archive_inverse_mastery")
            return

        try:
            cur = db.cursor()
            rows = cur.execute(
                """
                SELECT competency_id, mu, sigma, last_evidence
                FROM mastery
                WHERE mu >= 0.8 AND sigma <= 0.08
                """
            ).fetchall()
        except Exception as exc:
            report.notes.append(f"mastery_scan_failed:{exc}")
            report.phases_completed.append("archive_inverse_mastery")
            return

        if not rows:
            report.phases_completed.append("archive_inverse_mastery")
            return

        cutoff = self._cutoff_30d()
        ledger_path = self._evidence_dir / _HOT_LEDGER_NAME
        existing_evidence = self._load_hot_ledger(ledger_path)

        archive_targets: dict[str, list[dict[str, Any]]] = {}
        archived_ids: set[str] = set()
        kept: list[dict[str, Any]] = []
        for row in existing_evidence:
            comp_id = row.get("competency_id")
            occurred = row.get("occurred_at") or ""
            target_competencies = {
                r[0] for r in rows
                if r[3] is not None and self._iso_lt(str(r[3]), cutoff)
            }
            if (
                comp_id in target_competencies
                and occurred and self._iso_lt(occurred, cutoff)
            ):
                quarter = self._quarter_for(occurred)
                archive_targets.setdefault(quarter, []).append(row)
                archived_ids.add(comp_id)
            else:
                kept.append(row)

        if not archive_targets:
            report.phases_completed.append("archive_inverse_mastery")
            return

        archive_dir = self._evidence_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        for quarter, batch in archive_targets.items():
            if time.monotonic() >= deadline:
                report.notes.append("deadline_in_phase_archive")
                report.phases_skipped.append("archive_inverse_mastery")
                # Don't mutate the hot ledger if we couldn't finish.
                return
            archive_path = archive_dir / f"{quarter}.jsonl.zst"
            try:
                self._append_zst(archive_path, batch)
                report.archive_files.append(str(archive_path))
                report.archived_evidence_rows += len(batch)
            except Exception as exc:
                report.notes.append(f"archive_write_failed:{quarter}:{exc}")
                report.phases_skipped.append("archive_inverse_mastery")
                return

        # Only after every archive is durable do we rewrite the hot ledger.
        try:
            self._rewrite_hot_ledger(ledger_path, kept)
        except Exception as exc:
            report.notes.append(f"hot_ledger_rewrite_failed:{exc}")
            report.phases_skipped.append("archive_inverse_mastery")
            return

        report.archived_competencies = sorted(archived_ids)
        report.phases_completed.append("archive_inverse_mastery")

    # ── phase 3: vector rebuild (B.14) ─────────────────────────────────

    def _phase_vector_rebuild(
        self, report: CompactionReport, deadline: float
    ) -> None:
        if time.monotonic() >= deadline:
            report.phases_skipped.append("vector_rebuild")
            return
        try:
            from backend.memory import store_vector as _store_vector  # noqa: WPS433
        except Exception as exc:  # pragma: no cover
            report.notes.append(f"vector_unavailable:{exc}")
            report.phases_completed.append("vector_rebuild")
            return

        try:
            index = _store_vector.get_index()
        except Exception as exc:  # pragma: no cover
            report.notes.append(f"vector_index_unavailable:{exc}")
            report.phases_completed.append("vector_rebuild")
            return

        if isinstance(index, _store_vector.NullVectorIndex):
            report.notes.append("vector_index:null")
            report.phases_completed.append("vector_rebuild")
            return

        for source in ("transcripts", "evidence"):
            if time.monotonic() >= deadline:
                report.notes.append(f"deadline_in_phase_vector:{source}")
                report.phases_skipped.append("vector_rebuild")
                return
            try:
                rebuilt = index.rebuild_from_jsonl(source)
                report.notes.append(f"vector_rebuild:{source}={rebuilt}")
            except Exception as exc:  # pragma: no cover
                report.notes.append(f"vector_rebuild_failed:{source}:{exc}")
        report.phases_completed.append("vector_rebuild")

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _cutoff_30d() -> str:
        from datetime import timedelta

        return (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    @staticmethod
    def _iso_lt(a: str, b: str) -> bool:
        # Compare as ISO strings — DuckDB / Python emit the same shape so
        # lexicographic order is chronological.
        return a < b

    @staticmethod
    def _quarter_for(iso: str) -> str:
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return "unknown"
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{q}"

    def _load_hot_ledger(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return []
        return rows

    def _append_zst(self, path: Path, rows: list[dict[str, Any]]) -> None:
        """Atomic append to a zstd-compressed JSONL archive.

        We do read-decompress-append-recompress and replace via an
        ``os.replace`` so a crash mid-write leaves the previous archive
        intact. Cost is fine for B.16 — once-per-day workloads.
        """
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover — pyproject pins it
            raise RuntimeError(
                "zstandard is required for evidence archives; "
                "add `zstandard` to project dependencies"
            ) from exc

        existing_payload = b""
        if path.exists():
            try:
                with path.open("rb") as f:
                    existing_payload = zstd.ZstdDecompressor().decompress(f.read())
            except Exception:
                # Treat unreadable archive as empty so the partial-write
                # case (file existed but was truncated) self-heals.
                existing_payload = b""

        new_lines = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ).encode("utf-8")
        combined = existing_payload + new_lines
        compressed = zstd.ZstdCompressor(level=10).compress(combined)

        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            f.write(compressed)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:  # pragma: no cover
                pass
        os.replace(tmp, path)

    def _rewrite_hot_ledger(self, path: Path, rows: list[dict[str, Any]]) -> None:
        """Replace the hot ledger atomically with the surviving rows."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:  # pragma: no cover
                pass
        os.replace(tmp, path)

    def _persist_state(self, report: CompactionReport) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:  # pragma: no cover
            logger.warning("compactor: state write failed: %s", exc)


# ─── Module-level convenience used by MemoryManager.compact ─────────────


_compactor_lock = threading.Lock()


def run_for_manager(manager: Any, *, deadline_ms: int = 30_000) -> CompactionReport:
    """Run one compaction pass against a given MemoryManager."""
    compactor = Compactor(manager)
    with _compactor_lock:
        return compactor.run(deadline_ms=deadline_ms)
