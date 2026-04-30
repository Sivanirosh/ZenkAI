"""Runtime-mutable LLM options.

Keeps `Settings` (env-bound) immutable while exposing a small, JSON-backed
store for values the user can change from the UI: which Ollama model to use,
whether thinking mode is on, and the generation temperature.

On first use we hydrate from the user's `.env` (`OLLAMA_MODEL`); subsequent
changes are persisted to `data/runtime_config.json` so they survive restarts
without touching the environment file.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

from backend.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

_CONFIG_PATH: Path = PROJECT_ROOT / "data" / "runtime_config.json"
_lock = threading.RLock()


@dataclass(frozen=True)
class LlmOptions:
    model: str
    think: bool = False  # qwen3 thinking is off by default
    temperature: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


_options: Optional[LlmOptions] = None


def _load_from_disk() -> Optional[LlmOptions]:
    if not _CONFIG_PATH.exists():
        return None
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read runtime_config.json: %s", exc)
        return None
    return LlmOptions(
        model=data.get("model") or get_settings().ollama_model,
        think=bool(data.get("think", False)),
        temperature=float(data.get("temperature", 0.3)),
    )


def _persist(options: LlmOptions) -> None:
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(
            json.dumps(options.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write runtime_config.json: %s", exc)


def get_llm_options() -> LlmOptions:
    """Return the current runtime options (env → disk → defaults)."""
    global _options
    if _options is not None:
        return _options
    with _lock:
        if _options is not None:
            return _options
        settings = get_settings()
        loaded = _load_from_disk()
        _options = loaded or LlmOptions(model=settings.ollama_model)
        return _options


def set_llm_options(
    *,
    model: Optional[str] = None,
    think: Optional[bool] = None,
    temperature: Optional[float] = None,
) -> LlmOptions:
    """Patch runtime options and persist to disk."""
    global _options
    with _lock:
        current = get_llm_options()
        updated = replace(
            current,
            model=model if model is not None else current.model,
            think=think if think is not None else current.think,
            temperature=(
                temperature if temperature is not None else current.temperature
            ),
        )
        _options = updated
        _persist(updated)
        return updated


def reset_for_tests() -> None:
    """Drop the cached options — pytest use only."""
    global _options
    with _lock:
        _options = None
