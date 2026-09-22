"""Environment-driven settings for the ZenkAI backend.

All configuration comes from environment variables (or a `.env` file at the
project root). No file path should be hardcoded outside this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central settings object. Resolved once per process via `get_settings()`."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="mistral-nemo:12b")

    duckdb_path: Path = Field(default=PROJECT_ROOT / "data" / "corpus.duckdb")

    piper_binary: Path = Field(default=PROJECT_ROOT / "bin" / "piper")
    piper_model: Path = Field(
        default=PROJECT_ROOT / "data" / "piper_models" / "de_DE-thorsten-high.onnx"
    )

    whisper_model_size: str = Field(default="medium")
    whisper_device: str = Field(default="cpu")

    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
