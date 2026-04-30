"""Environment-driven settings for the Lesekamerad backend.

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
    # Default to Gemma 4 e4b (the 4B-class edge variant) for local-first dev.
    # Switch to gemma4:26b for richer reasoning on flagship hardware. The
    # AiSettingsPanel lets the user pick any installed model at runtime.
    ollama_model: str = Field(default="gemma4:e4b")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    qdrant_path: Path = Field(default=PROJECT_ROOT / "data" / "qdrant_storage")
    qdrant_collection: str = Field(default="lesekamerad_paragraphs")

    duckdb_path: Path = Field(default=PROJECT_ROOT / "data" / "corpus.duckdb")

    piper_binary: Path = Field(default=PROJECT_ROOT / "bin" / "piper")
    piper_model: Path = Field(
        default=PROJECT_ROOT / "data" / "piper_models" / "de_DE-thorsten-high.onnx"
    )

    # ``base`` (74 MB) is fast to download and decent for German short
    # utterances; bump to ``small`` / ``medium`` for higher fidelity on a
    # GPU or once the model has been pre-warmed.
    whisper_model_size: str = Field(default="base")
    # ``auto`` → CUDA if available, else CPU. Set to ``cpu`` to force CPU.
    whisper_device: str = Field(default="auto")

    frontend_url: str = Field(default="http://localhost:3000")
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
