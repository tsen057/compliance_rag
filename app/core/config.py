"""
app/core/config.py
──────────────────
Central settings loaded from environment / .env file.
All other modules import from here via get_settings().
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ── Paths ─────────────────────────────────────────────────────────────────
    docs_dir: Path = Field(default=Path("data/docs"))
    vectorstore_dir: Path = Field(default=Path("data/vectorstore"))

    # ── Embedding model (HuggingFace, runs locally, free) ────────────────────
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformers model name.",
    )

    # ── LLM (local, free — swap for Azure OpenAI by changing these) ──────────
    llm_model: str = Field(
        default="google/flan-t5-base",
        description="HuggingFace text2text-generation model.",
    )
    llm_max_new_tokens: int = Field(default=512)

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = Field(default=5)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
