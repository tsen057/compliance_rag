"""
app/core/schemas.py
───────────────────
Pydantic models shared across the API and core layers.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class QueryType(str, Enum):
    simple = "simple"
    complex = "complex"


# ── Request ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Natural language question over the regulatory documents.",
        examples=["What is the minimum CET1 ratio under Basel III?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve as context.",
    )

    @field_validator("question")
    @classmethod
    def strip_and_check_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


# ── Response ──────────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    document: str = Field(..., description="Source PDF filename.")
    page: Optional[int] = Field(None, description="Page number within the PDF.")
    excerpt: str = Field(..., description="Relevant text excerpt from the source.")


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    query_type: QueryType
    faithfulness_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Estimated RAGAS faithfulness score (0–1).",
    )


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    chunks_indexed: int
