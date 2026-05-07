"""
app/api/routes.py
------------------
API endpoints:

  GET  /health  -> is everything ready?
  POST /query   -> ask a question, get an answer + sources
  POST /ingest  -> rebuild the search index from all PDFs in data/docs/
  POST /upload  -> upload a new PDF and re-index immediately
"""

from __future__ import annotations
import shutil
import sys

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from app.core.agent import ComplianceAgent
from app.core.ingestion import run_ingestion
from app.core.retriever import chunk_count, is_ready
from app.core.schemas import (
    HealthResponse, IngestResponse, QueryRequest, QueryResponse, QueryType,
)
from app.core.config import get_settings

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

router = APIRouter()
_agent = ComplianceAgent()


def _faithfulness_score(answer: str, sources: list) -> float:
    if not answer or not sources:
        return 0.0
    context = " ".join(s.excerpt for s in sources).lower()
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words:
        return 0.0
    return round(sum(1 for w in words if w in context) / len(words), 2)


# ── GET /health ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    ready = is_ready()
    return HealthResponse(
        status="ok",
        vectorstore_ready=ready,
        chunks_indexed=chunk_count() if ready else 0,
    )


# ── POST /query ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest) -> QueryResponse:
    if not is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search index not ready. Run: python -m app.core.ingestion",
        )

    logger.info(f"Question: {request.question[:80]!r}")

    try:
        result = _agent.query(request.question, top_k=request.top_k)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        query_type=QueryType(result["query_type"]),
        faithfulness_score=_faithfulness_score(result["answer"], result["sources"]),
    )


# ── POST /ingest ───────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest() -> IngestResponse:
    logger.info("Re-ingestion started")
    try:
        result = run_ingestion()
        return IngestResponse(
            status="success",
            documents_processed=result["documents_processed"],
            chunks_indexed=result["chunks_indexed"],
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


# ── POST /upload ───────────────────────────────────────────────────────────────

@router.post("/upload", tags=["Admin"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF and make it immediately searchable.

    Steps:
      1. Validates the file is a PDF
      2. Saves it to data/docs/
      3. Re-runs ingestion so it joins the search index
      4. Clears the in-memory cache so the next query uses the fresh index
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    settings = get_settings()
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.docs_dir / file.filename

    logger.info(f"Receiving upload: {file.filename}")

    # Save to disk
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_kb = dest.stat().st_size / 1024
    logger.info(f"Saved {file.filename} ({size_kb:.0f} KB) — re-indexing...")

    # Re-index everything including the new file
    try:
        result = run_ingestion()

        # Clear retriever cache so next query loads the fresh index
        from app.core.retriever import _load_embeddings, _load_index
        _load_index.cache_clear()
        _load_embeddings.cache_clear()

        logger.info(f"Re-index complete after upload: {result}")
        return {
            "status": "success",
            "filename": file.filename,
            "size_kb": round(size_kb, 1),
            "documents_processed": result["documents_processed"],
            "chunks_indexed": result["chunks_indexed"],
        }
    except Exception as e:
        logger.error(f"Re-index failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File saved but indexing failed: {e}",
        )