"""
app/core/retriever.py
──────────────────────
FAISS vector store wrapper.

Loads the persisted index once and caches it in memory.
Exposes retrieve() for use by the agent, and helper functions
for health checks.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from app.core.config import get_settings
from app.core.schemas import SourceChunk

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the embedding model (called once on first use)."""
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _load_vectorstore() -> FAISS | None:
    """
    Load the persisted FAISS index from disk.
    Returns None if the index has not been built yet.
    """
    settings = get_settings()
    if not settings.vectorstore_dir.exists():
        logger.warning(f"Vector store not found at: {settings.vectorstore_dir}")
        return None

    try:
        embeddings = _get_embeddings()
        vs = FAISS.load_local(
            str(settings.vectorstore_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Vector store loaded — {vs.index.ntotal} chunks indexed")
        return vs
    except Exception as exc:
        logger.error(f"Failed to load vector store: {exc}")
        return None


def is_ready() -> bool:
    """Return True if the vector store is loaded and ready."""
    return _load_vectorstore() is not None


def chunk_count() -> int:
    """Return total number of indexed chunks (0 if not loaded)."""
    vs = _load_vectorstore()
    return vs.index.ntotal if vs else 0


def retrieve(query: str, top_k: int) -> list[SourceChunk]:
    """
    Retrieve the top_k most relevant chunks for a query.

    Returns a list of SourceChunk objects ready for the API response.
    """
    vs = _load_vectorstore()
    if vs is None:
        logger.error("retrieve() called but vector store is not loaded")
        return []

    results = vs.similarity_search_with_score(query, k=top_k)

    chunks = []
    for doc, score in results:
        chunks.append(
            SourceChunk(
                document=doc.metadata.get("source_file", "unknown"),
                page=doc.metadata.get("page"),
                excerpt=doc.page_content,
            )
        )
        logger.debug(f"  chunk from {doc.metadata.get('source_file')} score={score:.4f}")

    logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:60]!r}")
    return chunks
