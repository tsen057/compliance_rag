"""
app/core/ingestion.py
──────────────────────
Document ingestion pipeline.

Steps:
  1. Load all PDFs from data/docs/ using PyMuPDF (fast) with PyPDF fallback
  2. Split pages into overlapping chunks
  3. Generate HuggingFace embeddings (local, free)
  4. Build and persist a FAISS index to data/vectorstore/

Run directly:
    python -m app.core.ingestion
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from app.core.config import get_settings

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)


# ── Document loading ──────────────────────────────────────────────────────────

def _load_with_pymupdf(pdf_path: Path) -> list[Document]:
    """Fast PDF loading via PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    docs = []
    with fitz.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source_file": pdf_path.name, "page": page_num},
                    )
                )
    return docs


def _load_with_pypdf(pdf_path: Path) -> list[Document]:
    """Fallback PDF loading via pypdf."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        page.metadata["source_file"] = pdf_path.name
    return pages


def load_documents(docs_dir: Path) -> list[Document]:
    """Load all PDFs from docs_dir, trying PyMuPDF first."""
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {docs_dir.resolve()}")
        return []

    all_docs: list[Document] = []
    for pdf_path in pdf_files:
        logger.info(f"Loading: {pdf_path.name}")
        try:
            pages = _load_with_pymupdf(pdf_path)
        except Exception:
            logger.warning(f"PyMuPDF failed for {pdf_path.name}, falling back to pypdf")
            pages = _load_with_pypdf(pdf_path)

        all_docs.extend(pages)
        logger.info(f"  → {len(pages)} pages loaded")

    logger.info(f"Total pages loaded: {len(all_docs)}")
    return all_docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split page documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ── Embedding + FAISS ─────────────────────────────────────────────────────────

def build_and_save_vectorstore(
    chunks: list[Document],
    embedding_model: str,
    save_dir: Path,
) -> FAISS:
    """Embed chunks and persist a FAISS index."""
    logger.info(f"Loading embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info(f"Building FAISS index from {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    save_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_dir))
    logger.info(f"FAISS index saved to: {save_dir.resolve()}")
    return vectorstore


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_ingestion() -> dict:
    """Full ingestion pipeline. Returns a summary dict."""
    settings = get_settings()

    documents = load_documents(settings.docs_dir)
    if not documents:
        return {"documents_processed": 0, "chunks_indexed": 0}

    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    build_and_save_vectorstore(chunks, settings.embedding_model, settings.vectorstore_dir)

    unique_docs = len({d.metadata.get("source_file") for d in documents})
    result = {"documents_processed": unique_docs, "chunks_indexed": len(chunks)}
    logger.info(f"Ingestion complete: {result}")
    return result


if __name__ == "__main__":
    run_ingestion()
