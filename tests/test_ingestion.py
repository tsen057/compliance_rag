"""Tests for the document ingestion pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.core.ingestion import split_documents


def _make_doc(text: str, source: str = "test.pdf", page: int = 1) -> Document:
    return Document(page_content=text, metadata={"source_file": source, "page": page})


class TestSplitDocuments:
    def test_long_document_splits_into_multiple_chunks(self):
        doc = _make_doc("regulatory content " * 200)
        chunks = split_documents([doc], chunk_size=512, chunk_overlap=64)
        assert len(chunks) > 1

    def test_short_document_stays_as_single_chunk(self):
        doc = _make_doc("Basel III sets minimum capital requirements.")
        chunks = split_documents([doc], chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1

    def test_empty_input_returns_empty_list(self):
        assert split_documents([], chunk_size=512, chunk_overlap=64) == []

    def test_chunks_are_non_empty(self):
        doc = _make_doc("Capital adequacy. " * 100)
        chunks = split_documents([doc], chunk_size=256, chunk_overlap=32)
        for chunk in chunks:
            assert chunk.page_content.strip() != ""

    def test_metadata_propagated_to_chunks(self):
        doc = _make_doc("content " * 200, source="basel3.pdf", page=5)
        chunks = split_documents([doc], chunk_size=128, chunk_overlap=16)
        for chunk in chunks:
            assert chunk.metadata["source_file"] == "basel3.pdf"
