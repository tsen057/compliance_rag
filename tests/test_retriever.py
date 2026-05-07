"""Tests for the FAISS retriever wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.core.retriever import chunk_count, is_ready, retrieve


class TestIsReady:
    def test_returns_false_when_vectorstore_missing(self):
        with patch("app.core.retriever._load_vectorstore", return_value=None):
            assert is_ready() is False

    def test_returns_true_when_vectorstore_loaded(self):
        mock_vs = MagicMock()
        with patch("app.core.retriever._load_vectorstore", return_value=mock_vs):
            assert is_ready() is True


class TestRetrieve:
    def _make_mock_vs(self, content: str, source: str, page: int, score: float):
        mock_doc = MagicMock()
        mock_doc.page_content = content
        mock_doc.metadata = {"source_file": source, "page": page}
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [(mock_doc, score)]
        return mock_vs

    def test_returns_empty_when_vectorstore_not_loaded(self):
        with patch("app.core.retriever._load_vectorstore", return_value=None):
            result = retrieve("What is Basel III?", top_k=5)
        assert result == []

    def test_returns_source_chunk_objects(self):
        mock_vs = self._make_mock_vs(
            content="CET1 ratio must be 4.5%.",
            source="Basel_III_Framework.pdf",
            page=12,
            score=0.87,
        )
        with patch("app.core.retriever._load_vectorstore", return_value=mock_vs):
            chunks = retrieve("CET1 ratio", top_k=1)

        assert len(chunks) == 1
        assert chunks[0].document == "Basel_III_Framework.pdf"
        assert chunks[0].page == 12
        assert "CET1" in chunks[0].excerpt

    def test_top_k_passed_to_similarity_search(self):
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = []
        with patch("app.core.retriever._load_vectorstore", return_value=mock_vs):
            retrieve("leverage ratio", top_k=7)
        mock_vs.similarity_search_with_score.assert_called_once_with("leverage ratio", k=7)
