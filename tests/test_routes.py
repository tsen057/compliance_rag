"""Integration tests for FastAPI routes using TestClient."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.schemas import QueryType, SourceChunk
from main import app

client = TestClient(app)


class TestHealthRoute:
    def test_health_ok_when_ready(self):
        with (
            patch("app.api.routes.is_ready", return_value=True),
            patch("app.api.routes.chunk_count", return_value=150),
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["vectorstore_ready"] is True
        assert data["chunks_indexed"] == 150

    def test_health_when_vectorstore_missing(self):
        with patch("app.api.routes.is_ready", return_value=False):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["vectorstore_ready"] is False
        assert resp.json()["chunks_indexed"] == 0


class TestQueryRoute:
    def test_returns_503_when_not_ready(self):
        with patch("app.api.routes.is_ready", return_value=False):
            resp = client.post("/query", json={"question": "What is Basel III?"})
        assert resp.status_code == 503

    def test_returns_answer_with_sources(self):
        mock_result = {
            "answer": "The minimum CET1 ratio is 4.5%.",
            "sources": [
                SourceChunk(
                    document="Basel_III_Framework.pdf",
                    page=12,
                    excerpt="CET1 capital ratio of 4.5% of risk-weighted assets.",
                )
            ],
            "query_type": "simple",
        }
        with (
            patch("app.api.routes.is_ready", return_value=True),
            patch("app.api.routes._agent.query", return_value=mock_result),
        ):
            resp = client.post("/query", json={"question": "What is the CET1 ratio?"})

        assert resp.status_code == 200
        data = resp.json()
        assert "4.5" in data["answer"]
        assert data["query_type"] == "simple"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document"] == "Basel_III_Framework.pdf"

    def test_rejects_blank_question(self):
        resp = client.post("/query", json={"question": "   "})
        assert resp.status_code == 422

    def test_rejects_question_too_short(self):
        resp = client.post("/query", json={"question": "hi"})
        assert resp.status_code == 422

    def test_top_k_out_of_range_rejected(self):
        resp = client.post("/query", json={"question": "What is Basel III?", "top_k": 99})
        assert resp.status_code == 422


class TestRootRoute:
    def test_root_returns_service_info(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Compliance" in resp.json()["service"]
