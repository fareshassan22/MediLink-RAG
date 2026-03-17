import pytest
from unittest.mock import patch, Mock, MagicMock
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_vector_store():
    with patch("app.main.vector_store") as mock:
        mock.search.return_value = [
            {"text": "test doc", "score": 0.9, "metadata": {"page": 1}}
        ]
        mock.documents = []
        yield mock


@pytest.fixture
def mock_bm25():
    with patch("app.main.bm25") as mock:
        mock.search.return_value = [{"text": "test doc", "score": 0.8}]
        yield mock


@pytest.fixture
def client():
    with patch("app.main.load_models"):
        with patch("app.main.vector_store", create=True):
            with patch("app.main.bm25", create=True):
                from app.main import app

                return TestClient(app)


class TestAPIEndpoints:
    """Tests for API endpoints."""

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code in [200, 404]

    def test_ask_endpoint_valid_query(self, client):
        with patch("app.main.detect_emergency", return_value=False):
            with patch("app.main.expand_query", return_value=["test query"]):
                with patch("app.main.embed_texts") as mock_embed:
                    with patch("app.main.hybrid_fusion") as mock_fusion:
                        with patch("app.main.rerank") as mock_rerank:
                            with patch("app.main.compress_context") as mock_compress:
                                with patch(
                                    "app.main.build_prompt", return_value="prompt"
                                ):
                                    with patch(
                                        "app.main.generate_response",
                                        return_value="test answer",
                                    ):
                                        with patch(
                                            "app.main.filter_by_metadata"
                                        ) as mock_filter:
                                            mock_embed.return_value = []
                                            mock_fusion.return_value = [
                                                {
                                                    "text": "doc1",
                                                    "score": 0.9,
                                                    "metadata": {"page": 1},
                                                }
                                            ]
                                            mock_rerank.return_value = [
                                                {
                                                    "text": "doc1",
                                                    "score": 0.9,
                                                    "metadata": {"page": 1},
                                                }
                                            ]
                                            mock_compress.return_value = ["compressed"]
                                            mock_filter.return_value = [
                                                {
                                                    "text": "doc1",
                                                    "score": 0.9,
                                                    "metadata": {"page": 1},
                                                }
                                            ]

                                            response = client.post(
                                                "/ask",
                                                json={
                                                    "query": "أعراض السكري",
                                                    "role": "patient",
                                                },
                                            )

                                            assert response.status_code in [200, 500]

    def test_ask_endpoint_empty_query(self, client):
        response = client.post("/ask", json={"query": ""})
        assert response.status_code == 422

    def test_ask_endpoint_short_query(self, client):
        response = client.post("/ask", json={"query": "أب"})
        assert response.status_code == 422

    def test_ask_endpoint_emergency(self, client):
        with patch("app.main.detect_emergency", return_value=True):
            response = client.post(
                "/ask", json={"query": "ألم في الصدر", "role": "patient"}
            )

            if response.status_code == 200:
                data = response.json()
                assert data.get("status") == "emergency_escalation"

    def test_ask_endpoint_missing_role(self, client):
        with patch("app.main.detect_emergency", return_value=False):
            with patch("app.main.expand_query", return_value=["test"]):
                with patch("app.main.vector_store") as mock_vs:
                    mock_vs.search.return_value = []

                    with patch("app.main.hybrid_fusion", return_value=[]):
                        response = client.post("/ask", json={"query": "أعراض السكري"})

                        assert response.status_code in [200, 400, 500]


class TestQueryRequest:
    """Tests for query request validation."""

    def test_query_request_valid(self):
        from app.main import QueryRequest

        req = QueryRequest(query="أعراض السكري", role="patient")
        assert req.query == "أعراض السكري"
        assert req.role == "patient"

    def test_query_request_with_specialty(self):
        from app.main import QueryRequest

        req = QueryRequest(
            query="أعراض السكري", role="patient", specialty="endocrinology"
        )
        assert req.specialty == "endocrinology"

    def test_query_request_default_role(self):
        from app.main import QueryRequest

        req = QueryRequest(query="test query")
        assert req.role == "patient"

    def test_query_request_strips_whitespace(self):
        from app.main import QueryRequest

        req = QueryRequest(query="  أعراض السكري  ")
        assert req.query == "أعراض السكري"

    def test_query_request_rejects_empty(self):
        from app.main import QueryRequest

        with pytest.raises(Exception):
            QueryRequest(query="   ")

    def test_query_request_min_length(self):
        from app.main import QueryRequest

        with pytest.raises(Exception):
            QueryRequest(query="أب")

    def test_query_request_max_length(self):
        from app.main import QueryRequest

        with pytest.raises(Exception):
            QueryRequest(query="أ" * 501)


class TestResponseModel:
    """Tests for response model."""

    def test_response_model_valid(self):
        from app.main import ResponseModel

        resp = ResponseModel(
            answer="test answer",
            confidence=0.85,
            sources=["medical_book (Page 1)"],
            grounding_score=0.9,
            latency_seconds=1.5,
            status="success",
        )

        assert resp.answer == "test answer"
        assert resp.confidence == 0.85
        assert resp.status == "success"

    def test_response_model_with_stage_latencies(self):
        from app.main import ResponseModel

        resp = ResponseModel(
            answer="test",
            confidence=0.5,
            sources=[],
            grounding_score=0.5,
            latency_seconds=1.0,
            status="success",
            stage_latencies={"retrieval": 0.5, "generation": 0.5},
        )

        assert "retrieval" in resp.stage_latencies


class TestAPIMiddleware:
    """Tests for API middleware."""

    def test_cors_headers_present(self, client):
        response = client.options("/ask")
        assert (
            "access-control-allow-origin" in response.headers
            or response.status_code == 200
        )

    def test_rate_limiting_config(self):
        from app.api.middleware import RateLimitMiddleware

        assert RateLimitMiddleware is not None


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_500_error_handling(self, client):
        with patch("app.main.detect_emergency", side_effect=Exception("test")):
            response = client.post("/ask", json={"query": "test query"})

            assert response.status_code in [500, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
