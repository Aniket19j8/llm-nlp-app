"""
Tests for the NLP Microservice API
"""

import pytest
from fastapi.testclient import TestClient

# We import the app but mock the lifespan models for fast testing
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create a test client with mocked NLP engine."""
    # Patch model loading for fast tests
    with patch("app.main.NLPEngine") as MockEngine, \
         patch("app.main.FallbackRouter") as MockRouter:
        
        engine = MockEngine.return_value
        engine.is_loaded = True
        engine.analyse_sentiment.return_value = {
            "label": "POSITIVE",
            "score": 0.9876,
            "confidence": "HIGH",
            "all_scores": {"POSITIVE": 0.9876, "NEGATIVE": 0.0124},
        }
        engine.extract_entities.return_value = [
            {"text": "Google", "label": "ORG", "score": 0.99, "start": 0, "end": 6}
        ]
        engine.extract_keywords.return_value = [
            {"keyword": "technology", "score": 0.15, "frequency": 3}
        ]

        router = MockRouter.return_value
        router.summarize = MagicMock(
            return_value={"text": "This is a summary.", "backend": "local-t5"}
        )
        router.rewrite = MagicMock(
            return_value={"text": "Rewritten text here.", "backend": "openai"}
        )

        from app.main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_has_metrics(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "uptime" in data
        assert "total_requests" in data


class TestSentimentEndpoint:
    def test_positive_sentiment(self, client):
        resp = client.post(
            "/api/v1/sentiment",
            json={"text": "I love this product, it's amazing!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "POSITIVE"
        assert data["score"] > 0.5

    def test_empty_text_returns_422(self, client):
        resp = client.post("/api/v1/sentiment", json={"text": ""})
        assert resp.status_code == 422


class TestSummarizeEndpoint:
    def test_summarize_returns_summary(self, client):
        long_text = "This is a long text. " * 50
        resp = client.post(
            "/api/v1/summarize",
            json={"text": long_text, "max_length": 100, "min_length": 20},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "backend" in data


class TestNEREndpoint:
    def test_ner_returns_entities(self, client):
        resp = client.post(
            "/api/v1/ner",
            json={"text": "Google was founded by Larry Page and Sergey Brin."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "entities" in data
        assert "count" in data


class TestKeywordsEndpoint:
    def test_keywords_extraction(self, client):
        resp = client.post(
            "/api/v1/keywords",
            json={"text": "Machine learning and artificial intelligence are transforming technology."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "keywords" in data


class TestRewriteEndpoint:
    def test_rewrite_formal(self, client):
        resp = client.post(
            "/api/v1/rewrite",
            json={"text": "hey whats up, this thing is pretty cool", "style": "formal"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "rewritten" in data
        assert data["style"] == "formal"


class TestCaching:
    def test_second_call_uses_cache(self, client):
        payload = {"text": "Test caching behavior with identical input."}
        resp1 = client.post("/api/v1/sentiment", json=payload)
        resp2 = client.post("/api/v1/sentiment", json=payload)
        assert resp1.json() == resp2.json()


class TestLatencyHeader:
    def test_timing_header_present(self, client):
        resp = client.get("/health")
        assert "x-process-time-ms" in resp.headers
