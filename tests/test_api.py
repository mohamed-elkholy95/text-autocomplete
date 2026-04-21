"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestHealth:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_json(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_returns_version(self):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data


class TestAutocomplete:
    """Tests for the single autocomplete endpoint."""

    def test_autocomplete_ngram(self):
        resp = client.post("/autocomplete", json={
            "text": "machine learning",
            "top_k": 5,
            "model": "ngram",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suggestions"]) > 0
        assert "context" in data
        assert data["model"] == "ngram"

    def test_autocomplete_markov(self):
        resp = client.post("/autocomplete", json={
            "text": "machine learning",
            "top_k": 5,
            "model": "markov",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suggestions"]) > 0
        assert data["model"] == "markov"

    def test_autocomplete_default_model(self):
        """Default model should be ngram."""
        resp = client.post("/autocomplete", json={
            "text": "deep learning",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "ngram"

    def test_autocomplete_custom_top_k(self):
        resp = client.post("/autocomplete", json={
            "text": "the attention",
            "top_k": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suggestions"]) <= 2

    def test_autocomplete_suggestion_format(self):
        resp = client.post("/autocomplete", json={
            "text": "machine learning is",
            "top_k": 3,
        })
        data = resp.json()
        for sug in data["suggestions"]:
            assert "word" in sug
            assert "probability" in sug
            assert 0.0 <= sug["probability"] <= 1.0

    def test_autocomplete_empty_text(self):
        """Empty text should return 400."""
        resp = client.post("/autocomplete", json={
            "text": "",
            "top_k": 5,
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_autocomplete_invalid_model(self):
        """Invalid model name should return 422."""
        resp = client.post("/autocomplete", json={
            "text": "machine learning",
            "model": "invalid_model",
        })
        assert resp.status_code == 422


class TestBatchAutocomplete:
    """Tests for the batch autocomplete endpoint."""

    def test_batch_returns_results(self):
        resp = client.post("/autocomplete/batch", json={
            "texts": ["machine learning", "deep learning", "the attention"],
            "top_k": 3,
            "model": "ngram",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3

    def test_batch_single_text(self):
        resp = client.post("/autocomplete/batch", json={
            "texts": ["hello world"],
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1

    def test_batch_empty_texts(self):
        resp = client.post("/autocomplete/batch", json={
            "texts": [],
            "top_k": 5,
        })
        assert resp.status_code == 422  # min_length=1

    def test_batch_markov_model(self):
        resp = client.post("/autocomplete/batch", json={
            "texts": ["the", "machine"],
            "top_k": 3,
            "model": "markov",
        })
        assert resp.status_code == 200

    def test_batch_rejects_oversized_text(self):
        """Each batch entry must respect the 1000-char cap."""
        resp = client.post("/autocomplete/batch", json={
            "texts": ["a " * 600],  # 1200 characters
            "top_k": 1,
        })
        assert resp.status_code == 422


class TestRateLimiterIsolation:
    """Rate-limit bucket behaviour should not depend on forged headers."""

    def test_xff_header_ignored_by_default(self):
        """Spoofed X-Forwarded-For must not mint fresh buckets in default mode."""
        from src.api.main import _rate_buckets
        before = set(_rate_buckets.keys())
        for i in range(5):
            r = client.post(
                "/autocomplete",
                json={"text": "machine learning", "top_k": 1},
                headers={"x-forwarded-for": f"203.0.113.{i}"},
            )
            assert r.status_code == 200
        after = set(_rate_buckets.keys())
        # At most one new bucket should appear — the real peer IP.
        assert len(after - before) <= 1


class TestListModels:
    """Tests for the model listing endpoint."""

    def test_list_models_returns_200(self):
        resp = client.get("/models")
        assert resp.status_code == 200

    def test_list_models_structure(self):
        resp = client.get("/models")
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) >= 2

    def test_list_models_have_ids(self):
        resp = client.get("/models")
        data = resp.json()
        model_ids = [m["id"] for m in data["models"]]
        assert "ngram" in model_ids
        assert "markov" in model_ids

    def test_list_models_transformer_presence_tracks_torch(self):
        """The transformer entry must appear iff torch is importable —
        the API's optional-torch anchor says minimal installs see only
        the statistical models."""
        from src.neural_model import HAS_TORCH
        resp = client.get("/models")
        data = resp.json()
        model_ids = [m["id"] for m in data["models"]]
        if HAS_TORCH:
            assert "transformer" in model_ids
        else:
            assert "transformer" not in model_ids


class TestTransformerEndpoint:
    """Tests for the transformer path in /autocomplete."""

    def test_transformer_returns_503_without_torch(self, monkeypatch):
        """Selecting 'transformer' on a no-torch install must fail with
        a clear 503 instead of a generic 500. Simulated by pointing the
        helper at a fake HAS_TORCH=False."""
        from src.api import main as api_main
        monkeypatch.setattr(api_main, "_transformer_available", lambda: False)
        # Also evict any cached transformer so the helper has to check.
        api_main._model_cache.pop("transformer", None)
        resp = client.post("/autocomplete", json={
            "text": "machine learning is",
            "top_k": 3,
            "model": "transformer",
        })
        assert resp.status_code == 503
        assert "PyTorch" in resp.json().get("detail", "")


class TestVocabStats:
    """Tests for the vocabulary statistics endpoint."""

    def test_vocab_stats_returns_200(self):
        resp = client.get("/vocab/stats")
        assert resp.status_code == 200

    def test_vocab_stats_structure(self):
        resp = client.get("/vocab/stats")
        data = resp.json()
        assert "corpus" in data
        assert "ngram_vocab_size" in data
        assert "markov_vocab_size" in data
        assert data["corpus"]["total_tokens"] > 0
