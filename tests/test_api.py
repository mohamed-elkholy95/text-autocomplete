"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import HAS_PROMETHEUS, HAS_REDIS, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_buckets():
    """Clear the shared in-memory rate-limit bucket between every API test.

    Neural-model calls now cost 4–6 tokens each (see `_model_cost`), so
    a suite of ~40 tests can easily drain the 30-token bucket for the
    TestClient's localhost IP. Without this fixture, tests run in
    isolation but fail collectively with a 429 — order-dependent
    behaviour that's worse than the fix itself.
    """
    from src.api.main import _rate_buckets
    _rate_buckets.clear()
    yield
    _rate_buckets.clear()


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

    def test_list_models_lstm_presence_tracks_torch(self):
        """Same optional-torch anchor for the LSTM entry."""
        from src.neural_model import HAS_TORCH
        resp = client.get("/models")
        data = resp.json()
        model_ids = [m["id"] for m in data["models"]]
        if HAS_TORCH:
            assert "lstm" in model_ids
        else:
            assert "lstm" not in model_ids


class TestTransformerEndpoint:
    """Tests for the transformer path in /autocomplete."""

    def test_transformer_returns_503_without_torch(self, monkeypatch):
        """Selecting 'transformer' on a no-torch install must fail with
        a clear 503 instead of a generic 500. Simulated by pointing the
        helper at a fake HAS_TORCH=False."""
        from src.api import main as api_main
        monkeypatch.setattr(api_main, "_transformer_available", lambda: False)
        # Also evict any cached transformer so the helper has to check.
        for k in list(api_main._model_cache):
            if k.startswith("transformer"):
                api_main._model_cache.pop(k, None)
        resp = client.post("/autocomplete", json={
            "text": "machine learning is",
            "top_k": 3,
            "model": "transformer",
        })
        assert resp.status_code == 503
        assert "PyTorch" in resp.json().get("detail", "")


class TestLSTMEndpoint:
    """Tests for the LSTM path in /autocomplete."""

    def test_lstm_returns_503_without_torch(self, monkeypatch):
        """Selecting 'lstm' on a no-torch install must fail with a clear
        503 — same optional-torch anchor the transformer enforces."""
        from src.api import main as api_main
        monkeypatch.setattr(api_main, "_lstm_available", lambda: False)
        for k in list(api_main._model_cache):
            if k.startswith("lstm"):
                api_main._model_cache.pop(k, None)
        resp = client.post("/autocomplete", json={
            "text": "machine learning is",
            "top_k": 3,
            "model": "lstm",
        })
        assert resp.status_code == 503
        assert "PyTorch" in resp.json().get("detail", "")


class TestCheckpointLoader:
    """Tests for the env-var checkpoint loader.

    We don't exercise a real .safetensors bundle here — the loader's
    model-load path already has its own unit tests in test_neural.py
    / test_transformer.py. The API-layer concern is: env var missing =
    fall through to lazy-fit; env var set but file missing = log and
    fall through; env var set + load succeeds = skip training."""

    def test_env_var_unset_falls_through_to_lazy_fit(self, monkeypatch):
        from src.api import main as api_main
        for env in api_main._CHECKPOINT_ENV_VARS.values():
            monkeypatch.delenv(env, raising=False)
        assert api_main._try_load_checkpoint("lstm", "word") is None
        assert api_main._try_load_checkpoint("transformer", "bpe") is None

    def test_env_var_pointing_at_missing_file_falls_through(
        self, monkeypatch, tmp_path,
    ):
        """A bogus path must not crash the endpoint — we log and
        lazy-fit so the service keeps serving."""
        from src.api import main as api_main
        from src.neural_model import HAS_TORCH
        if not HAS_TORCH:
            return  # checkpoint path is skipped before the env var is read
        monkeypatch.setenv(
            "AUTOCOMPLETE_LSTM_CHECKPOINT_WORD",
            str(tmp_path / "does_not_exist"),
        )
        assert api_main._try_load_checkpoint("lstm", "word") is None

    def test_unknown_model_kind_returns_none(self):
        from src.api import main as api_main
        assert api_main._try_load_checkpoint("ngram", "word") is None


class TestTokenizerSelector:
    """Tests for the BPE tokenizer flag in /autocomplete and batch."""

    def test_bpe_with_ngram_rejected(self):
        """BPE only makes sense for neural models — ngram+bpe must 422."""
        resp = client.post("/autocomplete", json={
            "text": "machine learning",
            "model": "ngram",
            "tokenizer": "bpe",
        })
        assert resp.status_code == 422

    def test_bpe_with_markov_rejected(self):
        resp = client.post("/autocomplete", json={
            "text": "machine learning",
            "model": "markov",
            "tokenizer": "bpe",
        })
        assert resp.status_code == 422

    def test_bpe_batch_with_ngram_rejected(self):
        resp = client.post("/autocomplete/batch", json={
            "texts": ["hello"],
            "model": "ngram",
            "tokenizer": "bpe",
        })
        assert resp.status_code == 422

    def test_default_tokenizer_is_word(self):
        """Omitting the field must keep the pre-PR behaviour — the
        cache key lands under 'lstm:word' / 'transformer:word'."""
        from src.api import main as api_main
        from src.neural_model import HAS_TORCH
        if not HAS_TORCH:
            return  # covered by the 503 tests above
        # Drain any prior state so we observe the default key.
        for k in list(api_main._model_cache):
            if k.startswith("lstm"):
                api_main._model_cache.pop(k, None)
        resp = client.post("/autocomplete", json={
            "text": "machine learning",
            "model": "lstm",
        })
        assert resp.status_code == 200
        assert "lstm:word" in api_main._model_cache

    def test_tokenizer_name_without_bpe_rejected(self):
        """tokenizer_name is BPE-only — pairing it with tokenizer='word'
        (or omitting the tokenizer field) must 422."""
        resp = client.post("/autocomplete", json={
            "text": "hi",
            "model": "lstm",
            "tokenizer_name": "gpt2",
        })
        assert resp.status_code == 422

    def test_bogus_tokenizer_name_returns_503(self, monkeypatch):
        """An HF repo that doesn't resolve must 503, not crash. We
        short-circuit the fetch by stubbing BPETokenizer to raise."""
        from src.api import main as api_main
        if not api_main._bpe_available():
            return  # covered by the transformers-missing 503 test
        from src.bpe_tokenizer import BPETokenizer as _Real

        class BadTok(_Real):  # type: ignore[misc]
            def __init__(self, name: str = "x") -> None:
                raise RuntimeError(f"simulated HF resolve failure for {name}")

        monkeypatch.setattr("src.bpe_tokenizer.BPETokenizer", BadTok)
        for k in list(api_main._model_cache):
            if k.startswith(("lstm:", "transformer:")):
                api_main._model_cache.pop(k, None)
        resp = client.post("/autocomplete", json={
            "text": "hi",
            "model": "lstm",
            "tokenizer": "bpe",
            "tokenizer_name": "this-repo-does-not-exist",
        })
        assert resp.status_code == 503
        assert "Failed to load BPE tokenizer" in resp.json().get("detail", "")

    def test_bpe_returns_503_without_transformers(self, monkeypatch):
        """tokenizer='bpe' on an install without 'transformers' must 503
        with a clear message instead of a generic ImportError."""
        from src.api import main as api_main
        monkeypatch.setattr(api_main, "_bpe_available", lambda: False)
        for k in list(api_main._model_cache):
            if "bpe" in k:
                api_main._model_cache.pop(k, None)
        resp = client.post("/autocomplete", json={
            "text": "machine learning",
            "model": "lstm",
            "tokenizer": "bpe",
        })
        # Needs torch to even reach the bpe-available check; on a torch
        # install, 503 fires from _maybe_build_bpe_tokenizer. On a
        # no-torch install, the earlier lstm 503 fires instead — either
        # way, the client sees a 503 with a helpful message.
        assert resp.status_code == 503
        detail = resp.json().get("detail", "")
        assert ("BPE" in detail) or ("PyTorch" in detail)


class TestVocabStats:
    """Tests for the vocabulary statistics endpoint."""

    def setup_method(self):
        # The growing test suite can saturate the shared testclient
        # rate bucket before the last class runs. Reset it so these
        # read-only tests aren't order-dependent on bucket state.
        from src.api.main import _rate_buckets
        _rate_buckets.clear()

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


class TestPrometheus:
    """Prometheus exposition endpoint /metrics/prom.

    Only runs when prometheus-fastapi-instrumentator is installed; the
    package is optional so minimal installs skip cleanly.
    """

    def setup_method(self):
        from src.api.main import _rate_buckets
        _rate_buckets.clear()

    @pytest.mark.skipif(not HAS_PROMETHEUS, reason="prometheus instrumentator not installed")
    def test_prom_endpoint_returns_text_format(self):
        # Generate one request so there's something to count.
        client.post("/autocomplete", json={"text": "machine learning", "top_k": 3})
        resp = client.get("/metrics/prom")
        assert resp.status_code == 200
        body = resp.text
        assert body.startswith("# HELP") or body.startswith("# TYPE")
        assert "http_requests_total" in body or "http_request_duration" in body

    @pytest.mark.skipif(HAS_PROMETHEUS, reason="prometheus instrumentator is installed")
    def test_prom_endpoint_absent_without_dep(self):
        resp = client.get("/metrics/prom")
        assert resp.status_code == 404


class TestModelAliases:
    """Catalogue-level aliases: `lstm-bpe` / `transformer-bpe` in the `model`
    field should be normalised to `model=lstm` + `tokenizer=bpe` on the
    server side, without requiring the client to set the tokenizer."""

    def setup_method(self):
        from src.api.main import _rate_buckets
        _rate_buckets.clear()

    def test_alias_in_models_catalogue_when_bpe_available(self):
        from src.api.main import _bpe_available
        resp = client.get("/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["models"]]
        if _bpe_available():
            assert "lstm-bpe" in ids
            assert "transformer-bpe" in ids
        else:
            assert "lstm-bpe" not in ids
            assert "transformer-bpe" not in ids

    def test_alias_rejected_without_bpe_tokenizer_flag(self):
        """The request model does the normalisation; verify a regular
        request still rejects `tokenizer=bpe` paired with `model=ngram`."""
        resp = client.post("/autocomplete", json={
            "text": "hello world",
            "model": "ngram",
            "tokenizer": "bpe",
        })
        assert resp.status_code == 422  # model_validator complaint


class TestNeuralGenerate:
    """/generate supports `model=markov|lstm|transformer`."""

    def setup_method(self):
        from src.api.main import _rate_buckets
        _rate_buckets.clear()

    def test_markov_generate_default(self):
        resp = client.post("/generate", json={
            "start_word": "the", "max_length": 8, "temperature": 1.0,
        })
        assert resp.status_code == 200
        assert resp.json()["model"] == "markov"
        assert resp.json()["word_count"] >= 1

    def test_generate_model_validation(self):
        resp = client.post("/generate", json={"model": "nope"})
        assert resp.status_code == 422


class TestAttention:
    """Transformer attention-visualisation endpoint. Skipped without torch."""

    def setup_method(self):
        from src.api.main import _rate_buckets
        _rate_buckets.clear()

    def test_attention_requires_torch(self):
        from src.api.main import _transformer_available
        resp = client.post(
            "/attention",
            json={"text": "machine learning is", "tokenizer": "word"},
        )
        if not _transformer_available():
            assert resp.status_code == 503
            return
        assert resp.status_code == 200
        data = resp.json()
        # Structure contract: shape is [layers, heads, seq, seq].
        assert isinstance(data["tokens"], list) and len(data["tokens"]) > 0
        assert data["n_layers"] >= 1
        assert data["n_heads"] >= 1
        assert data["seq_len"] == len(data["tokens"])
        mat = data["attentions"][0][0]
        assert len(mat) == data["seq_len"]
        assert all(len(row) == data["seq_len"] for row in mat)
        # Causal mask: first row should attend only to itself.
        first_row = mat[0]
        assert first_row[0] == pytest.approx(1.0, abs=1e-4)
        assert all(v == pytest.approx(0.0, abs=1e-4) for v in first_row[1:])


class TestAPIKeyAuth:
    """Opt-in AUTOCOMPLETE_API_KEY gate on non-public endpoints."""

    def setup_method(self):
        from src.api.main import _rate_buckets
        _rate_buckets.clear()

    def test_no_key_set_allows_all(self):
        # Default path: env var unset → everything works without a key.
        import os
        assert not os.getenv("AUTOCOMPLETE_API_KEY")
        assert client.get("/models").status_code == 200

    def test_protected_path_requires_header(self, monkeypatch):
        monkeypatch.setenv("AUTOCOMPLETE_API_KEY", "unit-test-key")
        # /health is public and stays open.
        assert client.get("/health").status_code == 200
        # /models is protected.
        assert client.get("/models").status_code == 401
        # Wrong key still rejected.
        assert client.get(
            "/models", headers={"X-API-Key": "wrong"},
        ).status_code == 401
        # Correct key gets through.
        assert client.get(
            "/models", headers={"X-API-Key": "unit-test-key"},
        ).status_code == 200


class TestRedisRateLimit:
    """Redis-backed rate limiter (opt-in via REDIS_URL).

    We exercise the helper directly against fakeredis so there's no real
    Redis dependency. The test is skipped when neither redis nor
    fakeredis is available — which is the state on a minimal install.
    """

    def setup_method(self):
        from src.api.main import _rate_buckets
        _rate_buckets.clear()

    @pytest.mark.skipif(not HAS_REDIS, reason="redis package not installed")
    def test_redis_branch_shares_counter(self):
        fakeredis = pytest.importorskip("fakeredis")
        import asyncio
        from src.api import main as api_main

        api_main._redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
        try:
            # Fixed-window budget is 30; 35 calls in one window should
            # see exactly 30 allowed and 5 denied regardless of which
            # "worker" ran them — that's the whole point of Redis.
            async def fire():
                allowed = denied = 0
                for _ in range(35):
                    ok = await api_main._check_rate_limit("1.2.3.4")
                    allowed += int(ok)
                    denied += int(not ok)
                return allowed, denied
            allowed, denied = asyncio.run(fire())
            assert allowed == 30
            assert denied == 5
        finally:
            api_main._redis_client = None
