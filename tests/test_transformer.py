"""Tests for the HuggingFace TransformerModel wrapper."""

import os

import pytest

from src.transformer_model import HAS_TRANSFORMERS, TransformerModel

# SmolLM2-135M is ~300MB download. Gate live-model tests behind an env var
# so CI stays offline-safe; the mock-path tests cover the module shape.
RUN_HF_TESTS = os.getenv("RUN_HF_TESTS", "").lower() in ("1", "true", "yes")
hf_only = pytest.mark.skipif(
    not (RUN_HF_TESTS and HAS_TRANSFORMERS),
    reason="set RUN_HF_TESTS=1 and install transformers to exercise SmolLM2",
)


class TestTransformerContractWithoutModel:
    """Shape tests that run regardless of whether transformers is installed."""

    def test_predict_next_unfitted_returns_unk(self):
        m = TransformerModel.__new__(TransformerModel)  # skip __init__ network call
        m._is_fitted = False
        m.model = None
        m.tokenizer = None
        assert m.predict_next(["the"], top_k=3) == [("<UNK>", 0.0)]

    def test_perplexity_unfitted_is_inf(self):
        m = TransformerModel.__new__(TransformerModel)
        m._is_fitted = False
        m.model = None
        m.tokenizer = None
        assert m.perplexity("hello world") == float("inf")


@hf_only
class TestSmolLM2LiveContract:
    """Real SmolLM2-135M round trips. Gated behind RUN_HF_TESTS=1."""

    @pytest.fixture(scope="class")
    def model(self):
        from src.data_loader import tokenize, load_sample_data
        m = TransformerModel()
        m.fit(tokenize(load_sample_data()))
        return m

    def test_predict_next_returns_nonempty_words(self, model):
        preds = model.predict_next(["machine", "learning", "is", "a"], top_k=5)
        assert len(preds) == 5
        assert all(isinstance(w, str) and len(w) > 0 for w, _ in preds)
        assert all(0.0 <= p <= 1.0 for _, p in preds)

    def test_predict_next_sorted_descending(self, model):
        preds = model.predict_next(["the", "cat", "sat"], top_k=5)
        probs = [p for _, p in preds]
        assert probs == sorted(probs, reverse=True)

    def test_perplexity_short_text(self, model):
        ppl = model.perplexity("Machine learning is a subset of artificial intelligence.")
        # In-domain continuations of the sample corpus should be <30 even
        # for a 135M-param base model.
        assert 1.0 < ppl < 50.0

    def test_vocab_size_is_tokenizer_vocab(self, model):
        assert model.vocab_size > 10_000  # SmolLM2 tokenizer is ~49k
