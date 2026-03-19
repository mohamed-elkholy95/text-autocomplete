"""Tests for ngram model."""
import pytest
from src.data_loader import tokenize, load_sample_data
from src.ngram_model import NGramModel


@pytest.fixture
def trained_model():
    tokens = tokenize(load_sample_data())
    return NGramModel(n=3).fit(tokens)


class TestNGramModel:
    def test_fit(self, trained_model):
        assert trained_model.vocab_size > 0
        assert trained_model._is_fitted

    def test_predict_next(self, trained_model):
        preds = trained_model.predict_next(["machine", "learning"])
        assert isinstance(preds, list)
        assert len(preds) > 0
        assert isinstance(preds[0], tuple)

    def test_predict_top_k(self, trained_model):
        preds = trained_model.predict_next(["machine"], top_k=3)
        assert len(preds) <= 3

    def test_perplexity(self, trained_model):
        test_tokens = tokenize("machine learning is great")
        ppl = trained_model.perplexity(test_tokens)
        assert ppl > 0

    def test_empty_context(self, trained_model):
        preds = trained_model.predict_next([])
        assert len(preds) > 0

    def test_bigram(self):
        tokens = ["a", "b", "c", "a", "b", "d"] * 10
        model = NGramModel(n=2).fit(tokens)
        preds = model.predict_next(["a"])
        assert len(preds) > 0
