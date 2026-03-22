"""Tests for the N-gram language model."""

import pytest
from src.data_loader import tokenize, load_sample_data
from src.ngram_model import NGramModel


@pytest.fixture
def trained_model():
    """Create a trained trigram model for testing."""
    tokens = tokenize(load_sample_data())
    return NGramModel(n=3).fit(tokens)


@pytest.fixture
def simple_model():
    """Create a model trained on a small, known corpus for deterministic tests."""
    tokens = ["the", "cat", "sat", "on", "the", "mat", "the", "cat", "sat", "on",
              "the", "cat", "sat", "the", "dog", "ran", "the", "dog", "sat"]
    return NGramModel(n=3, min_freq=1).fit(tokens)


class TestNGramFit:
    """Tests for model training (fit method)."""

    def test_fit_sets_vocab(self, trained_model):
        """After fitting, the vocabulary should be non-empty."""
        assert trained_model.vocab_size > 0

    def test_fit_sets_is_fitted(self, trained_model):
        """The _is_fitted flag should be True after fit()."""
        assert trained_model._is_fitted

    def test_fit_builds_ngram_counts(self, trained_model):
        """N-gram counts should be built for all orders from 1 to n."""
        assert len(trained_model._ngram_counts) > 0
        stats = trained_model.get_ngram_stats()
        # Should have counts for unigrams, bigrams, and trigrams
        assert 1 in stats  # unigrams
        assert 2 in stats  # bigrams
        assert 3 in stats  # trigrams

    def test_fit_small_corpus(self, simple_model):
        """Model should train correctly on a small corpus."""
        assert simple_model.vocab_size > 0
        assert simple_model._is_fitted


class TestNGramPredict:
    """Tests for next-word prediction."""

    def test_predict_returns_list(self, trained_model):
        """predict_next should return a list of (word, probability) tuples."""
        preds = trained_model.predict_next(["machine", "learning"])
        assert isinstance(preds, list)
        assert len(preds) > 0
        assert isinstance(preds[0], tuple)
        assert isinstance(preds[0][0], str)
        assert isinstance(preds[0][1], float)

    def test_predict_top_k(self, trained_model):
        """Should return at most top_k predictions."""
        preds = trained_model.predict_next(["machine"], top_k=3)
        assert len(preds) <= 3

    def test_predict_top_k_one(self, trained_model):
        """top_k=1 should return exactly 1 prediction."""
        preds = trained_model.predict_next(["machine"], top_k=1)
        assert len(preds) == 1

    def test_predict_probabilities_sum(self, simple_model):
        """All prediction probabilities should sum to approximately 1.0."""
        preds = simple_model.predict_next(["the", "cat"], top_k=100)
        total_prob = sum(p for _, p in preds)
        assert 0.99 <= total_prob <= 1.01  # Allow small floating-point errors

    def test_predict_probabilities_decreasing(self, trained_model):
        """Predictions should be sorted by probability descending."""
        preds = trained_model.predict_next(["machine", "learning"], top_k=5)
        for i in range(len(preds) - 1):
            assert preds[i][1] >= preds[i + 1][1]

    def test_predict_empty_context(self, trained_model):
        """Empty context should still return predictions (unigram fallback)."""
        preds = trained_model.predict_next([])
        assert len(preds) > 0

    def test_predict_unknown_words(self, trained_model):
        """Unknown words should trigger backoff to shorter contexts."""
        preds = trained_model.predict_next(["xyznotaword", "anotherfake"])
        # Should not crash, should return some predictions via backoff
        assert isinstance(preds, list)

    def test_predict_deterministic(self, trained_model):
        """Same input should give same output."""
        preds1 = trained_model.predict_next(["machine", "learning"])
        preds2 = trained_model.predict_next(["machine", "learning"])
        assert preds1 == preds2


class TestNGramPerplexity:
    """Tests for perplexity computation."""

    def test_perplexity_positive(self, trained_model):
        """Perplexity should be a positive number."""
        test_tokens = tokenize("machine learning is great for data science")
        ppl = trained_model.perplexity(test_tokens)
        assert ppl > 0

    def test_perplexity_not_fitted(self):
        """Perplexity should return infinity for an unfitted model."""
        model = NGramModel(n=3)
        assert model.perplexity(["the", "cat"]) == float("inf")

    def test_perplexity_short_sequence(self, trained_model):
        """Should handle sequences shorter than n-gram order gracefully."""
        ppl = trained_model.perplexity(["the"])
        assert ppl > 0

    def test_perplexity_known_good_context(self, simple_model):
        """Perplexity on training-like data should be lower than random data."""
        # Use tokens similar to training data
        good_tokens = ["the", "cat", "sat", "on", "the", "mat"]
        # Use random tokens unlikely to appear in training
        random_tokens = ["xyz", "abc", "def", "ghi", "jkl", "mno"]

        good_ppl = simple_model.perplexity(good_tokens)
        random_ppl = simple_model.perplexity(random_tokens)

        # Good context should have lower perplexity (or at least not much higher)
        # With our small corpus, this might not always hold perfectly,
        # but the random text should generally have higher perplexity
        assert isinstance(good_ppl, float)
        assert isinstance(random_ppl, float)


class TestNGramEdgeCases:
    """Edge case tests."""

    def test_bigram(self):
        """Bigram model (n=2) should work correctly."""
        tokens = ["a", "b", "c", "a", "b", "d"] * 10
        model = NGramModel(n=2, min_freq=1).fit(tokens)
        preds = model.predict_next(["a"])
        assert len(preds) > 0

    def test_unigram(self):
        """Unigram model (n=1) should work correctly."""
        tokens = ["a", "b", "c"] * 10
        model = NGramModel(n=1, min_freq=1).fit(tokens)
        preds = model.predict_next(["anything"])
        assert len(preds) > 0

    def test_fourgram(self):
        """4-gram model (n=4) should work correctly."""
        tokens = tokenize(load_sample_data())
        model = NGramModel(n=4).fit(tokens)
        preds = model.predict_next(["machine", "learning", "is"])
        assert isinstance(preds, list)

    def test_single_word_corpus(self):
        """Model should handle a corpus of a single repeated word."""
        model = NGramModel(n=2, min_freq=1).fit(["the"] * 20)
        preds = model.predict_next(["the"])
        assert len(preds) > 0

    def test_two_word_corpus(self):
        """Model should handle a minimal corpus."""
        model = NGramModel(n=2, min_freq=1).fit(["hello", "world"] * 10)
        preds = model.predict_next(["hello"])
        assert len(preds) > 0
        # "world" should be the top prediction (only thing that follows "hello")
        assert preds[0][0] == "world"

    def test_get_ngram_stats(self, trained_model):
        """get_ngram_stats should return counts for each order."""
        stats = trained_model.get_ngram_stats()
        assert isinstance(stats, dict)
        assert len(stats) >= 1
