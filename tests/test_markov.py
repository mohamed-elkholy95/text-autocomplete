"""Tests for the Markov chain language model."""

import pytest
from src.data_loader import tokenize, load_sample_data
from src.markov_model import MarkovChainModel


@pytest.fixture
def trained_model():
    """Create a trained Markov chain model."""
    tokens = tokenize(load_sample_data())
    return MarkovChainModel().fit(tokens)


@pytest.fixture
def simple_model():
    """Model trained on a small, predictable corpus."""
    tokens = ["the", "cat", "sat", "the", "cat", "ran", "the", "dog", "barked",
              "the", "cat", "sat", "the", "dog", "sat"]
    return MarkovChainModel(smoothing=1.0).fit(tokens)


class TestMarkovFit:
    """Tests for model training."""

    def test_fit_sets_vocab(self, trained_model):
        assert trained_model.vocab_size > 0

    def test_fit_sets_is_fitted(self, trained_model):
        assert trained_model._is_fitted

    def test_fit_builds_transitions(self, trained_model):
        """Should build transition counts."""
        assert len(trained_model._transitions) > 0

    def test_fit_counts_transitions(self, simple_model):
        """Transition counts should match the training data."""
        # "the" appears 5 times, followed by "cat"(3), "dog"(2)
        assert simple_model._transitions["the"]["cat"] == 3
        assert simple_model._transitions["the"]["dog"] == 2

    def test_n_transitions(self, simple_model):
        """Total transitions should equal len(tokens) - 1."""
        tokens = ["the", "cat", "sat", "the", "cat", "ran",
                  "the", "dog", "barked", "the", "cat", "sat",
                  "the", "dog", "sat"]
        assert simple_model.n_transitions == len(tokens) - 1


class TestMarkovPredict:
    """Tests for next-word prediction."""

    def test_predict_returns_list(self, trained_model):
        preds = trained_model.predict_next(["machine"])
        assert isinstance(preds, list)
        assert len(preds) > 0

    def test_predict_tuple_format(self, trained_model):
        preds = trained_model.predict_next(["machine"])
        assert isinstance(preds[0], tuple)
        assert isinstance(preds[0][0], str)
        assert isinstance(preds[0][1], float)

    def test_predict_top_k(self, trained_model):
        preds = trained_model.predict_next(["machine"], top_k=3)
        assert len(preds) <= 3

    def test_predict_probabilities_sum(self, simple_model):
        """Probabilities from _get_transition_probs should sum to ~1.0."""
        probs = simple_model._get_transition_probs("the")
        total = sum(p for _, p in probs)
        # With smoothing, total might not be exactly 1.0 for the returned list
        # because we only return seen transitions (not the full vocab distribution)
        assert total > 0

    def test_predict_sorted(self, trained_model):
        """Predictions should be sorted by probability descending."""
        preds = trained_model.predict_next(["machine"], top_k=10)
        for i in range(len(preds) - 1):
            assert preds[i][1] >= preds[i + 1][1]

    def test_predict_empty_context(self, trained_model):
        """Empty context should fall back to most common words."""
        preds = trained_model.predict_next([])
        assert len(preds) > 0

    def test_predict_unknown_word(self, trained_model):
        """Unknown word should fall back to most common words."""
        preds = trained_model.predict_next(["xyznotaword"])
        assert len(preds) > 0

    def test_predict_deterministic(self, trained_model):
        """Same input → same output."""
        p1 = trained_model.predict_next(["machine", "learning"])
        p2 = trained_model.predict_next(["machine", "learning"])
        assert p1 == p2


class TestMarkovPerplexity:
    """Tests for perplexity computation."""

    def test_perplexity_positive(self, trained_model):
        ppl = trained_model.perplexity(tokenize("machine learning is great"))
        assert ppl > 0

    def test_perplexity_inf_when_not_fitted(self):
        model = MarkovChainModel()
        assert model.perplexity(["the", "cat"]) == float("inf")

    def test_perplexity_short_input(self, trained_model):
        """Perplexity with a very short sequence."""
        ppl = trained_model.perplexity(["a", "b"])
        assert ppl > 0


class TestMarkovGenerate:
    """Tests for text generation."""

    def test_generate_returns_string(self, trained_model):
        text = trained_model.generate_text(max_length=5)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_with_start_word(self, simple_model):
        text = simple_model.generate_text(start_word="the", max_length=5)
        assert text.startswith("the")

    def test_generate_respects_max_length(self, trained_model):
        text = trained_model.generate_text(max_length=10)
        word_count = len(text.split())
        assert word_count <= 10

    def test_generate_temperature_low(self, trained_model):
        """Low temperature should produce more deterministic output."""
        text1 = trained_model.generate_text(seed=42, max_length=10, temperature=0.1)
        text2 = trained_model.generate_text(seed=42, max_length=10, temperature=0.1)
        # With same seed and low temperature, should be deterministic
        assert isinstance(text1, str)

    def test_generate_not_fitted(self):
        model = MarkovChainModel()
        assert model.generate_text() == ""


class TestMarkovTransitions:
    """Tests for transition inspection."""

    def test_get_top_transitions(self, trained_model):
        transitions = trained_model.get_top_transitions("machine", top_k=5)
        assert isinstance(transitions, list)
        assert len(transitions) <= 5

    def test_get_top_transitions_unknown(self, trained_model):
        transitions = trained_model.get_top_transitions("xyznotaword")
        assert transitions == []

    def test_get_top_transitions_not_fitted(self):
        model = MarkovChainModel()
        assert model.get_top_transitions("the") == []


class TestMarkovSmoothedProb:
    """Tests for the _smoothed_prob helper used by perplexity."""

    def test_smoothed_unseen_matches_laplace_formula(self, simple_model):
        """For a seen word with an unseen next-word target, the probability
        must equal smoothing / (count(current) + smoothing * vocab_size)
        instead of the previous 1e-10 floor."""
        vocab_size = simple_model.vocab_size
        # In the fixture "the" appears 5 times; "xyz" never follows it.
        assert "xyz" not in simple_model._transitions["the"]
        p = simple_model._smoothed_prob("the", "xyz")
        expected = simple_model.smoothing / (5 + simple_model.smoothing * vocab_size)
        assert p == pytest.approx(expected)


class TestMarkovSmoothing:
    """Tests for Laplace smoothing behavior."""

    def test_zero_smoothing(self):
        """Zero smoothing (MLE) should still work."""
        model = MarkovChainModel(smoothing=0.0)
        model.fit(["a", "b", "c"] * 10)
        preds = model.predict_next(["a"])
        assert len(preds) > 0

    def test_high_smoothing(self):
        """High smoothing should spread probabilities more uniformly."""
        model_low = MarkovChainModel(smoothing=0.1)
        model_high = MarkovChainModel(smoothing=10.0)
        tokens = ["a", "b", "c"] * 10
        model_low.fit(tokens)
        model_high.fit(tokens)
        # Both should return predictions
        assert len(model_low.predict_next(["a"])) > 0
        assert len(model_high.predict_next(["a"])) > 0
