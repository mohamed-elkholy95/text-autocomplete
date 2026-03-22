"""Tests for the evaluation metrics module."""

import pytest
from src.evaluation import (
    autocomplete_accuracy,
    prediction_diversity,
    vocabulary_coverage,
    generate_report,
    compare_models,
    compute_perplexity,
)
from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.data_loader import tokenize, load_sample_data


@pytest.fixture
def trained_ngram():
    tokens = tokenize(load_sample_data())
    return NGramModel(n=3).fit(tokens)


@pytest.fixture
def trained_markov():
    tokens = tokenize(load_sample_data())
    return MarkovChainModel().fit(tokens)


class TestComputePerplexity:
    """Tests for the compute_perplexity helper."""

    def test_with_ngram(self, trained_ngram):
        ppl = compute_perplexity(trained_ngram, tokenize("machine learning is"))
        assert ppl > 0

    def test_with_markov(self, trained_markov):
        ppl = compute_perplexity(trained_markov, tokenize("machine learning is"))
        assert ppl > 0

    def test_with_unknown_type(self):
        """Unknown model types should return infinity."""
        ppl = compute_perplexity("not a model", ["a", "b"])
        assert ppl == float("inf")


class TestAccuracy:
    """Tests for autocomplete accuracy."""

    def test_perfect_accuracy(self):
        preds = [["a", "b", "c"], ["x", "y", "z"]]
        truth = ["a", "x"]
        assert autocomplete_accuracy(preds, truth) == 1.0

    def test_zero_accuracy(self):
        preds = [["a", "b"], ["x", "y"]]
        truth = ["z", "w"]
        assert autocomplete_accuracy(preds, truth) == 0.0

    def test_partial_accuracy(self):
        preds = [["a", "b"], ["x", "y"]]
        truth = ["a", "w"]
        assert autocomplete_accuracy(preds, truth) == 0.5

    def test_top_k_accuracy(self):
        preds = [["a", "b", "c", "d"], ["x", "y"]]
        truth = ["c", "x"]
        assert autocomplete_accuracy(preds, truth, top_k=3) == 1.0
        assert autocomplete_accuracy(preds, truth, top_k=1) == 0.5

    def test_empty_ground_truth(self):
        assert autocomplete_accuracy([], []) == 0.0

    def test_mismatched_lengths(self):
        """Should handle when preds and truth have different lengths gracefully."""
        preds = [["a"], ["b"], ["c"]]
        truth = ["a"]
        # Only first pair is compared due to zip
        assert autocomplete_accuracy(preds, truth) == 1.0


class TestDiversity:
    """Tests for prediction diversity metric."""

    def test_max_diversity(self):
        """All unique predictions → diversity = 1.0."""
        preds = [["a"], ["b"], ["c"], ["d"], ["e"]]
        assert prediction_diversity(preds) == 1.0

    def test_zero_diversity(self):
        """All same predictions → diversity = 1/N (not 0, because they're
        all technically the same single unique value)."""
        preds = [["a"], ["a"], ["a"]]
        # Only 1 unique top-1 prediction out of 3 total
        assert prediction_diversity(preds) == pytest.approx(1/3)

    def test_empty_predictions(self):
        assert prediction_diversity([]) == 0.0

    def test_empty_inner_lists(self):
        preds = [[], [], []]
        assert prediction_diversity(preds) == pytest.approx(1/3)


class TestVocabularyCoverage:
    """Tests for vocabulary coverage metric."""

    def test_full_coverage(self):
        preds = [["a", "b", "c"], ["d", "e", "f"]]
        vocab = {"a", "b", "c", "d", "e", "f"}
        assert vocabulary_coverage(preds, vocab) == 1.0

    def test_partial_coverage(self):
        preds = [["a", "b"], ["c"]]
        vocab = {"a", "b", "c", "d", "e"}
        assert vocabulary_coverage(preds, vocab) == pytest.approx(3/5)

    def test_zero_coverage(self):
        preds = [["x", "y"], ["z"]]
        vocab = {"a", "b", "c"}
        assert vocabulary_coverage(preds, vocab) == 0.0

    def test_empty_vocab(self):
        assert vocabulary_coverage([["a"]], set()) == 0.0

    def test_empty_predictions(self):
        assert vocabulary_coverage([], {"a", "b"}) == 0.0


class TestReport:
    """Tests for report generation."""

    def test_output_contains_headers(self):
        report = generate_report(45.2, 38.1, 22.5, 0.78, 0.85, 0.72)
        assert "# Text Autocomplete" in report
        assert "| N-gram Perplexity |" in report
        assert "| Markov Chain Perplexity |" in report

    def test_output_contains_values(self):
        report = generate_report(45.2, 38.1, 22.5, 0.78, 0.85, 0.72)
        assert "45.20" in report
        assert "78.00%" in report

    def test_output_contains_interpretation(self):
        report = generate_report(45.2, 38.1, 22.5, 0.78, 0.85, 0.72)
        assert "Interpretation Guide" in report


class TestCompareModels:
    """Tests for model comparison function."""

    def test_compare_two_models(self, trained_ngram, trained_markov):
        test_tokens = tokenize("machine learning deep neural networks")
        context = ["machine", "learning"]
        results = compare_models(
            {"ngram": trained_ngram, "markov": trained_markov},
            test_tokens,
            context,
        )
        assert "ngram" in results
        assert "markov" in results
        assert results["ngram"]["perplexity"] > 0
        assert results["markov"]["perplexity"] > 0
        assert len(results["ngram"]["top_predictions"]) > 0
        assert len(results["markov"]["top_predictions"]) > 0

    def test_compare_empty_models(self):
        results = compare_models({}, ["a", "b"], ["a"])
        assert results == {}
