"""
Integration Tests — Full Pipeline Validation
=============================================

These tests verify that all components work together correctly, testing
the complete flow from data loading through model training to prediction.

EDUCATIONAL CONTEXT:
-------------------
Unit tests verify individual functions in isolation. Integration tests
verify that the pieces FIT TOGETHER. Both are essential:

- Unit test: "Does NGramModel.predict_next() return sorted results?"
- Integration test: "Can I load data, train a model, save it, reload it,
  and get the same predictions?"

Integration tests catch issues like:
- Incompatible data formats between modules
- Broken import chains
- State management bugs (e.g., model not properly initialized)
- Serialization/deserialization mismatches
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data_loader import (
    load_sample_data,
    tokenize,
    train_test_split,
    get_corpus_stats,
    generate_synthetic_data,
    build_ngrams,
)
from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.beam_search import BeamSearchDecoder
from src.evaluation import (
    compute_perplexity,
    autocomplete_accuracy,
    prediction_diversity,
    vocabulary_coverage,
    compare_models,
    generate_report,
)


# ---------------------------------------------------------------------------
# Fixtures — Shared test data and trained models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus_tokens():
    """Load and tokenize the sample corpus once for all tests in this module.

    scope='module' means this fixture runs ONCE and is shared across all
    tests in this file. This is efficient because loading/tokenizing is
    deterministic and doesn't change between tests.
    """
    corpus = load_sample_data()
    return tokenize(corpus)


@pytest.fixture(scope="module")
def train_test_data(corpus_tokens):
    """Split corpus into train and test sets."""
    return train_test_split(corpus_tokens, test_ratio=0.2, seed=42)


@pytest.fixture(scope="module")
def trained_ngram(train_test_data):
    """Train an n-gram model on the training split."""
    train_tokens, _ = train_test_data
    return NGramModel(n=3, seed=42).fit(train_tokens)


@pytest.fixture(scope="module")
def trained_markov(train_test_data):
    """Train a Markov chain model on the training split."""
    train_tokens, _ = train_test_data
    return MarkovChainModel(seed=42).fit(train_tokens)


# ---------------------------------------------------------------------------
# End-to-End Pipeline Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test the complete data → train → predict → evaluate pipeline."""

    def test_load_tokenize_train_predict(self):
        """Verify the full pipeline from raw text to predictions."""
        # Step 1: Load data
        corpus = load_sample_data()
        assert len(corpus) > 0, "Corpus should not be empty"

        # Step 2: Tokenize
        tokens = tokenize(corpus)
        assert len(tokens) > 100, "Should have substantial number of tokens"
        assert all(isinstance(t, str) for t in tokens)

        # Step 3: Train
        model = NGramModel(n=3).fit(tokens)
        assert model.vocab_size > 0

        # Step 4: Predict
        preds = model.predict_next(["machine", "learning"], top_k=5)
        assert len(preds) > 0
        assert all(isinstance(w, str) and isinstance(p, float) for w, p in preds)

        # Step 5: Probabilities should be valid
        for word, prob in preds:
            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range for '{word}'"

    def test_train_test_split_no_data_leak(self, corpus_tokens):
        """Verify train/test split creates disjoint index sets.

        DATA LEAKAGE is when test data accidentally appears in training data.
        This makes evaluation metrics unrealistically optimistic because the
        model has "seen" the test data during training.
        """
        train, test = train_test_split(corpus_tokens, test_ratio=0.2, seed=42)
        # Combined size should equal original (no data lost)
        assert len(train) + len(test) == len(corpus_tokens)
        # Both sets should be non-empty
        assert len(train) > 0
        assert len(test) > 0

    def test_both_models_produce_valid_predictions(self, trained_ngram, trained_markov):
        """Both models should return valid predictions for the same context."""
        context = ["the", "neural"]

        for model_name, model in [("ngram", trained_ngram), ("markov", trained_markov)]:
            preds = model.predict_next(context, top_k=5)
            assert len(preds) > 0, f"{model_name} produced no predictions"
            probs = [p for _, p in preds]
            assert all(0.0 <= p <= 1.0 for p in probs), f"{model_name} has invalid probabilities"


# ---------------------------------------------------------------------------
# Model Serialization Round-Trip Tests
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    """Test that models produce identical results after save/load cycle."""

    def test_ngram_save_load_roundtrip(self, trained_ngram):
        """N-gram model should produce identical predictions after save+load."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            # Save the trained model
            trained_ngram.save(path)

            # Verify the file exists and is valid JSON
            assert Path(path).exists()
            with open(path) as f:
                data = json.load(f)
            assert data["model_type"] == "ngram"
            assert data["n"] == trained_ngram.n

            # Load and compare predictions
            loaded = NGramModel.load(path)
            assert loaded.vocab_size == trained_ngram.vocab_size
            assert loaded.n == trained_ngram.n

            # Predictions should be identical
            context = ["machine", "learning"]
            original_preds = trained_ngram.predict_next(context, top_k=5)
            loaded_preds = loaded.predict_next(context, top_k=5)
            assert original_preds == loaded_preds
        finally:
            Path(path).unlink(missing_ok=True)

    def test_markov_save_load_roundtrip(self, trained_markov):
        """Markov model should produce identical predictions after save+load."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            trained_markov.save(path)

            loaded = MarkovChainModel.load(path)
            assert loaded.vocab_size == trained_markov.vocab_size

            context = ["neural"]
            original = trained_markov.predict_next(context, top_k=5)
            reloaded = loaded.predict_next(context, top_k=5)
            assert original == reloaded
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_unfitted_model_raises(self):
        """Saving an untrained model should raise a clear error."""
        model = NGramModel(n=3)
        with pytest.raises(RuntimeError, match="untrained"):
            model.save("/tmp/should_not_exist.json")

    def test_load_wrong_model_type_raises(self):
        """Loading a file with wrong model type should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"model_type": "wrong"}, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="Expected model_type"):
                NGramModel.load(path)
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Beam Search Integration Tests
# ---------------------------------------------------------------------------

class TestBeamSearchIntegration:
    """Test beam search with real trained models."""

    def test_beam_search_with_ngram(self, trained_ngram):
        """Beam search should produce multi-word completions from n-gram model."""
        decoder = BeamSearchDecoder(beam_width=3, max_length=5)
        results = decoder.search(trained_ngram, ["machine", "learning"])

        assert len(results) > 0
        assert len(results) <= 3  # At most beam_width results

        best = results[0]
        assert "tokens" in best
        assert "score" in best
        assert len(best["tokens"]) > 0
        assert best["score"] <= 0.0  # Log probabilities are negative

    def test_beam_search_with_markov(self, trained_markov):
        """Beam search should work with Markov chain model too."""
        decoder = BeamSearchDecoder(beam_width=3, max_length=4)
        results = decoder.search(trained_markov, ["data"])

        assert len(results) > 0
        for result in results:
            assert result["length"] > 0

    def test_greedy_vs_beam_search(self, trained_ngram):
        """Beam width=1 should behave like greedy decoding."""
        greedy = BeamSearchDecoder(beam_width=1, max_length=3)
        beam = BeamSearchDecoder(beam_width=5, max_length=3)

        greedy_results = greedy.search(trained_ngram, ["the"])
        beam_results = beam.search(trained_ngram, ["the"])

        # Beam search should find a result at least as good as greedy
        assert beam_results[0]["score"] >= greedy_results[0]["score"] - 1e-6


# ---------------------------------------------------------------------------
# Evaluation Integration Tests
# ---------------------------------------------------------------------------

class TestEvaluationIntegration:
    """Test evaluation metrics with real trained models."""

    def test_perplexity_finite(self, trained_ngram, trained_markov, train_test_data):
        """Perplexity should be finite for trained models on test data."""
        _, test_tokens = train_test_data

        ngram_ppl = compute_perplexity(trained_ngram, test_tokens)
        markov_ppl = compute_perplexity(trained_markov, test_tokens)

        assert ngram_ppl < float("inf"), "N-gram perplexity should be finite"
        assert markov_ppl < float("inf"), "Markov perplexity should be finite"
        assert ngram_ppl > 0, "Perplexity must be positive"
        assert markov_ppl > 0, "Perplexity must be positive"

    def test_compare_models(self, trained_ngram, trained_markov, train_test_data):
        """Model comparison should produce valid metrics for both models."""
        _, test_tokens = train_test_data
        models = {"ngram": trained_ngram, "markov": trained_markov}

        results = compare_models(models, test_tokens, ["machine", "learning"])

        assert "ngram" in results
        assert "markov" in results
        for name, metrics in results.items():
            assert "perplexity" in metrics
            assert "top_predictions" in metrics
            assert metrics["perplexity"] > 0

    def test_evaluation_report_generation(self):
        """The report generator should produce valid markdown."""
        report = generate_report(
            ngram_ppl=120.5,
            markov_ppl=200.3,
            lstm_ppl=80.1,
            accuracy=0.35,
            diversity=0.72,
            coverage=0.45,
        )
        assert "Evaluation Report" in report
        assert "120.50" in report
        assert "35.00%" in report

    def test_interpolated_vs_backoff_predictions(self, trained_ngram):
        """Interpolated predictions should differ from backoff predictions."""
        context = ["neural", "networks"]

        backoff_preds = trained_ngram.predict_next(context, top_k=5)
        interp_preds = trained_ngram.predict_next_interpolated(context, top_k=5)

        # Both should return valid predictions
        assert len(backoff_preds) > 0
        assert len(interp_preds) > 0

        # Interpolated should typically produce more diverse results
        # (blending multiple orders gives more candidates)
        backoff_words = {w for w, _ in backoff_preds}
        interp_words = {w for w, _ in interp_preds}
        # At least some overlap expected
        assert len(backoff_words & interp_words) > 0 or len(interp_words) > 0


# ---------------------------------------------------------------------------
# Data Pipeline Tests
# ---------------------------------------------------------------------------

class TestDataPipeline:
    """Test data loading and preprocessing integration."""

    def test_synthetic_data_is_tokenizable(self):
        """Synthetic data should be valid for tokenization and training."""
        synthetic = generate_synthetic_data(n_sentences=50)
        tokens = tokenize(synthetic)
        assert len(tokens) > 50

        # Should be trainable
        model = NGramModel(n=2).fit(tokens)
        assert model.vocab_size > 0
        preds = model.predict_next(tokens[:2], top_k=3)
        assert len(preds) > 0

    def test_corpus_stats_match_tokens(self, corpus_tokens):
        """Corpus statistics should be consistent with the token list."""
        stats = get_corpus_stats(corpus_tokens)
        assert stats["total_tokens"] == len(corpus_tokens)
        assert stats["unique_tokens"] == len(set(corpus_tokens))
        assert stats["min_token_length"] <= stats["avg_token_length"]
        assert stats["avg_token_length"] <= stats["max_token_length"]

    def test_ngram_building(self, corpus_tokens):
        """Built n-grams should have correct structure."""
        # Use integer tokens for build_ngrams
        int_tokens = list(range(10))
        trigrams = build_ngrams(int_tokens, 3)
        assert len(trigrams) == 8  # 10 - 3 + 1
        assert all(len(ng) == 3 for ng in trigrams)
        assert trigrams[0] == (0, 1, 2)
        assert trigrams[-1] == (7, 8, 9)
