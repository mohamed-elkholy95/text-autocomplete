"""Tests for beam search decoding."""

import pytest
from src.beam_search import BeamSearchDecoder
from src.data_loader import tokenize, load_sample_data
from src.ngram_model import NGramModel


@pytest.fixture
def trained_ngram():
    """Create a trained n-gram model for beam search."""
    tokens = tokenize(load_sample_data())
    return NGramModel(n=3).fit(tokens)


class TestBeamSearchInit:
    """Tests for beam search initialization."""

    def test_default_params(self):
        decoder = BeamSearchDecoder()
        assert decoder.beam_width == 5
        assert decoder.max_length == 10
        assert decoder.length_penalty == 0.6

    def test_custom_params(self):
        decoder = BeamSearchDecoder(beam_width=3, max_length=5, length_penalty=0.8)
        assert decoder.beam_width == 3
        assert decoder.max_length == 5
        assert decoder.length_penalty == 0.8


class TestBeamSearchScoring:
    """Tests for length-normalized scoring."""

    def test_length_normalized_score(self):
        decoder = BeamSearchDecoder()
        # With penalty 0.6: score = log_prob / length^0.6
        score = decoder._length_normalized_score(-5.0, 10)
        assert isinstance(score, float)
        # Should be less negative than raw log_prob (division by > 1)
        assert score > -5.0

    def test_length_zero(self):
        decoder = BeamSearchDecoder()
        # Zero length should return raw log_prob
        score = decoder._length_normalized_score(-3.0, 0)
        assert score == -3.0

    def test_penalty_effects(self):
        """Different penalty values should produce different scores."""
        decoder_low = BeamSearchDecoder(length_penalty=0.3)

        log_prob = -5.0
        length = 10

        score_low = decoder_low._length_normalized_score(log_prob, length)
        score_high = decoder_low._length_normalized_score(log_prob, 1)

        # Longer sequences with penalty < 1.0 should have higher scores
        assert score_low > score_high


class TestBeamSearch:
    """Tests for the beam search algorithm."""

    def test_search_returns_results(self, trained_ngram):
        decoder = BeamSearchDecoder(beam_width=3, max_length=3)
        results = decoder.search(trained_ngram, ["machine", "learning"], steps=2)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_result_structure(self, trained_ngram):
        decoder = BeamSearchDecoder(beam_width=3, max_length=3)
        results = decoder.search(trained_ngram, ["machine"], steps=2)

        # Each result should have the required keys
        for result in results:
            assert "tokens" in result
            assert "score" in result
            assert "log_prob" in result
            assert "length" in result
            assert isinstance(result["tokens"], list)
            assert isinstance(result["score"], float)
            assert result["length"] == len(result["tokens"])

    def test_search_sorted_by_score(self, trained_ngram):
        """Results should be sorted by score (best first)."""
        decoder = BeamSearchDecoder(beam_width=5, max_length=5)
        results = decoder.search(trained_ngram, ["machine"], steps=3)

        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_search_beam_width_1(self, trained_ngram):
        """Beam width 1 is equivalent to greedy decoding."""
        decoder = BeamSearchDecoder(beam_width=1, max_length=3)
        results = decoder.search(trained_ngram, ["machine"], steps=2)
        assert len(results) == 1

    def test_search_multiple_steps(self, trained_ngram):
        """More steps should produce longer token sequences."""
        decoder = BeamSearchDecoder(beam_width=3, max_length=10)

        results_short = decoder.search(trained_ngram, ["machine"], steps=1)
        results_long = decoder.search(trained_ngram, ["machine"], steps=5)

        avg_short = sum(r["length"] for r in results_short) / len(results_short)
        avg_long = sum(r["length"] for r in results_long) / len(results_long)

        assert avg_long >= avg_short

    def test_search_empty_context(self, trained_ngram):
        """Should handle empty context gracefully."""
        decoder = BeamSearchDecoder(beam_width=3, max_length=3)
        results = decoder.search(trained_ngram, [], steps=2)
        assert isinstance(results, list)


class TestBeamSearchEdgeCases:
    """Edge case tests."""

    def test_single_step(self, trained_ngram):
        decoder = BeamSearchDecoder(beam_width=3, max_length=1)
        results = decoder.search(trained_ngram, ["the"], steps=1)
        assert all(r["length"] <= 1 for r in results)

    def test_large_beam_width(self, trained_ngram):
        """Large beam width should still work (just slower)."""
        decoder = BeamSearchDecoder(beam_width=10, max_length=3)
        results = decoder.search(trained_ngram, ["machine"], steps=2)
        assert len(results) <= 10


# ---------------------------------------------------------------------------
# Beam search across the neural contract (LSTM, Transformer)
# ---------------------------------------------------------------------------
# The decoder's contract is "any object with predict_next(context, top_k)";
# these tests confirm the neural families satisfy it end-to-end. Skipped on
# minimal installs (no torch) — same pattern as the neural unit tests.

from src.neural_model import HAS_TORCH  # noqa: E402


@pytest.fixture
def trained_lstm():
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    from src.neural_model import LSTMModel
    tokens = tokenize(load_sample_data())
    m = LSTMModel(embed_dim=16, hidden_dim=32, num_layers=1)
    return m.fit(tokens, epochs=1, seq_len=8, batch_size=8, lr=1e-3)


@pytest.fixture
def trained_transformer():
    if not HAS_TORCH:
        pytest.skip("PyTorch not installed")
    from src.transformer_model import TransformerModel
    tokens = tokenize(load_sample_data())
    m = TransformerModel(
        d_model=16, n_heads=2, n_layers=1, ff_dim=32, max_seq_len=16,
    )
    return m.fit(tokens, epochs=1, seq_len=8, batch_size=8, lr=1e-3)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestBeamSearchNeural:
    """Beam search works the same way on LSTM / Transformer as on n-gram.
    This catches contract drift — e.g. if a future change to
    ``predict_next`` returns the wrong tuple shape for neural models."""

    def test_lstm_beam_returns_results(self, trained_lstm):
        decoder = BeamSearchDecoder(beam_width=3, max_length=3)
        results = decoder.search(trained_lstm, ["machine", "learning"])
        assert 0 < len(results) <= 3
        # BeamSearchDecoder.search returns ONLY the generated continuation,
        # not the context. Each beam is a dict with tokens / score /
        # log_prob / length fields.
        for r in results:
            assert "tokens" in r and "score" in r and "log_prob" in r
            assert isinstance(r["score"], float)
            assert len(r["tokens"]) <= 3

    def test_transformer_beam_returns_results(self, trained_transformer):
        decoder = BeamSearchDecoder(beam_width=3, max_length=3)
        results = decoder.search(trained_transformer, ["the", "attention"])
        assert 0 < len(results) <= 3
        for r in results:
            assert len(r["tokens"]) <= 3
            assert all(isinstance(t, str) for t in r["tokens"])

    def test_neural_greedy_vs_beam_both_run(self, trained_transformer):
        """Beam-1 is effectively greedy; beam-5 should produce ≥ beam-1's
        length and usually more hypotheses. Both paths must at least
        execute without error on the neural contract."""
        greedy = BeamSearchDecoder(beam_width=1, max_length=3).search(
            trained_transformer, ["deep"],
        )
        wide = BeamSearchDecoder(beam_width=5, max_length=3).search(
            trained_transformer, ["deep"],
        )
        assert len(greedy) >= 1
        assert len(wide) >= len(greedy)
