"""Tests for the neural language model (LSTM)."""

import pytest
from src.neural_model import LSTMModel, train_lstm, predict_next_lstm, HAS_TORCH


class TestLSTMInit:
    """Tests for model initialization."""

    def test_init_default(self):
        """Model should initialize with default parameters."""
        model = LSTMModel(vocab_size=100, embed_dim=32, hidden_dim=64)
        assert model is not None

    def test_init_various_sizes(self):
        """Model should handle different vocabulary and hidden sizes."""
        model = LSTMModel(vocab_size=1000, embed_dim=128, hidden_dim=256, num_layers=3)
        assert model is not None

    def test_init_single_layer(self):
        """Single-layer LSTM should work."""
        model = LSTMModel(vocab_size=50, embed_dim=16, hidden_dim=32, num_layers=1)
        assert model is not None


class TestLSTMTraining:
    """Tests for the training function."""

    def test_train_mock(self):
        """Training with None model (no PyTorch) should return mock history."""
        history = train_lstm(None, [1, 2, 3, 4, 5], vocab_size=10, epochs=3)
        assert "loss" in history
        assert "perplexity" in history
        assert len(history["loss"]) == 3
        assert len(history["perplexity"]) == 3

    def test_train_loss_decreasing(self):
        """Mock training loss should decrease each epoch."""
        history = train_lstm(None, [1, 2, 3, 4, 5], vocab_size=10, epochs=5)
        for i in range(len(history["loss"]) - 1):
            assert history["loss"][i] > history["loss"][i + 1]

    def test_train_perplexity_from_loss(self):
        """Perplexity should be exp(loss) in mock training."""
        import numpy as np
        history = train_lstm(None, [1, 2, 3], vocab_size=10, epochs=3)
        for loss, ppl in zip(history["loss"], history["perplexity"]):
            assert abs(ppl - np.exp(min(loss, 20))) < 0.1

    def test_train_custom_epochs(self):
        """Should respect custom epoch count."""
        history = train_lstm(None, [1, 2, 3], vocab_size=10, epochs=7)
        assert len(history["loss"]) == 7


class TestLSTMPredict:
    """Tests for prediction function."""

    def test_predict_mock(self):
        """Prediction with None model should return default."""
        preds = predict_next_lstm(None, [1, 2, 3])
        assert len(preds) == 1
        assert preds[0] == ("<UNK>", 0.0)

    def test_predict_empty_input(self):
        """Empty input should return default prediction."""
        preds = predict_next_lstm(None, [])
        assert preds[0] == ("<UNK>", 0.0)

    def test_predict_single_token(self):
        """Single token input should work."""
        preds = predict_next_lstm(None, [42])
        assert isinstance(preds, list)

    def test_predict_long_input(self):
        """Long input should be truncated to last 20 tokens."""
        preds = predict_next_lstm(None, list(range(100)))
        assert isinstance(preds, list)


class TestTorchAvailability:
    """Tests that behave differently based on PyTorch availability."""

    def test_torch_flag_exists(self):
        """The HAS_TORCH flag should be a boolean."""
        assert isinstance(HAS_TORCH, bool)

    def test_no_torch_mock_behavior(self):
        """When torch is not available, functions should still work via mocks."""
        # This test always passes (mock mode) but documents expected behavior
        preds = predict_next_lstm(None, [1, 2, 3])
        assert isinstance(preds, list)


@pytest.mark.skipif(not HAS_TORCH, reason="LSTM persistence requires PyTorch")
class TestLSTMPersistence:
    """Round-trip save/load for the LSTMModel. Skipped without torch."""

    @staticmethod
    def _tiny_corpus():
        return ("the cat sat on the mat . the dog sat on the log . "
                "a cat and a dog are friends .").split()

    def test_save_then_load_round_trip(self, tmp_path):
        """A saved LSTM should reload with identical vocab and top-1 predictions."""
        tokens = self._tiny_corpus()
        original = LSTMModel(embed_dim=16, hidden_dim=32, num_layers=1)
        original.fit(tokens, epochs=1, seq_len=4, batch_size=2)

        path = tmp_path / "lstm_tiny"
        original.save(str(path))

        assert (tmp_path / "lstm_tiny.safetensors").exists()
        assert (tmp_path / "lstm_tiny.json").exists()

        reloaded = LSTMModel.load(str(path))
        assert reloaded.vocab_size == original.vocab_size
        assert reloaded._word_to_id == original._word_to_id

        ctx = ["the", "cat"]
        top_orig = original.predict_next(ctx, top_k=3)
        top_reload = reloaded.predict_next(ctx, top_k=3)
        assert [w for w, _ in top_orig] == [w for w, _ in top_reload]

    def test_save_untrained_raises(self, tmp_path):
        """Saving before fit() must fail loudly."""
        model = LSTMModel(vocab_size=10, embed_dim=8, hidden_dim=16, num_layers=1)
        with pytest.raises(RuntimeError, match="untrained"):
            model.save(str(tmp_path / "nope"))

    def test_load_rejects_wrong_model_type(self, tmp_path):
        """Loading a file whose model_type is not 'lstm' must fail loudly."""
        import json
        path = tmp_path / "wrong"
        # Write a dummy meta file only; weights don't need to exist to hit the check.
        (tmp_path / "wrong.json").write_text(
            json.dumps({"model_type": "ngram"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="model_type"):
            LSTMModel.load(str(path))

    def test_load_rejects_schema_v1(self, tmp_path):
        """v1 checkpoints (pre-weight-tying) must be refused, not silently loaded."""
        import json
        path = tmp_path / "oldbundle"
        (tmp_path / "oldbundle.json").write_text(
            json.dumps({
                "model_type": "lstm",
                "schema_version": 1,
                "vocab": ["a", "b"],
                "embed_dim": 8,
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema_version=1"):
            LSTMModel.load(str(path))
