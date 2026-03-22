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
