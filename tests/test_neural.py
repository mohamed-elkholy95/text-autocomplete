"""Tests for neural model."""
import pytest
from src.neural_model import LSTMModel, train_lstm, predict_next_lstm


class TestLSTMModel:
    def test_init(self):
        model = LSTMModel(vocab_size=100, embed_dim=32, hidden_dim=64)
        assert model is not None

    def test_train_mock(self):
        history = train_lstm(None, [1,2,3,4,5], 10, epochs=3)
        assert "loss" in history
        assert len(history["loss"]) == 3

    def test_predict_mock(self):
        preds = predict_next_lstm(None, [1,2,3])
        assert preds[0][1] == 0.0

    def test_empty_input(self):
        preds = predict_next_lstm(None, [])
        assert preds[0][1] == 0.0
