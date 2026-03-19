"""Tests for evaluation."""
import pytest
from src.evaluation import autocomplete_accuracy, generate_report


class TestAccuracy:
    def test_perfect(self):
        preds = [["a", "b", "c"], ["x", "y", "z"]]
        truth = ["a", "x"]
        assert autocomplete_accuracy(preds, truth) == 1.0

    def test_zero(self):
        preds = [["a", "b"], ["x", "y"]]
        truth = ["z", "w"]
        assert autocomplete_accuracy(preds, truth) == 0.0

    def test_top_k(self):
        preds = [["a", "b", "c", "d"], ["x", "y"]]
        truth = ["c", "x"]
        assert autocomplete_accuracy(preds, truth, top_k=3) == 1.0


class TestReport:
    def test_output(self):
        report = generate_report(45.2, 32.1, 0.78)
        assert "# Text Autocomplete" in report
        assert "| NGram" in report
