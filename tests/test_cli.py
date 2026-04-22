"""Smoke tests for cli.py.

The CLI is a thin shim over the library — we don't retest the models.
These tests confirm each subcommand parses its args, runs end to end on
the sample corpus, and prints something resembling the expected output.

Neural subcommands (`train --model lstm`, `predict --model lstm`) are
gated on HAS_TORCH and skip cleanly on minimal installs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "cli.py"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke cli.py with the same Python that's running this test."""
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


class TestInfo:
    def test_info_runs(self):
        proc = run_cli("info")
        assert proc.returncode == 0, proc.stderr
        # `info` prints corpus stats; the output should mention vocab or tokens.
        combined = proc.stdout + proc.stderr
        assert (
            "token" in combined.lower()
            or "vocab" in combined.lower()
            or "corpus" in combined.lower()
        )


class TestPredict:
    def test_predict_ngram(self):
        proc = run_cli("predict", "--text", "machine learning is", "--top-k", "3")
        assert proc.returncode == 0, proc.stderr
        assert "Predictions" in proc.stdout or "prediction" in proc.stdout.lower()

    def test_predict_markov(self):
        proc = run_cli(
            "predict", "--text", "the attention", "--model", "markov", "--top-k", "3",
        )
        assert proc.returncode == 0, proc.stderr

    def test_predict_empty_text_fails_cleanly(self):
        # argparse enforces required args; empty --text should still parse
        # but the model will emit "no valid tokens" and exit non-zero.
        proc = run_cli("predict", "--text", "   ", "--top-k", "3")
        # Either argparse rejects it (non-zero) or the model does.
        assert proc.returncode != 0


class TestEval:
    def test_eval_produces_report(self):
        proc = run_cli("eval", "--test-ratio", "0.2")
        assert proc.returncode == 0, proc.stderr
        # The report is a formatted table; look for one of its rows.
        assert "perplexity" in proc.stdout.lower() or "top" in proc.stdout.lower()


class TestTrainAndReload:
    def test_train_ngram_and_load_for_predict(self, tmp_path):
        save = tmp_path / "ngram.json"
        proc = run_cli(
            "train", "--model", "ngram", "--n", "3", "--save", str(save),
        )
        assert proc.returncode == 0, proc.stderr
        assert save.exists()

        # Reload path: predict with --load skips training and hits the JSON.
        proc = run_cli(
            "predict", "--text", "deep learning",
            "--load", str(save), "--top-k", "3",
        )
        assert proc.returncode == 0, proc.stderr


try:
    from src.neural_model import HAS_TORCH
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestNeuralCLI:
    def test_train_lstm_tiny_then_predict(self, tmp_path):
        save = tmp_path / "lstm_tiny"
        proc = run_cli(
            "train", "--model", "lstm",
            "--epochs", "1", "--embed-dim", "16", "--hidden-dim", "32",
            "--num-layers", "1", "--save", str(save),
        )
        assert proc.returncode == 0, proc.stderr
        assert (tmp_path / "lstm_tiny.safetensors").exists()

        proc = run_cli(
            "predict", "--text", "machine learning",
            "--model", "lstm", "--load", str(save), "--top-k", "3",
        )
        assert proc.returncode == 0, proc.stderr
