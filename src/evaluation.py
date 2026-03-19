from typing import Any, Dict, List
"""Evaluation metrics for language models."""
import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

def compute_perplexity(model: Any, tokens: List[str]) -> float:
    from src.ngram_model import NGramModel
    if isinstance(model, NGramModel):
        return model.perplexity(tokens)
    return float("inf")

def autocomplete_accuracy(predictions: List[List[str]], ground_truth: List[str], top_k: int = 1) -> float:
    hits = sum(1 for preds, truth in zip(predictions, ground_truth)
               if truth in preds[:top_k])
    return hits / max(len(ground_truth), 1)

def generate_report(ngram_ppl: float, lstm_ppl: float, accuracy: float) -> str:
    lines = ["# Text Autocomplete — Evaluation Report", "",
             "| Metric | Value |", "|--------|-------|",
             f"| NGram Perplexity | {ngram_ppl:.2f} |",
             f"| LSTM Perplexity | {lstm_ppl:.2f} |",
             f"| Top-1 Accuracy | {accuracy:.2%} |"]
    return "\n".join(lines)
