"""
Text Autocomplete System
========================

A multi-model text autocomplete library implementing n-gram, Markov chain,
and LSTM-based language models with beam search decoding.

PUBLIC API:
-----------
Models:
    NGramModel          — N-gram language model with backoff and interpolation
    MarkovChainModel    — First-order Markov chain with Laplace smoothing
    LSTMModel           — Optional PyTorch LSTM (word-level fit/predict_next)

Decoding:
    BeamSearchDecoder   — Multi-hypothesis beam search for better predictions

Data:
    tokenize            — Convert raw text to token lists
    load_sample_data    — Load the built-in sample corpus
    train_test_split    — Split tokens into training and test sets

Evaluation:
    compute_perplexity      — Measure model quality (lower is better)
    autocomplete_accuracy   — Top-k prediction accuracy
    prediction_diversity    — How varied predictions are across inputs
    vocabulary_coverage     — Fraction of reference vocab the model predicts
    prediction_confidence   — Top-1 prob / entropy / margin
    generate_report         — Markdown report across all metrics
    compare_models          — Side-by-side metric comparison

Example:
    >>> from src import NGramModel, tokenize, load_sample_data
    >>> tokens = tokenize(load_sample_data())
    >>> model = NGramModel(n=3).fit(tokens)
    >>> model.predict_next(["machine", "learning"], top_k=3)
    [('is', 0.5), ('models', 0.5)]
"""

from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.neural_model import LSTMModel
from src.beam_search import BeamSearchDecoder
from src.data_loader import tokenize, load_sample_data, train_test_split
from src.evaluation import (
    compute_perplexity,
    autocomplete_accuracy,
    prediction_diversity,
    vocabulary_coverage,
    prediction_confidence,
    generate_report,
    compare_models,
)
from src.config import API_VERSION as __version__

__all__ = [
    # Models
    "NGramModel",
    "MarkovChainModel",
    "LSTMModel",
    "BeamSearchDecoder",
    # Data utilities
    "tokenize",
    "load_sample_data",
    "train_test_split",
    # Evaluation
    "compute_perplexity",
    "autocomplete_accuracy",
    "prediction_diversity",
    "vocabulary_coverage",
    "prediction_confidence",
    "generate_report",
    "compare_models",
    # Version
    "__version__",
]
