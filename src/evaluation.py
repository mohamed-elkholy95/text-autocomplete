"""
Evaluation Metrics for Language Models
========================================

This module provides comprehensive evaluation metrics for measuring the quality
of language models and autocomplete systems.

EDUCATIONAL CONTEXT:
-------------------
Evaluating language models is different from evaluating classifiers. For
classification, accuracy is straightforward (correct/total). For language
models, we need to measure how well the model predicts UNSEEN text.

KEY METRICS:
1. PERPLEXITY: The primary metric for language models. Lower is better.
2. ACCURACY: How often the correct next word appears in top-k predictions.
3. DIVERSITY: Measures the variety of predictions (prevents trivial models).
4. COVERAGE: What fraction of the vocabulary the model can predict.
"""

from typing import Any, Dict, List, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_perplexity(model: Any, tokens: List[str]) -> float:
    """Compute perplexity on a token sequence using the model's built-in method.

    PERPLEXITY is the geometric mean of the inverse probabilities:
        PPL = P(w1, w2, ..., wn)^(-1/n)

    Intuitively: "on average, how many equally likely words could come next?"
    - PPL = 1: Perfect prediction (model always picks the right word)
    - PPL = 10: On average, model narrows it down to ~10 candidates
    - PPL = 100: Model is basically guessing from ~100 equally likely words
    - PPL = |V|: No better than random guessing from the vocabulary

    Args:
        model: A fitted model with a ``perplexity(tokens)`` method. All
            three first-class model families (NGramModel, MarkovChainModel,
            LSTMModel) implement it, using count-based probabilities for
            the statistical models and token-level cross-entropy for the
            neural one.
        tokens: Token sequence to evaluate on.

    Returns:
        Perplexity score (float). Lower is better. Returns ``inf`` for
        model types that don't advertise a ``perplexity`` method.
    """
    from src.ngram_model import NGramModel
    from src.markov_model import MarkovChainModel
    from src.neural_model import LSTMModel
    from src.transformer_model import TransformerModel

    if isinstance(model, (NGramModel, MarkovChainModel, LSTMModel, TransformerModel)):
        return model.perplexity(tokens)
    return float("inf")


def autocomplete_accuracy(
    predictions: List[List[str]],
    ground_truth: List[str],
    top_k: int = 1,
) -> float:
    """Compute top-k accuracy for autocomplete predictions.

    TOP-K ACCURACY counts a prediction as correct if the true next word
    appears anywhere in the model's top-k suggestions.

    Example with top_k=3:
        Prediction: ["learning", "models", "algorithms"]
        Ground truth: "models"
        → CORRECT! "models" is in the top-3 predictions.

    Why top-k instead of just top-1?
        Because in an autocomplete UI, we show multiple suggestions.
        If the user's intended word appears anywhere in the suggestion list,
        the system is still helpful. Top-1 would be too strict.

    Args:
        predictions: List of predicted word lists, one per test case.
        ground_truth: List of true next words, one per test case.
        top_k: Number of top predictions to consider.

    Returns:
        Accuracy as a float between 0.0 and 1.0.
    """
    if not ground_truth:
        return 0.0
    hits = sum(
        1 for preds, truth in zip(predictions, ground_truth)
        if truth in preds[:top_k]
    )
    return hits / len(ground_truth)


def prediction_diversity(predictions: List[List[str]]) -> float:
    """Measure how diverse the model's predictions are across test cases.

    DIVERSITY is important because a trivial model that always predicts
    the same top word (e.g., "the") might have decent accuracy on common
    test cases but is utterly useless for real autocomplete.

    We measure diversity as the ratio of unique predictions to total predictions.
    - 1.0 = Every prediction is unique (maximum diversity)
    - 0.0 = Every prediction is the same (no diversity — bad!)

    We look at the top-1 prediction for each test case.

    Args:
        predictions: List of predicted word lists.

    Returns:
        Diversity score between 0.0 and 1.0.
    """
    if not predictions:
        return 0.0

    # Extract top-1 prediction from each test case
    top_predictions = [preds[0] if preds else "" for preds in predictions]

    # Count unique top-1 predictions
    unique_count = len(set(top_predictions))
    total_count = len(top_predictions)

    return unique_count / total_count


def vocabulary_coverage(predictions: List[List[str]], reference_vocab: set) -> float:
    """Measure what fraction of the reference vocabulary the model can predict.

    COVERAGE tells us how many different words from a reference vocabulary
    appear in the model's predictions across all test cases.

    A model with high coverage but low accuracy is creative but unreliable.
    A model with high accuracy but low coverage is reliable but narrow.
    The ideal model has both high coverage AND high accuracy.

    Args:
        predictions: List of predicted word lists.
        reference_vocab: Set of reference vocabulary words.

    Returns:
        Coverage fraction between 0.0 and 1.0.
    """
    if not reference_vocab:
        return 0.0

    # Collect all unique words the model ever predicted
    predicted_words = set()
    for preds in predictions:
        predicted_words.update(preds)

    # What fraction of the reference vocab did the model predict?
    return len(predicted_words & reference_vocab) / len(reference_vocab)


def generate_report(
    ngram_ppl: float,
    markov_ppl: float,
    lstm_ppl: float,
    accuracy: float,
    diversity: float,
    coverage: float = 0.0,
) -> str:
    """Generate a human-readable evaluation report comparing all models.

    The report uses a markdown table format, suitable for rendering in
    GitHub READMEs or Jupyter notebooks.

    Args:
        ngram_ppl: N-gram model perplexity.
        markov_ppl: Markov chain model perplexity.
        lstm_ppl: LSTM model perplexity.
        accuracy: Top-1 autocomplete accuracy.
        diversity: Prediction diversity score.
        coverage: Vocabulary coverage fraction.

    Returns:
        Markdown-formatted report string.
    """
    lines = [
        "# Text Autocomplete — Evaluation Report",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| N-gram Perplexity | {ngram_ppl:.2f} |",
        f"| Markov Chain Perplexity | {markov_ppl:.2f} |",
        f"| LSTM Perplexity | {lstm_ppl:.2f} |",
        f"| Top-1 Accuracy | {accuracy:.2%} |",
        f"| Prediction Diversity | {diversity:.2%} |",
        f"| Vocabulary Coverage | {coverage:.2%} |",
        "",
        "## Interpretation Guide",
        "",
        "- **Perplexity**: Lower is better. Measures how well the model predicts unseen text.",
        "- **Accuracy**: Higher is better. Fraction of correct predictions in top-1 suggestions.",
        "- **Diversity**: Higher is better. Measures variety in predictions (prevents trivial models).",
        "- **Coverage**: Higher is better. Fraction of reference vocabulary the model can predict.",
    ]
    return "\n".join(lines)


def prediction_confidence(predictions: List[Tuple[str, float]]) -> Dict[str, float]:
    """Analyze the confidence level of a set of predictions.

    CONFIDENCE ANALYSIS:
    When an autocomplete system makes predictions, it's useful to know HOW
    confident the model is — not just what it predicts. This helps with:
    - UI decisions: show predictions only when confidence is high
    - Model debugging: low confidence everywhere → undertrained model
    - User experience: display confidence indicators (green/yellow/red)

    We compute three confidence metrics:

    1. TOP-1 PROBABILITY: How much probability mass is on the best prediction.
       - High (>0.5): Model is very sure about the next word
       - Low (<0.1): Model is uncertain, many words are roughly equally likely

    2. ENTROPY: Measures the "spread" of the probability distribution.
       - Low entropy: Predictions are concentrated (model is confident)
       - High entropy: Predictions are spread out (model is uncertain)
       - Maximum entropy = log2(k) where k is the number of predictions

    3. MARGIN: Difference between top-1 and top-2 probabilities.
       - Large margin: Clear winner — the top prediction stands out
       - Small margin: Close race — the model can't decide between options

    Args:
        predictions: List of (word, probability) tuples from a model.

    Returns:
        Dictionary with confidence metrics:
        - top1_prob: Probability of the best prediction
        - entropy: Shannon entropy of the prediction distribution (bits)
        - margin: Probability gap between #1 and #2 predictions
        - confidence_level: Categorical label ("high", "medium", "low")
    """
    if not predictions:
        return {
            "top1_prob": 0.0,
            "entropy": 0.0,
            "margin": 0.0,
            "confidence_level": "none",
        }

    probs = [p for _, p in predictions]
    top1 = probs[0]

    # Shannon entropy: H = -Σ p(x) × log2(p(x))
    # Measures information content / uncertainty in the distribution
    entropy = -sum(p * np.log2(p + 1e-10) for p in probs)

    # Margin between top-1 and top-2
    margin = probs[0] - probs[1] if len(probs) > 1 else probs[0]

    # Categorical confidence level based on heuristic thresholds
    if top1 > 0.5 or margin > 0.3:
        level = "high"
    elif top1 > 0.2 or margin > 0.1:
        level = "medium"
    else:
        level = "low"

    return {
        "top1_prob": round(top1, 4),
        "entropy": round(float(entropy), 4),
        "margin": round(margin, 4),
        "confidence_level": level,
    }


def compare_models(
    models: Dict[str, Any],
    test_tokens: List[str],
    context_tokens: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compare multiple models on the same test data.

    This function runs all models on identical inputs and collects metrics
    for fair comparison. It's the foundation for model selection in production.

    Args:
        models: Dictionary mapping model names to model objects.
            Each model must have predict_next() and perplexity() methods.
        test_tokens: Token sequence for perplexity evaluation.
        context_tokens: Context tokens for accuracy evaluation.

    Returns:
        Dictionary mapping model names to their metrics:
        {
            "model_name": {
                "perplexity": float,
                "top_k_predictions": [(word, prob), ...],
            },
            ...
        }
    """
    results = {}

    for name, model in models.items():
        try:
            # Compute perplexity on test set
            ppl = compute_perplexity(model, test_tokens)

            # Get predictions for the given context
            preds = model.predict_next(context_tokens, top_k=5)

            results[name] = {
                "perplexity": round(ppl, 2),
                "top_predictions": preds,
            }
            logger.info("Model '%s': perplexity=%.2f", name, ppl)
        except Exception as e:
            logger.warning("Failed to evaluate model '%s': %s", name, e)
            results[name] = {"perplexity": float("inf"), "top_predictions": []}

    return results
