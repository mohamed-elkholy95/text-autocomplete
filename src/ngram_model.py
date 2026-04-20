"""
N-gram Language Model with Enhanced Smoothing
==============================================

N-gram models are the foundation of statistical language modeling. They estimate
the probability of a word based on the preceding (n-1) words using simple
counting statistics from a training corpus.

EDUCATIONAL CONTEXT:
-------------------
THE MARKOV ASSUMPTION:
    The probability of a word depends only on the previous (n-1) words.
    P(w_n | w_1, ..., w_{n-1}) ≈ P(w_n | w_{n-2}, w_{n-1})  [for trigrams]

This is a simplification — in reality, words depend on the entire context.
But it works surprisingly well in practice and is computationally efficient.

WHY N-GRAMS MATTER:
- They're fast: O(1) lookup per prediction (just a dictionary lookup)
- They're interpretable: you can inspect the counts and see WHY the model
  predicts what it does
- They're the baseline: every language model paper compares against n-grams
- They're foundational: understanding n-grams helps you understand more
  complex models like transformers

THE SPARSE DATA PROBLEM:
With a vocabulary of V words:
  - Bigrams: V^2 possible combinations
  - Trigrams: V^3 possible combinations
  - 4-grams: V^4 possible combinations

For V=10,000: trigrams = 10^12 possible combinations!
Most combinations will never appear in training → we need smoothing.

SMOOTHING TECHNIQUES (implemented here):
1. Laplace (Add-1) Smoothing: Add 1 to every count → no zero probabilities
2. Backoff: If an n-gram wasn't seen, fall back to (n-1)-gram
3. Interpolation: Mix probabilities from different n-gram orders
"""

import logging
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import RANDOM_SEED, MIN_NGRAM, TOP_K, MIN_FREQUENCY

logger = logging.getLogger(__name__)


class NGramModel:
    """N-gram language model with backoff smoothing.

    This model builds probability tables for n-grams of orders 1 through n.
    For prediction, it uses the highest-order n-gram available and falls back
    to lower orders when the current context wasn't seen during training.

    Example:
        Model trained on: "the cat sat on the mat the cat sat"
        Query: predict_next(["the", "cat"])
        → Checks trigrams starting with ("the", "cat") → finds ("the", "cat", "sat")
        → Returns: [("sat", 1.0)]  # 100% of the time, "sat" follows "the cat"

    Attributes:
        n: Maximum n-gram order (e.g., 3 for trigrams).
        min_freq: Minimum frequency for an n-gram to be included.
        vocab_size: Number of unique words in the training data.
    """

    def __init__(self, n: int = 3, min_freq: int = MIN_FREQUENCY, seed: int = RANDOM_SEED) -> None:
        """Initialize the n-gram model.

        Args:
            n: Maximum n-gram order. Common values:
                - 1: Unigram (word frequencies only, no context)
                - 2: Bigram (one word of context)
                - 3: Trigram (two words of context) — most common default
                - 4: 4-gram (three words of context, needs lots of data)
            min_freq: Minimum count for an n-gram to be stored.
                Higher values remove rare n-grams (noise reduction).
                Lower values keep more data but increase memory usage.
            seed: Random seed for reproducibility.
        """
        self.n = n
        self.min_freq = min_freq
        self.seed = seed

        # Core data structures:
        # _ngram_counts: Maps n-gram tuples to their occurrence counts
        #   e.g., {("the", "cat", "sat"): 5, ("cat", "sat", "on"): 3}
        self._ngram_counts: Dict[Tuple, int] = Counter()

        # _context_counts: Maps context (all but last word) to total count
        #   e.g., {("the", "cat"): 5, ("cat",): 12}
        self._context_counts: Dict[Tuple, int] = Counter()

        # Vocabulary: set of all unique words seen during training
        self._vocab: set = set()
        self._vocab_size: int = 0
        self._is_fitted = False

    def fit(self, tokens: List[str]) -> "NGramModel":
        """Train the n-gram model on a token sequence.

        Training is just counting! We go through the corpus and tally:
        1. How many times each n-gram appears (for probability estimation)
        2. How many times each context appears (for normalization)

        We build counts for ALL orders from 1 to n, so the model can
        fall back from trigrams to bigrams to unigrams if needed.

        Args:
            tokens: List of string tokens from the training corpus.

        Returns:
            self (for method chaining).
        """
        logger.info("Training %d-gram model on %d tokens...", self.n, len(tokens))

        # Build vocabulary
        self._vocab = set(tokens)
        self._vocab_size = len(self._vocab)

        # Count n-grams for all orders from MIN_NGRAM to n
        for order in range(MIN_NGRAM, self.n + 1):
            for i in range(len(tokens) - order + 1):
                ngram = tuple(tokens[i:i + order])
                self._ngram_counts[ngram] += 1

                # Context = everything except the last word
                # For unigrams, context is empty tuple ()
                context = ngram[:-1] if len(ngram) > 1 else ()
                self._context_counts[context] += 1

        # Filter out rare n-grams (below minimum frequency threshold).
        # This reduces noise from random co-occurrences that happen too
        # few times to give a stable probability estimate.
        before = len(self._ngram_counts)
        self._ngram_counts = Counter({
            k: v for k, v in self._ngram_counts.items() if v >= self.min_freq
        })
        removed = before - len(self._ngram_counts)
        if removed > 0:
            logger.info(
                "Filtered %d n-grams below frequency threshold (%d), kept %d",
                removed, self.min_freq, len(self._ngram_counts),
            )

        self._is_fitted = True
        logger.info(
            "N-gram model fitted: n=%d, vocab=%d, unique ngrams=%d",
            self.n, self._vocab_size, len(self._ngram_counts),
        )
        return self

    def predict_next(self, context: List[str], top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Predict the most likely next word(s) given a context.

        PREDICTION STRATEGY:
        1. Look up n-grams matching the context (last n-1 words)
        2. Count how often each candidate word follows this context
        3. Normalize to get probabilities (count / total_count)
        4. Return the top-k most probable words

        BACKOFF:
        If no n-grams match the current context, we "back off" to a shorter
        context by dropping the first word. For example:
        - Trigram context ("the", "cat") → no matches
        - Fall back to bigram context ("cat",) → might find matches
        - If still no matches → fall back to unigrams (overall word frequencies)

        This hierarchical approach ensures we always return predictions,
        even for contexts we've never seen before.

        Args:
            context: List of preceding words.
            top_k: Number of predictions to return.

        Returns:
            List of (word, probability) tuples, sorted descending by probability.
        """
        if not self._is_fitted:
            return [("<UNK>", 0.0)]

        # Extract the relevant context for this model's n-gram order
        # For a trigram model (n=3), context = last 2 words
        ctx = tuple(context[-(self.n - 1):]) if self.n > 1 else ()

        # Find all n-grams that start with our context
        candidates = Counter()
        for ngram, count in self._ngram_counts.items():
            if ngram[:-1] == ctx:
                candidates[ngram[-1]] += count

        if not candidates:
            # BACKOFF: no matches at this order → try shorter context
            if len(ctx) > 0:
                return self.predict_next(list(ctx[1:]), top_k)

            # FINAL FALLBACK: unigram frequencies
            unigrams = Counter(t[-1] for t in self._ngram_counts if len(t) == 1)
            total = sum(unigrams.values()) or 1
            return [(w, c / total) for w, c in unigrams.most_common(top_k)]

        # Normalize counts to probabilities
        total = sum(candidates.values())
        results = [(w, c / total) for w, c in candidates.most_common(top_k)]
        return results

    def predict_next_interpolated(
        self, context: List[str], top_k: int = TOP_K, lambdas: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """Predict next word using interpolation across all n-gram orders.

        INTERPOLATION vs BACKOFF:
        Backoff only uses lower-order models when higher-order ones fail.
        Interpolation ALWAYS combines all orders, weighted by lambda values.

        The interpolated probability is:
            P_interp(w|context) = λ₁·P_unigram(w) + λ₂·P_bigram(w|w₋₁) + λ₃·P_trigram(w|w₋₂,w₋₁)

        WHY INTERPOLATION IS BETTER:
        - Backoff is all-or-nothing: it ignores lower-order information when
          the higher-order context exists, even if it's unreliable (low count)
        - Interpolation hedges its bets: even if the trigram count is 1
          (unreliable), it still considers the more robust bigram and unigram
        - In practice, interpolation often beats backoff, especially on
          smaller datasets where higher-order counts are sparse

        SETTING LAMBDAS:
        - Equal weights [0.33, 0.33, 0.33]: treats all orders equally
        - Higher weight on higher orders [0.1, 0.3, 0.6]: trusts context more
        - The optimal lambdas can be learned from a held-out validation set
          using the Expectation-Maximization (EM) algorithm

        Args:
            context: List of preceding words.
            top_k: Number of predictions to return.
            lambdas: Interpolation weights for each order (1 to n).
                Must sum to 1.0 and have length n. If None, uses equal weights.

        Returns:
            List of (word, probability) tuples sorted by probability descending.
        """
        if not self._is_fitted:
            return [("<UNK>", 0.0)]

        # Default: equal weights across all n-gram orders
        if lambdas is None:
            lambdas = [1.0 / self.n] * self.n

        if len(lambdas) != self.n or abs(sum(lambdas) - 1.0) > 1e-6:
            raise ValueError(
                f"lambdas must have length {self.n} and sum to 1.0, "
                f"got length {len(lambdas)} summing to {sum(lambdas):.4f}"
            )

        # Collect interpolated probabilities for all vocabulary words
        interpolated: Dict[str, float] = {}

        for word in self._vocab:
            prob = 0.0

            for order_idx in range(self.n):
                order = order_idx + 1  # 1-gram, 2-gram, ..., n-gram
                lam = lambdas[order_idx]

                if order == 1:
                    # Unigram probability: count(word) / total_count
                    unigram_key = (word,)
                    count = self._ngram_counts.get(unigram_key, 0)
                    total = self._context_counts.get((), 0)
                    p = count / total if total > 0 else 0.0
                else:
                    # Higher-order: P(word | last (order-1) context words)
                    ctx = tuple(context[-(order - 1):]) if len(context) >= order - 1 else tuple(context)
                    if len(ctx) < order - 1:
                        p = 0.0
                    else:
                        ngram_key = ctx + (word,)
                        count = self._ngram_counts.get(ngram_key, 0)
                        ctx_count = self._context_counts.get(ctx, 0)
                        p = count / ctx_count if ctx_count > 0 else 0.0

                prob += lam * p

            if prob > 0:
                interpolated[word] = prob

        # Sort by probability and return top-k
        sorted_preds = sorted(interpolated.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:top_k]

    def perplexity(self, tokens: List[str]) -> float:
        """Compute perplexity on a token sequence.

        Perplexity measures how well the model predicts a sequence of tokens.
        It's the exponentiated average negative log-probability:

            PPL = exp(-1/N × Σ log P(w_i | w_{i-2}, w_{i-1}))

        WHERE LOWER IS BETTER. A perplexity of 10 means the model is, on
        average, choosing from about 10 equally likely candidates at each step.

        INTERPRETATION SCALE (rough guide):
        - PPL < 50:  Excellent — model captures language structure well
        - PPL 50-150: Good — model learned common patterns
        - PPL 150-500: Fair — model captures some patterns
        - PPL > 500:  Poor — model barely better than random

        IMPLEMENTATION NOTE:
        We compute the true conditional probability directly from counts
        (count(context + token) / count(context)) with backoff to shorter
        orders. Earlier revisions delegated to predict_next() which caps at
        top_k and silently floors ranks beyond the cap — a subtle bug that
        inflated perplexity whenever the true token ranked outside the top-k.

        Args:
            tokens: Token sequence to evaluate on.

        Returns:
            Perplexity score (float). Returns infinity if not fitted.
        """
        if not self._is_fitted:
            return float("inf")

        log_prob = 0.0
        count = 0

        for i in range(self.n - 1, len(tokens)):
            token = tokens[i]
            prob = self._conditional_prob(tokens[i - self.n + 1:i], token)
            log_prob += log(max(prob, 1e-10))
            count += 1

        avg = log_prob / max(count, 1)
        return float(np.exp(-avg))

    def _conditional_prob(self, context: List[str], token: str) -> float:
        """Return P(token | context) with stupid-backoff across orders.

        Walks from the highest-order context the model knows down to the
        unigram. Uses the first order where the context was actually seen
        at least once, so ranks beyond top_k are not floored.
        """
        ctx = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        while True:
            ctx_count = self._context_counts.get(ctx, 0)
            if ctx_count > 0:
                ngram_count = self._ngram_counts.get(ctx + (token,), 0)
                if ngram_count > 0:
                    return ngram_count / ctx_count
            if not ctx:
                break
            ctx = ctx[1:]

        # Final unigram fallback using surviving unigrams.
        total_unigrams = self._context_counts.get((), 0)
        unigram_count = self._ngram_counts.get((token,), 0)
        if total_unigrams > 0 and unigram_count > 0:
            return unigram_count / total_unigrams
        return 0.0

    def get_ngram_stats(self) -> Dict[int, int]:
        """Get the count of unique n-grams for each order.

        Useful for understanding the model's knowledge:
        - Many high-order n-grams → model has seen lots of patterns
        - Few high-order n-grams → data is sparse for this order

        Returns:
            Dictionary mapping n-gram order to count of unique n-grams.
        """
        order_counts: Dict[int, int] = defaultdict(int)
        for ngram in self._ngram_counts:
            order = len(ngram)
            order_counts[order] += 1
        return dict(sorted(order_counts.items()))

    def save(self, path: str) -> None:
        """Serialize the trained model to disk using JSON.

        MODEL PERSISTENCE:
        Saving a trained model means we don't have to retrain every time
        we restart the application. For n-gram models, we save:
        - The n-gram counts (our "learned parameters")
        - The vocabulary and configuration

        WHY JSON (not pickle)?
        - JSON is human-readable — you can inspect what the model learned
        - JSON is language-agnostic — other tools can read the model
        - Pickle has security risks (arbitrary code execution on load)
        - The model is just counts and strings — JSON handles this perfectly

        Args:
            path: File path to save the model to (e.g., "models/ngram_3.json").
        """
        import json
        from pathlib import Path

        if not self._is_fitted:
            raise RuntimeError("Cannot save an untrained model. Call fit() first.")

        # Convert tuple keys to strings for JSON serialization
        # JSON only supports string keys, but our n-gram keys are tuples
        # e.g., ("the", "cat") → "the||cat"
        serialized_ngrams = {
            "||".join(k): v for k, v in self._ngram_counts.items()
        }
        serialized_contexts = {
            "||".join(k) if k else "": v for k, v in self._context_counts.items()
        }

        data = {
            "model_type": "ngram",
            "n": self.n,
            "min_freq": self.min_freq,
            "seed": self.seed,
            "vocab": sorted(self._vocab),
            "ngram_counts": serialized_ngrams,
            "context_counts": serialized_contexts,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("N-gram model saved to %s (%d n-grams)", path, len(self._ngram_counts))

    @classmethod
    def load(cls, path: str) -> "NGramModel":
        """Load a previously saved n-gram model from disk.

        This reconstructs the model exactly as it was when saved, including
        all n-gram counts, vocabulary, and configuration. No retraining needed.

        Args:
            path: File path to load the model from.

        Returns:
            A fitted NGramModel instance ready for predictions.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            ValueError: If the file doesn't contain a valid n-gram model.
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("model_type") != "ngram":
            raise ValueError(f"Expected model_type 'ngram', got '{data.get('model_type')}'")

        model = cls(n=data["n"], min_freq=data["min_freq"], seed=data["seed"])

        # Reconstruct tuple keys from "word1||word2" strings
        model._ngram_counts = Counter({
            tuple(k.split("||")): v for k, v in data["ngram_counts"].items()
        })
        model._context_counts = Counter({
            tuple(k.split("||")) if k else (): v
            for k, v in data["context_counts"].items()
        })
        model._vocab = set(data["vocab"])
        model._vocab_size = len(model._vocab)
        model._is_fitted = True

        logger.info("N-gram model loaded from %s (n=%d, vocab=%d)", path, model.n, model._vocab_size)
        return model

    @property
    def vocab_size(self) -> int:
        """Return the number of unique words in the training vocabulary."""
        return self._vocab_size
