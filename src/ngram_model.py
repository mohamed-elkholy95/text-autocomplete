"""N-gram language model."""
import logging
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import RANDOM_SEED, MAX_NGRAM, MIN_NGRAM, TOP_K, MIN_FREQUENCY

logger = logging.getLogger(__name__)


class NGramModel:
    """N-gram language model with Laplace smoothing."""

    def __init__(self, n: int = 3, min_freq: int = MIN_FREQUENCY, seed: int = RANDOM_SEED) -> None:
        self.n = n
        self.min_freq = min_freq
        self.seed = seed
        self._ngram_counts: Dict[Tuple, int] = Counter()
        self._context_counts: Dict[Tuple, int] = Counter()
        self._vocab: set = set()
        self._vocab_size: int = 0
        self._is_fitted = False

    def fit(self, tokens: List[str]) -> "NGramModel":
        """Train on token list."""
        self._vocab = set(tokens)
        self._vocab_size = len(self._vocab)
        for n in range(MIN_NGRAM, self.n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                self._ngram_counts[ngram] += 1
                context = ngram[:-1] if len(ngram) > 1 else ()
                self._context_counts[context] += 1
        # Filter by min frequency
        self._ngram_counts = Counter({k: v for k, v in self._ngram_counts.items() if v >= self.min_freq})
        self._is_fitted = True
        logger.info("NGramModel fitted: n=%d, vocab=%d, unique ngrams=%d", self.n, self._vocab_size, len(self._ngram_counts))
        return self

    def predict_next(self, context: List[str], top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Predict next tokens given context.

        Returns:
            List of (token, probability) sorted descending.
        """
        if not self._is_fitted:
            return [("<UNK>", 0.0)]

        ctx = tuple(context[-(self.n-1):]) if self.n > 1 else ()
        candidates = Counter()
        for ngram, count in self._ngram_counts.items():
            if ngram[:-1] == ctx:
                candidates[ngram[-1]] += count

        if not candidates:
            # Backoff: try shorter context
            if len(ctx) > 0:
                return self.predict_next(list(ctx[1:]), top_k)
            # Unigram fallback
            unigrams = Counter(t[-1] for t in self._ngram_counts if len(t) == 1)
            total = sum(unigrams.values()) or 1
            return [(w, c/total) for w, c in unigrams.most_common(top_k)]

        total = sum(candidates.values())
        results = [(w, c / total) for w, c in candidates.most_common(top_k)]
        return results

    def perplexity(self, tokens: List[str]) -> float:
        """Compute perplexity on a token sequence."""
        if not self._is_fitted:
            return float("inf")
        log_prob = 0.0
        count = 0
        for i in range(self.n - 1, len(tokens)):
            ctx = tokens[i - self.n + 1:i]
            preds = self.predict_next(ctx)
            token = tokens[i]
            prob = next((p for w, p in preds if w == token), 1e-10)
            log_prob += log(prob)
            count += 1
        avg = log_prob / max(count, 1)
        return np.exp(-avg)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
