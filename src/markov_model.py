"""
Markov Chain Language Model
============================

A first-order Markov chain is the simplest probabilistic language model.
Unlike n-gram models which use varying context lengths, a Markov chain
explicitly models state transitions between words.

EDUCATIONAL CONTEXT:
-------------------
Markov chains are named after Andrey Markov (1856–1922), who proved that
for certain types of random processes, the future depends only on the
present state — not on the entire history. This "memoryless property"
is called the Markov Property.

For text:
  P(word_n | word_1, word_2, ..., word_{n-1}) ≈ P(word_n | word_{n-1})

This means we only need to know the PREVIOUS word to predict the NEXT one.
It's a special case of bigram models but explicitly framed as a transition
graph — which makes the probability structure very clear and visual.

WHY THIS MATTERS:
- Markov chains are the foundation of many real-world systems: autocomplete,
  page ranking (PageRank IS a Markov chain!), weather prediction, finance
- Understanding them deeply prepares you for Hidden Markov Models (HMMs)
  used in speech recognition and bioinformatics
- The transition matrix visualization makes probability intuitive

ALGORITHM:
----------
1. Build a transition matrix: rows = current word, columns = next word
2. Each cell T[i][j] = count(word_j follows word_i) / count(word_i appears)
3. To predict: look up row for the current word, return top-k entries
4. To generate: sample from the transition distribution repeatedly
"""

import logging
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import RANDOM_SEED, TOP_K

logger = logging.getLogger(__name__)


class MarkovChainModel:
    """First-order Markov chain language model for text autocomplete.

    This model treats each word as a "state" and learns the probability
    of transitioning from one state (word) to another based on training data.

    The transition matrix P has shape (vocab_size × vocab_size) where:
        P[i][j] = P(next_word = j | current_word = i)

    For unseen transitions, we use Laplace (add-1) smoothing to avoid
    zero probabilities, which would make the perplexity infinite.

    Attributes:
        n_transitions: Total number of word-to-word transitions seen during training.
        vocab_size: Number of unique words in the training corpus.
        start_words: Counter of words that appear at the beginning of sentences.
            Useful for generating text from scratch.
    """

    def __init__(self, smoothing: float = 1.0, seed: int = RANDOM_SEED) -> None:
        """Initialize the Markov chain model.

        Args:
            smoothing: Laplace smoothing parameter (add-k smoothing).
                Higher values = more uniform distribution (less confident).
                Default 1.0 prevents zero probabilities for unseen transitions.
                Setting to 0.0 gives raw Maximum Likelihood Estimation (MLE).
            seed: Random seed for reproducible text generation.
        """
        self.smoothing = smoothing
        self.seed = seed

        # word_to_idx maps each unique word to a matrix row/column index
        self._word_to_idx: Dict[str, int] = {}
        self._idx_to_word: Dict[int, str] = {}

        # The core data structure: transitions[current_word][next_word] = count
        self._transitions: Dict[str, Counter] = defaultdict(Counter)
        self._word_counts: Counter = Counter()  # how often each word appears
        self._start_words: Counter = Counter()  # words that start sentences

        self._n_transitions: int = 0
        self._vocab_size: int = 0
        self._is_fitted: bool = False

    def fit(self, tokens: List[str]) -> "MarkovChainModel":
        """Train the Markov chain on a sequence of tokens.

        The training process builds a transition table by counting how often
        each word follows every other word. For example, if "learning" follows
        "machine" 5 times out of 8 total occurrences of "machine", then:
            P(learning | machine) = (5 + smoothing) / (8 + smoothing × vocab_size)

        Args:
            tokens: List of string tokens (already lowercased and split).
                The order matters — consecutive tokens form transitions.

        Returns:
            self (for method chaining, e.g., model.fit(tokens).predict(...))
        """
        logger.info("Training Markov chain on %d tokens...", len(tokens))

        # Build vocabulary — assign each unique word a matrix index
        vocab = sorted(set(tokens))
        self._word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self._idx_to_word = {idx: word for word, idx in self._word_to_idx.items()}
        self._vocab_size = len(vocab)

        # Count transitions: for each consecutive pair (tokens[i], tokens[i+1]),
        # increment the transition count
        for i in range(len(tokens) - 1):
            current_word = tokens[i]
            next_word = tokens[i + 1]

            self._transitions[current_word][next_word] += 1
            self._word_counts[current_word] += 1
            self._n_transitions += 1

            # Track sentence-starting words (heuristic: word after a period)
            if i > 0 and tokens[i - 1] in (".", "!", "?"):
                self._start_words[current_word] += 1

        self._is_fitted = True
        logger.info(
            "Markov chain trained: vocab=%d, transitions=%d, smoothing=%.1f",
            self._vocab_size, self._n_transitions, self.smoothing,
        )
        return self

    def _get_transition_probs(self, word: str) -> List[Tuple[str, float]]:
        """Get the probability distribution over next words given current word.

        Uses Laplace (add-k) smoothing:
            P(next | current) = (count(current → next) + k) / (count(current) + k × V)

        Where:
            k = smoothing parameter (default 1.0)
            V = vocabulary size

        WHY SMOOTHING?
        Without smoothing, any unseen transition gets probability 0.
        This is problematic because:
        1. Multiplying by 0 during perplexity calculation → infinite perplexity
        2. The model can never predict a word it hasn't seen follow the current word
        3. Log(0) is undefined, breaking logarithmic probability calculations

        Args:
            word: The current word (the "state" we're transitioning from).

        Returns:
            List of (next_word, probability) pairs, sorted by probability descending.
        """
        if word not in self._transitions:
            # This word was never seen — return uniform distribution over vocab
            # This is a reasonable fallback: any word is equally likely
            return [(w, 1.0 / self._vocab_size) for w in self._idx_to_word.values()]

        transitions = self._transitions[word]
        total_count = self._word_counts[word]

        # Laplace smoothing: add k to every possible transition
        # This "spreads" probability mass to unseen transitions
        smoothed_total = total_count + self.smoothing * self._vocab_size

        # Build probability list for words we've actually seen follow this word
        probs = []
        for next_word, count in transitions.items():
            # Smoothed probability = (observed_count + k) / (total + k × V)
            prob = (count + self.smoothing) / smoothed_total
            probs.append((next_word, prob))

        # Sort by probability descending so the most likely predictions come first
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs

    def predict_next(self, context: List[str], top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Predict the most likely next word(s) given a context.

        For a first-order Markov chain, we only use the LAST word in the context.
        All earlier words are ignored because of the Markov property — the future
        depends only on the present state.

        THINK OF IT LIKE THIS:
        Imagine you're playing a word-guessing game. Your partner says
        "The weather today is...". A Markov chain only looks at "is" to predict
        what comes next. It doesn't care about "The weather today".

        This seems limiting (and it is!), but:
        - It's computationally very fast — O(1) lookup per prediction
        - It's the building block for higher-order models
        - It works surprisingly well for common word pairs

        Args:
            context: List of preceding words. Only the last word is used.
            top_k: Number of predictions to return.

        Returns:
            List of (word, probability) tuples, sorted by probability descending.
        """
        if not self._is_fitted:
            return [("<UNK>", 0.0)]

        # The Markov property: only the current state matters
        current_word = context[-1] if context else None

        if not current_word or current_word not in self._transitions:
            # Unknown word or empty context — fall back to most common words overall
            # This is the "stationary distribution" fallback
            return self._most_common_words(top_k)

        probs = self._get_transition_probs(current_word)
        return probs[:top_k]

    def _most_common_words(self, top_k: int) -> List[Tuple[str, float]]:
        """Return the most frequently occurring words as a fallback prediction.

        When we can't make a context-based prediction (unknown word, empty context),
        we fall back to the global word frequency distribution. This is equivalent
        to a zeroth-order model (unigram model).

        The probabilities are normalized so they sum to 1.0.
        """
        total = sum(self._word_counts.values()) or 1
        return [(word, count / total) for word, count in self._word_counts.most_common(top_k)]

    def perplexity(self, tokens: List[str]) -> float:
        """Compute perplexity on a token sequence.

        PERPLEXITY EXPLAINED:
        Perplexity measures how "surprised" the model is by the test data.
        - Low perplexity = the model predicts the test tokens well (not surprised)
        - High perplexity = the model struggles to predict (very surprised)
        - A perfect model that always predicts correctly has perplexity = 1

        Mathematically:
            PPL = exp(-1/N × Σ log P(token_i | token_{i-1}))

        Where N is the number of predicted tokens. The exponential transforms
        the average log-probability into a more intuitive scale.

        BENCHMARKS (rough guide):
        - PPL < 50: Good model for the domain
        - PPL 50–200: Decent, captures some patterns
        - PPL > 200: Weak, barely better than random

        Args:
            tokens: Token sequence to evaluate on.

        Returns:
            Perplexity score (float). Lower is better. Returns infinity if not fitted.
        """
        if not self._is_fitted or len(tokens) < 2:
            return float("inf")

        total_log_prob = 0.0
        n_predictions = 0

        for i in range(1, len(tokens)):
            # Get probability of tokens[i] given tokens[i-1]
            current = tokens[i - 1]
            target = tokens[i]
            probs = self._get_transition_probs(current)

            # Find the probability assigned to the actual next word
            prob = next((p for word, p in probs if word == target), 1e-10)

            # Accumulate log probability (we use log to prevent underflow
            # when multiplying many small probabilities together)
            total_log_prob += log(prob)
            n_predictions += 1

        # Average log-probability per token
        avg_log_prob = total_log_prob / max(n_predictions, 1)

        # Exponentiate to get perplexity
        return float(np.exp(-avg_log_prob))

    def generate_text(self, start_word: Optional[str] = None, max_length: int = 20, temperature: float = 1.0, seed: Optional[int] = None) -> str:
        """Generate text by sampling from the Markov chain.

        TEXT GENERATION WITH MARKOV CHAINS:
        This is the classic "walk through the transition graph" algorithm:
        1. Start at a word (provided or randomly chosen)
        2. Look at the transition probabilities from that word
        3. Sample the next word from those probabilities
        4. Repeat until max_length or we hit a dead end

        TEMPERATURE CONTROLS RANDOMNESS:
        - temperature < 1.0: More focused/deterministic (peaked distribution)
        - temperature = 1.0: Original probabilities (natural diversity)
        - temperature > 1.0: More random/creative (flatter distribution)

        Mathematically, we divide logits by temperature before softmax:
            P(x) = softmax(logits / temperature)

        Args:
            start_word: Word to start generation from. If None, picks randomly
                from common starting words.
            max_length: Maximum number of words to generate.
            temperature: Controls randomness. 1.0 = normal, <1 = focused, >1 = creative.

        Returns:
            Generated text string.
        """
        if not self._is_fitted:
            return ""

        gen_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(gen_seed)

        # Choose starting word
        if start_word is None:
            # Pick from words that commonly start sentences
            if self._start_words:
                start_words = list(self._start_words.elements())
                word = rng.choice(start_words)
            else:
                # Fallback: pick from most common words
                word = self._word_counts.most_common(1)[0][0]
        else:
            word = start_word

        generated = [word]

        for _ in range(max_length - 1):
            if word not in self._transitions:
                break  # Dead end — no transitions from this word

            # Get transition probabilities for seen transitions only.
            # NOTE: These may NOT sum to 1.0 because we only return transitions
            # that were actually observed (not the full vocabulary distribution
            # that Laplace smoothing would give). We normalize below.
            probs = self._get_transition_probs(word)

            # Apply temperature: reshape the distribution
            # Lower temperature = sharper peaks (more predictable)
            # Higher temperature = flatter distribution (more random)
            words = [w for w, _ in probs]
            raw_probs = np.array([p for _, p in probs])

            if temperature != 1.0:
                # Convert to log-space, divide by temperature, back to prob-space
                # This is the standard "temperature sampling" technique
                log_probs = np.log(raw_probs + 1e-10) / temperature
                raw_probs = np.exp(log_probs - np.max(log_probs))  # numerical stability

            # Always normalize to ensure probabilities sum to 1.0 for sampling.
            # This is necessary because _get_transition_probs only returns
            # observed transitions, not the full Laplace-smoothed distribution.
            raw_probs = raw_probs / raw_probs.sum()

            # Sample the next word from the (possibly temperature-adjusted) distribution
            word = rng.choice(words, p=raw_probs)
            generated.append(word)

        return " ".join(generated)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique words in the vocabulary."""
        return self._vocab_size

    @property
    def n_transitions(self) -> int:
        """Return the total number of word-to-word transitions observed."""
        return self._n_transitions

    def save(self, path: str) -> None:
        """Serialize the trained Markov chain to disk using JSON.

        WHAT WE SAVE:
        The transition counts (not probabilities) — so we can reconstruct
        the exact same probability distribution on load, including the
        smoothing parameter. Saving counts instead of probabilities also
        means we could merge multiple saved models by adding their counts.

        Args:
            path: File path to save the model to.
        """
        import json
        from pathlib import Path

        if not self._is_fitted:
            raise RuntimeError("Cannot save an untrained model. Call fit() first.")

        data = {
            "model_type": "markov",
            "smoothing": self.smoothing,
            "seed": self.seed,
            "word_to_idx": self._word_to_idx,
            "transitions": {
                word: dict(counts) for word, counts in self._transitions.items()
            },
            "word_counts": dict(self._word_counts),
            "start_words": dict(self._start_words),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Markov model saved to %s (vocab=%d)", path, self._vocab_size)

    @classmethod
    def load(cls, path: str) -> "MarkovChainModel":
        """Load a previously saved Markov chain model from disk.

        Reconstructs all internal state: transition counts, vocabulary mapping,
        word frequencies, and sentence-starting words.

        Args:
            path: File path to load the model from.

        Returns:
            A fitted MarkovChainModel instance ready for predictions.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            ValueError: If the file doesn't contain a valid Markov model.
        """
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("model_type") != "markov":
            raise ValueError(f"Expected model_type 'markov', got '{data.get('model_type')}'")

        model = cls(smoothing=data["smoothing"], seed=data["seed"])
        model._word_to_idx = data["word_to_idx"]
        model._idx_to_word = {int(v): k for k, v in data["word_to_idx"].items()}
        model._vocab_size = len(model._word_to_idx)

        model._transitions = defaultdict(Counter, {
            word: Counter(counts) for word, counts in data["transitions"].items()
        })
        model._word_counts = Counter(data["word_counts"])
        model._start_words = Counter(data["start_words"])
        model._n_transitions = sum(model._word_counts.values())
        model._is_fitted = True

        logger.info("Markov model loaded from %s (vocab=%d)", path, model._vocab_size)
        return model

    def get_top_transitions(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get the most likely words to follow a given word.

        This is useful for visualization — you can show "after word X,
        the model expects: Y (45%), Z (20%), ..."

        Args:
            word: The word to look up transitions for.
            top_k: Number of transitions to return.

        Returns:
            List of (next_word, probability) tuples.
        """
        if not self._is_fitted or word not in self._transitions:
            return []
        return self._get_transition_probs(word)[:top_k]
