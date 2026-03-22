"""
Enhanced Data Loader
=====================

This module provides comprehensive text data loading, preprocessing, and
corpus management for the text autocomplete system.

EDUCATIONAL CONTEXT:
-------------------
The quality and quantity of training data is often MORE important than the
choice of model architecture. This module demonstrates several key concepts:

1. CORPUS DESIGN: A diverse corpus captures different writing styles,
   vocabulary, and grammatical patterns. This variety helps the model
   generalize better.

2. TEXT PREPROCESSING: Raw text needs cleaning before use:
   - Lowercasing: "Machine" and "machine" should be treated the same
   - Punctuation handling: Split into separate tokens or remove
   - Stop words: Optional removal of common words (the, is, a, ...)
   - Rare word handling: Words seen only once are unreliable for statistics

3. TRAIN/TEST SPLIT: Never evaluate on training data! The split ensures
   we measure how well the model GENERALIZES, not memorizes.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import RANDOM_SEED, DATA_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample Texts — Diverse Corpus
# ---------------------------------------------------------------------------
# A good training corpus should cover different domains, sentence structures,
# and vocabulary. Our corpus includes:
# - Machine learning and AI concepts (the main domain)
# - Software engineering and programming
# - Data science and statistics
# - General technology topics
# This diversity helps the model learn a broader range of word associations.

SAMPLE_TEXTS = [
    # --- Machine Learning & AI ---
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "Deep learning uses neural networks with multiple layers to learn representations of data.",
    "Natural language processing enables computers to understand and generate human language.",
    "The transformer architecture has revolutionized natural language processing and computer vision.",
    "Reinforcement learning trains agents to make decisions by maximizing cumulative rewards.",
    "Transfer learning allows models trained on one task to be applied to related tasks.",
    "Large language models are trained on massive datasets using self-supervised learning.",
    "The attention mechanism allows models to focus on relevant parts of the input sequence.",
    "Gradient descent is an optimization algorithm used to minimize the loss function.",
    "Batch normalization helps stabilize and accelerate the training of neural networks.",
    "Convolutional neural networks are particularly effective for image classification tasks.",
    "Recurrent neural networks process sequential data by maintaining a hidden state.",
    "Generative adversarial networks consist of a generator and a discriminator network.",
    "BERT uses a masked language modeling objective during pre-training.",
    "GPT models use autoregressive language modeling to generate text sequentially.",
    "The encoder-decoder architecture is used in sequence-to-sequence tasks like translation.",
    "Word embeddings map words to dense vector representations in continuous space.",
    "Cross-validation helps assess model performance and prevent overfitting.",
    "Hyperparameter tuning involves selecting optimal values for model configuration parameters.",
    "Feature engineering transforms raw data into meaningful representations for machine learning models.",

    # --- Software Engineering & Programming ---
    "Version control systems like Git help developers track changes and collaborate on code.",
    "Agile methodology promotes iterative development with frequent feedback cycles.",
    "Test-driven development encourages writing tests before implementing features.",
    "Design patterns provide reusable solutions to common software design problems.",
    "Microservices architecture decomposes applications into small independent services.",
    "Containerization with Docker enables consistent deployment across environments.",
    "Continuous integration and deployment automate the testing and release process.",
    "Code review improves software quality by having peers examine changes before merging.",

    # --- Data Science & Statistics ---
    "Statistical significance testing helps determine whether results are meaningful or due to chance.",
    "Data visualization transforms complex datasets into intuitive graphical representations.",
    "Feature selection reduces model complexity by identifying the most informative variables.",
    "Anomaly detection identifies data points that deviate significantly from expected patterns.",
    "Time series analysis examines data collected at regular intervals over time.",
    "Dimensionality reduction techniques like PCA help visualize high-dimensional data.",

    # --- Technology & General ---
    "Cloud computing provides on-demand access to computing resources over the internet.",
    "Edge computing processes data closer to where it is generated, reducing latency.",
    "Blockchain technology provides a decentralized and tamper-proof ledger system.",
    "The Internet of Things connects everyday devices to enable smart automation.",
    "Cybersecurity involves protecting systems and networks from digital attacks.",
    "Quantum computing leverages quantum mechanics to solve certain problems exponentially faster.",
]


def load_sample_data() -> str:
    """Load and return the full sample corpus as a single string.

    The corpus is repeated 5 times to create a larger training set.
    Repetition is acceptable here because:
    1. It gives the n-gram model enough examples to estimate probabilities
    2. The model learns from word co-occurrence patterns, not exact memorization
    3. In production, you'd use a much larger real-world corpus instead

    Returns:
        Concatenated corpus string with spaces between sentences.
    """
    return " ".join(SAMPLE_TEXTS * 5)


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Convert text into a list of tokens.

    TOKENIZATION EXPLAINED:
    Tokenization is the process of splitting text into meaningful units (tokens).
    It's the first step in almost every NLP pipeline.

    Our strategy:
    1. Lowercase everything — ensures "The" and "the" map to the same token
    2. Use regex to split into words and punctuation — keeps punctuation as
       separate tokens, which is useful because punctuation carries meaning
       (e.g., "?" indicates a question, "," indicates a pause/list)
    3. Optionally remove stopwords — common words like "the", "is", "a" that
       don't carry much semantic meaning but dominate the frequency counts

    WHY KEEP PUNCTUATION?
    In autocomplete, knowing that a period just appeared is useful because
    the next word is likely the start of a new sentence (capitalized word
    or a common sentence starter like "The", "This", etc.).

    Args:
        text: Raw text string to tokenize.
        remove_stopwords: If True, filter out common English stopwords.

    Returns:
        List of string tokens.
    """
    # \w+ matches word characters (letters, digits, underscores)
    # [^\w\s] matches punctuation and special characters
    # The result: each word and each punctuation mark becomes a separate token
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())

    if remove_stopwords:
        tokens = _remove_stopwords(tokens)

    return tokens


def _remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove common English stopwords from a token list.

    STOPWORDS are extremely common words that appear in virtually every text.
    They're often removed in information retrieval and classification tasks
    because they don't carry much discriminative meaning.

    Common stopwords: the, is, at, which, on, a, an, and, or, but, ...
    For language modeling, we usually KEEP stopwords because the model needs
    to learn natural word order (e.g., "the cat" is much more natural than "cat the").

    Args:
        tokens: Token list to filter.

    Returns:
        Token list with stopwords removed.
    """
    # Minimal stopwords list — not exhaustive, just the most common ones
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "and",
        "but", "or", "nor", "not", "so", "yet", "it", "its", "this", "that",
    }
    return [t for t in tokens if t not in stopwords]


def build_ngrams(tokens: List[int], n: int) -> List[Tuple]:
    """Build n-gram tuples from a list of token IDs.

    An n-gram is a contiguous sequence of n items from a text.
    For tokens [A, B, C, D, E] with n=3:
    - Trigrams: (A,B,C), (B,C,D), (C,D,E)

    N-grams capture local word co-occurrence patterns:
    - Unigrams (n=1): Individual word frequencies → "machine" appears 15 times
    - Bigrams (n=2): Word pair frequencies → "machine learning" appears 8 times
    - Trigrams (n=3): Three-word patterns → "machine learning is" appears 5 times

    Higher n captures more context but requires exponentially more data
    to get reliable probability estimates (the "sparse data problem").

    Args:
        tokens: List of token IDs (integers).
        n: The n-gram order (length of each tuple).

    Returns:
        List of n-gram tuples.
    """
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def generate_synthetic_data(n_sentences: int = 100, seed: int = RANDOM_SEED) -> str:
    """Generate synthetic text data by sampling from the corpus.

    This creates a training set by randomly selecting sentences from our
    corpus with replacement. The random seed ensures reproducibility.

    WHY SYNTHETIC DATA?
    - Quick prototyping without downloading large datasets
    - Reproducible experiments (same seed → same data)
    - Testing edge cases (very small or very large datasets)

    In production, replace this with real-world text corpora like:
    - Wikipedia dumps
    - Project Gutenberg books
    - Common Crawl web text
    - Domain-specific corpora

    Args:
        n_sentences: Number of sentences to sample.
        seed: Random seed for reproducibility.

    Returns:
        Generated text string.
    """
    rng = np.random.default_rng(seed)
    selected = rng.choice(SAMPLE_TEXTS, size=n_sentences).tolist()
    return " ".join(selected)


def train_test_split(
    tokens: List[str],
    test_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str]]:
    """Split a token list into training and test sets.

    WHY TRAIN/TEST SPLIT?
    The fundamental principle of machine learning evaluation: never test on
    training data. If you do, you're measuring memorization, not learning.

    The test set simulates "unseen data" — words and sequences the model
    hasn't encountered during training. A good model generalizes well to
    this unseen data.

    COMMON SPLIT RATIOS:
    - 80/20: Most common, good balance for medium datasets
    - 70/30: Use when you have lots of data
    - 90/10: Use when data is scarce (but test set might be too small)

    We use random shuffling before splitting to ensure both sets represent
    the full distribution of the data. Without shuffling, if the data is
    ordered (e.g., all ML sentences first, then all stats sentences),
    the test set would only contain statistics sentences.

    Args:
        tokens: Full token list to split.
        test_ratio: Fraction of data for the test set (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_tokens, test_tokens).
    """
    rng = np.random.default_rng(seed)

    # Create a random permutation of indices
    indices = rng.permutation(len(tokens))

    # Calculate the split point
    test_size = int(len(tokens) * test_ratio)

    # Split: first (1 - test_ratio) for training, rest for testing
    test_indices = set(indices[:test_size].tolist())
    train_tokens = [tokens[i] for i in range(len(tokens)) if i not in test_indices]
    test_tokens = [tokens[i] for i in range(len(tokens)) if i in test_indices]

    logger.info(
        "Train/test split: train=%d tokens (%.0f%%), test=%d tokens (%.0f%%)",
        len(train_tokens), (1 - test_ratio) * 100,
        len(test_tokens), test_ratio * 100,
    )

    return train_tokens, test_tokens


def get_corpus_stats(tokens: List[str]) -> Dict[str, int]:
    """Compute basic statistics about a token corpus.

    These statistics help understand the data and diagnose potential issues:
    - Very small vocab → model won't have much to work with
    - Very high type-token ratio → very diverse text, hard to learn patterns
    - Very low type-token ratio → repetitive text, might overfit

    Args:
        tokens: List of tokens.

    Returns:
        Dictionary with corpus statistics:
        - total_tokens: Total number of tokens (with repetitions)
        - unique_tokens: Number of distinct token types
        - avg_token_length: Average character length per token
        - min_token_length: Length of shortest token
        - max_token_length: Length of longest token
    """
    if not tokens:
        return {"total_tokens": 0, "unique_tokens": 0, "avg_token_length": 0,
                "min_token_length": 0, "max_token_length": 0}

    lengths = [len(t) for t in tokens]
    return {
        "total_tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "avg_token_length": round(sum(lengths) / len(lengths), 2),
        "min_token_length": min(lengths),
        "max_token_length": max(lengths),
    }
