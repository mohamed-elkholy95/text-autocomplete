"""Text data loading and generation."""
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
from src.config import RANDOM_SEED, DATA_DIR

logger = logging.getLogger(__name__)

SAMPLE_TEXTS = [
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
]

def load_sample_data() -> str:
    return " ".join(SAMPLE_TEXTS * 5)

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with punctuation splitting."""
    import re
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens

def build_ngrams(tokens: List[int], n: int) -> List[Tuple]:
    """Build n-gram tuples from token IDs."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def generate_synthetic_data(n_sentences: int = 100, seed: int = RANDOM_SEED) -> str:
    rng = np.random.default_rng(seed)
    selected = rng.choice(SAMPLE_TEXTS, size=n_sentences).tolist()
    return " ".join(selected)
