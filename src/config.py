"""
Configuration Module
=====================

Central configuration for the text autocomplete project.
All tunable parameters, paths, and settings live here.

EDUCATIONAL CONTEXT:
-------------------
Centralizing configuration is a best practice because:
1. It makes hyperparameter tuning easy — change one file, not twenty
2. It prevents "magic numbers" scattered throughout the code
3. It makes the project reproducible — anyone can see and modify settings
4. It separates WHAT the model does (code) from HOW it's configured (params)

HYPERPARAMETERS vs PARAMETERS:
- Model PARAMETERS: Learned during training (weights, biases, counts)
- HYPERPARAMETERS: Set before training (learning rate, n-gram order, etc.)
  These are the "knobs and dials" you adjust to improve performance.
"""

import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# General Settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42  # Reproducibility: same seed → same results

# ---------------------------------------------------------------------------
# N-gram Model Settings
# ---------------------------------------------------------------------------
# N-gram order determines how many previous words the model considers.
# Higher n = more context, but needs exponentially more data.
#
# WHY MAX 4? Beyond 4-grams, the data becomes extremely sparse.
# With a vocabulary of 10,000 words:
#   - 4-grams: up to 10,000^4 = 10^16 possible combinations
#   - Most will never appear in training → unreliable probability estimates
MAX_NGRAM = 4
MIN_NGRAM = 1  # Unigrams are always included as a fallback

# Only keep n-grams that appear at least this many times.
# This is a form of smoothing — rare n-grams have unreliable statistics.
MIN_FREQUENCY = 2

# How many predictions to return for autocomplete suggestions
TOP_K = 5

# Maximum context window for predictions (in tokens).
# Longer context gives better predictions but is slower.
CONTEXT_WINDOW = 50

# ---------------------------------------------------------------------------
# Neural Network Settings
# ---------------------------------------------------------------------------
NEURAL_CONFIG = {
    # Embedding dimension: maps each word to a dense vector of this size.
    # Higher = more expressive but more parameters (risk of overfitting).
    "embed_dim": 64,

    # LSTM hidden dimension: size of the internal state.
    # This is the "memory" of the LSTM — larger = remembers more patterns.
    "hidden_dim": 128,

    # Number of LSTM layers stacked on top of each other.
    # More layers = deeper model, but harder to train.
    "num_layers": 2,

    # Dropout probability: randomly zeroes neurons during training.
    # Prevents overfitting by forcing the model to not rely on any single neuron.
    # 0.0 = no dropout, 0.5 = heavy dropout (half the neurons zeroed).
    "dropout": 0.2,

    # Training hyperparameters
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 10,

    # Sequence length for training: how many tokens per training example.
    # Longer = more context, but needs more memory.
    "seq_len": 20,
}

# ---------------------------------------------------------------------------
# Markov Chain Settings
# ---------------------------------------------------------------------------
# Laplace smoothing parameter for the Markov chain.
# See markov_model.py for detailed explanation.
MARKOV_SMOOTHING = 1.0

# ---------------------------------------------------------------------------
# Beam Search Settings
# ---------------------------------------------------------------------------
BEAM_CONFIG = {
    # Number of parallel hypotheses to maintain.
    # beam_width=1 is equivalent to greedy decoding.
    "beam_width": 5,

    # Maximum number of tokens to generate beyond the input.
    "max_length": 10,

    # Length penalty exponent (see beam_search.py for details).
    # < 1.0 favors longer sequences, > 1.0 favors shorter sequences.
    "length_penalty": 0.6,
}

# ---------------------------------------------------------------------------
# API Settings
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"  # Listen on all network interfaces
API_PORT = 8010       # Port number for the FastAPI server
API_TITLE = "Text Autocomplete API"
API_VERSION = "2.2.0"  # keep in sync with src.__version__

# ---------------------------------------------------------------------------
# Streamlit Theme
# ---------------------------------------------------------------------------
# Dark theme colors for consistent UI appearance.
# These are applied to the Streamlit app for a professional look.
STREAMLIT_THEME: Dict[str, str] = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#0e1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#ffffff",
}
