# Text Autocomplete — Architecture Guide

## System Overview

This project implements a text autocomplete system using multiple language
modeling approaches. Each model demonstrates a different strategy for the
fundamental NLP task: **given a sequence of words, predict what comes next.**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                             │
│  load_sample_data() → tokenize() → train_test_split()           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ tokens
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐  ┌─────────────┐  ┌────────────┐
   │  N-gram    │  │   Markov    │  │    LSTM    │
   │  Model     │  │   Chain     │  │   Neural   │
   │            │  │             │  │   Model    │
   │ Backoff +  │  │  Laplace    │  │  PyTorch   │
   │ Interpol.  │  │  Smoothing  │  │  Sequence  │
   └─────┬──────┘  └──────┬──────┘  └─────┬──────┘
         │                │                │
         └────────┬───────┘                │
                  ▼                        │
          ┌──────────────┐                 │
          │  Beam Search │◄────────────────┘
          │  Decoder     │
          └──────┬───────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│  CLI   │ │ FastAPI  │ │Streamlit │
│        │ │  REST    │ │Dashboard │
│ cli.py │ │  API     │ │          │
└────────┘ └──────────┘ └──────────┘
```

## Component Breakdown

### 1. Data Layer (`src/data_loader.py`)

**Responsibility:** Load, preprocess, and split text data.

| Function | Purpose | Educational Concept |
|----------|---------|-------------------|
| `load_sample_data()` | Load built-in corpus | Corpus design |
| `tokenize()` | Text → token list | Tokenization strategies |
| `train_test_split()` | Prevent data leakage | Evaluation methodology |
| `build_ngrams()` | Create n-gram tuples | N-gram construction |
| `get_corpus_stats()` | Corpus analysis | Exploratory data analysis |

**Design Decision:** The sample corpus is hardcoded rather than loaded from
files. This keeps the project self-contained — anyone can clone and run it
without downloading datasets. For production, replace `SAMPLE_TEXTS` with
a real corpus loader.

### 2. Statistical Models

#### N-gram Model (`src/ngram_model.py`)

**Core Idea:** Estimate P(next_word | previous_words) by counting co-occurrences.

```
Training:   "the cat sat" → count[("the", "cat", "sat")] += 1
Prediction: P("sat" | "the", "cat") = count("the cat sat") / count("the cat")
```

**Key Features:**
- Multi-order support (1-gram through 4-gram)
- Backoff smoothing: falls back to lower orders when context unseen
- Interpolated smoothing: blends all orders simultaneously with λ weights
- Model persistence via JSON serialization

**When to use:** Fast prototyping, small datasets, interpretable results.

#### Markov Chain (`src/markov_model.py`)

**Core Idea:** Model language as a state machine where each word is a state
and transitions have probabilities.

```
State: "learning" → {("is", 0.3), ("models", 0.2), ("algorithms", 0.15), ...}
```

**Key Features:**
- Explicit transition matrix (inspectable, visualizable)
- Laplace smoothing prevents zero-probability transitions
- Text generation via random walk through the state graph
- Temperature-controlled sampling for creativity vs. coherence

**When to use:** Visualization, text generation, teaching probabilistic models.

### 3. Neural Model (`src/neural_model.py`)

**Core Idea:** Learn dense vector representations that capture semantic
similarity, then predict the next token from the hidden state.

```
Embedding → LSTM layers → Linear → Softmax → next token distribution
```

**Key Features:**
- 2-layer LSTM with dropout regularization
- Graceful fallback when PyTorch is not installed
- Configurable via `NEURAL_CONFIG` in `config.py`

**When to use:** Larger datasets, capturing long-range dependencies.

### 4. Beam Search Decoder (`src/beam_search.py`)

**The Problem:** Greedy decoding (always pick the top-1 prediction) can lead
to locally optimal but globally suboptimal sequences.

**The Solution:** Maintain `beam_width` parallel hypotheses, expand all of
them at each step, then prune to keep the best.

```
Step 0: ["the"]
Step 1: ["the cat", "the neural", "the machine"]     ← expand & prune
Step 2: ["the cat sat", "the neural network", ...]    ← expand & prune
```

**Length Penalty:** Without normalization, shorter sequences always score
higher (fewer probability multiplications). The length penalty `α` controls
the trade-off: `score = log_prob / length^α`.

### 5. Evaluation (`src/evaluation.py`)

| Metric | Measures | Good Value |
|--------|----------|------------|
| Perplexity | How surprised the model is by test data | < 100 |
| Top-k Accuracy | Correct word in top-k suggestions | > 0.3 |
| Diversity | Variety of predictions across inputs | > 0.5 |
| Coverage | Fraction of vocab the model can predict | > 0.3 |

### 6. API Layer (`src/api/main.py`)

FastAPI REST endpoints with:
- **Rate limiting** via token bucket algorithm (per-client-IP)
- **Model caching** to avoid retraining on each request
- **Request metrics** for monitoring and observability
- **CORS** enabled for cross-origin web app integration
- **Pydantic validation** for type-safe request/response schemas

### 7. Interfaces

| Interface | Technology | Purpose |
|-----------|------------|---------|
| `cli.py` | argparse | Terminal-based train/predict/eval |
| `src/api/main.py` | FastAPI | REST API for integration |
| `streamlit_app/` | Streamlit | Interactive visual dashboard |

## Design Decisions

### Why Multiple Models?

Educational value. Each model represents a different point on the
complexity–interpretability spectrum:

| Model | Complexity | Interpretability | Data Needs |
|-------|-----------|-------------------|------------|
| N-gram | Low | High (inspect counts) | Small |
| Markov | Low | High (transition matrix) | Small |
| LSTM | High | Low (black box) | Large |

### Why JSON for Model Persistence?

Pickle is faster but has security risks (arbitrary code execution) and
isn't human-readable. JSON is:
- Safe to load from untrusted sources
- Inspectable with any text editor
- Language-agnostic (could load in JavaScript, Go, etc.)

### Why In-Memory Rate Limiting?

For a portfolio project, simplicity wins. In production, you'd use Redis
or a service mesh (Istio, Envoy) for distributed rate limiting. The token
bucket algorithm is the same either way — the storage backend changes,
not the logic.

## File Map

```
10-text-autocomplete/
├── src/
│   ├── __init__.py          # Public API exports
│   ├── config.py            # Centralized configuration
│   ├── data_loader.py       # Corpus loading and preprocessing
│   ├── ngram_model.py       # N-gram model (backoff + interpolation)
│   ├── markov_model.py      # Markov chain model
│   ├── neural_model.py      # LSTM neural model
│   ├── beam_search.py       # Beam search decoder
│   ├── evaluation.py        # Metrics (perplexity, accuracy, diversity)
│   └── api/
│       └── main.py          # FastAPI REST API with rate limiting
├── tests/
│   ├── test_ngram.py        # N-gram unit tests
│   ├── test_markov.py       # Markov chain unit tests
│   ├── test_beam_search.py  # Beam search unit tests
│   ├── test_evaluation.py   # Evaluation metric tests
│   ├── test_data_loader.py  # Data pipeline tests
│   ├── test_api.py          # API endpoint tests
│   ├── test_neural.py       # Neural model tests
│   └── test_integration.py  # End-to-end pipeline tests
├── streamlit_app/           # Interactive dashboard
├── cli.py                   # Command-line interface
├── docs/
│   └── ARCHITECTURE.md      # This file
├── requirements.txt
└── README.md
```
