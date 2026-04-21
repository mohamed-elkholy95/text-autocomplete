<div align="center">

# ✍️ Text Autocomplete

**N-gram, Markov Chain & Beam Search language models** for text completion with perplexity evaluation

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-173%20passed-success?style=flat-square)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?style=flat-square)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-0E4F88?style=flat-square&logo=github)](.github/workflows/codeql.yml)

</div>

## Overview

A comprehensive **text autocomplete system** that demonstrates multiple approaches to next-word prediction. Built with an educational focus — every algorithm is heavily commented with explanations of WHY, not just WHAT.

## Features

### 🧠 Language Models
All three models implement the same `fit(tokens)` / `predict_next(context, top_k)` contract so they are interchangeable behind the beam-search decoder and the API.
- **N-gram Model** — Unigram through 4-gram with **backoff** *and* **interpolation** smoothing, configurable min-frequency pruning, and JSON save/load
- **Markov Chain Model** — First-order Markov chain with Laplace smoothing, temperature-controlled text generation, and transition inspection
- **LSTM Neural Model** — PyTorch LSTM (word-level, CUDA-aware) with a deterministic mock fallback when torch isn't installed. Trained checkpoints persist as a `safetensors`-weights + JSON-metadata bundle (no code execution on load — `torch.save` is deliberately avoided for the same reason JSON is used elsewhere).

### 🔍 Advanced Decoding
- **Beam Search** — Multi-hypothesis decoding with length-normalized scoring (`score = log_prob / length^α`), configurable beam width and search depth

### 📏 Evaluation
- **Perplexity** — Standard language model metric with interpretation guide
- **Top-k Accuracy** — Measures prediction correctness at different ranking levels
- **Prediction Diversity** — Detects trivial models that always predict the same word
- **Vocabulary Coverage** — Measures what fraction of the vocabulary the model can predict
- **Prediction Confidence** — Top-1 probability, Shannon entropy, and #1–#2 margin with a categorical `high`/`medium`/`low` label
- **Model Comparison** — Side-by-side evaluation of all models on identical test data via `compare_models()`

### 📚 Data Pipeline
- **Built-in Corpus** — `load_sample_data()` ships a curated teaching corpus
- **File-based Corpus** — `load_corpus_from_file(path)` for training on your own text
- **Unicode-safe Tokenizer** — NFKC normalization, smart-quote folding, optional stop-word removal
- **Deterministic Train/Test Split** — seeded split for reproducible evaluation

### 🚀 API (FastAPI, port 8010)
- **Health Check** — `GET /health` — for load balancers and uptime monitoring
- **Single Autocomplete** — `POST /autocomplete` with model selection (`ngram`/`markov`)
- **Batch Autocomplete** — `POST /autocomplete/batch` for processing up to 50 texts per call
- **Text Generation** — `POST /generate` with temperature control and optional seed for reproducibility
- **Model Listing** — `GET /models` — discover available models and their parameters
- **Vocabulary Stats** — `GET /vocab/stats` — corpus and model statistics
- **Metrics** — `GET /metrics` — request counts, rate-limit hits, per-endpoint breakdown
- **Rate Limiting** — token-bucket per client IP (30 burst, 2/s refill) with bounded-memory eviction
- **Proxy-aware** — honours `X-Forwarded-For` only when `TRUST_FORWARDED_HEADERS=1` is set, to prevent spoofed-IP bucket minting
- **Model Caching** — Trained models persist across requests for fast inference

### 📊 Streamlit Dashboard
- **Overview** — Architecture explanation, model comparison, corpus statistics, Zipf's law
- **Autocomplete** — Interactive predictions with both models, text generation, beam search demo
- **Metrics** — Perplexity evaluation, n-gram order impact, prediction analysis, Markov transitions

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/text-autocomplete.git
cd text-autocomplete
pip install -r requirements.txt          # torch is optional — LSTM falls back to a mock if it's missing
python -m pytest tests/ -v                # 173 tests across 8 files

# Pick one frontend:
streamlit run streamlit_app/app.py                              # interactive dashboard
uvicorn src.api.main:app --host 0.0.0.0 --port 8010 --reload    # REST API
python cli.py info                                              # command-line
```

## CLI Usage

The project includes a full command-line interface for training, prediction, and evaluation:

```bash
# Train and save an n-gram model (with optional held-out evaluation)
python cli.py train --model ngram --n 3 --save models/ngram_3.json --eval

# Train a Markov chain model
python cli.py train --model markov --save models/markov.json

# Train an LSTM (requires PyTorch). Save path produces a pair of files:
#   models/lstm.safetensors  — tensor weights
#   models/lstm.json         — vocabulary, hyperparameters, schema version
python cli.py train --model lstm --epochs 3 --embed-dim 64 --hidden-dim 128 \
    --num-layers 2 --save models/lstm

# Get autocomplete predictions (visualised with probability bars)
python cli.py predict --text "machine learning is" --top-k 5
python cli.py predict --text "neural networks" --model markov

# Load a saved model for faster predictions (skips training)
python cli.py predict --text "deep learning" --load models/ngram_3.json
python cli.py predict --text "the model is" --model lstm --load models/lstm

# Run full evaluation comparing both models — prints a formatted report
python cli.py eval --test-ratio 0.2

# View corpus statistics and Zipf-style word frequencies
python cli.py info
```

## Project Structure

```
text-autocomplete/
├── src/
│   ├── __init__.py        # Public API re-exports (models, evaluation, data utils)
│   ├── config.py          # Central configuration & hyperparameters
│   ├── data_loader.py     # Corpus loading, Unicode normalization, tokenization, split
│   ├── ngram_model.py     # N-gram LM with backoff + interpolation, JSON save/load
│   ├── markov_model.py    # Markov chain LM with text generation, JSON save/load
│   ├── beam_search.py     # Beam search decoder with length penalty
│   ├── neural_model.py    # LSTM (PyTorch) with torch-optional mock fallback
│   ├── evaluation.py      # Perplexity, accuracy, diversity, coverage, confidence, compare_models
│   └── api/
│       └── main.py        # FastAPI REST API (rate-limited, metrics, model cache)
├── streamlit_app/
│   ├── app.py             # Main Streamlit entry point
│   └── pages/
│       ├── 1_📊_Overview.py
│       ├── 2_✍️_Autocomplete.py
│       └── 3_📈_Metrics.py
├── tests/                  # 173 tests across 8 files
│   ├── test_ngram.py
│   ├── test_markov.py
│   ├── test_beam_search.py
│   ├── test_neural.py
│   ├── test_data_loader.py
│   ├── test_evaluation.py
│   ├── test_api.py
│   └── test_integration.py   # End-to-end pipeline tests
├── cli.py                     # Command-line interface (train / predict / eval / info)
├── docs/
│   ├── ARCHITECTURE.md        # System design & diagrams
│   └── GLOSSARY.md            # NLP terminology reference
├── .github/
│   ├── workflows/ci.yml       # pytest on push/PR
│   ├── workflows/codeql.yml   # CodeQL security scanning
│   └── dependabot.yml         # Weekly pip + actions updates
├── SECURITY.md                # Vulnerability disclosure policy
├── LICENSE                    # MIT
├── requirements.txt
└── README.md
```

## Educational Concepts

### How N-gram Models Work

N-gram models estimate word probabilities by counting how often word sequences appear in training data:

```
P("learning" | "machine") = count("machine learning") / count("machine")
```

For higher-order n-grams (trigrams, 4-grams), the model considers more context words but faces the **sparse data problem** — with a 10,000-word vocabulary, there are 10^12 possible trigrams, most of which never appear in training.

**Backoff smoothing** solves this by falling back to shorter contexts: if a trigram isn't found, try the bigram; if the bigram isn't found, use the unigram.

### How Markov Chains Work

A Markov chain models text as a **state transition graph** where each word is a state and edges carry transition probabilities. Given the current word, we look up its outgoing edges and return the most probable destinations.

The **Markov Property** states that the future depends only on the present — not on the entire history. This makes prediction fast (O(1) lookup) but limits context to just the previous word.

**Laplace smoothing** (add-k) ensures no transition has zero probability by adding k to every count, preventing infinite perplexity on unseen transitions.

### What is Beam Search?

Instead of greedily picking the single best word at each step, beam search maintains **multiple candidate sequences** (beams) in parallel. At each step:
1. Expand every beam with top-k next words
2. Score all expanded sequences (length-normalized log probability)
3. Keep only the top `beam_width` sequences
4. Repeat

This produces better multi-word predictions than greedy decoding, especially with `length_penalty < 1.0` which encourages longer, more coherent sequences.

### Understanding Perplexity

**Perplexity** measures how "surprised" a model is by test data:

| PPL Range | Quality | Intuition |
|-----------|---------|-----------|
| < 50 | Excellent | Narrows it down to ~50 candidates |
| 50–150 | Good | Captures common patterns well |
| 150–500 | Fair | Some useful predictions |
| > 500 | Poor | Barely better than random |

Formula: `PPL = exp(-1/N × Σ log P(word_i | context_i))`

## API Usage

```python
import requests

# Single autocomplete
resp = requests.post("http://localhost:8010/autocomplete", json={
    "text": "machine learning is a subset of",
    "top_k": 5,
    "model": "ngram"  # or "markov"
})
print(resp.json())
# {"suggestions": [...], "context": "learning is a subset of", "model": "ngram"}

# Batch autocomplete
resp = requests.post("http://localhost:8010/autocomplete/batch", json={
    "texts": ["the attention", "deep learning", "gradient descent"],
    "top_k": 3,
    "model": "markov"
})

# Generate text with temperature control
resp = requests.post("http://localhost:8010/generate", json={
    "start_word": "machine",
    "max_length": 20,
    "temperature": 1.2  # Higher = more creative
})

# List available models
resp = requests.get("http://localhost:8010/models")

# Vocabulary statistics
resp = requests.get("http://localhost:8010/vocab/stats")

# API metrics (request counts, rate limit stats)
resp = requests.get("http://localhost:8010/metrics")

# Health check (skipped by rate limiter — safe to poll)
resp = requests.get("http://localhost:8010/health")
```

Interactive docs are available at `http://localhost:8010/docs` (Swagger UI) and `http://localhost:8010/redoc` once the server is running.

### Deploying behind a proxy

The rate limiter keys off `request.client.host` by default. If you run the API behind a trusted reverse proxy (nginx, Traefik, Cloudflare), set `TRUST_FORWARDED_HEADERS=1` so the left-most `X-Forwarded-For` entry is honoured. Do **not** enable this when the server is exposed directly — any client could otherwise spoof the header and mint a fresh rate-limit bucket per request.

## Python API

```python
from src import (
    NGramModel, MarkovChainModel, LSTMModel, BeamSearchDecoder,
    tokenize, load_sample_data, train_test_split,
    compute_perplexity, autocomplete_accuracy,
    prediction_diversity, vocabulary_coverage,
    prediction_confidence, compare_models,
)

tokens = tokenize(load_sample_data())
train, test = train_test_split(tokens, test_ratio=0.2, seed=42)

ngram = NGramModel(n=3).fit(train)
markov = MarkovChainModel().fit(train)

# Side-by-side comparison on identical data
results = compare_models(
    {"ngram": ngram, "markov": markov},
    test_tokens=test,
    context_tokens=["machine", "learning"],
)

# Confidence analysis on a single prediction
conf = prediction_confidence(ngram.predict_next(["machine", "learning"], top_k=5))
# -> {'top1_prob': ..., 'entropy': ..., 'margin': ..., 'confidence_level': 'high'|'medium'|'low'}

# Multi-word completion via beam search (works with any model that has predict_next)
decoder = BeamSearchDecoder(beam_width=5, max_length=5, length_penalty=0.6)
beams = decoder.search(ngram, context_tokens=["machine", "learning"])
# beams is a list of dicts sorted by score (best first), each with tokens, score, log_prob
```

## Running Tests

```bash
# All tests (173 across 8 test files)
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_ngram.py -v

# With coverage
python -m pytest tests/ -v --cov=src
```

## Real-Data Benchmarks

The numbers below come from training on the **WikiText-2 raw train split**
(2,076,893 tokens, 65,920 unique types) with a deterministic 90/10 split
(`seed=42`) and evaluating on the held-out slice. The n-gram and Markov
runs are reproducible via `scripts/bench_real_data.py`; the LSTM column
was produced in the same harness on the same split (hyperparameters noted
below the table).

| Model | Corpus | Perplexity ↓ | Top-1 Acc ↑ | Top-5 Acc ↑ | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Trigram (n=3, min_freq=2) | WikiText-2 (full 1.87 M train) | 1,643.6 | 5.00 % | 11.00 % | Backoff smoothing |
| Markov chain (Laplace) | WikiText-2 (full 1.87 M train) | 13,998.8 | 4.80 % | 12.70 % | First-order only |
| LSTM (embed 96 / hidden 192 / 2 layers) | WikiText-2 (200 k-token subset, 1 epoch) | **1,301.1** | 4.60 % | **13.80 %** | 17.7 k vocab, 12 s on an RTX 5080 |

The LSTM was trained on a 200 k-token subset so the end-to-end run fits
in ~12 s on a single GPU — the numbers are *deliberately under-trained*
to document the minimum-effort baseline. They still beat the trigram on
perplexity (−21 %) and top-5 accuracy (+2.8 pp), and crush the Markov
chain on perplexity (−91 %). Top-1 is slightly below the trigram because
one epoch isn't enough to sharpen the softmax — add epochs or train on
the full corpus to close that gap.

**Reproducing the LSTM run** (requires PyTorch + a compatible GPU; see
`scripts/bench_real_data.py` for the corpus setup):

```bash
# After scripts/bench_real_data.py has cached data/wikitext2/wikitext2_train.txt
python cli.py train --model lstm \
    --epochs 1 --seq-len 32 --embed-dim 96 --hidden-dim 192 --num-layers 2 \
    --save models/lstm_wikitext2
python cli.py predict --model lstm --load models/lstm_wikitext2 \
    --text "in the nineteenth century" --top-k 5
```

Perplexity on the LSTM is computed via token-level cross-entropy (the
`compute_perplexity()` helper in `src/evaluation.py` only handles the
n-gram and Markov contracts; neural models get the direct softmax path).

### Small-corpus note

On the built-in 40-sentence teaching corpus (~2,465 tokens), perplexity
is in the millions for every model — the held-out split contains many
n-grams that were never seen in training, which is a small-data artefact
rather than a model bug. Top-k accuracy and prediction diversity are the
useful signals on the teaching corpus; perplexity becomes meaningful
again on the WikiText-2 scale shown above.

## Security

- **MIT licensed** — see [`LICENSE`](LICENSE)
- **GitHub Advanced Security**: secret scanning + push protection, CodeQL (`security-and-quality` suite), Dependabot alerts on every push/PR to `main`
- **CI**: pytest runs on push/PR with least-privilege permissions (`persist-credentials: false`, `contents: read`)
- **Signed commits** required on `main` (branch-protected, squash-only, linear history)
- **Vulnerability disclosure** — see [`SECURITY.md`](SECURITY.md)

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
