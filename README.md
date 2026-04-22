<div align="center">

# ✍️ Text Autocomplete

**N-gram, Markov Chain & Beam Search language models** for text completion with perplexity evaluation

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-212%20passed-success?style=flat-square)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-v4-06B6D4?style=flat-square&logo=tailwindcss)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-0E4F88?style=flat-square&logo=github)](.github/workflows/codeql.yml)

</div>

## Overview

A comprehensive **text autocomplete system** that demonstrates multiple approaches to next-word prediction. Built with an educational focus — every algorithm is heavily commented with explanations of WHY, not just WHAT.

## Features

### 🧠 Language Models
All four models implement the same `fit(tokens)` / `predict_next(context, top_k)` / `perplexity(tokens)` contract so they are interchangeable behind the beam-search decoder and the API.
- **N-gram Model** — Unigram through 4-gram with **backoff** *and* **interpolation** smoothing, configurable min-frequency pruning, and JSON save/load
- **Markov Chain Model** — First-order Markov chain with Laplace smoothing, temperature-controlled text generation, and transition inspection
- **LSTM Neural Model** — PyTorch LSTM with weight tying (Press & Wolf 2016), a configurable `vocab_cap` routing rare tokens to `<unk>`, stateful-BPTT option, optional `torch.compile`, and `bfloat16` autocast on CUDA. Schema-v2 checkpoints (word-level) and schema-v3 (subword, tokenizer-backed) both persist as `safetensors`-weights + JSON-metadata bundles.
- **Decoder-only Transformer** — Causal self-attention with tied LM head, learned absolute positional embeddings, pre-norm blocks, AdamW + cosine LR. Same persistence discipline as the LSTM; own `schema_version` (v1 word-level, v2 subword).
- **Subword Tokenizer** — Optional `BPETokenizer` wrapper around `transformers.AutoTokenizer` (defaults to SmolLM2's ~49 k-piece vocab). When passed to `LSTMModel.fit` or `TransformerModel.fit`, the model trains on subword ids directly, with the tokenizer identity captured in the saved meta for reproducible reload.
- **SentencePiece Tokenizer** — Alternative `SPTokenizer` wrapping Google SentencePiece (Unigram or BPE). Unlike the HF adapter, it can **train a fresh tokenizer on the project's own corpus** via `SPTokenizer.train_from_corpus()` — useful for teaching how subword models are fit in the first place. Same `encode`/`decode`/`vocab_size`/`unk_id`/`name` surface as `BPETokenizer`, so it's a drop-in for the neural models' `tokenizer=` kwarg.

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
- **Health Check** — `GET /health` — for load balancers and uptime monitoring. Response includes a `capabilities` map so operators can confirm which optional paths are active (torch, transformer, BPE, Prometheus, Redis, API-key auth)
- **Single Autocomplete** — `POST /autocomplete` with model selection: `ngram`, `markov`, `lstm`, `transformer`, plus BPE aliases `lstm-bpe` / `transformer-bpe` (auto-imply `tokenizer=bpe`)
- **Batch Autocomplete** — `POST /autocomplete/batch` for processing up to 50 texts per call (same catalogue aliases as `/autocomplete`)
- **Text Generation** — `POST /generate` with temperature control, optional seed, and `model: markov|lstm|transformer` — neural generation samples from the softmax top-20 each step
- **Model Listing** — `GET /models` — discover available models and their parameters
- **Vocabulary Stats** — `GET /vocab/stats` — corpus and model statistics
- **Attention Visualisation** — `POST /attention` — transformer only; returns per-layer per-head causal self-attention weights for a short prompt (consumed by the React Attention page)
- **Metrics (JSON)** — `GET /metrics` — request counts, rate-limit hits, per-endpoint breakdown
- **Metrics (Prometheus)** — `GET /metrics/prom` — Prometheus exposition format with route-level counters, latency histograms, and a custom `autocomplete_model_hits_total{model, tokenizer}` business counter (opt-in; install `prometheus-fastapi-instrumentator`)
- **Rate Limiting** — token-bucket per client IP (30 burst, 2/s refill) with bounded-memory eviction; transparently upgrades to a shared Redis counter when `REDIS_URL` is set. Per-model quotas: neural calls drain more tokens than statistical calls (`ngram`/`markov` = 1, `lstm` = 4, `transformer` = 6 by default; override via `AUTOCOMPLETE_COST_<MODEL>`). Every response carries `X-RateLimit-Limit` and `X-RateLimit-Remaining` hint headers.
- **Request IDs** — every response carries `X-Request-ID` (echoed from the client header, or a freshly minted 16-char UUID). Paired with `LOG_FORMAT=json` to trace a single request across log lines.
- **Eval Summary** — `GET /eval/summary` — perplexity + top-1/5 accuracy for every available model on a deterministic 80/20 split of the sample corpus, cached in-process. Feeds the React Metrics page's accuracy bar chart.
- **API Key Auth (opt-in)** — set `AUTOCOMPLETE_API_KEY` and every non-public endpoint requires `X-API-Key: <value>`. `/health`, `/docs`, `/redoc`, `/openapi.json` stay open.
- **CORS** — dev default allows `http://localhost:5173` (Vite). Override with `CORS_ALLOWED_ORIGINS=https://app.example.com,...` or set to `*` to restore the old wildcard.
- **Proxy-aware** — honours `X-Forwarded-For` only when `TRUST_FORWARDED_HEADERS=1` is set, to prevent spoofed-IP bucket minting
- **Model Caching** — Trained models persist across requests for fast inference

### 🎨 React Dashboard (`frontend/`)
- **Overview** — Corpus stats + four model cards pulled live from `/models`
- **Autocomplete** — Form hitting `/autocomplete` with model and tokenizer selectors, top-k suggestion table with a probability-bar column
- **Metrics** — JSON `/metrics` counters, a per-endpoint bar chart, and a preview of the Prometheus `/metrics/prom` exposition
- Built with **React 19 + Vite 8 + TypeScript + Tailwind CSS v4 + shadcn/ui (`new-york-v4`)**, custom 4-color palette (blue / lime / slate / bone), dark mode included

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/text-autocomplete.git
cd text-autocomplete

# --- Backend (Python 3.12+) ---
pip install -r requirements.txt          # torch is optional — LSTM falls back to a mock if it's missing
python -m pytest tests/ -q               # 212 tests, ~5 s
uvicorn src.api.main:app --host 0.0.0.0 --port 8010 --reload    # REST API on :8010

# --- CLI ---
python cli.py info                       # corpus stats + Zipf-style frequencies
```

For the interactive UI, see the [Web UI](#web-ui) section below.

## Web UI

A **React 19 single-page app** lives under `frontend/`. It's built with
Vite 8, TypeScript, Tailwind CSS v4, and shadcn/ui (`new-york-v4` style),
and talks to the FastAPI backend via a `/api` dev proxy. Three pages
mirror the REST surface:

| Page | Route | What it shows |
| --- | --- | --- |
| Overview | `/` | Corpus stats + four model cards pulled live from `GET /models` |
| Autocomplete | `/autocomplete` | Form hitting `POST /autocomplete` with model + tokenizer selectors, top-k suggestion table with per-row probability bars |
| Attention | `/attention` | Per-layer, per-head causal self-attention heatmaps from the trained transformer (`POST /attention`) |
| Metrics | `/metrics` | JSON `/metrics` counters, a per-endpoint bar chart (Recharts), and a preview of the Prometheus `/metrics/prom` exposition |

Dark mode, a health pill polled from `/health`, and a 4-color brand
palette (blue `#3686C9`, lime `#B4C540`, slate `#575A6C`, bone `#E0E2D2`)
ship out of the box.

### Prerequisites

- **Node.js 20+** and **npm 10+** (the project is tested on Node 24 / npm 11).
  Install via [nvm](https://github.com/nvm-sh/nvm) (`nvm install --lts`) or
  your package manager — no global dependencies required beyond Node itself.
- The FastAPI backend from the quick-start above, running on port 8010.

### Run in development

```bash
# 1. Start the API (in one terminal)
uvicorn src.api.main:app --host 127.0.0.1 --port 8010 --reload

# 2. Start the UI (in another terminal)
cd frontend
npm install                # first time only; installs ~250 MB into frontend/node_modules
npm run dev                # Vite dev server on http://localhost:5173
```

Open <http://localhost:5173> in your browser. The dev server hot-reloads
on save, and its `/api/*` proxy transparently forwards every fetch to
`http://127.0.0.1:8010`, so the UI and API share the same origin from
the browser's perspective — no CORS config needed during development.

### Serve pre-trained checkpoints (optional)

Point the four `AUTOCOMPLETE_*_CHECKPOINT_*` env vars at checkpoint
bundles before starting the API and the neural models load from disk
instead of lazy-fitting on the sample corpus:

```bash
export AUTOCOMPLETE_LSTM_CHECKPOINT_WORD=models/lstm_wikitext2_long
export AUTOCOMPLETE_TRANSFORMER_CHECKPOINT_WORD=models/transformer_wikitext2_long
export AUTOCOMPLETE_LSTM_CHECKPOINT_BPE=models/lstm_wikitext2_bpe_long
export AUTOCOMPLETE_TRANSFORMER_CHECKPOINT_BPE=models/transformer_wikitext2_bpe_long
uvicorn src.api.main:app --host 127.0.0.1 --port 8010
```

The API's `lifespan` hook pre-loads the checkpoints at startup, so the
first `/autocomplete` request with `model=lstm` or `model=transformer`
returns in milliseconds instead of seconds. Generate compatible
checkpoints with `python scripts/bench_full_train.py` (see
[Real-Data Benchmarks](#real-data-benchmarks)).

### Build for production

```bash
cd frontend
npm run build                 # outputs frontend/dist/ (tree-shaken, minified, ~225 kB gzip)
npm run preview               # optional: serve the built bundle locally on :4173
```

`dist/` is a plain static bundle. Two deployment shapes:

1. **Same process** — `frontend/dist/` is auto-mounted on the FastAPI app
   when it's present (see `src/api/main.py`). `uvicorn src.api.main:app`
   then serves both the UI and the API on port 8010.
2. **Two hosts** — drop `dist/` behind nginx / Caddy / a CDN and point
   it at an origin that proxies `/api/*` to the FastAPI. Useful if the
   UI is fronted by a CDN and the API lives in a cluster.

### Docker

A two-stage `Dockerfile` builds the React bundle and bakes it into a
Python 3.12 image that serves both the API and the SPA on port 8010.

```bash
docker build -t text-autocomplete .
docker run --rm -p 8010:8010 text-autocomplete
# → http://localhost:8010  (UI + /docs + /metrics/prom all on one port)
```

For the neural paths (LSTM, Transformer, BPE, `/attention`), build with
`--build-arg INSTALL_TORCH=1` to include torch + transformers (~3 GB
image). `docker-compose.yml` wires an optional Redis service for the
shared rate-limit counter; uncomment and set `REDIS_URL` to enable.

### API key auth (opt-in)

Set `AUTOCOMPLETE_API_KEY` to any non-empty string and every non-public
endpoint will require the header `X-API-Key: <your value>`. Public
paths (`/health`, `/docs`, `/redoc`, `/openapi.json`) stay open so
load balancers and the Swagger UI keep working.

```bash
AUTOCOMPLETE_API_KEY=change-me uvicorn src.api.main:app --host 0.0.0.0 --port 8010
curl -H "X-API-Key: change-me" http://localhost:8010/models    # 200
curl http://localhost:8010/models                              # 401
```

This is the smallest educational surface — a real deployment should use
OAuth, mTLS, or an API gateway.

### Project layout (frontend/)

```
frontend/
├── index.html
├── package.json
├── vite.config.ts               # Tailwind v4 plugin, @/ alias, /api dev proxy
├── tsconfig.json / tsconfig.app.json
├── components.json              # shadcn config (style: "new-york-v4")
└── src/
    ├── main.tsx                 # Router setup
    ├── App.tsx                  # Sidebar + nav layout, health pill, theme toggle
    ├── index.css                # Tailwind v4 @theme + shadcn semantic tokens
    ├── pages/
    │   ├── Overview.tsx
    │   ├── Autocomplete.tsx     # uses React 19 useActionState for form submit
    │   └── Metrics.tsx          # uses React 19 useTransition for refresh
    ├── components/ui/           # shadcn components (button, card, table, …)
    └── lib/
        ├── api.ts               # typed fetch wrappers for every REST endpoint
        ├── types.ts             # mirrors FastAPI Pydantic models
        └── utils.ts             # shadcn cn() helper
```

## CLI Usage

The project includes a full command-line interface for training, prediction, and evaluation:

```bash
# Train and save an n-gram model (with optional held-out evaluation)
python cli.py train --model ngram --n 3 --save models/ngram_3.json --eval

# Train a Markov chain model
python cli.py train --model markov --save models/markov.json

# Train an LSTM (requires PyTorch). Save path produces a pair of files:
#   models/lstm.safetensors  — tensor weights (tied embedding, schema v2)
#   models/lstm.json         — vocabulary, hyperparameters, schema version
python cli.py train --model lstm --epochs 3 --embed-dim 64 --hidden-dim 128 \
    --num-layers 2 --vocab-cap 10000 --save models/lstm

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
│   ├── neural_model.py    # LSTM (PyTorch) with schema v2/v3 persistence, stateful BPTT, compile opt-in
│   ├── transformer_model.py # Decoder-only transformer with causal attention and tied LM head
│   ├── bpe_tokenizer.py   # BPETokenizer adapter around transformers.AutoTokenizer (optional)
│   ├── evaluation.py      # Perplexity, accuracy, diversity, coverage, confidence, compare_models
│   └── api/
│       └── main.py        # FastAPI REST API (rate-limited, metrics, model cache)
├── frontend/               # React 19 + Vite 8 + Tailwind v4 + shadcn/ui
│   ├── src/
│   │   ├── App.tsx         # Sidebar + nav layout
│   │   ├── pages/          # Overview / Autocomplete / Metrics
│   │   ├── components/ui/  # shadcn components (button, card, table, ...)
│   │   └── lib/            # typed fetch client + Pydantic-mirror types
│   └── components.json     # shadcn config (style: new-york-v4)
├── tests/                  # 212+ tests
│   ├── test_ngram.py
│   ├── test_markov.py
│   ├── test_beam_search.py
│   ├── test_neural.py        # LSTM (+ BPE round-trip, gated on `transformers`)
│   ├── test_transformer.py   # Decoder-only transformer (+ BPE round-trip)
│   ├── test_bpe_tokenizer.py # BPETokenizer adapter (gated on `transformers`)
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
    NGramModel, MarkovChainModel, LSTMModel, TransformerModel,
    BPETokenizer, BeamSearchDecoder,
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
# All tests (212 across 9 test files; BPE tests skip without `transformers`)
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_ngram.py -v

# With coverage
python -m pytest tests/ -v --cov=src
```

## Real-Data Benchmarks

All rows below come from training on the **WikiText-2 raw train split**
(2,076,893 tokens, 65,920 unique types) with a deterministic 90/10 split
(`seed=42`) and evaluating on the same held-out slice — every number is
measured on session hardware (RTX 5080, bf16 autocast) and reproducible
via `scripts/bench_real_data.py`. No projections.

| Model | Hyperparameters | Fit | Held-out PPL ↓ | Top-1 ↑ | Top-5 ↑ | Diversity ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Trigram (n=3, min_freq=2) | Backoff smoothing | 3.3 s | 1643.6 | 5.00 % | 11.00 % | 13.80 % |
| Markov chain | Laplace, first-order | 1.1 s | 13998.8 | 4.80 % | 12.70 % | 9.60 % |
| LSTM (stateful, vocab_cap=20k) | embed 128 / hidden 256 / 2 layers, 10 epochs | 29.5 s | 1537.0 | 4.50 % | 13.00 % | 4.10 % |
| **Transformer** | d_model 128 / 4 heads / 4 layers / ff 512, 5 epochs | **16.0 s** | **1148.5** | **5.60 %** | **16.90 %** | 1.60 % |

### Takeaways

- **The transformer wins every ranking metric.** 25 % lower held-out PPL than the LSTM (1148 vs 1537), +3.9 pp top-5, +1.1 pp top-1, and it does it in **half the training time and half the epochs**. Attention pays off on WikiText-2 scale.
- **Both neural models beat the trigram on top-5** (+2.0 pp for the LSTM, +5.9 pp for the transformer) even on word-level tokens.
- **Diversity is low for both neural rows**, not because the models collapsed but because they're underfit: held-out PPL 1148 vs train PPL 1024 for the transformer is a near-zero generalisation gap. More epochs, more params, or subword tokens would push both numbers.

### Subword (BPE) path

The project ships a `BPETokenizer` adapter around SmolLM2's ~49 k-piece vocabulary. `LSTMModel.fit` and `TransformerModel.fit` both accept a `tokenizer=` kwarg that swaps the word-level vocab for BPE subwords; checkpoints then save under schema v3 (LSTM) or v2 (Transformer) with the tokenizer identity captured in the JSON meta, and reload into a fully working model. A worked example lives at `scripts/bpe_train_lstm.py`.

Measured BPE-trained rows (3 epochs each, `scripts/bench_bpe.py`, SmolLM2 tokenizer at 49 k subwords):

| Model | Fit | Held-out PPL ↓ (subword) | Top-1 ↑ | Top-5 ↑ | Diversity ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPE-LSTM | 25.6 s | 2861.9 | 5.00 % | 13.80 % | 7.20 % |
| **BPE-Transformer** | **24.0 s** | **3073.4** | **5.20 %** | **17.80 %** | 3.20 % |

Subword PPL is in units of *subword pieces*, not whole words, so it isn't directly comparable to the word-level PPL column above — don't read "2862 vs 1537" as a regression. Top-k *is* comparable (it just asks "does the right subword appear in the top-k?") and the transformer still wins the ranking metrics (+0.2 pp top-1, +0.9 pp top-5 over its word-level self; LSTM also gains +0.8 pp top-5 from subwords).

### Longer training — `scripts/bench_full_train.py`

Scaling configs roughly 2–4× (LSTM embed=256/hidden=512, Transformer d=256/6 layers/ff=1024) and running 25/15 epochs took ~10 min on RTX 5080 and told a clear story: **word-level saturates fast on 2M tokens; BPE benefits the most from extra capacity.**

| Variant | 3–10-ep baseline PPL ↓ | Long-train PPL ↓ | Δ |
| --- | ---: | ---: | ---: |
| LSTM / word | 1537.0 (10 ep) | 3335.1 (25 ep) | **+117 %** — overfit |
| Transformer / word | 1148.5 (5 ep) | 1197.7 (15 ep) | +4 % — saturated |
| LSTM / BPE (subword) | 2861.9 (3 ep) | **1951.2** (25 ep) | **−32 %** ✓ |
| Transformer / BPE (subword) | 3073.4 (3 ep) | **1406.2** (15 ep) | **−54 %** ✓ |

Takeaways: more parameters + more epochs aren't universally better — they pay off only where the previous config was under-capacity relative to the data. The word-level LSTM memorised the train set past the point where held-out PPL turns around, and the word-level transformer was already saturating. The BPE path (49k softmax) genuinely needed both the extra capacity and the epoch budget. Checkpoints land at `models/{lstm,transformer}_wikitext2{,_bpe}_long.{safetensors,json}` and can be served by the API via the `AUTOCOMPLETE_*_CHECKPOINT_*` env vars — see the [Web UI](#web-ui) section.

### SLM reference point

`scripts/bench_real_data.py` also loads **SmolLM2-135M** (134.5 M params, bf16 on CUDA) for a side-by-side autocomplete demo. It's a teaching anchor, not a competitor to the project's own models — it has orders of magnitude more parameters and data. On the same hardware:

- Weights load in ~0.7 s from the HF cache.
- Greedy decode of 12 new tokens: ~390 ms cold, ~90 ms warm.
- Example output on "Machine learning is a subset of" → `"machine learning. It is a branch of artificial intelligence that uses"`.

The trigram + LSTM + transformer rows above are the things this repo is teaching; the SmolLM2 row is there to keep the scale honest.

### Reproducing

```bash
# N-gram + Markov + LSTM + Transformer + SmolLM2 demo, one command:
python scripts/bench_real_data.py

# Train just the transformer end-to-end:
python cli.py train --model transformer \
    --epochs 5 --seq-len 64 --batch-size 64 \
    --d-model 128 --n-heads 4 --num-layers 4 --ff-dim 512 --max-seq-len 128 \
    --vocab-cap 20000 --save models/transformer_wikitext2

# Four-way head-to-head on the sample corpus:
python cli.py eval --test-ratio 0.2 --include-lstm --include-transformer \
    --lstm-epochs 2 --transformer-epochs 2
```

Perplexity on the neural models is computed via token-level cross-entropy
— `compute_perplexity()` in `src/evaluation.py` dispatches on model type
and routes LSTM / Transformer through the direct-softmax path, while the
statistical models use count-based probabilities.

**Checkpoint format.** All neural bundles are a two-file pair —
`<path>.safetensors` for the weights plus `<path>.json` for the
vocabulary / tokenizer identity / hyperparameters. Older schema versions
are refused with a clear retrain message; this repo has never shipped
a format that can execute code on load.

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
