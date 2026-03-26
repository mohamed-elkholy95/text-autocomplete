<div align="center">

# ✍️ Text Autocomplete

**N-gram, Markov Chain & Beam Search language models** for text completion with perplexity evaluation

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-77%2B%20passed-success?style=flat-square)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

A comprehensive **text autocomplete system** that demonstrates multiple approaches to next-word prediction. Built with an educational focus — every algorithm is heavily commented with explanations of WHY, not just WHAT.

## Features

### 🧠 Language Models
- **N-gram Model** — Bigram through 4-gram with backoff smoothing and configurable frequency thresholding
- **Markov Chain Model** — First-order Markov chain with Laplace smoothing, text generation, and transition visualization
- **LSTM Neural Model** — PyTorch-based LSTM for next-token prediction (with graceful mock fallback)

### 🔍 Advanced Decoding
- **Beam Search** — Multi-hypothesis decoding with length penalty, configurable beam width and search depth

### 📏 Evaluation
- **Perplexity** — Standard language model metric with interpretation guide
- **Top-k Accuracy** — Measures prediction correctness at different ranking levels
- **Prediction Diversity** — Detects trivial models that always predict the same word
- **Vocabulary Coverage** — Measures what fraction of the vocabulary the model can predict
- **Model Comparison** — Side-by-side evaluation of all models on identical test data

### 🚀 API
- **Single Autocomplete** — `POST /autocomplete` with model selection (ngram/markov)
- **Batch Autocomplete** — `POST /autocomplete/batch` for processing multiple texts
- **Model Listing** — `GET /models` — discover available models
- **Vocabulary Stats** — `GET /vocab/stats` — corpus and model statistics
- **Model Caching** — Trained models persist across requests for fast inference

### 📊 Streamlit Dashboard
- **Overview** — Architecture explanation, model comparison, corpus statistics, Zipf's law
- **Autocomplete** — Interactive predictions with both models, text generation, beam search demo
- **Metrics** — Perplexity evaluation, n-gram order impact, prediction analysis, Markov transitions

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/text-autocomplete.git
cd text-autocomplete
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## CLI Usage

The project includes a full command-line interface for training, prediction, and evaluation:

```bash
# Train and save an n-gram model
python cli.py train --model ngram --n 3 --save models/ngram_3.json --eval

# Get autocomplete predictions
python cli.py predict --text "machine learning is" --top-k 5
python cli.py predict --text "neural networks" --model markov

# Load a saved model for faster predictions
python cli.py predict --text "deep learning" --load models/ngram_3.json

# Run full evaluation comparing both models
python cli.py eval --test-ratio 0.2

# View corpus statistics and word frequencies
python cli.py info
```

## Project Structure

```
text-autocomplete/
├── src/
│   ├── config.py          # Central configuration & hyperparameters
│   ├── data_loader.py     # Text loading, tokenization, train/test split
│   ├── ngram_model.py     # N-gram LM with backoff smoothing
│   ├── markov_model.py    # Markov chain LM with text generation
│   ├── beam_search.py     # Beam search decoder with length penalty
│   ├── neural_model.py    # LSTM neural language model (PyTorch)
│   ├── evaluation.py      # Comprehensive evaluation metrics
│   └── api/
│       └── main.py        # FastAPI REST API
├── streamlit_app/
│   ├── app.py             # Main Streamlit entry point
│   └── pages/
│       ├── 1_📊_Overview.py
│       ├── 2_✍️_Autocomplete.py
│       └── 3_📈_Metrics.py
├── tests/
│   ├── test_ngram.py
│   ├── test_markov.py
│   ├── test_beam_search.py
│   ├── test_neural.py
│   ├── test_data_loader.py
│   ├── test_evaluation.py
│   ├── test_api.py
│   └── test_integration.py   # End-to-end pipeline tests
├── cli.py                     # Command-line interface
├── docs/
│   ├── ARCHITECTURE.md        # System design & diagrams
│   └── GLOSSARY.md            # NLP terminology reference
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
```

## Running Tests

```bash
# All tests (77+ across 8 test files)
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_ngram.py -v

# With coverage
python -m pytest tests/ -v --cov=src
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
