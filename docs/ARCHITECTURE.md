# Architecture Guide

> Plain-language walkthrough of how the autocomplete system is put together,
> why each piece exists, and what would change if this were a production service.

The project implements three language models behind a single
`predict_next(context, top_k)` interface, composes them with a beam-search
decoder, and exposes the whole thing through three interchangeable frontends
(CLI, FastAPI, React 19 SPA). The goal is breadth of technique with a small,
honest surface area — not a toy, not a pretend ChatGPT.

---

## 1. System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         Data Pipeline                               │
│   load_sample_data() / load_corpus_from_file()                      │
│        → normalize_text() → tokenize() → train_test_split()         │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ tokens
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐   ┌─────────────┐   ┌────────────┐
   │  N-gram    │   │   Markov    │   │    LSTM    │
   │  Model     │   │   Chain     │   │   Neural   │
   │            │   │             │   │   Model    │
   │ Backoff +  │   │  Laplace    │   │  PyTorch   │
   │ Interpol.  │   │  Smoothing  │   │ (optional) │
   └─────┬──────┘   └──────┬──────┘   └─────┬──────┘
         │                 │                │
         └─────────┬───────┘                │
                   ▼                        │
           ┌──────────────┐                 │
           │  Beam Search │◄────────────────┘
           │  Decoder     │
           └──────┬───────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│  CLI   │  │ FastAPI  │  │  React   │
│ cli.py │  │  REST    │  │ frontend │
└────────┘  └──────────┘  └──────────┘
```

All three models implement the same `fit(tokens)` / `predict_next(context, top_k)`
contract, which is what lets the beam-search decoder, the CLI, the API, and the
React pages stay model-agnostic.

---

## 2. Request Lifecycle (API)

What actually happens when a client POSTs to `/autocomplete`:

```
1. Client  ──POST /autocomplete {text, top_k, model}─▶ FastAPI
2. Middleware
      a. resolve client IP (X-Forwarded-For aware)
      b. token-bucket rate-limit check (30 burst, 2 rps refill)
      c. start latency timer
3. Pydantic validates AutocompleteRequest schema
4. tokenize(text) → lowercase, split words/punct, NFC normalize
5. _get_trained_ngram() / _get_trained_markov()
      - first call: fit on sample corpus, cache in module-level dict
      - later calls: hit cache (≈μs)
6. model.predict_next(tokens, top_k)
      - N-gram: backoff from n → n-1 → ... → 1
      - Markov: transition row lookup + Laplace smoothing
7. Build AutocompleteResponse (suggestions + last-5-word context)
8. Middleware stamps X-Response-Time-Ms, logs structured line
9. Client  ◄──200 JSON── FastAPI
```

Latency on this corpus is dominated by tokenization and the first-request
training step; subsequent requests are cache hits and finish in under a
millisecond on a laptop CPU.

---

## 3. Components

### 3.1 Data Layer — `src/data_loader.py`

| Function                  | Purpose                               | Concept                    |
| ------------------------- | ------------------------------------- | -------------------------- |
| `load_sample_data()`      | Built-in 40-sentence corpus × 5       | Corpus design              |
| `load_corpus_from_file()` | Read UTF-8 text from disk             | Real-world ingestion       |
| `normalize_text()`        | NFC + smart-quote/dash cleanup        | Unicode hygiene            |
| `tokenize()`              | `\w+|[^\w\s]` regex, lowercase        | Word/punct tokenization    |
| `build_ngrams()`          | Sliding n-gram tuples                 | N-gram construction        |
| `train_test_split()`      | Seeded random split                   | Evaluation methodology     |
| `get_corpus_stats()`      | Type/token counts, lengths            | Exploratory data analysis  |

**Design choice.** The sample corpus is hardcoded so the repo is runnable the
moment you clone it — no downloads, no auth keys. `load_corpus_from_file()` is
the hook for real data.

### 3.2 Statistical Models

**N-gram (`src/ngram_model.py`).**
Counts word co-occurrences of orders 1..n and estimates
`P(w_n | w_{n-1}, ..., w_{n-k+1}) = count(ngram) / count(context)`.
`MIN_FREQUENCY=2` prunes rare n-grams (noisy estimates on small corpora).
Two smoothing modes: **backoff** (use the highest-order ngram that was seen;
fall back otherwise) and **interpolation** (blend orders with λ weights).
Persisted as JSON via `save()` / `load()`.

**Markov chain (`src/markov_model.py`).**
First-order transition graph: `P[word_i][word_j] = P(next=j | current=i)`
with Laplace (add-k) smoothing so unseen transitions never get probability 0
(which would push perplexity to infinity). Exposes `generate_text()` with a
temperature parameter for sampling-based generation.

**LSTM (`src/neural_model.py`).**
`Embedding → 2-layer LSTM (dropout 0.2) → hidden→embed projection →
tied-weight output` (Press & Wolf 2016). The projection layer lets
`hidden_dim` and `embed_dim` diverge without breaking tying. Training
supports `vocab_cap` (top-N frequent tokens; rest collapse to `<unk>`),
gradient clipping, a cosine LR schedule, `bfloat16` autocast on CUDA,
opt-in `torch.compile`, opt-in stateful BPTT (hidden state carried
across `seq_len` windows, detached between them), and an optional
external `BPETokenizer` for subword training. Perplexity is computed
via token-level cross-entropy (`LSTMModel.perplexity()`) so the
evaluation driver can compare all four model families head-to-head.
Checkpoints persist as a `safetensors` weights file plus a JSON
metadata sidecar; word-level bundles use `schema_version=2`, subword
bundles bump to `schema_version=3` with the tokenizer identity captured
in meta. Older bundles refuse to load rather than silently breaking.
Torch itself is an *optional* dependency — when it's not installed,
`HAS_TORCH=False` and the module returns deterministic mock outputs so
`pytest` stays green on a minimal install.

**Decoder-only Transformer (`src/transformer_model.py`).**
`Embedding (tied) + learned absolute positional embedding → N pre-norm
decoder blocks (causal multi-head self-attention + GELU MLP) →
LayerNorm → tied LM head`. Same `fit` / `predict_next` / `perplexity`
contract as the LSTM so `compute_perplexity` and the beam-search
decoder stay model-agnostic; `vocab_cap` + `<unk>` + BPE-tokenizer
wiring all mirror the LSTM. Training uses AdamW with weight decay
0.01, a cosine LR schedule, and `bfloat16` autocast on CUDA. No
stateful-BPTT option — transformers attend directly, so there's no
hidden state to carry. Checkpoints persist identically (safetensors
+ JSON); word-level is `schema_version=1`, subword bumps to
`schema_version=2`. The transformer and LSTM version their schemas
independently because their state-dict layouts evolve separately.
On the full WikiText-2 benchmark it beats the LSTM on perplexity
(−24 %) and top-5 accuracy (+4 pp) with half the training time — see
the README for the measured numbers.

**Subword Tokenizer (`src/bpe_tokenizer.py`).**
`BPETokenizer` wraps `transformers.AutoTokenizer` (SmolLM2's ~49 k
byte-level BPE by default) and exposes the minimal `encode` / `decode`
/ `vocab_size` / `unk_id` / `name` surface the neural models need.
Passing `tokenizer=` to `LSTMModel.fit` or `TransformerModel.fit`
replaces the word-level vocab with BPE subword ids; callers that don't
want BPE don't need to install `transformers` at all. The tokenizer
identity is captured in the saved meta so reloaded checkpoints
re-instantiate the same tokenizer automatically.

### 3.3 Beam Search Decoder — `src/beam_search.py`

Wraps any model implementing `predict_next` and performs multi-step search.
At each step: expand every beam with `candidates_per_step` next tokens, score
in log-space (`new_log_prob = log_prob + log(prob)` — avoids underflow), prune
to `beam_width`, repeat.

**Length penalty.** This project uses `score = log_prob / length^α` with
`α = 0.6`. A common alternative in the literature is the GNMT form
`((5 + length) / 6)^α` (Wu et al., 2016); both aim to stop short hypotheses
from winning just because they multiply fewer probabilities. The simple form
is kept here for clarity.

### 3.4 Evaluation — `src/evaluation.py`

| Metric               | What it measures                                   | Code                          |
| -------------------- | -------------------------------------------------- | ----------------------------- |
| Perplexity           | Geometric mean of inverse prob on test data        | `compute_perplexity()`        |
| Top-k accuracy       | True next word appears in top-k predictions        | `autocomplete_accuracy()`     |
| Prediction diversity | `unique(top-1) / total` — guards against trivial models | `prediction_diversity()` |
| Vocabulary coverage  | Fraction of vocab the model ever predicts          | `vocabulary_coverage()`       |
| Confidence           | Top-1 prob, Shannon entropy, top-1/top-2 margin    | `prediction_confidence()`     |

`compute_perplexity()` dispatches on model type — n-gram and Markov use
count-based probabilities, LSTMs use token-level cross-entropy over
`seq_len` windows. The neural path maps OOV test tokens to `<unk>` and
clamps `seq_len` against the test slice length so short slices don't
silently collapse to `inf`.

### 3.5 API Layer — `src/api/main.py`

FastAPI with seven endpoints:

| Method | Path                    | Purpose                                      |
| ------ | ----------------------- | -------------------------------------------- |
| GET    | `/health`               | Liveness probe (skips rate-limit)            |
| POST   | `/autocomplete`         | Single prediction, model selectable          |
| POST   | `/autocomplete/batch`   | Up to 50 texts in one round trip             |
| POST   | `/generate`             | Temperature-controlled Markov generation     |
| GET    | `/models`               | Discoverable model catalogue                 |
| GET    | `/vocab/stats`          | Corpus + vocab numbers                       |
| GET    | `/metrics`              | Per-endpoint counters + rate-limit rejects   |

Cross-cutting concerns live in middleware: X-Forwarded-For-aware IP
resolution, in-memory token-bucket rate limiting (30 burst, 2 rps refill),
request counting, and latency logging via `X-Response-Time-Ms`. Models are
trained once and cached in a module-level dict so requests don't retrain.

### 3.6 Frontends

| Interface         | Technology | Entry point                    |
| ----------------- | ---------- | ------------------------------ |
| CLI               | argparse   | `cli.py {train,predict,eval,info}` (train takes `--model {ngram,markov,lstm}`, `--vocab-cap`, `--compile`; `eval --include-lstm` adds a three-way comparison column) |
| REST API          | FastAPI         | `uvicorn src.api.main:app`                        |
| Interactive app   | React 19 SPA    | `cd frontend && npm run dev` (proxies `/api` → :8010) |

---

## 4. Key Design Decisions

### Why three models instead of one?
The three models sit on different points of the
complexity-vs-interpretability curve, which makes the project a useful
teaching surface:

| Model  | Training cost | Interpretability       | Data needs | Latency |
| ------ | ------------- | ---------------------- | ---------- | ------- |
| N-gram | O(N) counts   | High (inspect counts)  | Small      | μs      |
| Markov | O(N) counts   | High (transition table)| Small      | μs      |
| LSTM   | GPU-hours     | Low (black box)        | Large      | ms      |

### Why JSON for persistence instead of pickle?
Pickle is faster but executes arbitrary code on load, so a malicious model
file is a shell on the server. JSON is:
- **Safe** — no code execution path on load
- **Inspectable** — any editor can read it
- **Portable** — JavaScript, Go, anything can parse it

### Why in-memory rate limiting and model caching?
Deliberate simplicity. The token-bucket algorithm is identical whether the
bucket state lives in a dict, in Redis, or behind an API gateway — only the
storage changes. Same for the model cache. See §5 for what breaks at scale.

### Why absolute imports everywhere?
`from src.config import ...` works uniformly from the test runner, from
`uvicorn src.api.main:app`, and from the CLI entry point (which
prepends the project root to `sys.path`). Relative imports would couple to
the runner.

### Why keep PyTorch optional?
The n-gram and Markov models are the teaching surface and don't need it. A
new contributor can install `requirements.txt`, run every test, hit every
API endpoint, and never see a torch error.

---

## 5. Production Gaps (known and intentional)

What this project does *not* do, flagged so nobody is surprised:

- **No subword tokenization.** Whitespace + regex tokenization means
  `"pretraining"` and `"pre-training"` are different tokens. Real systems
  use BPE, WordPiece, or SentencePiece.
- **Per-process state.** Rate-limit buckets and model caches live in
  module globals. Run two uvicorn workers and you get two caches, two
  independently-refilling buckets per IP, and no shared metrics.
  - *Rate limiter:* optional Redis backend is wired in. Set
    `REDIS_URL=redis://host:6379` and install the `redis` package and
    every worker shares one fixed-window counter per IP. Unset, the
    in-memory token bucket keeps running.
  - *Model cache:* best handled by (a) warming at startup — already done
    for the count-based models via the `lifespan` hook — and (b) sharing
    neural weights across workers via the `AUTOCOMPLETE_*_CHECKPOINT_*`
    env vars. Redis is the wrong tool here: models are live Python
    objects, not bytes.
- **No authentication.** Every endpoint is public. A real deployment would
  sit behind API-key auth, OAuth, or an API gateway.
- **CORS wildcard.** `allow_origins=["*"]` is fine for a demo. In
  production, pin it to the actual frontend origin.
- **Small corpus, inflated perplexity.** On the 40-sentence sample corpus,
  evaluated perplexity is huge (millions) because the test split contains
  many n-grams that were never seen in training. This is a small-data
  artefact, not a model bug — swap in WikiText-103 or Gutenberg and the
  numbers collapse into the 50–300 range. Top-k accuracy and diversity
  are the better signals on this corpus.
- **Observability: two endpoints, two audiences.** `/metrics` returns a
  hand-rolled JSON summary — small enough to read in one screen and used
  by the React Metrics page. When the optional
  `prometheus-fastapi-instrumentator` package is installed, `/metrics/prom`
  additionally serves Prometheus text format (counters + latency
  histograms) for Grafana / Alertmanager scrapes. A real deployment would
  keep just the Prometheus path; this repo keeps both for the teaching
  contrast.

---

## 5a. Checkpoint Schema History

Neural checkpoint bundles are versioned **per model family** — their
state-dict layouts evolve independently. Each bundle is a pair of files:
`<path>.safetensors` for the weights and `<path>.json` for the
vocabulary/tokenizer/hyperparameters.

### `LSTMModel` schema versions

| Version | Introduced | Layout change | Reloadable on `main`? |
|---|---|---|---|
| v1 | pre-PR-#6 | `Embedding + LSTM + Linear(hidden→vocab)` (no tying). Meta fields: `vocab` list. | **No** — refused with a clear retrain message (no migration shim; the tied-head saves different state-dict keys). |
| v2 | PR #6 | Adds a `hidden→embed` projection and ties the output head to the embedding weight (Press & Wolf 2016). State-dict gains `proj.*`, drops `fc.bias`. | Yes. |
| v3 | PR #14 | Same layer layout as v2, but vocab is driven by an external `BPETokenizer`. Meta stores `tokenizer: {type, name}` + `vocab_size` instead of a word list. | Yes — `load()` branches on `schema_version` and re-instantiates the tokenizer. |

### `TransformerModel` schema versions

| Version | Introduced | Layout change | Reloadable on `main`? |
|---|---|---|---|
| v1 | PR #11 | Tied embedding + learned absolute positional embedding + N pre-norm decoder blocks + LayerNorm + tied LM head. Word-level vocab. | Yes. |
| v2 | PR #14 | Same layer layout, subword vocab via `BPETokenizer`. Meta carries tokenizer identity. | Yes — same branching pattern as LSTMModel v3. |

### Rules of thumb

- Bumping a schema version is a **breaking change for that family's
  bundles**. Older bundles refuse to load with a clear retrain message;
  there are no migration shims unless explicitly requested.
- Families version independently — upgrading the LSTM from v2 to v3
  doesn't affect Transformer bundles, and vice versa.
- `torch.save` is intentionally never used. Bundles are `safetensors`
  (weights) + JSON (metadata) — no format that can execute code on
  load. The threat model is loading a checkpoint from an untrusted
  source.
- The four long-trained bundles under `models/*_long.*` (generated by
  `scripts/bench_full_train.py`) are on the latest schema for each
  family (LSTM v2 word / v3 BPE, Transformer v1 word / v2 BPE).

---

## 6. Scaling Path

If the goal were to take this from demo to service, here is the shortest
credible path without a full rewrite:

1. **Move state out of the process.** Rate-limit buckets and request
   metrics → Redis. Model cache → a shared read-only artefact on disk (or
   object storage) loaded by every worker at startup.
2. **Run multiple workers.** `uvicorn --workers N` or gunicorn with
   uvicorn workers. With step 1 done, this is safe.
3. **Put a gateway in front.** TLS termination, auth, per-route rate limits,
   request-size caps.
4. **Export real metrics.** `prometheus_fastapi_instrumentator`, ship to
   Grafana. Alert on error rate and p95 latency, not CPU.
5. **Replace the model if needed.** The `predict_next` interface is the
   seam — swap in a transformer or a hosted model behind the same signature
   without touching the frontends.

---

## 7. File Map

```
text-autocomplete/
├── src/
│   ├── __init__.py          # Public API exports (keep in sync)
│   ├── config.py            # Hyperparameters, paths, API settings
│   ├── data_loader.py       # Corpus, normalize, tokenize, split
│   ├── ngram_model.py       # N-gram LM, backoff + interpolation, JSON persist
│   ├── markov_model.py      # First-order Markov, Laplace, text generation
│   ├── neural_model.py      # Optional LSTM (torch-guarded)
│   ├── beam_search.py       # Beam search with length penalty
│   ├── evaluation.py        # Perplexity, top-k, diversity, coverage, confidence
│   └── api/
│       └── main.py          # FastAPI app, middleware, endpoints
├── frontend/                # React 19 SPA (Vite + Tailwind v4 + shadcn)
├── tests/                   # 212+ tests
├── cli.py                   # train | predict | eval | info
├── docs/
│   ├── ARCHITECTURE.md      # This file
│   └── GLOSSARY.md          # Terminology reference
├── requirements.txt
└── README.md
```
