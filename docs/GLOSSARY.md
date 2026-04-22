# NLP & Language Modeling Glossary

A reference for every term used in this project. Each entry gives a
plain-English definition and points to the code that implements it, so the
glossary stays anchored to real behaviour rather than drifting into
textbook material the project doesn't actually use.

---

## 1. Core Concepts

### Token
A unit of text after splitting — usually a word or a punctuation mark.
`"Machine learning is great."` → `["machine", "learning", "is", "great", "."]`.
- **Code:** `src/data_loader.py → tokenize()`

### Vocabulary (vocab, `V`)
The set of unique tokens seen during training. `|V|` is the upper bound on
what the model can ever predict; a model never outputs a word it hasn't
seen.
- **Code:** `NGramModel._vocab`, `MarkovChainModel._word_to_idx`

### Corpus
A body of text used for training. This project ships a 40-sentence
built-in corpus covering ML, software engineering, data science, and
general tech, repeated 5× to give the statistical models enough counts.
- **Code:** `src/data_loader.py → SAMPLE_TEXTS`, `load_corpus_from_file()`

### Language Model (LM)
A model that assigns probability to sequences of words. The basic
question: *given some words, what's likely to come next?*
- **Code:** every model in `src/` implementing `predict_next()`

### Context
The window of preceding tokens fed to the model. For an n-gram model, the
context is the last `n-1` tokens; for a Markov chain, just the last one.
- **Code:** sliced inside each `predict_next()`. `CONTEXT_WINDOW = 50` in
  `src/config.py` is declared as a future cap for long inputs but is not
  yet referenced by the models — a deliberate placeholder, not live code.

### Normalization
Cleaning raw text before tokenization: Unicode NFC, smart-quote/dash
folding, whitespace collapse. Without this, `"café"` (composed) and
`"cafe\u0301"` (decomposed) count as different words.
- **Code:** `src/data_loader.py → normalize_text()`

---

## 2. Statistical Concepts

### N-gram
A contiguous sequence of `n` tokens.
- **Unigram** (n=1): `"cat"`
- **Bigram** (n=2): `"the cat"`
- **Trigram** (n=3): `"the cat sat"`
- **Code:** `src/ngram_model.py`, `src/data_loader.py → build_ngrams()`

### Markov Property
The assumption that the future depends only on the present state, not on
the full history: `P(w_n | w_1..w_{n-1}) ≈ P(w_n | w_{n-1})` for a
first-order chain. An n-gram model uses an `(n-1)`-order Markov
assumption.
- **Code:** `src/markov_model.py` (first-order), `src/ngram_model.py`

### Maximum Likelihood Estimation (MLE)
The "just count" approach: `P(w | context) = count(context, w) / count(context)`.
No smoothing. In this project, you get pure MLE by setting
`MarkovChainModel(smoothing=0.0)`.
- **Code:** `MarkovChainModel._get_transition_probs()` with `smoothing=0.0`

### Sparse data problem
With vocab `V`, there are `V^n` possible n-grams; for `V=10,000` and
`n=3` that's `10^12`. Almost all will be unseen in any realistic corpus,
and MLE assigns them probability 0 — which makes perplexity infinite
and breaks prediction for any unseen context. The fix is smoothing.

### Smoothing
Techniques that shift some probability mass onto unseen events.

| Technique       | How it works                          | Trade-off                                          |
| --------------- | ------------------------------------- | -------------------------------------------------- |
| Laplace (add-k) | Add `k` to every count                | Simple, over-smooths when `V` is large             |
| Backoff         | If higher-order unseen, drop to lower | Ignores lower orders when a higher one is seen     |
| Interpolation   | Weighted blend of all orders with λ   | Better quality, needs λ tuning                     |

- **Code:** `NGramModel.predict_next()` (backoff),
  `predict_next_interpolated()` (interpolation),
  `MarkovChainModel._get_transition_probs()` (Laplace)

### Transition matrix
A matrix where `T[i][j] = P(next_word = j | current_word = i)`, with
each row summing to 1.
- **Code:** `MarkovChainModel._transitions` (stored as
  `Dict[str, Counter]` for sparsity, not a dense matrix)

### Log-probability
`log P(sequence) = Σ log P(w_i | context_i)`. Working in log-space turns
multiplication into addition and avoids floating-point underflow when
multiplying hundreds of small probabilities.
- **Code:** `src/beam_search.py:174` (`new_log_prob = log_prob + np.log(prob)`)

---

## 3. Evaluation Metrics

### Perplexity (PPL)
The dominant metric for language models. Geometric mean of inverse
probabilities on test data. Lower is better.

| PPL     | Intuition                                    |
| ------- | -------------------------------------------- |
| 1       | Perfect prediction (impossible in practice)  |
| ~10     | Model narrows each choice to ~10 candidates  |
| ~`|V|`  | No better than uniform random guessing       |

Formula: `PPL = exp(-1/N × Σ log P(w_i | context_i))`. On a small corpus
with many unseen test n-grams, PPL inflates dramatically — so treat
small-corpus PPL as a sanity check, not a benchmark.
- **Code:** `src/evaluation.py → compute_perplexity()`

### Top-k accuracy
Fraction of test cases where the true next word appears in the model's
top-k predictions. Top-1 is strict; top-5 matches how autocomplete UIs
actually surface suggestions.
- **Code:** `src/evaluation.py → autocomplete_accuracy()`

### Prediction diversity
`unique(top-1) / total_predictions`. A model that always predicts
`"the"` has high accuracy on common positions but zero diversity — and
is useless as a product. Diversity catches those trivial wins.
- **Code:** `src/evaluation.py → prediction_diversity()`

### Vocabulary coverage
Fraction of the reference vocab the model ever predicts across the test
set. High coverage + high accuracy = a model worth shipping.
- **Code:** `src/evaluation.py → vocabulary_coverage()`

### Confidence (top-1 prob, entropy, margin)
Three ways to ask *how sure* the model is:
- **Top-1 probability** — mass on the best prediction
- **Shannon entropy** — `H = −Σ p log₂ p`; high H = spread-out, uncertain
- **Margin** — `p(top-1) − p(top-2)`; big margin = clear winner

- **Code:** `src/evaluation.py → prediction_confidence()`

---

## 4. Decoding Strategies

### Greedy decoding
Always take the argmax at each step. Fastest option; equivalent to
beam search with `beam_width=1`. Often produces repetitive or
locally-optimal-but-globally-bad sequences.

### Beam search
Keep `beam_width` parallel hypotheses; at each step expand each one
with top-k candidates, score, and prune back to `beam_width`. Better
sequences at a cost proportional to `beam_width × candidates_per_step`.
- **Code:** `src/beam_search.py → BeamSearchDecoder`

### Length penalty
Without normalization, shorter sequences always win because each
additional token multiplies the probability by `<1.0`. This project
uses the simple form:

```
score = log_prob / length^α     (α = 0.6 by default)
```

A common alternative is Wu et al. (2016) GNMT:

```
score = log_prob / ((5 + length) / 6)^α
```

Both aim to remove the short-sequence bias; GNMT's form gives a
smoother penalty for very short sequences.
- **Code:** `BeamSearchDecoder._length_normalized_score()`

### Temperature sampling
Rescales logits before softmax: `p'_i ∝ p_i^(1/T)`.
- `T < 1.0` — sharper, more deterministic (boring but safe)
- `T = 1.0` — original distribution
- `T > 1.0` — flatter, more random (creative but noisier)

Used for text generation, not for ranked autocomplete.
- **Code:** `MarkovChainModel.generate_text()` (temperature arg)

---

## 5. Neural Concepts (optional path)

### Embedding
A learned mapping from discrete tokens to dense vectors. Similar words
end up close in vector space; classic example:
`vec(king) − vec(man) + vec(woman) ≈ vec(queen)`.
- **Code:** `LSTMModel.embedding` (`nn.Embedding`)

### LSTM (Long Short-Term Memory)
A recurrent network with gated memory cells (forget/input/output gates)
designed to learn long-range dependencies without the vanishing-gradient
problem of vanilla RNNs.
- **Code:** `src/neural_model.py → LSTMModel`

### Dropout
Randomly zero a fraction of neurons during training so the network
can't rely on any single unit. Off at inference.
- **Code:** `LSTMModel.__init__()` — `dropout=0.2` from `NEURAL_CONFIG`

### Torch-optional fallback
If `import torch` fails, the neural module sets `HAS_TORCH = False` and
all training/prediction functions return deterministic mock values so
the rest of the project and the test suite still work.
- **Code:** `src/neural_model.py → HAS_TORCH` guard

---

## 6. API & Engineering Concepts

### Token bucket (rate limiting)
Each client gets a bucket that holds up to `MAX_TOKENS` tokens. Every
request consumes one token; tokens refill at `REFILL_RATE` per second.
Empty bucket → HTTP 429. This is the same algorithm AWS API Gateway
and Stripe use — simple, memory-efficient, and allows short bursts
while enforcing a long-term rate. In this project: 30-token burst,
2-token/sec refill.
- **Code:** `src/api/main.py → _check_rate_limit()`

### Model caching
Trained models are stored in a module-level dict (`_model_cache`) so
requests after the first don't retrain. Per-process, not shared across
workers — a deliberate simplification (see ARCHITECTURE §5).
- **Code:** `src/api/main.py → _get_trained_ngram()`, `_get_trained_markov()`

### CORS (Cross-Origin Resource Sharing)
Browser-enforced rule that a page on origin A can only call APIs on
origin B if B sends explicit Access-Control-Allow-* headers. The API
opens CORS fully (`allow_origins=["*"]`) for the demo; production would
pin it to the real frontend.
- **Code:** `src/api/main.py → CORSMiddleware`

### Pydantic validation
Every request and response is a typed Pydantic model. FastAPI rejects
malformed input with a structured 422 before it reaches the endpoint,
and auto-generates OpenAPI/Swagger docs from the same schemas.
- **Code:** `AutocompleteRequest`, `AutocompleteResponse`, `BatchRequest`,
  `GenerateRequest`, `GenerateResponse` in `src/api/main.py`

### Health check
A dependency-free `GET /health` endpoint used by load balancers and
orchestrators (k8s, Docker Compose) to decide whether to route traffic.
Skipped by the rate limiter so external probes can't lock themselves out.
- **Code:** `src/api/main.py → @app.get("/health")`

### Middleware
Code that wraps every request/response cycle. Order matters here:
IP resolution → rate-limit check → start timer → endpoint → stop timer
→ stamp `X-Response-Time-Ms` → log. Middleware is where cross-cutting
concerns (auth, rate limits, logging) live in FastAPI.
- **Code:** `src/api/main.py → rate_limit_and_metrics_middleware`

### JSON model persistence
Models serialize to human-readable JSON rather than pickle. Two
reasons: pickle executes arbitrary code on load (a security risk for
models downloaded over the network), and JSON is portable and
debuggable with any text editor.
- **Code:** `NGramModel.save()` / `load()`

### Prometheus exposition format
The `/metrics/prom` endpoint serves metrics in Prometheus's text-based
format: one metric per line as `<name>{<label>=<value>,...} <float>`,
preceded by `# HELP` and `# TYPE` comments. Prometheus servers scrape
this endpoint on an interval and store the values in a time-series
database. Contrast with the hand-rolled JSON `/metrics` which is a
one-shot counter snapshot readable by humans.
- **Code:** `src/api/main.py` (optional, needs `prometheus-fastapi-instrumentator`)

### Token bucket vs fixed window (rate limiting)
Two common rate-limit algorithms. **Token bucket** (in-memory path):
each client has a bucket that refills at a constant rate; each request
drains some tokens; empty bucket → 429. Allows short bursts while
enforcing long-term limits. **Fixed window** (Redis path): count
requests in discrete time buckets (e.g. 15-second windows); when the
count exceeds the budget, reject. Simpler to implement atomically with
INCR/EXPIRE, but allows burstier traffic at window boundaries.
- **Code:** `src/api/main.py → _check_rate_limit_memory` / `_check_rate_limit_redis`

### Per-model rate-limit cost
Neural predictions cost more CPU/GPU than n-gram lookups, so charging
every endpoint the same token rate under-charges the neural paths.
This project assigns each model a cost (ngram=1, markov=1, lstm=4,
transformer=6) and drains the bucket accordingly, so a malicious
client that floods the transformer endpoint hits 429 ~6× sooner than
one flooding n-gram.
- **Code:** `src/api/main.py → _model_cost`

### Causal self-attention
The "causal" variant of multi-head attention used by decoder-only
transformers. Its mask zeros out the upper-right triangle of the
attention matrix, so query token *i* can only attend to key tokens
*0..i* (not *i+1..T*). This is what lets a transformer be trained to
predict next tokens and sampled autoregressively at inference without
leaking future information.
- **Code:** `src/transformer_model.py → forward()` (`torch.triu(...)` mask)

### Attention-visualisation heatmap
The React `/attention` page renders per-layer, per-head attention
weight matrices as colour-coded grids so you can see *what the model
looks at* when predicting each token. Every row sums to 1; near-zero
columns past the diagonal confirm the causal mask is active.
- **Code:** `src/transformer_model.py → attention_matrices()`, `frontend/src/pages/Attention.tsx`

### AWD-LSTM dropout variants
Beyond stock nn.LSTM dropout between layers, the AWD-LSTM recipe
(Merity et al. 2018) adds: **input dropout** on the embedded sequence
before the LSTM, **embedding dropout** applied per-row on the
embedding matrix during training (equivalent to randomly dropping
tokens to `<unk>` but continuously differentiable), and optional
**weight dropout** (DropConnect on the recurrent weight matrices; not
shipped here). This project exposes the first two as opt-in kwargs
that default to 0.0 so existing checkpoints behave bit-identically.
- **Code:** `src/neural_model.py → LSTMModel.__init__(input_dropout=..., embedding_dropout=...)`

### SPA dev proxy
In development, the Vite dev server runs on port 5173 and proxies
`/api/*` requests to the FastAPI backend on port 8010. From the
browser's perspective, UI and API share the same origin, so there's
no CORS preflight handshake in the dev loop. In production the same
effect is achieved by mounting the built `frontend/dist/` bundle as
StaticFiles on the FastAPI app itself.
- **Code:** `frontend/vite.config.ts → server.proxy`, `src/api/main.py → StaticFiles mount`

### `*-bpe` catalogue alias
A request-level convenience: `model: "transformer-bpe"` in the
`/autocomplete` body is auto-normalised to
`model=transformer, tokenizer=bpe` by the Pydantic request model.
Saves clients from having to set two fields to mean one thing. The
base model id is still what the cache and metrics see.
- **Code:** `src/api/main.py → AutocompleteRequest._validate_tokenizer_flags`
