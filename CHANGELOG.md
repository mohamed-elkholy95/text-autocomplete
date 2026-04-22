# Changelog

All notable changes to this project are documented here. Dates are
ISO-8601. The project does not yet publish versioned releases; each
section below corresponds to a landmark on `main`.

## Unreleased

### Added
- **Pretrained-HF eval (`scripts/bench_hf_eval.py`)** — loads any `AutoModelForCausalLM` and scores it on a CONTIGUOUS WikiText-2 test split (90/10 character-level cut) with sliding-window perplexity + 300-probe top-1/top-5 next-word accuracy. Outputs markdown; covers local families + HF side-by-side. Writes to `outputs/hf_eval.md`.
- **XL in-house Transformer (`scripts/bench_train_xl.py`)** — d_model=512 / 8 heads / 12 layers / 20 epochs / seed 42. Produces `models/transformer_wikitext2_xl.{safetensors,json}` (223 MB). Train ~5.8 min on RTX 5080. Held-out PPL 1647.8 on the *shuffled* split — 44 % worse than the d=256/6L baseline (1147 PPL) because training loss saturated at 249 PPL while test stayed at 1647, i.e. the bigger model overfit on the 2 M-token corpus.
- **Stateful-BPTT bench (`scripts/bench_stateful.py`)** — runs stateless vs stateful LSTM fits back-to-back on WikiText-2 with identical config/seed and prints a markdown delta table. Closes roadmap R10: at hidden 256 / 2 layers / 12 epochs, stateful was +3.5 % *worse* on held-out PPL, so the `stateful=False` default stays.
- **Lazy-loaded React routes** — Autocomplete, Attention, and Metrics pages are now code-split. Initial page load ships only the Overview bundle + shared chunks (react, radix); Recharts-heavy routes load on demand.
- **Request-ID middleware** — every response carries `X-Request-ID` (echoed from the request, or freshly minted UUID16). Paired with structured JSON logs (`LOG_FORMAT=json`) for cross-layer request tracing.
- **Rate-limit hint headers** — every response includes `X-RateLimit-Limit` and `X-RateLimit-Remaining` so clients can back off before hitting 429.
- **OpenAPI tag grouping** — `/docs` now groups endpoints under `prediction`, `evaluation`, `system`, and `observability`.
- **SentencePiece bench script** — `scripts/bench_sp_train.py` trains an SP tokenizer on WikiText-2 then fits a small LSTM through it, for comparison against the HuggingFace BPE path.

## 2026-04-21 — `d68111a` (bundle split + expanded `/health` + JSON logs + SentencePiece + mypy)

### Added
- Vite `manualChunks` splits the production bundle (react / charts / radix / app); no more 500 kB warning.
- `/health` now returns a `capabilities` map (torch, transformer, bpe, prometheus, redis_connected, auth_required).
- `LOG_FORMAT=json` switches to structured one-line JSON logs suitable for Loki/CloudWatch.
- `SPTokenizer` in `src/sp_tokenizer.py` — Google SentencePiece adapter (Unigram or BPE) with `train_from_corpus()` helper.
- `tests/test_cli.py` (7 tests) and `tests/test_sp_tokenizer.py` (5 tests).
- `mypy` job in CI with permissive `mypy.ini`.

### Fixed
- Missing `import torch.nn.functional as F` at module level in `src/neural_model.py` (was only imported inside one method).

## 2026-04-21 — `5e6ddc6` (AWD dropout + per-model cost + eval summary + GH templates + GPU workflow)

### Added
- `LSTMModel` optional AWD-style knobs: `input_dropout` and `embedding_dropout` (both default 0.0 → no behaviour change on existing checkpoints).
- Per-model rate-limit cost: `ngram`/`markov` = 1, `lstm` = 4, `transformer` = 6 tokens per call; overridable via `AUTOCOMPLETE_COST_<MODEL>` env vars.
- `GET /eval/summary` — live held-out perplexity + top-1/5 for every available model; React Metrics page renders it as a grouped bar chart.
- `.github/ISSUE_TEMPLATE/*`, `.github/pull_request_template.md`, and a self-hosted-GPU workflow `.github/workflows/gpu-bench.yml`.

## 2026-04-21 — `a28940e` (neural /generate + BPE aliases + beam bench + schema doc)

### Added
- `POST /generate` supports `model: markov|lstm|transformer` (neural paths sample from softmax top-20).
- Catalogue aliases `lstm-bpe` / `transformer-bpe` on `/autocomplete` and `/autocomplete/batch` (auto-imply `tokenizer=bpe`).
- `scripts/bench_beam.py` — greedy vs beam search comparison across all four families.
- `docs/ARCHITECTURE.md §5a` — per-family checkpoint schema history.
- CLI `predict` strips leading-space BPE markers for display (raw token shown in parens when different).

## 2026-04-21 — `7ec9279` (CORS pin + auth + model-hit metric + Docker + Attention + codegen)

### Added
- CORS pin (`CORS_ALLOWED_ORIGINS` env var, defaults to Vite dev origin).
- Opt-in API-key auth via `AUTOCOMPLETE_API_KEY`; `/health`, `/docs`, `/redoc`, `/openapi.json` stay public.
- Custom Prometheus counter `autocomplete_model_hits_total{model,tokenizer}`.
- `POST /attention` endpoint + `TransformerModel.attention_matrices()` + React `/attention` page with lime-scaled heatmap grid.
- Two-stage `Dockerfile` (Node build + Python runtime) and `docker-compose.yml`.
- FastAPI auto-mounts `frontend/dist/` via `StaticFiles` when present — one-process deployment.
- `openapi-typescript` codegen — `npm run gen:api` regenerates TS types from `/openapi.json`.
- `sonner` toast on Autocomplete and Metrics error paths.

## 2026-04-21 — `5ddae90` + `5514113` (Streamlit → React 19 migration)

### Added
- Full React 19 + Vite 8 + TypeScript + Tailwind CSS v4 + shadcn/ui (`new-york-v4` style) SPA under `frontend/`.
- Feature-parity React pages: Overview, Autocomplete, Metrics.
- 4-colour brand palette from colours.cafe wired through `@theme` + `@theme inline` (blue primary, lime accent, slate foreground, bone muted). Dark mode included.
- README "Web UI" section with prerequisites, dev flow, checkpoint env vars, production build, and frontend tree.

### Removed
- `streamlit_app/` (app.py + 3 pages, ~1,200 LOC).
- `streamlit`, `plotly`, `pandas`, `pyarrow` from `requirements.txt` (~80 MB install savings).
- `STREAMLIT_THEME` constant and unused `Dict` import from `src/config.py`.
- Every Streamlit reference from README, ARCHITECTURE, docstrings, and test comments.

## 2026-04-21 — `d254808` (full-corpus multi-epoch training driver)

### Added
- `scripts/bench_full_train.py` — trains LSTM and Transformer in both word-level and BPE variants (2–4× larger configs, 15–25 epochs) on WikiText-2. Saves four checkpoint bundles under `models/*_long.*`.

### Measured
- LSTM/word 25 ep → PPL 3335 (overfit vs 10-ep 1537 baseline).
- Transformer/word 15 ep → PPL 1197 (tied with 5-ep 1147 baseline).
- LSTM/BPE 25 ep → PPL 1951 (−32 % vs 3-ep 2861).
- Transformer/BPE 15 ep → PPL 1406 (−54 % vs 3-ep 3073).

## 2026-04-21 — `b378952` (optional Prometheus + Redis rate limiter)

### Added
- `/metrics/prom` Prometheus exposition endpoint (opt-in via `prometheus-fastapi-instrumentator`). Hand-rolled JSON `/metrics` kept for human inspection.
- Redis-backed fixed-window rate limiter (opt-in via `REDIS_URL` + `redis` package). Falls back to the in-memory token bucket when either is missing.
- FastAPI `lifespan` hook preloads ngram + markov models at startup (and the four neural checkpoints when `AUTOCOMPLETE_*_CHECKPOINT_*` env vars are set).
