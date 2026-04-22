"""
Enhanced FastAPI for Text Autocomplete
========================================

REST API providing real-time text autocomplete predictions using multiple
language models. Designed for integration into text editors, search bars,
and chat applications.

EDUCATIONAL CONTEXT:
-------------------
A REST API (REpresentational State Transfer) is the most common way to expose
machine learning models as web services. The key idea: HTTP requests in,
JSON responses out.

API DESIGN PRINCIPLES:
1. Clear endpoints: Each URL path does ONE thing
2. Validation: Input is validated before processing (Pydantic models)
3. Documentation: Auto-generated from code (FastAPI + OpenAPI)
4. Statelessness: Each request contains all info needed to process it
5. Error handling: Graceful failures with informative error messages

ARCHITECTURE:
    Client (browser/app) → HTTP POST /autocomplete → FastAPI server
                                                        ↓
                                                    Model inference
                                                        ↓
                              Client ← HTTP 200 JSON response ← Results
"""

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints, model_validator

from src.config import API_TITLE, API_VERSION, TOP_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional observability + distributed-state dependencies
# ---------------------------------------------------------------------------
# Both blocks follow the same "optional dependency" pattern used for torch
# and transformers: import guarded, feature flag exposed, everything still
# works on a minimal install.
#
# 1. prometheus-fastapi-instrumentator exposes a proper Prometheus scrape
#    endpoint at /metrics/prom. The hand-rolled JSON /metrics stays — the
#    two coexist so learners can compare "what I built" against "what the
#    ecosystem ships".
# 2. redis is used to share the rate-limit state across uvicorn workers
#    when REDIS_URL is set. Without it, the in-memory token bucket keeps
#    working (fine for a single-worker demo).

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

# Populated by the lifespan hook when REDIS_URL is set AND redis is
# installed. Stays None otherwise, which flips _check_rate_limit back to
# the in-memory bucket.
_redis_client = None  # type: ignore[var-annotated]

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hook
# ---------------------------------------------------------------------------
# FastAPI runs this async context once at process start (before the first
# request) and once at shutdown. Two jobs:
#   1. Warm the cheap models so the first real request isn't a cold train.
#   2. Connect to Redis if REDIS_URL is set — used by the rate limiter to
#      share state across workers. A failed connect is logged and the API
#      falls back to in-memory limiting; we don't abort boot for it.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the two count-based models (both fit in ~tens of ms on the
    # sample corpus). Neural models stay lazy — they're larger and the
    # "torch optional" anchor means we can't assume they're importable.
    try:
        _get_trained_ngram()
        _get_trained_markov()
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Model preload skipped: %s", exc)

    # Redis rate-limit backend (opt-in).
    global _redis_client
    redis_url = os.getenv("REDIS_URL")
    if redis_url and HAS_REDIS:
        try:
            _redis_client = aioredis.from_url(redis_url, decode_responses=True)
            await _redis_client.ping()
            logger.info("Redis rate-limit backend active at %s", redis_url)
        except Exception as exc:
            logger.warning(
                "Redis connect failed (%s); falling back to in-memory rate limit.",
                exc,
            )
            _redis_client = None
    elif redis_url and not HAS_REDIS:
        logger.warning(
            "REDIS_URL set but 'redis' package not installed — "
            "using in-memory rate limit instead."
        )

    yield

    if _redis_client is not None:
        await _redis_client.aclose()


# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan,
    description="""
    Text Autocomplete API — Get real-time word completion suggestions.

    Supports multiple language models:
    - **ngram**: Classic n-gram model with backoff smoothing (fast, interpretable)
    - **markov**: Markov chain model (simple transition probabilities)
    - **lstm**: 2-layer LSTM with tied weights (requires PyTorch)
    - **transformer**: Decoder-only transformer with causal self-attention (requires PyTorch)

    ## How Autocomplete Works
    1. Send your text as input
    2. The API tokenizes it and extracts the context
    3. The selected model predicts the most likely next words
    4. Top-k suggestions are returned with probabilities
    """,
)

# ---------------------------------------------------------------------------
# Rate Limiting — Token Bucket Algorithm
# ---------------------------------------------------------------------------
# Rate limiting prevents abuse and ensures fair access. We use a simple
# in-memory token bucket per client IP. In production, you'd use Redis
# or a dedicated rate limiter like slowapi.
#
# TOKEN BUCKET ALGORITHM:
# - Each client has a "bucket" that holds up to MAX_TOKENS tokens
# - Each request consumes one token
# - Tokens are refilled at REFILL_RATE per second
# - If the bucket is empty, the request is rejected with HTTP 429
#
# This is the same algorithm used by AWS API Gateway, Stripe, and
# most cloud rate limiters. It's simple, memory-efficient, and allows
# short bursts while enforcing long-term rate limits.

RATE_LIMIT_MAX_TOKENS = 30      # Maximum burst size
RATE_LIMIT_REFILL_RATE = 2.0    # Tokens added per second
# A bucket is evicted once it has been idle long enough to have fully
# refilled twice over. Bounds memory growth under the spoofable-IP threat.
RATE_LIMIT_BUCKET_TTL = (RATE_LIMIT_MAX_TOKENS / RATE_LIMIT_REFILL_RATE) * 2
# Hard cap on the number of live buckets; beyond this we evict the oldest
# first even if their TTL hasn't expired. Prevents a flood of unique IPs
# from exhausting memory.
RATE_LIMIT_MAX_BUCKETS = 10_000

_rate_buckets: Dict[str, Dict] = {}


async def _check_rate_limit(client_ip: str, cost: int = 1) -> bool:
    """Dispatch to the Redis limiter if it's connected, else in-memory.

    ``cost`` is the number of bucket tokens this request consumes (see
    ``_model_cost``). cost=1 matches the pre-cost-aware behaviour.
    Kept async because the Redis path awaits network I/O. The memory
    path is still sync under the hood — we just return its result.
    """
    if _redis_client is not None:
        return await _check_rate_limit_redis(client_ip, cost)
    return _check_rate_limit_memory(client_ip, cost)


# Redis key layout: one counter per (IP, window). INCR is atomic, so
# two workers racing on the same key still share one bucket. We use a
# fixed-window counter (simpler than a full token bucket) with the same
# budget — 30 requests per 15 s — as the in-memory limiter's steady state.
_RATE_LIMIT_WINDOW_SECONDS = 15
_RATE_LIMIT_WINDOW_BUDGET = 30


async def _check_rate_limit_redis(client_ip: str, cost: int = 1) -> bool:
    """Fixed-window limiter backed by Redis INCRBY + EXPIRE.

    The teaching contrast with the in-memory token bucket: here the state
    lives outside the process, so every uvicorn worker sees the same
    counter. That's exactly what the in-memory bucket *can't* give you
    once you scale beyond one worker. If Redis errors, we fail-open
    (allow the request) rather than locking everyone out — the API
    should survive a flaky cache.
    """
    try:
        window = int(time.time() // _RATE_LIMIT_WINDOW_SECONDS)
        key = f"ratelimit:{client_ip}:{window}"
        # INCRBY is atomic and lets a single request drain `cost` tokens
        # in one round trip — matches the in-memory bucket's semantics.
        count = await _redis_client.incrby(key, cost)
        if count == cost:
            # First hit in this window — set the TTL so the key expires.
            await _redis_client.expire(key, _RATE_LIMIT_WINDOW_SECONDS)
        return count <= _RATE_LIMIT_WINDOW_BUDGET
    except Exception as exc:
        logger.warning("Redis rate-limit check failed, failing open: %s", exc)
        return True


def _check_rate_limit_memory(client_ip: str, cost: int = 1) -> bool:
    """Check if a client has remaining rate limit tokens.

    Returns True if the request is allowed, False if rate-limited. The
    bucket table is pruned opportunistically so it can't grow without
    bound when many unique client IPs hit the API.
    """
    now = time.monotonic()

    # Opportunistic eviction: drop any bucket that's been idle past the
    # TTL (i.e. would already be at full tokens again anyway).
    if _rate_buckets:
        stale = [
            ip for ip, b in _rate_buckets.items()
            if now - b["last_refill"] > RATE_LIMIT_BUCKET_TTL
        ]
        for ip in stale:
            _rate_buckets.pop(ip, None)

    # Hard cap: if we're still over the limit, drop the oldest buckets.
    if len(_rate_buckets) >= RATE_LIMIT_MAX_BUCKETS:
        for ip, _ in sorted(
            _rate_buckets.items(), key=lambda kv: kv[1]["last_refill"]
        )[: len(_rate_buckets) - RATE_LIMIT_MAX_BUCKETS + 1]:
            _rate_buckets.pop(ip, None)

    bucket = _rate_buckets.get(client_ip)
    if bucket is None:
        bucket = {"tokens": RATE_LIMIT_MAX_TOKENS, "last_refill": now}
        _rate_buckets[client_ip] = bucket

    # Refill tokens based on elapsed time since last refill
    elapsed = now - bucket["last_refill"]
    bucket["tokens"] = min(
        RATE_LIMIT_MAX_TOKENS,
        bucket["tokens"] + elapsed * RATE_LIMIT_REFILL_RATE,
    )
    bucket["last_refill"] = now

    # Consume `cost` tokens if available (cost=1 matches the prior
    # per-request behaviour; neural endpoints pass higher values).
    if bucket["tokens"] >= cost:
        bucket["tokens"] -= cost
        return True
    return False


# Only trust X-Forwarded-For when the deploy explicitly opts in (e.g. when
# running behind a known reverse proxy). Otherwise we use the direct peer
# address, which a client can't spoof in HTTP/1.1.
TRUST_FORWARDED_HEADERS = os.getenv("TRUST_FORWARDED_HEADERS", "").lower() in (
    "1", "true", "yes",
)


# ---------------------------------------------------------------------------
# Per-model rate-limit costs
# ---------------------------------------------------------------------------
# A single /autocomplete call with model=ngram is pennies of CPU; the same
# call with model=transformer does a forward pass through a neural net.
# Charging both one token is under-charging the neural paths — a client
# that hammers model=transformer can burn far more server resources than
# a client that hammers model=ngram, despite hitting the same rate limit.
#
# Fix: every autocomplete endpoint drains the bucket by `cost` tokens
# where cost reflects the model's relative work. Cheap models cost 1
# (the baseline); neural models cost more. Tunable via env vars so
# operators can match their deployment's actual latency distribution.

_MODEL_COSTS_DEFAULT = {
    "ngram": 1,
    "markov": 1,
    "lstm": 4,
    "transformer": 6,
}


def _model_cost(model_id: str) -> int:
    """Return the per-request drain for a given model id.

    Falls back to ``AUTOCOMPLETE_COST_<MODEL>`` env vars so operators can
    override without a code change (e.g. if they run a larger transformer
    and want it to cost 10 rather than 6).
    """
    base = model_id.rstrip("-bpe")  # aliases cost the same as their base
    override = os.getenv(f"AUTOCOMPLETE_COST_{base.upper()}")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return _MODEL_COSTS_DEFAULT.get(base, 1)


def _resolve_client_ip(request: Request) -> str:
    """Resolve the client IP for rate-limit bucketing.

    Defaults to ``request.client.host`` because the X-Forwarded-For header
    is freely settable by any HTTP client and would otherwise let a caller
    mint a fresh bucket per request. Operators that really do run behind a
    trusted proxy can set ``TRUST_FORWARDED_HEADERS=1`` to honour the
    first IP in the header chain.
    """
    if TRUST_FORWARDED_HEADERS:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # Standard: the left-most entry is the original client.
            first = xff.split(",")[0].strip()
            if first:
                return first
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Request Metrics Tracking
# ---------------------------------------------------------------------------
# Simple in-memory metrics for monitoring API health. In production,
# you'd use Prometheus, DataDog, or similar observability tools.

_request_metrics: Dict[str, int] = defaultdict(int)


# CORS (Cross-Origin Resource Sharing).
# Dev default is the Vite dev server on :5173. Override with a
# comma-separated env var when deploying, e.g.
#   CORS_ALLOWED_ORIGINS=https://autocomplete.example.com,https://admin.example.com
# Set to "*" only if you really want to expose the API to every origin.
_cors_env = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).strip()
CORS_ALLOWED_ORIGINS = (
    ["*"] if _cors_env == "*"
    else [o.strip() for o in _cors_env.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation (optional). When the package is installed,
# every route gets automatic counters + latency histograms emitted in
# Prometheus text format at /metrics/prom. The hand-rolled JSON /metrics
# below stays put — it's a small learning surface for "what's inside a
# metrics endpoint". Prometheus uses a parallel path at /metrics/prom so
# the two don't collide.
if HAS_PROMETHEUS:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics/prom")
    # Business-level metric on top of the route-level HTTP counters: one
    # counter per (model, tokenizer) combination served. Lets a dashboard
    # answer "how often is transformer-bpe actually used?" without parsing
    # request bodies. Only defined when the instrumentator is installed
    # (no registry to attach to otherwise).
    from prometheus_client import Counter as _PromCounter
    AUTOCOMPLETE_MODEL_HITS = _PromCounter(
        "autocomplete_model_hits_total",
        "Autocomplete predictions served, labeled by model and tokenizer.",
        ["model", "tokenizer"],
    )
else:
    AUTOCOMPLETE_MODEL_HITS = None


@app.middleware("http")
async def rate_limit_and_metrics_middleware(request: Request, call_next):
    """Middleware that enforces rate limiting and tracks request metrics.

    MIDDLEWARE EXPLAINED:
    Middleware wraps every request/response cycle. It runs BEFORE the
    endpoint handler (for rate limiting, auth, logging) and AFTER
    (for response timing, error handling).

    The execution order is:
        Client → Middleware(before) → Endpoint → Middleware(after) → Client
    """
    # Extract client IP (honours X-Forwarded-For only when explicitly trusted)
    client_ip = _resolve_client_ip(request)

    # API-key auth (opt-in). Set AUTOCOMPLETE_API_KEY to any non-empty string
    # and every non-/health request must send `X-API-Key: <that value>`.
    # Not set: auth is disabled. This is the smallest educational surface —
    # a real deployment should use OAuth / mTLS / an API gateway.
    _api_key = os.getenv("AUTOCOMPLETE_API_KEY", "")
    _public_paths = ("/health", "/docs", "/redoc", "/openapi.json")
    if _api_key and not request.url.path.startswith(_public_paths):
        if request.headers.get("x-api-key") != _api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key."},
            )

    # Rate limit check (skip health endpoint to allow monitoring)
    if request.url.path != "/health" and not await _check_rate_limit(client_ip):
        _request_metrics["rate_limited"] += 1
        logger.warning("Rate limited: %s on %s", client_ip, request.url.path)
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again shortly."},
        )

    # Track request count per endpoint
    _request_metrics[f"requests:{request.url.path}"] += 1
    _request_metrics["total_requests"] += 1

    # Time the request
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Stamp the timing header first so the logged line reflects what the
    # client will actually see (matches the order described in
    # docs/ARCHITECTURE.md §2 step 8).
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

    logger.info(
        "request: method=%s path=%s status=%d duration=%.1fms client=%s",
        request.method, request.url.path, response.status_code,
        duration_ms, client_ip,
    )
    return response


# ---------------------------------------------------------------------------
# Shared Model Cache
# ---------------------------------------------------------------------------
# Loading and training models is expensive. We cache them in module-level
# variables so they persist across requests. This is a simple form of caching.
# In production, you'd use a proper caching system like Redis.

_model_cache: Dict[str, object] = {}


def _get_trained_ngram():
    """Get or create a cached trained n-gram model."""
    if "ngram" not in _model_cache:
        from src.data_loader import tokenize, load_sample_data
        from src.ngram_model import NGramModel
        model = NGramModel(n=3)
        model.fit(tokenize(load_sample_data()))
        _model_cache["ngram"] = model
        logger.info("N-gram model trained and cached")
    return _model_cache["ngram"]


def _get_trained_markov():
    """Get or create a cached trained Markov chain model."""
    if "markov" not in _model_cache:
        from src.data_loader import tokenize, load_sample_data
        from src.markov_model import MarkovChainModel
        model = MarkovChainModel()
        model.fit(tokenize(load_sample_data()))
        _model_cache["markov"] = model
        logger.info("Markov chain model trained and cached")
    return _model_cache["markov"]


def _transformer_available() -> bool:
    """Return True when the transformer path can train/serve.

    The API's optional-torch anchor means the transformer only appears
    in /models and is only selectable in /autocomplete when PyTorch is
    importable. Installs without torch keep the pre-PR behaviour
    (ngram + markov only) with no functional change.
    """
    try:
        from src.neural_model import HAS_TORCH
        return bool(HAS_TORCH)
    except Exception:
        return False


def _lstm_available() -> bool:
    """Return True when the LSTM path can train/serve.

    Same optional-torch anchor as the transformer: the LSTM entry only
    shows up in /models and is only selectable in /autocomplete when
    PyTorch is importable. No-torch installs see ngram + markov only.
    """
    try:
        from src.neural_model import HAS_TORCH
        return bool(HAS_TORCH)
    except Exception:
        return False


def _bpe_available() -> bool:
    """Return True when the BPE subword path can train/serve.

    BPE needs torch (for the neural models it feeds) AND the optional
    'transformers' package (for the HF tokenizer). The selector stays
    a no-op on minimal installs — tokenizer='bpe' returns 503 there.
    """
    if not _lstm_available():
        return False
    try:
        from src.bpe_tokenizer import HAS_TRANSFORMERS
        return bool(HAS_TRANSFORMERS)
    except Exception:
        return False


_CHECKPOINT_ENV_VARS = {
    ("lstm", "word"): "AUTOCOMPLETE_LSTM_CHECKPOINT_WORD",
    ("lstm", "bpe"): "AUTOCOMPLETE_LSTM_CHECKPOINT_BPE",
    ("transformer", "word"): "AUTOCOMPLETE_TRANSFORMER_CHECKPOINT_WORD",
    ("transformer", "bpe"): "AUTOCOMPLETE_TRANSFORMER_CHECKPOINT_BPE",
}


def _try_load_checkpoint(model_kind: str, tokenizer_flavour: str):
    """Return a loaded neural model, or None when no usable checkpoint
    is configured.

    The helper intentionally swallows load errors (file missing, schema
    mismatch, corrupt bundle) and returns None so the caller falls back
    to the existing lazy-fit path. Nothing here makes PyTorch required
    — the caller already gates on ``_lstm_available`` / ``_transformer_available``.
    """
    env_var = _CHECKPOINT_ENV_VARS.get((model_kind, tokenizer_flavour))
    if not env_var:
        return None
    path = os.getenv(env_var)
    if not path:
        return None
    try:
        if model_kind == "lstm":
            from src.neural_model import LSTMModel
            model = LSTMModel.load(path)
        else:
            from src.transformer_model import TransformerModel
            model = TransformerModel.load(path)
    except Exception as exc:
        logger.warning(
            "Checkpoint load failed for %s/%s at %s: %s — falling back to lazy-fit.",
            model_kind, tokenizer_flavour, path, exc,
        )
        return None
    # Guard against a word-level checkpoint being served on a BPE request
    # (and vice versa). The checkpoint's own _tokenizer attribute is the
    # source of truth for schema-v3 BPE bundles.
    has_bpe = getattr(model, "_tokenizer", None) is not None
    expected_bpe = (tokenizer_flavour == "bpe")
    if has_bpe != expected_bpe:
        logger.warning(
            "Checkpoint at %s is %s but the request asked for %s — "
            "refusing to serve. Point the right env var at the right file.",
            path,
            "BPE-trained" if has_bpe else "word-level",
            tokenizer_flavour,
        )
        return None
    logger.info(
        "Loaded %s/%s checkpoint from %s",
        model_kind, tokenizer_flavour, path,
    )
    return model


def _maybe_build_bpe_tokenizer(tokenizer_flavour: str, tokenizer_name: Optional[str] = None):
    """Return a BPETokenizer instance or None.

    Raises 503 when the caller asked for BPE but 'transformers' isn't
    installed, or when the requested HF repo can't be resolved — so
    the error surfaces at the API boundary instead of reaching the
    model layer as a generic ImportError."""
    if tokenizer_flavour != "bpe":
        return None
    if not _bpe_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "BPE tokenizer unavailable: install the optional "
                "'transformers' package to enable subword mode."
            ),
        )
    from src.bpe_tokenizer import BPETokenizer, DEFAULT_BPE_NAME
    try:
        return BPETokenizer(name=tokenizer_name or DEFAULT_BPE_NAME)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Failed to load BPE tokenizer "
                f"'{tokenizer_name or DEFAULT_BPE_NAME}': {exc}"
            ),
        )


def _get_trained_transformer(
    tokenizer_flavour: str = "word",
    tokenizer_name: Optional[str] = None,
):
    """Get or create a cached trained decoder-only transformer.

    Uses a deliberately small config (d_model=64, 2 layers, 3 epochs)
    so the first request finishes in a few seconds on CPU. This mirrors
    the CLI eval's --include-transformer defaults.

    ``tokenizer_flavour`` picks the vocabulary: ``"word"`` (default)
    uses the project's whitespace+regex tokenizer; ``"bpe"`` wires in
    a ``BPETokenizer`` so the model trains on subword ids. BPE and
    word-level instances are cached independently. ``tokenizer_name``
    is only meaningful when ``tokenizer_flavour='bpe'`` and identifies
    the HF repo — the cache key embeds the name so different BPE
    vocabularies don't share a cached model."""
    suffix = tokenizer_flavour
    if tokenizer_flavour == "bpe" and tokenizer_name:
        suffix = f"bpe:{tokenizer_name}"
    cache_key = f"transformer:{suffix}"
    if cache_key not in _model_cache:
        if not _transformer_available():
            raise HTTPException(
                status_code=503,
                detail="Transformer model unavailable: PyTorch is not installed.",
            )
        if tokenizer_flavour == "word":
            loaded = _try_load_checkpoint("transformer", tokenizer_flavour)
            if loaded is not None:
                _model_cache[cache_key] = loaded
                return loaded
        elif tokenizer_name is None:
            # Default BPE repo still honours the checkpoint env var.
            loaded = _try_load_checkpoint("transformer", tokenizer_flavour)
            if loaded is not None:
                _model_cache[cache_key] = loaded
                return loaded
        from src.data_loader import tokenize, load_sample_data
        from src.transformer_model import TransformerModel
        bpe = _maybe_build_bpe_tokenizer(tokenizer_flavour, tokenizer_name)
        # Raw text for BPE (tokenizer encodes internally); pre-tokenized
        # list for the word-level path.
        raw_text = load_sample_data()
        data = raw_text if bpe is not None else tokenize(raw_text)
        length = len(data) if isinstance(data, list) else len(data.split())
        model = TransformerModel(
            d_model=64, n_heads=4, n_layers=2, ff_dim=128, max_seq_len=64,
        )
        model.fit(
            data, epochs=3,
            seq_len=min(16, max(length // 4, 4)),
            batch_size=16, lr=3e-4,
            tokenizer=bpe,
        )
        _model_cache[cache_key] = model
        logger.info("Transformer model (%s) trained and cached", suffix)
    return _model_cache[cache_key]


def _get_trained_lstm(
    tokenizer_flavour: str = "word",
    tokenizer_name: Optional[str] = None,
):
    """Get or create a cached trained LSTM.

    Tiny config (embed=32, hidden=64, 1 layer, 3 epochs) so the first
    request finishes in a second or two on CPU. Mirrors the transformer
    helper's lazy-fit-on-sample-corpus pattern; production deployments
    would load a pre-trained checkpoint via LSTMModel.load() instead.

    ``tokenizer_flavour`` picks the vocabulary the same way as the
    transformer helper. ``tokenizer_name`` is only meaningful with
    ``tokenizer_flavour='bpe'`` and selects the HF repo."""
    suffix = tokenizer_flavour
    if tokenizer_flavour == "bpe" and tokenizer_name:
        suffix = f"bpe:{tokenizer_name}"
    cache_key = f"lstm:{suffix}"
    if cache_key not in _model_cache:
        if not _lstm_available():
            raise HTTPException(
                status_code=503,
                detail="LSTM model unavailable: PyTorch is not installed.",
            )
        if tokenizer_flavour == "word" or tokenizer_name is None:
            loaded = _try_load_checkpoint("lstm", tokenizer_flavour)
            if loaded is not None:
                _model_cache[cache_key] = loaded
                return loaded
        from src.data_loader import tokenize, load_sample_data
        from src.neural_model import LSTMModel
        bpe = _maybe_build_bpe_tokenizer(tokenizer_flavour, tokenizer_name)
        raw_text = load_sample_data()
        data = raw_text if bpe is not None else tokenize(raw_text)
        length = len(data) if isinstance(data, list) else len(data.split())
        model = LSTMModel(
            embed_dim=32, hidden_dim=64, num_layers=1, dropout=0.0,
        )
        model.fit(
            data, epochs=3,
            seq_len=min(16, max(length // 4, 4)),
            batch_size=16, lr=1e-3,
            tokenizer=bpe,
        )
        _model_cache[cache_key] = model
        logger.info("LSTM model (%s) trained and cached", suffix)
    return _model_cache[cache_key]


# ---------------------------------------------------------------------------
# Request/Response Models (Pydantic)
# ---------------------------------------------------------------------------
# Pydantic models validate incoming requests and serialize outgoing responses.
# They also auto-generate the interactive API documentation (Swagger UI).

class AutocompleteRequest(BaseModel):
    """Request body for the autocomplete endpoint.

    Attributes:
        text: The input text to get completions for.
        top_k: Number of suggestions to return (1-20).
        model: Which language model to use.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Input text to complete. Last few words are used as context.",
        examples=["machine learning is a", "the attention mechanism", "deep learning"],
    )
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=20,
        description="Number of suggestions to return.",
    )
    model: str = Field(
        default="ngram",
        pattern="^(ngram|markov|lstm|lstm-bpe|transformer|transformer-bpe)$",
        description=(
            "Language model to use. Choices: 'ngram', 'markov', 'lstm', "
            "'transformer' (word-level), and the subword aliases "
            "'lstm-bpe' / 'transformer-bpe' (equivalent to the base id "
            "with tokenizer='bpe'). Neural options require PyTorch; "
            "BPE options also require the 'transformers' package."
        ),
    )
    tokenizer: str = Field(
        default="word",
        pattern="^(word|bpe)$",
        description=(
            "Tokenizer flavour: 'word' (default) uses the project's "
            "whitespace+regex tokenizer; 'bpe' wires in a byte-level "
            "BPE tokenizer (needs the optional 'transformers' package, "
            "and only valid with model='lstm' or 'transformer'). "
            "Ignored when model='lstm-bpe' or 'transformer-bpe' — those "
            "aliases imply BPE already."
        ),
    )
    tokenizer_name: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "Optional HF repo for the BPE tokenizer (e.g. 'gpt2'). "
            "Ignored when tokenizer='word'. Defaults to SmolLM2's "
            "tokenizer so the first BPE request doesn't force a new "
            "download. Different names are cached independently."
        ),
    )

    @model_validator(mode="after")
    def _validate_tokenizer_flags(self) -> "AutocompleteRequest":
        # Catalogue aliases: "*-bpe" means "the base model with tokenizer=bpe".
        # Normalise here so downstream code only ever sees the base name.
        if self.model.endswith("-bpe"):
            self.model = self.model[: -len("-bpe")]
            self.tokenizer = "bpe"
        if self.tokenizer == "bpe" and self.model not in ("lstm", "transformer"):
            raise ValueError(
                "tokenizer='bpe' only applies to model='lstm' or 'transformer'"
            )
        if self.tokenizer_name is not None and self.tokenizer != "bpe":
            raise ValueError("tokenizer_name requires tokenizer='bpe'")
        return self


class Suggestion(BaseModel):
    """A single autocomplete suggestion with its probability."""
    word: str = Field(..., description="The suggested next word.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of this suggestion.")


class AutocompleteResponse(BaseModel):
    """Response body for the autocomplete endpoint."""
    suggestions: List[Suggestion] = Field(..., description="Top-k word suggestions.")
    context: str = Field(..., description="The context words used for prediction.")
    model: str = Field(..., description="Which model generated these suggestions.")


class BatchRequest(BaseModel):
    """Request body for batch autocomplete."""
    texts: List[
        Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    ] = Field(
        ...,
        min_length=1,
        max_length=50,
        description=(
            "List of input texts to complete. Each text is capped at 1000 "
            "characters to match the single-text endpoint."
        ),
    )
    top_k: int = Field(default=5, ge=1, le=20)
    model: str = Field(
        default="ngram",
        pattern="^(ngram|markov|lstm|lstm-bpe|transformer|transformer-bpe)$",
    )
    tokenizer: str = Field(default="word", pattern="^(word|bpe)$")
    tokenizer_name: Optional[str] = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _validate_tokenizer_flags(self) -> "BatchRequest":
        if self.model.endswith("-bpe"):
            self.model = self.model[: -len("-bpe")]
            self.tokenizer = "bpe"
        if self.tokenizer == "bpe" and self.model not in ("lstm", "transformer"):
            raise ValueError(
                "tokenizer='bpe' only applies to model='lstm' or 'transformer'"
            )
        if self.tokenizer_name is not None and self.tokenizer != "bpe":
            raise ValueError("tokenizer_name requires tokenizer='bpe'")
        return self


class BatchResponse(BaseModel):
    """Response body for batch autocomplete."""
    results: List[AutocompleteResponse] = Field(..., description="Predictions for each input text.")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint — used by monitoring systems to verify the API is running.

    Returns a simple JSON object with status. This is a standard practice:
    load balancers and orchestration systems (Kubernetes, Docker Compose)
    call this endpoint to check if the service is healthy.

    Also reports which optional capabilities are currently active so
    operators can confirm their env is wired correctly (torch present,
    Prometheus instrumentator loaded, Redis rate limiter connected, etc.)
    without digging through logs. All fields are booleans except `version`.
    """
    return {
        "status": "healthy",
        "version": API_VERSION,
        "capabilities": {
            "torch": _lstm_available(),
            "transformer": _transformer_available(),
            "bpe": _bpe_available(),
            "prometheus": HAS_PROMETHEUS,
            "redis_connected": _redis_client is not None,
            "auth_required": bool(os.getenv("AUTOCOMPLETE_API_KEY", "")),
        },
    }


async def _charge_model_cost(request: Request, model_id: str) -> None:
    """Drain additional bucket tokens for expensive models.

    Middleware already consumed 1 token. If the selected model costs more,
    top up the drain here so neural calls actually cost more than
    statistical calls against the same bucket. Raises 429 if the bucket
    doesn't have the remaining balance.
    """
    extra = _model_cost(model_id) - 1
    if extra <= 0:
        return
    client_ip = _resolve_client_ip(request)
    if not await _check_rate_limit(client_ip, cost=extra):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded — model '{model_id}' costs "
                f"{extra + 1} tokens per request."
            ),
        )


@app.post("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(req: AutocompleteRequest, request: Request):
    """Get autocomplete suggestions for a given text.

    This is the primary endpoint — it takes text input and returns
    the most likely next words based on the selected language model.

    HOW IT WORKS:
    1. Tokenize the input text (split into words, lowercase, handle punctuation)
    2. Select the requested model (n-gram or Markov chain)
    3. Ask the model for top-k next-word predictions
    4. Return predictions with probabilities

    Args:
        req: AutocompleteRequest with text, top_k, and model selection.

    Returns:
        AutocompleteResponse with suggestions, context, and model name.
    """
    from src.data_loader import tokenize

    tokens = tokenize(req.text)

    if not tokens:
        # Empty after tokenization — can't make predictions
        raise HTTPException(status_code=400, detail="Text contains no valid tokens.")

    # Extra rate-limit drain proportional to the model's cost (noop for
    # cheap models; 429 if the bucket can't cover a neural call).
    await _charge_model_cost(request, req.model)

    # Select the model
    if req.model == "markov":
        model = _get_trained_markov()
    elif req.model == "lstm":
        model = _get_trained_lstm(
            tokenizer_flavour=req.tokenizer,
            tokenizer_name=req.tokenizer_name,
        )
    elif req.model == "transformer":
        model = _get_trained_transformer(
            tokenizer_flavour=req.tokenizer,
            tokenizer_name=req.tokenizer_name,
        )
    else:
        model = _get_trained_ngram()

    # Get predictions
    preds = model.predict_next(tokens, top_k=req.top_k)

    if not preds:
        raise HTTPException(status_code=404, detail="No predictions available for this context.")

    if AUTOCOMPLETE_MODEL_HITS is not None:
        AUTOCOMPLETE_MODEL_HITS.labels(
            model=req.model, tokenizer=req.tokenizer,
        ).inc()

    return AutocompleteResponse(
        suggestions=[Suggestion(word=w, probability=round(p, 6)) for w, p in preds],
        context=" ".join(tokens[-5:]),  # Show last 5 words as context
        model=req.model,
    )


class AttentionRequest(BaseModel):
    """Request body for the transformer attention-visualisation endpoint."""
    text: str = Field(..., min_length=1, max_length=500)
    tokenizer: str = Field(default="word", pattern="^(word|bpe)$")
    tokenizer_name: Optional[str] = Field(default=None, max_length=128)


class AttentionResponse(BaseModel):
    """Per-layer, per-head attention weights on the request tokens."""
    tokens: List[str] = Field(..., description="Token strings for rows/cols.")
    attentions: List[List[List[List[float]]]] = Field(
        ...,
        description=(
            "List of per-layer matrices. Each matrix has shape "
            "[n_heads, seq_len, seq_len]."
        ),
    )
    n_layers: int
    n_heads: int
    seq_len: int


@app.post("/attention", response_model=AttentionResponse)
async def attention(req: AttentionRequest):
    """Return causal self-attention weights for a short text (transformer only).

    Meant for visualisation — the React frontend renders each layer's
    per-head matrix as a heatmap so you can see "what the model is
    looking at" at each position. Returns 503 if torch isn't installed.
    """
    if not _transformer_available():
        raise HTTPException(
            status_code=503,
            detail="Transformer model unavailable: PyTorch is not installed.",
        )

    model = _get_trained_transformer(
        tokenizer_flavour=req.tokenizer,
        tokenizer_name=req.tokenizer_name,
    )

    # Word path uses the project's tokenizer; BPE path hands raw text.
    if req.tokenizer == "word":
        from src.data_loader import tokenize
        context = tokenize(req.text)
    else:
        context = [req.text]

    if not context:
        raise HTTPException(status_code=400, detail="Text contains no valid tokens.")

    result = model.attention_matrices(context)
    attns = result["attentions"]
    if not attns:
        raise HTTPException(
            status_code=404,
            detail="No attention produced — model is likely unfit.",
        )

    return AttentionResponse(
        tokens=result["tokens"],
        attentions=attns,
        n_layers=len(attns),
        n_heads=len(attns[0]),
        seq_len=len(attns[0][0]),
    )


@app.post("/autocomplete/batch", response_model=BatchResponse)
async def autocomplete_batch(req: BatchRequest, request: Request):
    """Batch autocomplete — get predictions for multiple texts at once.

    BATCH PROCESSING is more efficient than making N individual requests:
    - Reduces HTTP overhead (1 request instead of N)
    - Models are loaded/cached once
    - Easier for clients to manage

    Use this when you need predictions for many texts, e.g., processing
    a document line-by-line or a search query log.

    Args:
        req: BatchRequest with list of texts and model settings.

    Returns:
        BatchResponse with predictions for each input text.
    """
    from src.data_loader import tokenize

    # Batch costs model_cost × len(texts) — neural * 50 on a long batch
    # should 429 rather than silently doing 50 forward passes.
    extra_per_req = _model_cost(req.model) - 1
    if extra_per_req > 0:
        total_extra = extra_per_req * len(req.texts)
        client_ip = _resolve_client_ip(request)
        if not await _check_rate_limit(client_ip, cost=total_extra):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded — batch of {len(req.texts)} "
                    f"× model '{req.model}' costs "
                    f"{total_extra + 1} tokens."
                ),
            )

    results = []

    # Select the model once (shared across all texts)
    if req.model == "markov":
        model = _get_trained_markov()
    elif req.model == "transformer":
        model = _get_trained_transformer()
    else:
        model = _get_trained_ngram()

    for text in req.texts:
        tokens = tokenize(text)
        if not tokens:
            results.append(AutocompleteResponse(
                suggestions=[],
                context="",
                model=req.model,
            ))
            continue

        preds = model.predict_next(tokens, top_k=req.top_k)
        results.append(AutocompleteResponse(
            suggestions=[Suggestion(word=w, probability=round(p, 6)) for w, p in preds],
            context=" ".join(tokens[-5:]),
            model=req.model,
        ))

    return BatchResponse(results=results)


@app.get("/models")
async def list_models():
    """List available language models with their descriptions.

    This endpoint helps API consumers discover what models are available
    and choose the right one for their use case.
    """
    catalogue = [
        {
            "id": "ngram",
            "name": "N-gram Language Model",
            "description": "Classic statistical model using word co-occurrence counts. Fast and interpretable.",
            "max_ngram_order": 3,
        },
        {
            "id": "markov",
            "name": "Markov Chain Model",
            "description": "First-order Markov chain with Laplace smoothing. Simple and effective for common word pairs.",
        },
    ]
    if _lstm_available():
        catalogue.append({
            "id": "lstm",
            "name": "LSTM Language Model",
            "description": (
                "2-layer LSTM with tied embedding/output weights. Trained "
                "lazily on first request with a tiny config (embed=32, "
                "hidden=64); production would load a pre-trained checkpoint."
            ),
        })
    if _transformer_available():
        catalogue.append({
            "id": "transformer",
            "name": "Decoder-only Transformer",
            "description": (
                "Causal self-attention with tied LM head. Beats the LSTM on "
                "WikiText-2 (-24% PPL, +4 pp top-5) with half the training "
                "time. Trained lazily on first request; takes a few seconds."
            ),
        })
    if _bpe_available():
        catalogue.append({
            "id": "lstm-bpe",
            "name": "LSTM (BPE subwords)",
            "description": (
                "Same LSTM architecture but trained on SmolLM2's ~49k "
                "subword tokenizer instead of whitespace-split words. "
                "Alias for model='lstm' + tokenizer='bpe'."
            ),
        })
        catalogue.append({
            "id": "transformer-bpe",
            "name": "Transformer (BPE subwords)",
            "description": (
                "Same transformer architecture but trained on subword "
                "tokens. Alias for model='transformer' + tokenizer='bpe'."
            ),
        })
    return {"models": catalogue}


class EvalSummaryRow(BaseModel):
    """One row of the /eval/summary response — per-model held-out metrics
    on a fixed split of the built-in sample corpus."""
    model: str
    perplexity: float
    top1: float
    top5: float


class EvalSummaryResponse(BaseModel):
    """Held-out accuracy + perplexity for every available model, run on
    the same deterministic split so the numbers are comparable."""
    rows: List[EvalSummaryRow]
    test_tokens: int


@app.get("/eval/summary", response_model=EvalSummaryResponse)
async def eval_summary():
    """Run a fast side-by-side evaluation on the sample corpus.

    The React Metrics page renders this as a bar chart so you can
    eyeball "which model is best on the teaching corpus today?" without
    leaving the UI. Numbers are cached in-process until the model cache
    is rebuilt — re-fitting every request on the 40-sentence corpus
    would just churn CPU for no benefit.
    """
    from src.data_loader import load_sample_data, tokenize, train_test_split
    from src.evaluation import (
        compute_perplexity,
        autocomplete_accuracy,
    )

    cache_key = "eval:summary:v1"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    tokens = tokenize(load_sample_data())
    train, test = train_test_split(tokens, test_ratio=0.2, seed=42)

    # Slide a small window over the test split for top-k scoring so every
    # model gets exactly the same probes.
    ctx_len = 3
    probes = [
        (test[i - ctx_len:i], test[i])
        for i in range(ctx_len, min(len(test), ctx_len + 200))
    ]

    rows: List[EvalSummaryRow] = []

    # autocomplete_accuracy wants pred WORDS (not (word, prob) tuples).
    def _score(m) -> tuple[float, float, float]:
        preds = [
            [w for w, _ in m.predict_next(list(ctx), top_k=5)]
            for ctx, _ in probes
        ]
        truths = [t for _, t in probes]
        return (
            float(compute_perplexity(m, test)),
            float(autocomplete_accuracy(preds, truths, top_k=1)),
            float(autocomplete_accuracy(preds, truths, top_k=5)),
        )

    for model_id, getter in (
        ("ngram", _get_trained_ngram),
        ("markov", _get_trained_markov),
    ):
        ppl, t1, t5 = _score(getter())
        rows.append(EvalSummaryRow(
            model=model_id, perplexity=ppl, top1=t1, top5=t5,
        ))

    if _lstm_available():
        try:
            ppl, t1, t5 = _score(_get_trained_lstm())
            rows.append(EvalSummaryRow(
                model="lstm", perplexity=ppl, top1=t1, top5=t5,
            ))
        except HTTPException:
            pass

    if _transformer_available():
        try:
            ppl, t1, t5 = _score(_get_trained_transformer())
            rows.append(EvalSummaryRow(
                model="transformer", perplexity=ppl, top1=t1, top5=t5,
            ))
        except HTTPException:
            pass

    resp = EvalSummaryResponse(rows=rows, test_tokens=len(test))
    _model_cache[cache_key] = resp
    return resp


class GenerateRequest(BaseModel):
    """Request body for text generation endpoint."""
    start_word: Optional[str] = Field(
        default=None,
        description="Word to start generation from. If omitted, picks a common starting word.",
    )
    max_length: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of words to generate.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Sampling temperature: <1.0 = focused, 1.0 = natural, >1.0 = creative.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation.",
    )
    model: str = Field(
        default="markov",
        pattern="^(markov|lstm|transformer)$",
        description=(
            "Generator model: 'markov' (default, sampling from transition "
            "probabilities), or a neural 'lstm' / 'transformer' that "
            "samples from the softmax each step with the given temperature."
        ),
    )


class GenerateResponse(BaseModel):
    """Response body for text generation."""
    text: str = Field(..., description="Generated text.")
    word_count: int = Field(..., description="Number of words generated.")
    temperature: float = Field(..., description="Temperature used for generation.")
    model: str = Field(default="markov", description="Model used for generation.")


def _neural_generate(neural_model, start_word, max_length, temperature, seed):
    """Temperature-sampled iterative generation for LSTM / Transformer.

    Each step, we ask the model for top-20 softmax candidates, scale the
    probabilities by 1/temperature, renormalise, and sample one token.
    Falls back to the top-1 word if the model returns nothing (e.g. on
    a minimal install where predict_next returns mock outputs)."""
    import random
    rng = random.Random(seed)
    words = [start_word] if start_word else ["the"]
    for _ in range(max_length):
        preds = neural_model.predict_next(words, top_k=20)
        if not preds:
            break
        # Temperature rescale. Low T sharpens, high T flattens.
        scaled = [(w, p ** (1.0 / max(temperature, 1e-3))) for w, p in preds]
        total = sum(p for _, p in scaled) or 1.0
        r = rng.random() * total
        cum = 0.0
        pick = preds[0][0]
        for w, p in scaled:
            cum += p
            if r <= cum:
                pick = w
                break
        words.append(pick)
    return " ".join(words)


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    """Generate text using the selected language model.

    TEXT GENERATION is a natural extension of autocomplete — instead of
    predicting ONE next word, we iteratively predict many words to produce
    coherent (or creative) text passages.

    The temperature parameter controls the randomness:
    - 0.1: Very deterministic, repeats common patterns
    - 1.0: Natural diversity, balanced between common and rare words
    - 2.0+: Wild and creative, may produce unusual word combinations

    Markov generation samples from the transition table directly; neural
    generation samples from the softmax top-20 each step. Neural
    generation on a tiny-corpus lazy-fit model tends to produce junk —
    use a trained checkpoint via AUTOCOMPLETE_*_CHECKPOINT_* for
    coherent output.

    Args:
        req: GenerateRequest with start_word, max_length, temperature, model.

    Returns:
        GenerateResponse with the generated text and metadata.
    """
    if req.model == "lstm":
        model = _get_trained_lstm()
        text = _neural_generate(
            model, req.start_word, req.max_length, req.temperature, req.seed,
        )
    elif req.model == "transformer":
        model = _get_trained_transformer()
        text = _neural_generate(
            model, req.start_word, req.max_length, req.temperature, req.seed,
        )
    else:
        model = _get_trained_markov()
        text = model.generate_text(
            start_word=req.start_word,
            max_length=req.max_length,
            temperature=req.temperature,
            seed=req.seed,
        )

    return GenerateResponse(
        text=text,
        word_count=len(text.split()),
        temperature=req.temperature,
        model=req.model,
    )


@app.get("/metrics")
async def metrics():
    """Get API usage metrics (hand-rolled JSON format).

    Returns request counts, rate limit hits, and endpoint-level breakdowns.
    Useful for quick human inspection and for the React Metrics page.

    For a scrape-friendly format, install prometheus-fastapi-instrumentator
    and hit /metrics/prom — that endpoint speaks the Prometheus exposition
    format and adds latency histograms out of the box.
    """
    return {
        "total_requests": _request_metrics.get("total_requests", 0),
        "rate_limited": _request_metrics.get("rate_limited", 0),
        "endpoints": {
            k.split(":", 1)[1]: v
            for k, v in _request_metrics.items()
            if k.startswith("requests:")
        },
    }


@app.get("/vocab/stats")
async def vocab_stats():
    """Get vocabulary statistics for the trained models.

    Returns information about the training corpus and vocabulary size,
    useful for understanding the model's coverage and limitations.
    """
    from src.data_loader import tokenize, load_sample_data, get_corpus_stats

    tokens = tokenize(load_sample_data())
    stats = get_corpus_stats(tokens)

    available = ["ngram", "markov"]
    if _lstm_available():
        available.append("lstm")
    if _transformer_available():
        available.append("transformer")
    return {
        "corpus": stats,
        "ngram_vocab_size": _get_trained_ngram().vocab_size,
        "markov_vocab_size": _get_trained_markov().vocab_size,
        "available_models": available,
    }


# ---------------------------------------------------------------------------
# Static frontend (optional, production single-process mode)
# ---------------------------------------------------------------------------
# If the React app has been built (`cd frontend && npm run build`), mount
# the resulting `frontend/dist/` at the root so one process serves both
# the API and the UI. In dev you don't need this — Vite serves the UI
# and proxies /api/* to FastAPI. The mount is last so every real route
# above it wins; the StaticFiles `html=True` option serves index.html
# for any unmatched GET, which is how SPA routing works.
_FRONTEND_DIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend", "dist",
)
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="spa")
    logger.info("Mounted built frontend at / from %s", _FRONTEND_DIST)


# ---------------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
