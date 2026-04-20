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
from typing import Annotated, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from src.config import API_TITLE, API_VERSION, TOP_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="""
    Text Autocomplete API — Get real-time word completion suggestions.

    Supports multiple language models:
    - **ngram**: Classic n-gram model with backoff smoothing (fast, interpretable)
    - **markov**: Markov chain model (simple transition probabilities)
    - **beam**: N-gram model with beam search decoding (higher quality, slower)

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


def _check_rate_limit(client_ip: str) -> bool:
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

    # Consume a token if available
    if bucket["tokens"] >= 1.0:
        bucket["tokens"] -= 1.0
        return True
    return False


# Only trust X-Forwarded-For when the deploy explicitly opts in (e.g. when
# running behind a known reverse proxy). Otherwise we use the direct peer
# address, which a client can't spoof in HTTP/1.1.
TRUST_FORWARDED_HEADERS = os.getenv("TRUST_FORWARDED_HEADERS", "").lower() in (
    "1", "true", "yes",
)


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


# Enable CORS (Cross-Origin Resource Sharing) so web apps can call this API
# from a different domain. The wildcard (*) allows all origins — in production,
# you'd restrict this to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Rate limit check (skip health endpoint to allow monitoring)
    if request.url.path != "/health" and not _check_rate_limit(client_ip):
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
        pattern="^(ngram|markov)$",
        description="Language model to use: 'ngram' or 'markov'.",
    )


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
    model: str = Field(default="ngram", pattern="^(ngram|markov)$")


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
    """
    return {"status": "healthy", "version": API_VERSION}


@app.post("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(req: AutocompleteRequest):
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

    # Select the model
    if req.model == "markov":
        model = _get_trained_markov()
    else:
        model = _get_trained_ngram()

    # Get predictions
    preds = model.predict_next(tokens, top_k=req.top_k)

    if not preds:
        raise HTTPException(status_code=404, detail="No predictions available for this context.")

    return AutocompleteResponse(
        suggestions=[Suggestion(word=w, probability=round(p, 6)) for w, p in preds],
        context=" ".join(tokens[-5:]),  # Show last 5 words as context
        model=req.model,
    )


@app.post("/autocomplete/batch", response_model=BatchResponse)
async def autocomplete_batch(req: BatchRequest):
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

    results = []

    # Select the model once (shared across all texts)
    if req.model == "markov":
        model = _get_trained_markov()
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
    return {
        "models": [
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
    }


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


class GenerateResponse(BaseModel):
    """Response body for text generation."""
    text: str = Field(..., description="Generated text.")
    word_count: int = Field(..., description="Number of words generated.")
    temperature: float = Field(..., description="Temperature used for generation.")
    model: str = Field(default="markov", description="Model used for generation.")


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    """Generate text using the Markov chain model.

    TEXT GENERATION is a natural extension of autocomplete — instead of
    predicting ONE next word, we iteratively predict many words to produce
    coherent (or creative) text passages.

    The temperature parameter controls the randomness:
    - 0.1: Very deterministic, repeats common patterns
    - 1.0: Natural diversity, balanced between common and rare words
    - 2.0+: Wild and creative, may produce unusual word combinations

    Try it: generate with temperature=0.3 vs temperature=2.0 and compare!

    Args:
        req: GenerateRequest with start_word, max_length, and temperature.

    Returns:
        GenerateResponse with the generated text and metadata.
    """
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
        model="markov",
    )


@app.get("/metrics")
async def metrics():
    """Get API usage metrics.

    Returns request counts, rate limit hits, and endpoint-level breakdowns.
    In production, these would feed into a monitoring dashboard (Grafana,
    DataDog) for alerting on anomalies like sudden traffic spikes.
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

    return {
        "corpus": stats,
        "ngram_vocab_size": _get_trained_ngram().vocab_size,
        "markov_vocab_size": _get_trained_markov().vocab_size,
        "available_models": ["ngram", "markov"],
    }


# ---------------------------------------------------------------------------
# Run Server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
