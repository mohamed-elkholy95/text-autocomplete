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
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

# Enable CORS (Cross-Origin Resource Sharing) so web apps can call this API
# from a different domain. The wildcard (*) allows all origins — in production,
# you'd restrict this to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of input texts to complete.",
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
