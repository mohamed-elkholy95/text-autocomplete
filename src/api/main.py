"""FastAPI for text autocomplete."""
import logging
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
app = FastAPI(title="Text Autocomplete API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AutocompleteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)

class Suggestion(BaseModel):
    word: str
    probability: float

class AutocompleteResponse(BaseModel):
    suggestions: List[Suggestion]
    context: str

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(req: AutocompleteRequest):
    from src.ngram_model import NGramModel
    from src.data_loader import tokenize, load_sample_data
    tokens = tokenize(req.text)
    if not tokens:
        tokens = ["the"]
    model = NGramModel(n=3)
    model.fit(tokenize(load_sample_data()))
    preds = model.predict_next(tokens, top_k=req.top_k)
    return AutocompleteResponse(
        suggestions=[Suggestion(word=w, probability=p) for w, p in preds],
        context=" ".join(tokens[-5:])
    )

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8010)
