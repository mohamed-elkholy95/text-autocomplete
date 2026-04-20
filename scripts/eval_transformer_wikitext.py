#!/usr/bin/env python3
"""Measure real sliding-window perplexity of a HuggingFace causal LM on WikiText.

Produces the headline number that goes into README / docs. Uses the same
stride-based methodology as the published WikiText-103 numbers for
distilgpt2 (21.1) and GPT-2 (16.3), so the result is directly comparable.

Usage:
    python scripts/eval_transformer_wikitext.py               # SmolLM2-135M on WikiText-103 test
    python scripts/eval_transformer_wikitext.py --size 2      # quick sanity on WikiText-2
    python scripts/eval_transformer_wikitext.py --model distilbert/distilgpt2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running this script directly via `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_wikitext
from src.transformer_model import DEFAULT_MODEL_ID, HAS_TRANSFORMERS, TransformerModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL_ID,
                    help=f"HF model ID (default: {DEFAULT_MODEL_ID})")
    ap.add_argument("--size", choices=["2", "103"], default="103",
                    help="WikiText size: 2 (fast) or 103 (benchmark)")
    ap.add_argument("--split", choices=["train", "validation", "test"], default="test",
                    help="Which split to score (default: test)")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="Max context length per sliding window")
    ap.add_argument("--stride", type=int, default=512,
                    help="Number of new tokens scored per window")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Cap the number of articles (useful for quick runs)")
    args = ap.parse_args()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers is not installed.")
        sys.exit(2)

    print(f"Loading WikiText-{args.size} ({args.split} split)...")
    t0 = time.perf_counter()
    text = load_wikitext(split=args.split, size=args.size, max_docs=args.max_docs)
    print(f"  {len(text):,} chars loaded in {time.perf_counter() - t0:.1f}s")

    print(f"Loading {args.model}...")
    t0 = time.perf_counter()
    model = TransformerModel(model_id=args.model)
    model.fit([])  # warmup without per-corpus vocab
    print(f"  loaded on {model.device} ({model.dtype}) in {time.perf_counter() - t0:.1f}s")

    print(f"Computing stride-{args.stride} perplexity (window={args.max_length})...")
    t0 = time.perf_counter()
    ppl = model.perplexity(text, max_length=args.max_length, stride=args.stride)
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print(f"  Model:        {args.model}")
    print(f"  Dataset:      WikiText-{args.size} ({args.split} split)")
    print(f"  Methodology:  sliding window, ctx={args.max_length}, stride={args.stride}")
    print(f"  Perplexity:   {ppl:.3f}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
