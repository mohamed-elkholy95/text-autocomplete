"""Beam-search benchmark: greedy vs beam top-1 across all four model families.

For each model, we ask the same predict_next() for greedy top-1, then run
BeamSearchDecoder to produce the best multi-step continuation. Output is
a markdown table of wall-time and the two top continuations per model so
you can eyeball whether beam search actually helps the neural models.

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bench_beam.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_sample_data, tokenize  # noqa: E402
from src.ngram_model import NGramModel  # noqa: E402
from src.markov_model import MarkovChainModel  # noqa: E402
from src.beam_search import BeamSearchDecoder  # noqa: E402
from src.neural_model import LSTMModel, HAS_TORCH  # noqa: E402
from src.transformer_model import TransformerModel  # noqa: E402

CONTEXTS = [
    ["machine", "learning", "is"],
    ["the", "attention"],
    ["deep", "learning"],
]

BEAM_WIDTH = 5
MAX_LENGTH = 5
LENGTH_PENALTY = 0.6


def greedy(model, ctx):
    """Pure greedy decode — the top-1 at each step."""
    out = list(ctx)
    for _ in range(MAX_LENGTH):
        preds = model.predict_next(out, top_k=1)
        if not preds:
            break
        out.append(preds[0][0])
    return " ".join(out[len(ctx):])


def beam(model, ctx):
    dec = BeamSearchDecoder(
        beam_width=BEAM_WIDTH,
        max_length=MAX_LENGTH,
        length_penalty=LENGTH_PENALTY,
    )
    beams = dec.search(model, ctx)
    if not beams:
        return ""
    # beams[0] holds the full sequence including the context.
    tokens = beams[0]["tokens"][len(ctx):]
    return " ".join(tokens)


def bench(name: str, model, contexts):
    rows = []
    t0 = time.time()
    for ctx in contexts:
        g = greedy(model, ctx)
        rows.append((ctx, "greedy", g))
    g_s = time.time() - t0

    t0 = time.time()
    for ctx in contexts:
        b = beam(model, ctx)
        rows.append((ctx, f"beam{BEAM_WIDTH}", b))
    b_s = time.time() - t0

    print(f"\n## {name}  (greedy: {g_s*1000:.1f} ms total, beam: {b_s*1000:.1f} ms total)")
    print("| context | mode | continuation |")
    print("| --- | --- | --- |")
    for ctx, mode, cont in rows:
        ctx_str = " ".join(ctx)
        print(f"| `{ctx_str}` | {mode} | {cont} |")


def main() -> None:
    tokens = tokenize(load_sample_data())

    bench("NGramModel (n=3)", NGramModel(n=3).fit(tokens), CONTEXTS)
    bench("MarkovChainModel", MarkovChainModel().fit(tokens), CONTEXTS)

    if HAS_TORCH:
        lstm = LSTMModel(embed_dim=32, hidden_dim=64, num_layers=1)
        lstm.fit(tokens, epochs=3, seq_len=8, batch_size=16, lr=1e-3)
        bench("LSTMModel (tiny, 3 ep on sample corpus)", lstm, CONTEXTS)

        xform = TransformerModel(
            d_model=32, n_heads=2, n_layers=1, ff_dim=64, max_seq_len=32,
        )
        xform.fit(tokens, epochs=3, seq_len=8, batch_size=16, lr=1e-3)
        bench("TransformerModel (tiny, 3 ep on sample corpus)", xform, CONTEXTS)
    else:
        print("\n(skipping neural rows — torch not installed)")


if __name__ == "__main__":
    main()
