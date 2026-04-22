"""Stateful vs stateless BPTT comparison on WikiText-2 (R10).

The roadmap leaves R10 open until a full-corpus / multi-epoch run shows
stateful beating stateless by >=5% held-out perplexity. This driver runs
both flavours back-to-back with identical config + seed and prints a
markdown table with the delta.

Scale: word-level LSTM, hidden 256 / 2 layers / 12 epochs / 20k vocab.
Roughly 3x bigger than the 200k-subset bench that originally looked
favourable for stateful; still comfortably under 10 min on an RTX 5080.

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bench_stateful.py
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

from src.data_loader import tokenize, train_test_split  # noqa: E402
from src.neural_model import LSTMModel, HAS_TORCH  # noqa: E402

if HAS_TORCH:
    import random
    import numpy as np
    import torch

CORPUS_PATH = ROOT / "data" / "wikitext2" / "wikitext2_train.txt"

EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
VOCAB_CAP = 20000
EPOCHS = 12
SEQ_LEN = 64
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run(tokens_train, tokens_test, *, stateful: bool) -> dict:
    _seed_all(SEED)
    model = LSTMModel(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        vocab_cap=VOCAB_CAP,
    )
    t0 = time.perf_counter()
    model.fit(
        tokens_train,
        epochs=EPOCHS,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        lr=LR,
        stateful=stateful,
    )
    fit_s = time.perf_counter() - t0
    ppl = model.perplexity(tokens_test, seq_len=SEQ_LEN)
    return {"fit_s": fit_s, "ppl": ppl}


def main() -> int:
    if not HAS_TORCH:
        print("PyTorch not installed; skipping.")
        return 0
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found at {CORPUS_PATH}. "
            "Run `scripts/bench_real_data.py` once to cache it."
        )

    raw = CORPUS_PATH.read_text(encoding="utf-8")
    tokens = tokenize(raw)
    train, test = train_test_split(tokens, test_ratio=0.1, seed=SEED)
    _log(
        f"Corpus ready: {len(train):,} train / {len(test):,} test tokens, "
        f"vocab cap {VOCAB_CAP}"
    )

    _log("Training stateless LSTM...")
    stateless = _run(train, test, stateful=False)
    _log(
        f"  stateless: fit={stateless['fit_s']:.1f}s  ppl={stateless['ppl']:.2f}"
    )

    _log("Training stateful LSTM...")
    stateful = _run(train, test, stateful=True)
    _log(
        f"  stateful:  fit={stateful['fit_s']:.1f}s  ppl={stateful['ppl']:.2f}"
    )

    delta_ppl = stateful["ppl"] - stateless["ppl"]
    delta_pct = 100.0 * delta_ppl / stateless["ppl"]
    decision = "flip" if delta_pct <= -5.0 else "keep opt-in"

    print()
    print("| Variant    | Fit (s) | Held-out PPL | Δ vs stateless |")
    print("| ---------- | ------: | -----------: | -------------: |")
    print(
        f"| stateless  | {stateless['fit_s']:7.1f} | "
        f"{stateless['ppl']:12.2f} | — |"
    )
    sign = "+" if delta_pct >= 0 else ""
    print(
        f"| stateful   | {stateful['fit_s']:7.1f} | "
        f"{stateful['ppl']:12.2f} | {sign}{delta_pct:5.2f} % |"
    )
    print()
    print(f"Decision rule (>=5% improvement required): {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
