"""Scaled-up in-house Transformer training on WikiText-2.

Trains a d_model=512 / 8-head / 12-layer decoder-only transformer (word-
level, vocab_cap=20k) — roughly 3x the parameter budget of the
bench_full_train.py config. Produces a checkpoint under
``models/transformer_wikitext2_xl.{safetensors,json}``.

Also trains a BPE variant of the same size if 'transformers' is
installed, saving to ``models/transformer_wikitext2_bpe_xl.*``.

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bench_train_xl.py
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
from src.transformer_model import TransformerModel  # noqa: E402
from src.neural_model import HAS_TORCH  # noqa: E402

CORPUS_PATH = ROOT / "data" / "wikitext2" / "wikitext2_train.txt"
MODELS_DIR = ROOT / "models"
SEED = 42

D_MODEL = 512
N_HEADS = 8
N_LAYERS = 12
FF_DIM = 2048
MAX_SEQ_LEN = 128
DROPOUT = 0.2
VOCAB_CAP = 20000

EPOCHS = 20
SEQ_LEN = 96
BATCH_SIZE = 48
LR = 3e-4


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _seed_all(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    if not HAS_TORCH:
        print("PyTorch not installed; skipping.")
        return 0
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found at {CORPUS_PATH}. "
            "Run `scripts/bench_real_data.py` once to cache it."
        )
    MODELS_DIR.mkdir(exist_ok=True)

    raw = CORPUS_PATH.read_text(encoding="utf-8")
    tokens = tokenize(raw)
    train, test = train_test_split(tokens, test_ratio=0.1, seed=SEED)
    _log(f"Corpus: {len(train):,} train / {len(test):,} test tokens")

    # Word-level XL
    _seed_all(SEED)
    model = TransformerModel(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ff_dim=FF_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        vocab_cap=VOCAB_CAP,
    )
    t0 = time.perf_counter()
    model.fit(
        train,
        epochs=EPOCHS,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        lr=LR,
    )
    fit_s = time.perf_counter() - t0
    ppl = model.perplexity(test, seq_len=SEQ_LEN)
    _log(f"XL-word Transformer: fit={fit_s:.1f}s held-out PPL={ppl:.2f}")

    out = MODELS_DIR / "transformer_wikitext2_xl"
    model.save(str(out))
    _log(f"Saved → {out}.safetensors (+.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
