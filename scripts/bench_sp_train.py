"""Train a SentencePiece tokenizer on WikiText-2 and fit a small LSTM
through it, for comparison against the byte-level BPE path.

The educational point: byte-level BPE (SmolLM2 via HuggingFace) and
SentencePiece Unigram are two different approaches to subword
tokenization. This script trains a SentencePiece Unigram model *on the
project's own corpus*, unlike bench_bpe.py which downloads a pretrained
HF tokenizer. Unigram also handles spaces differently (the `▁` marker).

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bench_sp_train.py

Requires the optional deps: `pip install torch safetensors sentencepiece`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import train_test_split  # noqa: E402
from src.neural_model import LSTMModel, HAS_TORCH  # noqa: E402
from src.sp_tokenizer import SPTokenizer, HAS_SENTENCEPIECE  # noqa: E402

CORPUS_PATH = ROOT / "data" / "wikitext2" / "wikitext2_train.txt"
MODELS_DIR = ROOT / "models"

SP_VOCAB_SIZE = 8000
SP_MODEL_TYPE = "unigram"  # or "bpe"

LSTM_EMBED_DIM = 128
LSTM_HIDDEN_DIM = 256
LSTM_EPOCHS = 3
LSTM_SEQ_LEN = 64
LSTM_BATCH = 64
LSTM_LR = 1e-3


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    if not HAS_TORCH:
        raise SystemExit("PyTorch not installed — nothing to train.")
    if not HAS_SENTENCEPIECE:
        raise SystemExit(
            "sentencepiece not installed. "
            "Install with `pip install sentencepiece`."
        )
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus missing at {CORPUS_PATH}. Run "
            "scripts/bench_real_data.py once to cache WikiText-2."
        )

    MODELS_DIR.mkdir(exist_ok=True)
    raw_text = CORPUS_PATH.read_text(encoding="utf-8")

    # Word-level split of the raw string so we can train SP on the same
    # slice the BPE bench uses for reproducibility.
    words = raw_text.split()
    train_words, test_words = train_test_split(words, test_ratio=0.1, seed=42)
    train_text = " ".join(train_words)
    test_text = " ".join(test_words)
    _log(f"Corpus: train={len(train_words):,} words, test={len(test_words):,}")

    # Train the tokenizer.
    t0 = time.time()
    sp_prefix = str(MODELS_DIR / f"sp_{SP_MODEL_TYPE}_wiki2")
    tok = SPTokenizer.train_from_corpus(
        train_text, sp_prefix,
        vocab_size=SP_VOCAB_SIZE,
        model_type=SP_MODEL_TYPE,
    )
    train_time = time.time() - t0
    _log(
        f"SP({SP_MODEL_TYPE}) trained in {train_time:.1f}s "
        f"— vocab={tok.vocab_size}, model={sp_prefix}.model"
    )

    # Fit an LSTM through the fresh tokenizer.
    _log(
        f"LSTM fit: embed={LSTM_EMBED_DIM} hidden={LSTM_HIDDEN_DIM} "
        f"epochs={LSTM_EPOCHS}"
    )
    t0 = time.time()
    model = LSTMModel(
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=2,
        dropout=0.2,
    )
    model.fit(
        train_text,
        epochs=LSTM_EPOCHS,
        seq_len=LSTM_SEQ_LEN,
        batch_size=LSTM_BATCH,
        lr=LSTM_LR,
        tokenizer=tok,
    )
    fit_time = time.time() - t0
    ppl = model.perplexity(test_text[:200_000], seq_len=128)
    _log(f"LSTM done in {fit_time:.1f}s — held-out PPL(subword)={ppl:.1f}")

    # Save LSTM bundle.
    out = MODELS_DIR / f"lstm_sp_{SP_MODEL_TYPE}_wiki2"
    model.save(str(out))
    _log(f"Saved {out}.safetensors (+ .json meta)")


if __name__ == "__main__":
    main()
