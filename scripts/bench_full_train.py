"""Full-corpus multi-epoch training on WikiText-2 for all four neural variants.

Scales up the existing bench configs roughly 2-4x and runs enough epochs to
push test perplexity meaningfully below the 10-epoch baseline. Saves each
checkpoint under ``models/`` so the FastAPI can pick them up via the
``AUTOCOMPLETE_{LSTM,TRANSFORMER}_CHECKPOINT_{WORD,BPE}`` env vars.

Variants trained:
    word  LSTM          ->  models/lstm_wikitext2_long.{safetensors,json}
    word  Transformer   ->  models/transformer_wikitext2_long.{safetensors,json}
    bpe   LSTM          ->  models/lstm_wikitext2_bpe_long.{safetensors,json}
    bpe   Transformer   ->  models/transformer_wikitext2_bpe_long.{safetensors,json}

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bench_full_train.py
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
from src.transformer_model import TransformerModel  # noqa: E402
from src.bpe_tokenizer import BPETokenizer, HAS_TRANSFORMERS  # noqa: E402

CORPUS_PATH = ROOT / "data" / "wikitext2" / "wikitext2_train.txt"
MODELS_DIR = ROOT / "models"

# Scaled-up from the 10-epoch baseline in bench_real_data.py (embed=128,
# hidden=256). ~3x parameter budget, ~2.5x epochs. Still comfortably
# under 10 min on an RTX 5080 per model.
LSTM_EMBED_DIM = 256
LSTM_HIDDEN_DIM = 512
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_VOCAB_CAP = 20000
LSTM_EPOCHS = 25
LSTM_SEQ_LEN = 64
LSTM_BATCH_SIZE = 64
LSTM_LR = 1e-3

# ~3x the baseline's 2-layer d_model=128 config. 6 layers is the
# sweet spot cited in the literature review for WikiText-2 under 50M params.
TF_D_MODEL = 256
TF_N_HEADS = 8
TF_N_LAYERS = 6
TF_FF_DIM = 1024
TF_MAX_SEQ_LEN = 128
TF_DROPOUT = 0.1
TF_VOCAB_CAP = 20000
TF_EPOCHS = 15
TF_SEQ_LEN = 64
TF_BATCH_SIZE = 64
TF_LR = 3e-4


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_corpus() -> str:
    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found at {CORPUS_PATH}. "
            "Run `scripts/bench_real_data.py` once to cache it."
        )
    return CORPUS_PATH.read_text(encoding="utf-8")


def _split_text(raw_text: str) -> tuple[list[str], list[str]]:
    tokens = tokenize(raw_text)
    train_tokens, test_tokens = train_test_split(tokens, test_ratio=0.1, seed=42)
    return train_tokens, test_tokens


def _split_raw_for_bpe(raw_text: str) -> tuple[str, str]:
    # BPE path wants raw text. Split on whitespace so the slice matches the
    # word-level train/test partition roughly.
    words = raw_text.split()
    tokens_train, tokens_test = train_test_split(words, test_ratio=0.1, seed=42)
    return " ".join(tokens_train), " ".join(tokens_test)


def train_lstm_word(train_tokens, test_tokens) -> LSTMModel:
    _log(f"LSTM/word: embed={LSTM_EMBED_DIM} hidden={LSTM_HIDDEN_DIM} "
         f"layers={LSTM_NUM_LAYERS} epochs={LSTM_EPOCHS} cap={LSTM_VOCAB_CAP}")
    model = LSTMModel(
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        vocab_cap=LSTM_VOCAB_CAP,
    )
    t0 = time.time()
    model.fit(
        train_tokens,
        epochs=LSTM_EPOCHS,
        seq_len=LSTM_SEQ_LEN,
        batch_size=LSTM_BATCH_SIZE,
        lr=LSTM_LR,
        use_compile=True,
    )
    train_s = time.time() - t0
    ppl = model.perplexity(test_tokens[:50_000], seq_len=128)
    _log(f"LSTM/word done in {train_s:.1f}s, held-out PPL={ppl:.1f}")
    out = MODELS_DIR / "lstm_wikitext2_long"
    model.save(str(out))
    _log(f"LSTM/word saved to {out}.safetensors")
    return model


def train_transformer_word(train_tokens, test_tokens) -> TransformerModel:
    _log(f"Xform/word: d={TF_D_MODEL} heads={TF_N_HEADS} layers={TF_N_LAYERS} "
         f"epochs={TF_EPOCHS} cap={TF_VOCAB_CAP}")
    model = TransformerModel(
        d_model=TF_D_MODEL,
        n_heads=TF_N_HEADS,
        n_layers=TF_N_LAYERS,
        ff_dim=TF_FF_DIM,
        max_seq_len=TF_MAX_SEQ_LEN,
        dropout=TF_DROPOUT,
        vocab_cap=TF_VOCAB_CAP,
    )
    t0 = time.time()
    model.fit(
        train_tokens,
        epochs=TF_EPOCHS,
        seq_len=TF_SEQ_LEN,
        batch_size=TF_BATCH_SIZE,
        lr=TF_LR,
    )
    train_s = time.time() - t0
    ppl = model.perplexity(test_tokens[:50_000], seq_len=128)
    _log(f"Xform/word done in {train_s:.1f}s, held-out PPL={ppl:.1f}")
    out = MODELS_DIR / "transformer_wikitext2_long"
    model.save(str(out))
    _log(f"Xform/word saved to {out}.safetensors")
    return model


def train_lstm_bpe(raw_train: str, raw_test: str) -> LSTMModel:
    if not HAS_TRANSFORMERS:
        _log("LSTM/bpe: skipped — 'transformers' not installed.")
        return None
    _log(f"LSTM/bpe: embed={LSTM_EMBED_DIM} hidden={LSTM_HIDDEN_DIM} "
         f"layers={LSTM_NUM_LAYERS} epochs={LSTM_EPOCHS}")
    tok = BPETokenizer()
    model = LSTMModel(
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    )
    t0 = time.time()
    model.fit(
        raw_train,
        epochs=LSTM_EPOCHS,
        seq_len=LSTM_SEQ_LEN,
        batch_size=LSTM_BATCH_SIZE,
        lr=LSTM_LR,
        use_compile=True,
        tokenizer=tok,
    )
    train_s = time.time() - t0
    ppl = model.perplexity(raw_test[:200_000], seq_len=128)
    _log(f"LSTM/bpe done in {train_s:.1f}s, held-out PPL(subword)={ppl:.1f}")
    out = MODELS_DIR / "lstm_wikitext2_bpe_long"
    model.save(str(out))
    _log(f"LSTM/bpe saved to {out}.safetensors")
    return model


def train_transformer_bpe(raw_train: str, raw_test: str) -> TransformerModel:
    if not HAS_TRANSFORMERS:
        _log("Xform/bpe: skipped — 'transformers' not installed.")
        return None
    _log(f"Xform/bpe: d={TF_D_MODEL} heads={TF_N_HEADS} layers={TF_N_LAYERS} "
         f"epochs={TF_EPOCHS}")
    tok = BPETokenizer()
    model = TransformerModel(
        d_model=TF_D_MODEL,
        n_heads=TF_N_HEADS,
        n_layers=TF_N_LAYERS,
        ff_dim=TF_FF_DIM,
        max_seq_len=TF_MAX_SEQ_LEN,
        dropout=TF_DROPOUT,
    )
    t0 = time.time()
    model.fit(
        raw_train,
        epochs=TF_EPOCHS,
        seq_len=TF_SEQ_LEN,
        batch_size=TF_BATCH_SIZE,
        lr=TF_LR,
        tokenizer=tok,
    )
    train_s = time.time() - t0
    ppl = model.perplexity(raw_test[:200_000], seq_len=128)
    _log(f"Xform/bpe done in {train_s:.1f}s, held-out PPL(subword)={ppl:.1f}")
    out = MODELS_DIR / "transformer_wikitext2_bpe_long"
    model.save(str(out))
    _log(f"Xform/bpe saved to {out}.safetensors")
    return model


def main() -> None:
    if not HAS_TORCH:
        raise SystemExit("PyTorch not installed — nothing to train.")

    MODELS_DIR.mkdir(exist_ok=True)
    raw = _load_corpus()
    _log(f"Corpus loaded: {len(raw):,} chars")

    train_tokens, test_tokens = _split_text(raw)
    _log(f"Word-level split: train={len(train_tokens):,} test={len(test_tokens):,}")

    raw_train, raw_test = _split_raw_for_bpe(raw)
    _log(f"BPE raw split: train={len(raw_train):,} chars, test={len(raw_test):,} chars")

    overall = time.time()

    train_lstm_word(train_tokens, test_tokens)
    train_transformer_word(train_tokens, test_tokens)
    train_lstm_bpe(raw_train, raw_test)
    train_transformer_bpe(raw_train, raw_test)

    _log(f"All models done in {time.time() - overall:.1f}s total")


if __name__ == "__main__":
    main()
