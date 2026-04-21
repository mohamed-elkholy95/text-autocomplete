"""Subword-trained LSTM + Transformer benchmark on WikiText-2.

Trains both neural families on SmolLM2-tokenized WikiText-2 (the full
train split) and measures token-level perplexity on the held-out slice.
Separate from ``bench_real_data.py`` because:

- It requires ``transformers`` (the BPE path). The main benchmark keeps
  that optional so the teaching surface stays minimal.
- Subword runs are larger (49k-vocab softmax vs 20k word-level) and
  slower — splitting them keeps the main bench's wall time bounded.

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bench_bpe.py

The numbers it prints are not directly comparable to the word-level
rows — PPL is in different units (subword pieces vs whole words). For
an apples-to-apples view the README compares *within* tokenization
scheme, not across.
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

from src.bpe_tokenizer import BPETokenizer, HAS_TRANSFORMERS
from src.data_loader import train_test_split
from src.neural_model import LSTMModel, HAS_TORCH
from src.transformer_model import TransformerModel

CORPUS_PATH = ROOT / "data" / "wikitext2" / "wikitext2_train.txt"

LSTM_EMBED_DIM = 128
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_EPOCHS = 3
LSTM_SEQ_LEN = 64
LSTM_BATCH_SIZE = 64
LSTM_LR = 1e-3

TF_D_MODEL = 128
TF_N_HEADS = 4
TF_N_LAYERS = 2
TF_FF_DIM = 512
TF_MAX_SEQ_LEN = 128
TF_EPOCHS = 3
TF_SEQ_LEN = 64
TF_BATCH_SIZE = 64
TF_LR = 3e-4


def _topk_metrics(model, test_ids, tokenizer, n_probe: int = 500, top_k: int = 5):
    """Top-k accuracy + diversity over subword ids (not words)."""
    # Build a list of subword strings from ids so predict_next can be
    # driven from the project's existing string-based contract.
    import random
    ctx_len = 5
    rng = random.Random(42)
    positions = sorted(rng.sample(
        range(ctx_len, len(test_ids) - 1),
        min(n_probe, len(test_ids) - ctx_len - 1),
    ))
    hits1 = 0
    hitsk = 0
    predicted_first = []
    for pos in positions:
        ctx_text = tokenizer.decode(test_ids[pos - ctx_len:pos])
        preds = model.predict_next([ctx_text], top_k=top_k)
        truth_sub = tokenizer.decode([test_ids[pos]])
        pred_subs = [w for w, _ in preds]
        if pred_subs:
            predicted_first.append(pred_subs[0])
        if pred_subs and pred_subs[0] == truth_sub:
            hits1 += 1
        if truth_sub in pred_subs:
            hitsk += 1
    n = len(positions)
    return {
        "top1": hits1 / max(n, 1),
        f"top{top_k}": hitsk / max(n, 1),
        "diversity": len(set(predicted_first)) / max(len(predicted_first), 1),
    }


def _row(name, fit_s, ppl, m):
    return (f"     {name:<16}  fit={fit_s:>5.1f}s  ppl={ppl:>8.1f}  "
            f"top1={m['top1']:6.2%}  top5={m['top5']:6.2%}  div={m['diversity']:6.2%}")


def main():
    if not HAS_TORCH:
        print("torch not installed; aborting.")
        return
    if not HAS_TRANSFORMERS:
        print("transformers not installed; aborting.")
        return
    if not CORPUS_PATH.exists():
        print(f"{CORPUS_PATH} missing — run scripts/bench_real_data.py first.")
        return

    print("[1/3] Loading BPE tokenizer and tokenizing WikiText-2")
    tok = BPETokenizer()
    text = CORPUS_PATH.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    all_ids = tok.encode(text)
    print(f"    tokenizer={tok.name}  vocab_size={tok.vocab_size:,}")
    print(f"    {len(text):,} chars -> {len(all_ids):,} subword ids ({time.perf_counter()-t0:.1f}s)")

    # Use the same 90/10 split logic as the word-level bench. Pass the
    # ids through stringify + train_test_split to match, but split on
    # the ids directly for accuracy.
    str_ids = [str(i) for i in all_ids]
    train_s, test_s = train_test_split(str_ids, test_ratio=0.1, seed=42)
    train_ids = [int(s) for s in train_s]
    test_ids = [int(s) for s in test_s]
    print(f"    split: train={len(train_ids):,}  test={len(test_ids):,}")
    # Pre-join the training ids as text so LSTMModel/TransformerModel's
    # tokenizer path can re-encode. This is a bit wasteful but keeps the
    # model contract (data is str or List[str]) untouched.
    train_text = tok.decode(train_ids)
    test_text = tok.decode(test_ids)

    print(f"\n[2/3] Training BPE-LSTM (embed={LSTM_EMBED_DIM} hidden={LSTM_HIDDEN_DIM} layers={LSTM_NUM_LAYERS})")
    t0 = time.perf_counter()
    lstm = LSTMModel(
        embed_dim=LSTM_EMBED_DIM, hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS, dropout=0.2,
    )
    lstm.fit(
        train_text, epochs=LSTM_EPOCHS, seq_len=LSTM_SEQ_LEN,
        batch_size=LSTM_BATCH_SIZE, lr=LSTM_LR, tokenizer=tok, stateful=True,
    )
    lstm_fit = time.perf_counter() - t0
    lstm_ppl = lstm.perplexity(test_text, seq_len=128)
    lstm_metrics = _topk_metrics(lstm, test_ids, tok)

    print(f"\n[3/3] Training BPE-Transformer (d_model={TF_D_MODEL} heads={TF_N_HEADS} layers={TF_N_LAYERS})")
    t0 = time.perf_counter()
    xformer = TransformerModel(
        d_model=TF_D_MODEL, n_heads=TF_N_HEADS, n_layers=TF_N_LAYERS,
        ff_dim=TF_FF_DIM, max_seq_len=TF_MAX_SEQ_LEN, dropout=0.1,
    )
    xformer.fit(
        train_text, epochs=TF_EPOCHS, seq_len=TF_SEQ_LEN,
        batch_size=TF_BATCH_SIZE, lr=TF_LR, tokenizer=tok,
    )
    tf_fit = time.perf_counter() - t0
    tf_ppl = xformer.perplexity(test_text, seq_len=128)
    tf_metrics = _topk_metrics(xformer, test_ids, tok)

    print("\n     ── subword models on WikiText-2 held-out slice ──")
    print(_row("bpe-lstm", lstm_fit, lstm_ppl, lstm_metrics))
    print(_row("bpe-xformer", tf_fit, tf_ppl, tf_metrics))
    print("\nNote: PPL is over subword pieces, not whole words — don't "
          "compare directly against the word-level rows in bench_real_data.py.")


if __name__ == "__main__":
    main()
