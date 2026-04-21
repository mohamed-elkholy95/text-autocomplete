"""Train an LSTM on BPE-encoded WikiText-2 (worked example).

The model classes in ``src/`` still expect ``tokens: List[str]``, so
this script demonstrates the bridge until a follow-up PR teaches them
to consume an external tokenizer directly:

    raw text → BPETokenizer.encode → [int] → stringified [str]
             → LSTMModel.fit via the standard fit() path

Each integer subword id is rendered as its string form (``"123"``) so
``LSTMModel._word_to_id`` treats it as an atomic vocabulary entry. This
is deliberately ugly — the hand-stringification exists to make the
contract mismatch visible. A proper follow-up should let the model
accept ``List[int]`` directly when a tokenizer is wired up.

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bpe_train_lstm.py
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

CORPUS_PATH = ROOT / "data" / "wikitext2" / "wikitext2_train.txt"
N_TRAIN_CHARS = 400_000   # keep this short so the demo finishes in seconds


def main():
    if not HAS_TRANSFORMERS:
        print("transformers not installed; skipping BPE demo.")
        return
    if not HAS_TORCH:
        print("torch not installed; skipping BPE demo.")
        return
    if not CORPUS_PATH.exists():
        print(f"{CORPUS_PATH} missing — run scripts/bench_real_data.py first.")
        return

    print(f"[1/3] Loading BPE tokenizer ({BPETokenizer.__init__.__defaults__[0]})")
    tok = BPETokenizer()
    print(f"      vocab_size={tok.vocab_size:,}")

    print(f"[2/3] Encoding first {N_TRAIN_CHARS:,} chars of WikiText-2")
    t0 = time.perf_counter()
    text = CORPUS_PATH.read_text(encoding="utf-8")[:N_TRAIN_CHARS]
    ids = tok.encode(text)
    print(f"      {len(text):,} chars → {len(ids):,} subword ids ({time.perf_counter()-t0:.1f}s)")

    # Bridge: the LSTM's fit() currently demands List[str]. Stringify ids
    # so each subword id becomes an atomic vocab entry. A retrofit PR
    # should let fit() accept List[int] + a tokenizer directly.
    str_ids = [str(i) for i in ids]
    train, test = train_test_split(str_ids, test_ratio=0.1, seed=42)

    print(f"[3/3] Training LSTM on BPE subwords (3 epochs, tiny config)")
    t0 = time.perf_counter()
    model = LSTMModel(embed_dim=96, hidden_dim=192, num_layers=2, dropout=0.2)
    model.fit(train, epochs=3, seq_len=64, batch_size=32, lr=1e-3, stateful=True)
    print(f"      trained in {time.perf_counter()-t0:.1f}s on vocab={model.vocab_size:,}")

    ppl = model.perplexity(test, seq_len=128)
    print(f"      held-out PPL on BPE ids = {ppl:.1f}")

    # Demo: autocomplete a prompt. Encode → predict → stringify back.
    prompt = "Machine learning is a"
    prompt_ids_str = [str(i) for i in tok.encode(prompt)]
    top = model.predict_next(prompt_ids_str, top_k=5)
    decoded = [(tok.decode([int(w)]) if w.isdigit() else w, p) for w, p in top]
    print(f"\n      '{prompt}' → top 5 next subwords:")
    for token_text, prob in decoded:
        print(f"        {token_text!r:<20} {prob:6.2%}")


if __name__ == "__main__":
    main()
