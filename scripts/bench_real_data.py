"""Real-data benchmark: WikiText-2 (HF) + SmolLM2-135M on the RTX 5080.

Trains the project's statistical LMs (n-gram, Markov) on WikiText-2 raw text,
reports perplexity and top-k accuracy, then loads SmolLM2-135M on CUDA for a
side-by-side autocomplete demo. The transformer is the "real" SLM baseline;
the n-gram/Markov models are the project's teaching surface.

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bench_real_data.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Tame HF chatter before importing.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import tokenize, train_test_split, get_corpus_stats
from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.neural_model import LSTMModel, HAS_TORCH
from src.transformer_model import TransformerModel
from src.evaluation import (
    autocomplete_accuracy,
    prediction_diversity,
    vocabulary_coverage,
    compute_perplexity,
)

LSTM_EMBED_DIM = 128
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_VOCAB_CAP = 20000
LSTM_EPOCHS = 10
LSTM_SEQ_LEN = 64
LSTM_BATCH_SIZE = 64
LSTM_LR = 1e-3

TF_D_MODEL = 128
TF_N_HEADS = 4
TF_N_LAYERS = 4
TF_FF_DIM = 512
TF_MAX_SEQ_LEN = 128
TF_VOCAB_CAP = 20000
TF_EPOCHS = 5
TF_SEQ_LEN = 64
TF_BATCH_SIZE = 64
TF_LR = 3e-4

CACHE_DIR = ROOT / "data" / "wikitext2"
CORPUS_PATH = CACHE_DIR / "wikitext2_train.txt"
SLM_NAME = "HuggingFaceTB/SmolLM2-135M"
SLM_PROMPTS = [
    "Machine learning is a subset of",
    "The attention mechanism allows",
    "Convolutional neural networks are",
    "In the nineteenth century ,",
]


def fetch_wikitext2() -> str:
    """Download WikiText-2-raw train split once, cache as a flat .txt."""
    if CORPUS_PATH.exists():
        return CORPUS_PATH.read_text(encoding="utf-8")

    from datasets import load_dataset
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # WikiText rows are one paragraph each, many empty. Drop the heading-only
    # and empty rows so the tokenizer sees continuous prose.
    text = "\n".join(r.strip() for r in ds["text"] if r.strip() and not r.strip().startswith("="))
    CORPUS_PATH.write_text(text, encoding="utf-8")
    return text


def eval_stat_model(model, train_tokens, test_tokens, *, n_probe: int = 1000, top_k: int = 5):
    """Compute perplexity + top-k accuracy/diversity/coverage on a slice."""
    ppl = compute_perplexity(model, test_tokens)

    if isinstance(model, NGramModel):
        ctx_len = max(model.n - 1, 1)
    elif isinstance(model, (LSTMModel, TransformerModel)):
        ctx_len = 5  # same probe window as the SmolLM2 row — fair shot for neural
    else:
        ctx_len = 1

    preds, truth = [], []
    probe_end = min(len(test_tokens) - 1, ctx_len + n_probe)
    for i in range(ctx_len, probe_end):
        ctx = test_tokens[i - ctx_len:i]
        preds.append([w for w, _ in model.predict_next(ctx, top_k=top_k)])
        truth.append(test_tokens[i])

    top1 = autocomplete_accuracy(preds, truth, top_k=1)
    topk = autocomplete_accuracy(preds, truth, top_k=top_k)
    div = prediction_diversity(preds)
    cov = vocabulary_coverage(preds, set(test_tokens))
    return {
        "perplexity": ppl,
        "top1_acc": top1,
        f"top{top_k}_acc": topk,
        "diversity": div,
        "coverage": cov,
        "probes": len(truth),
    }


def run_slm_demo():
    """Load SmolLM2-135M on CUDA and greedy-complete a few prompts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"\n=== SLM: {SLM_NAME} on {device} ({dtype}) ===")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(SLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(SLM_NAME, torch_dtype=dtype).to(device).eval()
    print(f"loaded in {time.perf_counter() - t0:.1f}s  |  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    for prompt in SLM_PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            t0 = time.perf_counter()
            out = model.generate(
                **inputs, max_new_tokens=12,
                do_sample=False, pad_token_id=tok.eos_token_id,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        text = tok.decode(out[0], skip_special_tokens=True)
        new = text[len(prompt):].strip()
        print(f"  '{prompt}'\n    → {new!r}   ({dt*1000:.0f} ms)")


def train_lstm_full(train: list[str]) -> LSTMModel:
    """Train the LSTM on the full WikiText-2 train split.

    Uses the PR 1 + PR 2 + PR 10 stack: batched training, gradient clipping,
    cosine LR, bf16 autocast, weight tying, vocab cap, stateful BPTT.
    """
    model = LSTMModel(
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=0.3,
        vocab_cap=LSTM_VOCAB_CAP,
    )
    model.fit(
        train,
        epochs=LSTM_EPOCHS,
        seq_len=LSTM_SEQ_LEN,
        batch_size=LSTM_BATCH_SIZE,
        lr=LSTM_LR,
        stateful=True,
    )
    return model


def train_transformer_full(train: list[str]) -> TransformerModel:
    """Train the decoder-only transformer on the full WikiText-2 train split.

    Mirrors the LSTM config above (same ~20k vocab cap, word-level tokens)
    so the two neural rows in the report table are apples-to-apples. AdamW
    + cosine LR + bf16 autocast come from the model's default training
    loop. Takes a couple minutes on RTX 5080.
    """
    model = TransformerModel(
        d_model=TF_D_MODEL,
        n_heads=TF_N_HEADS,
        n_layers=TF_N_LAYERS,
        ff_dim=TF_FF_DIM,
        max_seq_len=TF_MAX_SEQ_LEN,
        dropout=0.1,
        vocab_cap=TF_VOCAB_CAP,
    )
    model.fit(
        train,
        epochs=TF_EPOCHS,
        seq_len=TF_SEQ_LEN,
        batch_size=TF_BATCH_SIZE,
        lr=TF_LR,
    )
    return model


def main():
    print(f"project: {ROOT}")
    print("\n[1/4] Fetching WikiText-2 raw train split…")
    t0 = time.perf_counter()
    text = fetch_wikitext2()
    print(f"    cache: {CORPUS_PATH}  ({len(text):,} chars, {time.perf_counter()-t0:.1f}s)")

    print("\n[2/4] Tokenize + train n-gram (n=3) and Markov chain on real data.")
    tokens = tokenize(text)
    print("     corpus:", get_corpus_stats(tokens))
    train, test = train_test_split(tokens, test_ratio=0.1, seed=42)
    print(f"     split: train={len(train):,}  test={len(test):,}")

    t0 = time.perf_counter()
    ngram = NGramModel(n=3, min_freq=2).fit(train)
    ng_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    markov = MarkovChainModel().fit(train)
    mk_fit = time.perf_counter() - t0

    print(f"     fit times: ngram={ng_fit:.2f}s  markov={mk_fit:.2f}s")

    print("\n     evaluating on held-out slice…")
    ng_metrics = eval_stat_model(ngram, train, test, n_probe=1000)
    mk_metrics = eval_stat_model(markov, train, test, n_probe=1000)

    def row(name, m):
        return (f"     {name:<8}  ppl={m['perplexity']:>10.1f}  "
                f"top1={m['top1_acc']:6.2%}  top5={m['top5_acc']:6.2%}  "
                f"div={m['diversity']:6.2%}  cov={m['coverage']:6.2%}")

    print("\n     ── statistical models ──")
    print(row("ngram", ng_metrics))
    print(row("markov", mk_metrics))

    print("\n[3/4] Train LSTM + Transformer on full corpus (minutes, not seconds).")
    if HAS_TORCH:
        t0 = time.perf_counter()
        lstm = train_lstm_full(train)
        lstm_fit = time.perf_counter() - t0
        print(f"     lstm fit: {lstm_fit:.1f}s (vocab={lstm.vocab_size})")
        lstm_metrics = eval_stat_model(lstm, train, test, n_probe=1000)

        t0 = time.perf_counter()
        xformer = train_transformer_full(train)
        tf_fit = time.perf_counter() - t0
        print(f"     transformer fit: {tf_fit:.1f}s (vocab={xformer.vocab_size})")
        tf_metrics = eval_stat_model(xformer, train, test, n_probe=1000)

        print("\n     ── neural models ──")
        print(row("lstm", lstm_metrics))
        print(row("xformer", tf_metrics))
    else:
        print("     torch not available; skipping neural rows.")

    print("\n[4/4] Small language model baseline.")
    try:
        run_slm_demo()
    except Exception as e:
        print(f"     SLM demo failed: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
