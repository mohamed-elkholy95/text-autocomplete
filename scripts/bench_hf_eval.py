"""Evaluate pretrained HuggingFace causal LMs on WikiText-2 next-word.

Compares modern small open-weight LMs (SmolLM2-360M, Qwen2.5-0.5B,
Llama-3.2-1B when accessible) against the project's four in-house
models on the same WikiText-2 held-out split.

Two metrics per model:
    * Held-out perplexity — sliding-window token-level, in the model's
      own tokenisation. Subword PPL is NOT comparable to word-level PPL
      (different base units), so this column is only meaningful within a
      tokeniser family. Cross-family comparisons use top-k.
    * Top-1 / Top-5 next-WORD accuracy — at each probe position, feed
      the left context, greedy/argmax over next-token candidates, decode
      to text, take the first whitespace-split word, compare to the
      gold word. Identical protocol across every model.

Probe set is seeded so the same 500 held-out word positions are scored
by every model.

Run:
    /home/ai/miniforge3/envs/ai/bin/python scripts/bench_hf_eval.py \\
        --hf HuggingFaceTB/SmolLM2-135M \\
        --hf HuggingFaceTB/SmolLM2-360M \\
        --hf Qwen/Qwen2.5-0.5B \\
        --include-local
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import tokenize  # noqa: E402

CORPUS_PATH = ROOT / "data" / "wikitext2" / "wikitext2_train.txt"
SEED = 42
N_PROBES = 500
PROBE_CTX = 32  # words of left context per probe


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_corpus_split() -> Tuple[List[str], List[str], str]:
    """Contiguous 90/10 split at the character level.

    The project's ``train_test_split`` shuffles tokens with a permutation,
    which breaks sequential LM evaluation — a 32-word left context becomes
    a random word bag, and no LM can predict the next word from it. For
    this bench we want REAL next-word prediction, so we split the raw text
    at a newline boundary near the 90 % mark and tokenise each side
    independently. Returns (train_tokens, test_tokens, test_raw_text).
    """
    raw = CORPUS_PATH.read_text(encoding="utf-8")
    cut = int(len(raw) * 0.9)
    # Snap to the nearest newline so we don't slice mid-paragraph.
    cut = raw.find("\n", cut)
    if cut < 0:
        cut = int(len(raw) * 0.9)
    train_raw, test_raw = raw[:cut], raw[cut:]
    return tokenize(train_raw), tokenize(test_raw), test_raw


def _build_probes(test_tokens: List[str], n: int, ctx: int) -> List[Tuple[List[str], str]]:
    """Sample ``n`` probe positions from a CONTIGUOUS test stream.

    Each probe carries ``ctx`` real preceding words and the gold word at
    the sampled position — so "next-word prediction" is a meaningful
    task for any language model."""
    rng = random.Random(SEED)
    positions = rng.sample(range(ctx, len(test_tokens) - 1), k=min(n, len(test_tokens) - ctx - 1))
    return [(test_tokens[p - ctx:p], test_tokens[p]) for p in sorted(positions)]


def _hf_model_size_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2


def _hf_perplexity(model, tokenizer, text: str, max_tokens: int = 20000, seq_len: int = 512) -> float:
    """Non-overlapping sliding-window PPL over up to ``max_tokens`` of text.

    This is the chunk-PPL variant from the HF perplexity cookbook — a
    mild upper bound on the stride=1 number and what most papers report."""
    import torch

    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    ids = ids[:max_tokens].to(model.device)
    if ids.numel() < 2:
        return float("nan")

    total_nll = 0.0
    total_tokens = 0
    model.train(False)
    with torch.no_grad():
        for i in range(0, ids.numel() - 1, seq_len):
            chunk = ids[i:i + seq_len + 1]
            if chunk.numel() < 2:
                break
            inp = chunk[:-1].unsqueeze(0)
            tgt = chunk[1:].unsqueeze(0)
            logits = model(inp).logits
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            total_nll += nll
            total_tokens += tgt.numel()
    return float(math.exp(total_nll / max(total_tokens, 1)))


def _hf_next_word_topk(model, tokenizer, left_words: List[str], k_max: int = 5) -> List[str]:
    """Top-k predicted next WORDS after the given left context.

    Subword LMs usually take 1-3 subword tokens per word. For each of the
    top-k first-token candidates we greedy-continue up to 6 subword
    tokens, decode to text, and then run the result through the PROJECT'S
    ``tokenize`` regex so punctuation is split off the same way as it is
    for the gold word. Take the first non-empty normalised token as the
    predicted word. Without this re-tokenisation step HF output like
    ``"apple,"`` fails to match gold ``"apple"``."""
    import torch

    prompt = " ".join(left_words) + " "
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
        top_k_ids = torch.topk(logits, k=min(k_max, logits.numel())).indices.tolist()

    words: List[str] = []
    seen: set[str] = set()
    for tok_id in top_k_ids:
        seed = torch.cat([input_ids[0], torch.tensor([tok_id], device=model.device)]).unsqueeze(0)
        with torch.no_grad():
            gen = model.generate(
                seed,
                max_new_tokens=6,
                do_sample=False,
                temperature=1.0,
                pad_token_id=getattr(tokenizer, "eos_token_id", None) or 0,
            )
        decoded = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        normalised = tokenize(decoded)
        first_word = normalised[0] if normalised else ""
        if first_word and first_word not in seen:
            seen.add(first_word)
            words.append(first_word)
        if len(words) >= k_max:
            break
    return words


def _score_hf(model_id: str, test_raw: str, probes: List[Tuple[List[str], str]]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _log(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    size_mb = _hf_model_size_mb(model)

    t0 = time.perf_counter()
    ppl = _hf_perplexity(model, tokenizer, test_raw)
    ppl_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    correct1 = 0
    correct5 = 0
    for left, gold in probes:
        preds = _hf_next_word_topk(model, tokenizer, left, k_max=5)
        g = gold.lower()
        if preds and preds[0] == g:
            correct1 += 1
        if g in preds:
            correct5 += 1
    acc_s = time.perf_counter() - t0

    del model
    del tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_id,
        "size_mb": size_mb,
        "ppl": ppl,
        "top1": correct1 / len(probes),
        "top5": correct5 / len(probes),
        "ppl_s": ppl_s,
        "acc_s": acc_s,
    }


def _score_local(test_tokens: List[str], probes: List[Tuple[List[str], str]]) -> List[dict]:
    """Score the four in-house families on the SAME probe set."""
    from src.ngram_model import NGramModel
    from src.markov_model import MarkovChainModel
    try:
        from src.neural_model import LSTMModel
    except ImportError:
        LSTMModel = None  # type: ignore
    try:
        from src.transformer_model import TransformerModel
    except ImportError:
        TransformerModel = None  # type: ignore
    from src.evaluation import compute_perplexity

    train, _, _ = _load_corpus_split()

    def _probe_topk(model, left: List[str]) -> List[str]:
        preds = model.predict_next(list(left), top_k=5)
        return [w.lower() for (w, _) in preds]

    def _top_counts(model) -> Tuple[int, int]:
        c1 = 0
        c5 = 0
        for left, gold in probes:
            preds = _probe_topk(model, left)
            g = gold.lower()
            if preds and preds[0] == g:
                c1 += 1
            if g in preds:
                c5 += 1
        return c1, c5

    out: List[dict] = []

    _log("Fitting local ngram (n=3)...")
    ng = NGramModel(n=3, min_freq=2).fit(train)
    c1, c5 = _top_counts(ng)
    out.append({
        "model": "local:ngram",
        "size_mb": 0.0,
        "ppl": compute_perplexity(ng, test_tokens),
        "top1": c1 / len(probes),
        "top5": c5 / len(probes),
        "ppl_s": 0.0, "acc_s": 0.0,
    })

    _log("Fitting local markov...")
    mk = MarkovChainModel().fit(train)
    c1, c5 = _top_counts(mk)
    out.append({
        "model": "local:markov",
        "size_mb": 0.0,
        "ppl": compute_perplexity(mk, test_tokens),
        "top1": c1 / len(probes),
        "top5": c5 / len(probes),
        "ppl_s": 0.0, "acc_s": 0.0,
    })

    from pathlib import Path as _P
    models_dir = ROOT / "models"
    for kind, cls, base in (
        ("lstm", LSTMModel, "lstm_wikitext2_long"),
        ("transformer", TransformerModel, "transformer_wikitext2_long"),
        ("transformer_xl", TransformerModel, "transformer_wikitext2_xl"),
    ):
        bundle = models_dir / base
        if cls is None or not (models_dir / f"{base}.safetensors").exists():
            continue
        _log(f"Loading local {kind} checkpoint {base}...")
        m = cls.load(str(bundle))
        c1, c5 = _top_counts(m)
        out.append({
            "model": f"local:{kind}_long",
            "size_mb": _P(f"{bundle}.safetensors").stat().st_size / 1024 ** 2,
            "ppl": compute_perplexity(m, test_tokens),
            "top1": c1 / len(probes),
            "top5": c5 / len(probes),
            "ppl_s": 0.0, "acc_s": 0.0,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", action="append", default=[], help="Hugging Face model ID (repeatable).")
    ap.add_argument("--include-local", action="store_true", help="Also score the 4 in-house families.")
    ap.add_argument("--n-probes", type=int, default=N_PROBES)
    ap.add_argument("--output-md", type=str, default="")
    args = ap.parse_args()

    if not CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found at {CORPUS_PATH}. "
            "Run `scripts/bench_real_data.py` once to cache it."
        )

    train, test, test_raw = _load_corpus_split()
    probes = _build_probes(test, args.n_probes, PROBE_CTX)
    _log(f"Corpus: {len(train):,} train / {len(test):,} test tokens; {len(probes):,} probes, ctx={PROBE_CTX}")

    rows: List[dict] = []

    if args.include_local:
        rows.extend(_score_local(test, probes))

    for model_id in args.hf:
        try:
            rows.append(_score_hf(model_id, test_raw, probes))
        except Exception as exc:
            _log(f"  {model_id}: SKIPPED ({type(exc).__name__}: {exc})")

    lines: List[str] = []
    lines.append("")
    lines.append("| Model | Size (MB) | Held-out PPL* | Top-1 | Top-5 | PPL s | Acc s |")
    lines.append("| ----- | --------: | ------------: | ----: | ----: | ----: | ----: |")
    for r in rows:
        lines.append(
            f"| `{r['model']}` | {r['size_mb']:.1f} | {r['ppl']:.2f} | "
            f"{100*r['top1']:.2f} % | {100*r['top5']:.2f} % | "
            f"{r['ppl_s']:.1f} | {r['acc_s']:.1f} |"
        )
    lines.append("")
    lines.append(
        "\\* PPL is in the model's native tokeniser units. Subword PPL "
        "(SmolLM2 / Qwen) is NOT comparable to word-level PPL (local ngram "
        "/ markov / word-LSTM / word-Transformer). Use top-k for cross-"
        "tokenizer comparisons — they all score against the same gold word."
    )

    md = "\n".join(lines)
    print(md)
    if args.output_md:
        Path(args.output_md).write_text(md + "\n", encoding="utf-8")
        _log(f"wrote {args.output_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
