#!/usr/bin/env python3
"""
Command-Line Interface for Text Autocomplete
=============================================

A unified CLI for training models, making predictions, evaluating performance,
and managing the text autocomplete system from the terminal.

EDUCATIONAL CONTEXT:
-------------------
A good CLI makes your project accessible beyond Jupyter notebooks and web UIs.
Reviewers and hiring managers often clone your repo and run it locally —
a clean CLI with --help documentation makes a strong first impression.

DESIGN PATTERN: Subcommand Architecture
    cli.py train   → Train and save models
    cli.py predict → Get autocomplete suggestions
    cli.py eval    → Run evaluation metrics
    cli.py info    → Show model and corpus statistics

This mirrors popular tools like git (git commit, git push) and docker
(docker build, docker run). Each subcommand has its own arguments.

Usage:
    python cli.py train --model ngram --n 3 --save models/ngram_3.json
    python cli.py predict --text "machine learning is" --model ngram --top-k 5
    python cli.py eval --model ngram --test-ratio 0.2
    python cli.py info
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import TOP_K, RANDOM_SEED
from src.data_loader import (
    load_sample_data,
    tokenize,
    train_test_split,
    get_corpus_stats,
)
from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.neural_model import LSTMModel, HAS_TORCH
from src.transformer_model import TransformerModel
from src.evaluation import (
    compute_perplexity,
    autocomplete_accuracy,
    prediction_diversity,
    vocabulary_coverage,
)


def cmd_train(args: argparse.Namespace) -> None:
    """Train a language model and optionally save it to disk.

    Training flow:
    1. Load and tokenize the corpus
    2. Split into train/test sets (if --eval flag is set)
    3. Train the selected model on the training tokens
    4. Optionally save the trained model to a JSON file
    """
    print(f"📚 Loading corpus...")
    corpus = load_sample_data()
    tokens = tokenize(corpus)
    print(f"   Tokenized: {len(tokens):,} tokens, {len(set(tokens)):,} unique")

    # Train/test split for optional evaluation during training
    if args.eval:
        train_tokens, test_tokens = train_test_split(tokens, test_ratio=0.2)
        print(f"   Split: {len(train_tokens):,} train, {len(test_tokens):,} test")
    else:
        train_tokens = tokens
        test_tokens = None

    # Train the selected model
    start = time.perf_counter()

    if args.model == "ngram":
        n = args.n or 3
        model = NGramModel(n=n, seed=RANDOM_SEED)
        model.fit(train_tokens)
        print(f"✅ N-gram model (n={n}) trained in {time.perf_counter() - start:.2f}s")
        print(f"   Vocabulary: {model.vocab_size:,} words")
        stats = model.get_ngram_stats()
        for order, count in stats.items():
            print(f"   {order}-grams: {count:,}")
    elif args.model == "markov":
        model = MarkovChainModel(seed=RANDOM_SEED)
        model.fit(train_tokens)
        elapsed = time.perf_counter() - start
        print(f"✅ Markov chain model trained in {elapsed:.2f}s")
        print(f"   Vocabulary: {model.vocab_size:,} words")
        print(f"   Transitions: {model.n_transitions:,}")
    elif args.model == "lstm":
        if not HAS_TORCH:
            print("❌ LSTM training requires PyTorch. Install it or pick ngram/markov.")
            sys.exit(1)
        model = LSTMModel(
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            vocab_cap=args.vocab_cap,
        )
        model.fit(
            train_tokens,
            epochs=args.epochs,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            use_compile=args.compile,
            stateful=args.stateful,
        )
        elapsed = time.perf_counter() - start
        print(f"✅ LSTM trained in {elapsed:.2f}s")
        print(f"   Vocabulary: {model.vocab_size:,} words")
        print(f"   Params: embed={args.embed_dim}, hidden={args.hidden_dim}, layers={args.num_layers}")
    elif args.model == "transformer":
        if not HAS_TORCH:
            print("❌ Transformer training requires PyTorch. Install it or pick ngram/markov.")
            sys.exit(1)
        model = TransformerModel(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.num_layers,
            ff_dim=args.ff_dim,
            max_seq_len=args.max_seq_len,
            vocab_cap=args.vocab_cap,
        )
        model.fit(
            train_tokens,
            epochs=args.epochs,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        elapsed = time.perf_counter() - start
        print(f"✅ Transformer trained in {elapsed:.2f}s")
        print(f"   Vocabulary: {model.vocab_size:,} words")
        print(f"   Params: d_model={args.d_model}, heads={args.n_heads}, layers={args.num_layers}, ff={args.ff_dim}")
    else:
        print(f"❌ Unknown model: {args.model}")
        sys.exit(1)

    # Optional evaluation on test set
    if test_tokens:
        ppl = compute_perplexity(model, test_tokens)
        print(f"📏 Test perplexity: {ppl:.2f}")

    # Save model if requested
    if args.save:
        model.save(args.save)
        print(f"💾 Model saved to {args.save}")


def cmd_predict(args: argparse.Namespace) -> None:
    """Get autocomplete predictions for input text.

    If a saved model path is provided via --load, it loads that model.
    Otherwise, it trains a fresh model on the sample corpus.
    """
    # Load or train model
    if args.load:
        print(f"📂 Loading model from {args.load}...")
        if args.model == "ngram":
            model = NGramModel.load(args.load)
        elif args.model == "markov":
            model = MarkovChainModel.load(args.load)
        elif args.model == "transformer":
            if not HAS_TORCH:
                print("❌ Loading a transformer requires PyTorch.")
                sys.exit(1)
            model = TransformerModel.load(args.load)
        else:
            if not HAS_TORCH:
                print("❌ Loading an LSTM requires PyTorch.")
                sys.exit(1)
            model = LSTMModel.load(args.load)
    else:
        print(f"🔄 Training fresh {args.model} model...")
        corpus = load_sample_data()
        tokens = tokenize(corpus)
        if args.model == "ngram":
            model = NGramModel(n=args.n or 3)
            model.fit(tokens)
        elif args.model == "markov":
            model = MarkovChainModel()
            model.fit(tokens)
        else:
            if not HAS_TORCH:
                print("❌ LSTM requires PyTorch. Use --load or pick ngram/markov.")
                sys.exit(1)
            model = LSTMModel()
            model.fit(tokens, epochs=3)

    # Tokenize input and predict
    input_tokens = tokenize(args.text)
    if not input_tokens:
        print("❌ No valid tokens in input text.")
        sys.exit(1)

    predictions = model.predict_next(input_tokens, top_k=args.top_k)

    # Display results
    print(f"\n🔮 Predictions for: \"{args.text}\"")
    print(f"   Context tokens: {input_tokens[-5:]}")
    print(f"   Model: {args.model}\n")

    if not predictions:
        print("   (no predictions available)")
        return

    # Show predictions as a formatted table. BPE/byte-level tokens can
    # carry leading spaces (e.g. " of") or continuation marks (e.g. "##ing");
    # the raw form is useful when debugging, but the visual form is what
    # the user wanted to predict next. `_display_token` renders the
    # human-visible version; `_raw_token` keeps the original for callers
    # that need to round-trip through the vocab.
    def _display_token(tok: str) -> str:
        return tok.lstrip(" ") if tok.startswith(" ") else tok

    display = [(_display_token(w), w, p) for w, p in predictions]
    max_word_len = max(len(d) for d, _, _ in display)
    for rank, (shown, raw, prob) in enumerate(display, 1):
        bar = "█" * int(prob * 40)  # Visual probability bar
        suffix = f"  (raw: {raw!r})" if shown != raw else ""
        print(f"   {rank}. {shown:<{max_word_len}}  {prob:>7.2%}  {bar}{suffix}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run comprehensive evaluation on all models.

    Trains both n-gram and Markov models, computes metrics on a held-out
    test set, and prints a formatted comparison report.
    """
    print("📚 Loading and splitting corpus...")
    corpus = load_sample_data()
    tokens = tokenize(corpus)
    train_tokens, test_tokens = train_test_split(
        tokens, test_ratio=args.test_ratio, seed=RANDOM_SEED,
    )

    # Train the statistical models.
    print("🔧 Training n-gram model...")
    ngram = NGramModel(n=args.n or 3)
    ngram.fit(train_tokens)

    print("🔧 Training Markov chain model...")
    markov = MarkovChainModel()
    markov.fit(train_tokens)

    models: "list[tuple[str, object, int]]" = [
        ("N-gram", ngram, (args.n or 3) - 1),
        ("Markov", markov, 1),
    ]

    if args.include_lstm:
        if not HAS_TORCH:
            print("⚠️  --include-lstm requested but PyTorch is not installed; skipping.")
        else:
            print(f"🔧 Training LSTM ({args.lstm_epochs} epoch(s))...")
            lstm = LSTMModel(embed_dim=64, hidden_dim=128, num_layers=2)
            lstm.fit(
                train_tokens,
                epochs=args.lstm_epochs,
                seq_len=min(32, max(len(train_tokens) // 4, 4)),
                batch_size=32,
                lr=1e-3,
            )
            models.append(("LSTM", lstm, max((args.n or 3) - 1, 1)))

    if args.include_transformer:
        if not HAS_TORCH:
            print("⚠️  --include-transformer requested but PyTorch is not installed; skipping.")
        else:
            print(f"🔧 Training Transformer ({args.transformer_epochs} epoch(s))...")
            xformer = TransformerModel(
                d_model=64, n_heads=4, n_layers=2, ff_dim=128, max_seq_len=64,
            )
            xformer.fit(
                train_tokens,
                epochs=args.transformer_epochs,
                seq_len=min(32, max(len(train_tokens) // 4, 4)),
                batch_size=32,
                lr=3e-4,
            )
            models.append(("Transformer", xformer, max((args.n or 3) - 1, 1)))

    predictions: "dict[str, list[list[str]]]" = {name: [] for name, _, _ in models}
    ground_truth: "list[str]" = []

    max_ctx = max(ctx for _, _, ctx in models)
    for i in range(max_ctx, min(len(test_tokens) - 1, 500)):
        truth = test_tokens[i]
        ground_truth.append(truth)
        for name, model, ctx_len in models:
            preds = model.predict_next(test_tokens[i - ctx_len:i], top_k=args.top_k)
            predictions[name].append([w for w, _ in preds])

    vocab_set = set(test_tokens)
    per_model = {}
    for name, model, _ in models:
        per_model[name] = {
            "ppl": compute_perplexity(model, test_tokens),
            "top1": autocomplete_accuracy(predictions[name], ground_truth, top_k=1),
            "div": prediction_diversity(predictions[name]),
            "cov": vocabulary_coverage(predictions[name], vocab_set),
        }

    names = [name for name, _, _ in models]
    # Widen columns enough that long model names ("Transformer") and
    # huge perplexity values from undertrained models don't collide.
    col_w = max(13, max(len(n) for n in names) + 2)
    header = f"{'Metric':<30}" + "".join(f"{name:>{col_w}}" for name in names)
    print("\n" + "=" * len(header))
    print("          TEXT AUTOCOMPLETE — EVALUATION REPORT")
    print("=" * len(header))
    print("\n" + header)
    print("-" * len(header))
    rows = (
        ("Perplexity (↓ better)", "ppl", False),
        ("Top-1 Accuracy (↑ better)", "top1", True),
        ("Prediction Diversity", "div", True),
        ("Vocabulary Coverage", "cov", True),
    )
    for label, key, pct in rows:
        row = f"{label:<30}"
        for name in names:
            v = per_model[name][key]
            if pct:
                cell = f"{v:.2%}"
            elif v > 1e6:
                cell = f"{v:.2e}"  # huge PPL from undertrained runs → scientific
            else:
                cell = f"{v:.2f}"
            row += f"{cell:>{col_w}}"
        print(row)
    print("-" * len(header))

    # Rank-sum winner: lower perplexity wins; higher top1/div/cov win.
    def rank(metric_key: str, higher_is_better: bool) -> dict:
        vals = sorted(
            ((name, per_model[name][metric_key]) for name in names),
            key=lambda x: x[1], reverse=higher_is_better,
        )
        return {name: i for i, (name, _) in enumerate(vals)}  # 0 = best

    rankings = [
        rank("ppl", False),
        rank("top1", True),
        rank("div", True),
        rank("cov", True),
    ]
    scores = {name: sum(r[name] for r in rankings) for name in names}
    winner = min(scores, key=scores.get)
    wins = sum(1 for r in rankings if r[winner] == 0)
    print(f"\n🏆 Overall winner: {winner} (best on {wins}/{len(rankings)} metrics)")


def cmd_info(args: argparse.Namespace) -> None:
    """Display corpus statistics and project information."""
    corpus = load_sample_data()
    tokens = tokenize(corpus)
    stats = get_corpus_stats(tokens)

    print("📊 Corpus Statistics")
    print(f"   Total tokens:      {stats['total_tokens']:,}")
    print(f"   Unique tokens:     {stats['unique_tokens']:,}")
    print(f"   Avg token length:  {stats['avg_token_length']:.1f} chars")
    print(f"   Token length range: {stats['min_token_length']}–{stats['max_token_length']} chars")
    print(f"   Type-token ratio:  {stats['unique_tokens'] / max(stats['total_tokens'], 1):.4f}")
    print()

    # Show most frequent words
    from collections import Counter
    freq = Counter(tokens)
    print("📝 Top 15 Most Frequent Tokens:")
    for word, count in freq.most_common(15):
        pct = count / len(tokens) * 100
        bar = "█" * int(pct * 2)
        print(f"   {word:<15} {count:>5}  ({pct:>5.1f}%)  {bar}")


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog="autocomplete",
        description="Text Autocomplete CLI — Train, predict, and evaluate language models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py train --model ngram --n 3 --save models/ngram_3.json
  python cli.py predict --text "machine learning is" --top-k 5
  python cli.py eval --test-ratio 0.2
  python cli.py info
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train a language model")
    train_parser.add_argument(
        "--model", choices=["ngram", "markov", "lstm", "transformer"], default="ngram",
        help="Model type to train (default: ngram)",
    )
    train_parser.add_argument(
        "--n", type=int, default=3,
        help="N-gram order (only for ngram model, default: 3)",
    )
    train_parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save the trained model. ngram/markov → single JSON; "
             "lstm → pair of <path>.safetensors + <path>.json.",
    )
    train_parser.add_argument(
        "--eval", action="store_true",
        help="Evaluate on a held-out test set after training",
    )
    train_parser.add_argument("--epochs", type=int, default=5, help="LSTM: training epochs (default: 5)")
    train_parser.add_argument("--seq-len", type=int, default=20, help="LSTM: sequence length (default: 20)")
    train_parser.add_argument("--batch-size", type=int, default=32, help="LSTM: batch size (default: 32)")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="LSTM: learning rate (default: 1e-3)")
    train_parser.add_argument("--embed-dim", type=int, default=64, help="LSTM: embedding dim (default: 64)")
    train_parser.add_argument("--hidden-dim", type=int, default=128, help="LSTM: hidden dim (default: 128)")
    train_parser.add_argument("--num-layers", type=int, default=2, help="LSTM/transformer: number of layers (default: 2)")
    train_parser.add_argument("--d-model", type=int, default=128, help="Transformer: model dim (default: 128)")
    train_parser.add_argument("--n-heads", type=int, default=4, help="Transformer: attention heads (default: 4)")
    train_parser.add_argument("--ff-dim", type=int, default=512, help="Transformer: feed-forward dim (default: 512)")
    train_parser.add_argument("--max-seq-len", type=int, default=256, help="Transformer: max sequence length (default: 256)")
    train_parser.add_argument(
        "--vocab-cap", type=int, default=None,
        help="LSTM: cap vocab at top-N most frequent tokens; rest map to <unk>. "
             "Default None = keep every token.",
    )
    train_parser.add_argument(
        "--compile", action="store_true",
        help="LSTM: run the training forward/backward through torch.compile. "
             "Has warmup cost; best on longer runs. Falls back to eager on failure.",
    )
    train_parser.add_argument(
        "--stateful", action="store_true",
        help="LSTM: enable stateful BPTT — carry the (h, c) hidden state "
             "across seq_len windows (detached between windows). Default is "
             "stateless; stateful is opt-in because on short training budgets "
             "the two are within a few percent of each other and stateless "
             "has the nicer baseline.",
    )

    # --- predict subcommand ---
    pred_parser = subparsers.add_parser("predict", help="Get autocomplete predictions")
    pred_parser.add_argument(
        "--text", type=str, required=True,
        help="Input text to complete",
    )
    pred_parser.add_argument(
        "--model", choices=["ngram", "markov", "lstm", "transformer"], default="ngram",
        help="Model type (default: ngram)",
    )
    pred_parser.add_argument(
        "--n", type=int, default=3,
        help="N-gram order (default: 3)",
    )
    pred_parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of suggestions (default: {TOP_K})",
    )
    pred_parser.add_argument(
        "--load", type=str, default=None,
        help="Path to a saved model file (skips training)",
    )

    # --- eval subcommand ---
    eval_parser = subparsers.add_parser("eval", help="Evaluate model performance")
    eval_parser.add_argument(
        "--n", type=int, default=3,
        help="N-gram order (default: 3)",
    )
    eval_parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    eval_parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Top-k for accuracy (default: {TOP_K})",
    )
    eval_parser.add_argument(
        "--include-lstm", action="store_true",
        help="Also train and report an LSTM alongside ngram + markov (requires PyTorch).",
    )
    eval_parser.add_argument(
        "--lstm-epochs", type=int, default=3,
        help="Training epochs when --include-lstm is passed (default: 3).",
    )
    eval_parser.add_argument(
        "--include-transformer", action="store_true",
        help="Also train and report a decoder-only transformer (requires PyTorch).",
    )
    eval_parser.add_argument(
        "--transformer-epochs", type=int, default=3,
        help="Training epochs when --include-transformer is passed (default: 3).",
    )

    # --- info subcommand ---
    subparsers.add_parser("info", help="Show corpus statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "predict": cmd_predict,
        "eval": cmd_eval,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
