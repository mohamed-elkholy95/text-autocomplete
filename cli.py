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
import json
import sys
import time
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import TOP_K, RANDOM_SEED
from src.data_loader import (
    load_sample_data,
    load_wikitext,
    tokenize,
    train_test_split,
    get_corpus_stats,
)
from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.transformer_model import HAS_TRANSFORMERS, TransformerModel
from src.evaluation import (
    compute_perplexity,
    autocomplete_accuracy,
    prediction_diversity,
    vocabulary_coverage,
    generate_report,
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
    if args.model == "transformer":
        if not HAS_TRANSFORMERS:
            print("❌ Transformer requires `transformers` to be installed.")
            sys.exit(1)
        if args.load:
            print("⚠️  --load is ignored for transformer; weights come from HuggingFace.")
        print("🤗 Loading SmolLM2-135M from HuggingFace...")
        model = TransformerModel()
        model.fit(tokenize(load_sample_data()))  # calibration + warmup
    elif args.load:
        print(f"📂 Loading model from {args.load}...")
        if args.model == "ngram":
            model = NGramModel.load(args.load)
        else:
            model = MarkovChainModel.load(args.load)
    else:
        print(f"🔄 Training fresh {args.model} model...")
        tokens = tokenize(load_sample_data())
        if args.model == "ngram":
            model = NGramModel(n=args.n or 3)
            model.fit(tokens)
        else:
            model = MarkovChainModel()
            model.fit(tokens)

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

    # Show predictions as a formatted table
    max_word_len = max(len(w) for w, _ in predictions)
    for rank, (word, prob) in enumerate(predictions, 1):
        bar = "█" * int(prob * 40)  # Visual probability bar
        print(f"   {rank}. {word:<{max_word_len}}  {prob:>7.2%}  {bar}")


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

    # Train both models
    print("🔧 Training n-gram model...")
    ngram = NGramModel(n=args.n or 3)
    ngram.fit(train_tokens)

    print("🔧 Training Markov chain model...")
    markov = MarkovChainModel()
    markov.fit(train_tokens)

    # Compute perplexity
    ngram_ppl = compute_perplexity(ngram, test_tokens)
    markov_ppl = compute_perplexity(markov, test_tokens)

    # Compute accuracy and diversity
    # Generate predictions for each test position
    predictions_ngram = []
    predictions_markov = []
    ground_truth = []

    context_len = (args.n or 3) - 1
    for i in range(context_len, min(len(test_tokens) - 1, 500)):
        ctx = test_tokens[i - context_len:i]
        truth = test_tokens[i]
        ground_truth.append(truth)

        preds_n = ngram.predict_next(ctx, top_k=args.top_k)
        preds_m = markov.predict_next(ctx, top_k=args.top_k)
        predictions_ngram.append([w for w, _ in preds_n])
        predictions_markov.append([w for w, _ in preds_m])

    acc_ngram = autocomplete_accuracy(predictions_ngram, ground_truth, top_k=1)
    acc_markov = autocomplete_accuracy(predictions_markov, ground_truth, top_k=1)
    div_ngram = prediction_diversity(predictions_ngram)
    div_markov = prediction_diversity(predictions_markov)

    vocab_set = set(test_tokens)
    cov_ngram = vocabulary_coverage(predictions_ngram, vocab_set)
    cov_markov = vocabulary_coverage(predictions_markov, vocab_set)

    # Print report
    print("\n" + "=" * 60)
    print("          TEXT AUTOCOMPLETE — EVALUATION REPORT")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'N-gram':>12} {'Markov':>12}")
    print("-" * 54)
    print(f"{'Perplexity (↓ better)':<30} {ngram_ppl:>12.2f} {markov_ppl:>12.2f}")
    print(f"{'Top-1 Accuracy (↑ better)':<30} {acc_ngram:>11.2%} {acc_markov:>11.2%}")
    print(f"{'Prediction Diversity':<30} {div_ngram:>11.2%} {div_markov:>11.2%}")
    print(f"{'Vocabulary Coverage':<30} {cov_ngram:>11.2%} {cov_markov:>11.2%}")
    print("-" * 54)

    # Determine winner
    ngram_wins = sum([
        ngram_ppl < markov_ppl,
        acc_ngram > acc_markov,
        div_ngram > div_markov,
    ])
    winner = "N-gram" if ngram_wins >= 2 else "Markov"
    print(f"\n🏆 Overall winner: {winner} (won {max(ngram_wins, 3 - ngram_wins)}/3 metrics)")


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
        "--model", choices=["ngram", "markov"], default="ngram",
        help="Model type to train (default: ngram)",
    )
    train_parser.add_argument(
        "--n", type=int, default=3,
        help="N-gram order (only for ngram model, default: 3)",
    )
    train_parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save the trained model (JSON format)",
    )
    train_parser.add_argument(
        "--eval", action="store_true",
        help="Evaluate on a held-out test set after training",
    )

    # --- predict subcommand ---
    pred_parser = subparsers.add_parser("predict", help="Get autocomplete predictions")
    pred_parser.add_argument(
        "--text", type=str, required=True,
        help="Input text to complete",
    )
    pred_parser.add_argument(
        "--model", choices=["ngram", "markov", "transformer"], default="ngram",
        help=(
            "Model type: 'ngram' (classical), 'markov' (1st-order), or "
            "'transformer' (SmolLM2-135M from HuggingFace; returns BPE "
            "subword pieces). Default: ngram."
        ),
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
