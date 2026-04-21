"""
Metrics Page — Model Evaluation and Analysis
==============================================
"""

import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON_BIN = sys.executable

from src.data_loader import tokenize, load_sample_data, train_test_split, get_corpus_stats
from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.evaluation import (
    compute_perplexity,
    autocomplete_accuracy,
    prediction_diversity,
    vocabulary_coverage,
    compare_models,
)

st.title("📈 Model Evaluation Metrics")

# ---------------------------------------------------------------------------
# Load and split data
# ---------------------------------------------------------------------------
@st.cache_resource
def prepare_evaluation():
    """Prepare evaluation data and models."""
    tokens = tokenize(load_sample_data())
    train, test = train_test_split(tokens, test_ratio=0.2)

    ngram = NGramModel(n=3)
    ngram.fit(train)

    markov = MarkovChainModel()
    markov.fit(train)

    return train, test, ngram, markov

train_tokens, test_tokens, ngram_model, markov_model = prepare_evaluation()

# ---------------------------------------------------------------------------
# Corpus Stats
# ---------------------------------------------------------------------------
st.header("📋 Corpus Statistics")

train_stats = get_corpus_stats(train_tokens)
test_stats = get_corpus_stats(test_tokens)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Training Set")
    st.metric("Tokens", f"{train_stats['total_tokens']:,}")
    st.metric("Unique Words", f"{train_stats['unique_tokens']:,}")
with col2:
    st.subheader("Test Set")
    st.metric("Tokens", f"{test_stats['total_tokens']:,}")
    st.metric("Unique Words", f"{test_stats['unique_tokens']:,}")

st.caption("The test set is 20% of the data, held out during training to measure generalization.")

# ---------------------------------------------------------------------------
# Perplexity Comparison
# ---------------------------------------------------------------------------
st.header("🧮 Perplexity — The Primary Language Model Metric")

st.markdown("""
**Perplexity** measures how "surprised" the model is by the test data.

| PPL Range | Quality | Meaning |
|-----------|---------|---------|
| < 50 | Excellent | Model predicts test text very well |
| 50-150 | Good | Captures common patterns |
| 150-500 | Fair | Some useful predictions |
| > 500 | Poor | Barely better than random |

*Formula: PPL = exp(-1/N × Σ log P(word_i | context_i))*
""")

if st.button("📊 Compute Perplexity", type="primary", use_container_width=True):
    with st.spinner("Evaluating models on test set..."):
        ngram_ppl = compute_perplexity(ngram_model, test_tokens)
        markov_ppl = compute_perplexity(markov_model, test_tokens)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("N-gram Perplexity", f"{ngram_ppl:.1f}")
        # Color indicator
        if ngram_ppl < 50:
            st.success("✅ Excellent")
        elif ngram_ppl < 150:
            st.info("ℹ️ Good")
        else:
            st.warning("⚠️ Fair — model needs more training data")
    with col2:
        st.metric("Markov Chain Perplexity", f"{markov_ppl:.1f}")
        if markov_ppl < 50:
            st.success("✅ Excellent")
        elif markov_ppl < 150:
            st.info("ℹ️ Good")
        else:
            st.warning("⚠️ Fair — model needs more training data")

    # Comparison bar chart
    fig = go.Figure(go.Bar(
        x=["N-gram", "Markov Chain"],
        y=[ngram_ppl, markov_ppl],
        marker_color=["#1f77b4", "#ff7f0e"],
        text=[f"{ngram_ppl:.1f}", f"{markov_ppl:.1f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Perplexity Comparison (Lower is Better ↓)",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#262730",
        font_color="white",
        yaxis_title="Perplexity",
    )
    st.plotly_chart(fig, use_container_width=True)

    winner = "N-gram" if ngram_ppl < markov_ppl else "Markov Chain"
    st.info(f"🏆 **{winner}** has lower perplexity — it's better at predicting unseen text!")

# ---------------------------------------------------------------------------
# N-gram Order Impact
# ---------------------------------------------------------------------------
st.divider()
st.header("📈 Impact of N-gram Order")

st.markdown("""
How does the number of context words affect model quality?
We train models with different n-gram orders and compare perplexity.
""")

if st.button("🔍 Compare N-gram Orders", use_container_width=True):
    orders = [1, 2, 3, 4]
    perplexities = []
    ngram_counts_list = []

    for n in orders:
        model = NGramModel(n=n)
        model.fit(train_tokens)
        ppl = compute_perplexity(model, test_tokens)
        perplexities.append(ppl)
        stats = model.get_ngram_stats()
        ngram_counts_list.append(sum(stats.values()))

    # Plot perplexity vs n-gram order
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"{n}-gram" for n in orders],
        y=perplexities,
        mode="lines+markers",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=12),
        text=[f"PPL={p:.1f}" for p in perplexities],
        textposition="top center",
    ))
    fig.update_layout(
        title="Perplexity vs N-gram Order",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#262730",
        font_color="white",
        yaxis_title="Perplexity (lower is better)",
        xaxis_title="N-gram Order",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show n-gram counts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Unique N-grams per Order")
        for n, count in zip(orders, ngram_counts_list):
            st.markdown(f"- **{n}-gram**: {count:,} unique n-grams")
    with col2:
        st.subheader("Key Insight")
        st.info("""
        📊 Higher n-grams capture more context but need exponentially more data.
        With our small corpus, 4-grams often perform WORSE than trigrams due to
        data sparsity — there simply aren't enough examples to estimate 4-gram
        probabilities reliably.
        """)

# ---------------------------------------------------------------------------
# Prediction Analysis
# ---------------------------------------------------------------------------
st.divider()
st.header("🔍 Prediction Analysis")

st.markdown("Detailed analysis of what each model predicts for specific contexts.")

context_input = st.text_input(
    "Enter context words (comma-separated):",
    value="machine, learning",
    help="These words will be used as the context for prediction.",
)

if st.button("🔍 Analyze Predictions", use_container_width=True):
    if not context_input.strip():
        st.error("Please enter context words!")
    else:
        ctx = [w.strip().lower() for w in context_input.split(",") if w.strip()]

        ngram_preds = ngram_model.predict_next(ctx, top_k=10)
        markov_preds = markov_model.predict_next(ctx, top_k=10)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("N-gram Predictions")
            fig1 = go.Figure(go.Bar(
                x=[w for w, _ in ngram_preds],
                y=[p * 100 for _, p in ngram_preds],
                marker_color="#1f77b4",
                text=[f"{p:.1%}" for _, p in ngram_preds],
                textposition="outside",
            ))
            fig1.update_layout(
                title="Top 10 N-gram Predictions",
                paper_bgcolor="#0e1117", plot_bgcolor="#262730",
                font_color="white",
                yaxis_title="Probability (%)",
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Markov Predictions")
            fig2 = go.Figure(go.Bar(
                x=[w for w, _ in markov_preds],
                y=[p * 100 for _, p in markov_preds],
                marker_color="#ff7f0e",
                text=[f"{p:.1%}" for _, p in markov_preds],
                textposition="outside",
            ))
            fig2.update_layout(
                title="Top 10 Markov Predictions",
                paper_bgcolor="#0e1117", plot_bgcolor="#262730",
                font_color="white",
                yaxis_title="Probability (%)",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Diversity analysis — count how many distinct words each model
        # surfaced for this context (a rough per-context diversity signal).
        for name, preds in zip(["N-gram", "Markov"], [ngram_preds, markov_preds]):
            unique_words = len({w for w, _ in preds})
            st.metric(f"{name} — Unique predictions", unique_words)

# ---------------------------------------------------------------------------
# Markov Chain Transition Visualization
# ---------------------------------------------------------------------------
st.divider()
st.header("🔗 Markov Chain Transitions")

st.markdown("Explore the transition probabilities for any word in the Markov chain.")

word_input = st.text_input(
    "Enter a word to see its transitions:",
    value="learning",
    help="Shows the most likely words that follow this word.",
)

if st.button("🔗 Show Transitions", use_container_width=True):
    if not word_input.strip():
        st.error("Please enter a word!")
    else:
        transitions = markov_model.get_top_transitions(word_input.strip().lower(), top_k=10)

        if not transitions:
            st.warning(f"No transitions found for '{word_input}'. Try a different word.")
        else:
            words, probs = zip(*transitions)
            fig = go.Figure(go.Bar(
                x=list(words),
                y=[p * 100 for p in probs],
                marker_color="#2ca02c",
                text=[f"{p:.1%}" for _, p in transitions],
                textposition="outside",
            ))
            fig.update_layout(
                title=f'Words Most Likely to Follow "{word_input}"',
                paper_bgcolor="#0e1117", plot_bgcolor="#262730",
                font_color="white",
                yaxis_title="Transition Probability (%)",
                xaxis_title="Next Word",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"""
            💡 The Markov chain learned these transitions from the training corpus.
            "{word_input}" appeared {sum(p * 100 for p in probs):.0f}% of the time
            with these {len(transitions)} words as its successor.
            """)


# ---------------------------------------------------------------------------
# Quick In-Memory Benchmark (sample corpus, sub-5s)
# ---------------------------------------------------------------------------
# Trains all available model families on the already-loaded sample
# corpus and reports PPL / top-5 / diversity / coverage in one table.
# The subprocess benchmarks below run against WikiText-2 and take
# minutes; this one is instant so the page has a useful default action.
st.divider()
st.header("⚡ Quick Benchmark — All Models (sample corpus)")

st.markdown(
    """
Runs on the same 40-sentence sample corpus already loaded for this
page. The neural rows appear only when PyTorch is importable, and
they use a deliberately tiny config so the whole benchmark finishes
in a couple of seconds. Numbers on a sample this small are the usual
small-data artefact (inflated PPL, low coverage) — the value is the
side-by-side comparison, not absolute scores.
"""
)


def _run_quick_benchmark(train, test, probes=40):
    """Train and score all model families we can reach. Returns one
    dict per model: name, ppl, top5, diversity, coverage, fit_s."""
    from src.evaluation import (
        compute_perplexity,
        autocomplete_accuracy,
        prediction_diversity,
        vocabulary_coverage,
    )
    from src.ngram_model import NGramModel
    from src.markov_model import MarkovChainModel

    # Shared probe layout: at each position i in test, context is
    # test[:i] (capped at the last 5 tokens for cheap prediction) and
    # the ground truth is test[i]. Works for every family because
    # every family implements predict_next(list, top_k).
    stops = list(range(5, min(len(test), probes + 5)))
    gts = [test[i] for i in stops]
    contexts = [test[max(0, i - 5):i] for i in stops]
    ref_vocab = set(test)

    results = []

    def score(name: str, model, fit_s: float):
        preds = [[w for w, _ in model.predict_next(ctx, top_k=5)] for ctx in contexts]
        ppl = compute_perplexity(model, test)
        results.append({
            "Model": name,
            "PPL": ppl,
            "Top-5 accuracy": autocomplete_accuracy(preds, gts, top_k=5),
            "Diversity": prediction_diversity(preds),
            "Coverage": vocabulary_coverage(preds, ref_vocab),
            "Fit (s)": fit_s,
        })

    t0 = time.perf_counter()
    ng = NGramModel(n=3)
    ng.fit(train)
    score("N-gram (n=3)", ng, time.perf_counter() - t0)

    t0 = time.perf_counter()
    mk = MarkovChainModel()
    mk.fit(train)
    score("Markov", mk, time.perf_counter() - t0)

    try:
        from src.neural_model import LSTMModel, HAS_TORCH
    except Exception:
        HAS_TORCH = False
    if HAS_TORCH:
        t0 = time.perf_counter()
        lstm = LSTMModel(embed_dim=32, hidden_dim=64, num_layers=1, dropout=0.0)
        lstm.fit(train, epochs=3, seq_len=8, batch_size=16, lr=1e-3)
        score("LSTM (tiny)", lstm, time.perf_counter() - t0)

        try:
            from src.transformer_model import TransformerModel
            t0 = time.perf_counter()
            tfm = TransformerModel(
                d_model=64, n_heads=4, n_layers=2, ff_dim=128, max_seq_len=64,
            )
            tfm.fit(train, epochs=3, seq_len=8, batch_size=16, lr=3e-4)
            score("Transformer (tiny)", tfm, time.perf_counter() - t0)
        except Exception:
            pass

    return results


if st.button("⚡ Run quick benchmark", type="primary", use_container_width=True):
    with st.spinner("Training and scoring all available models..."):
        rows = _run_quick_benchmark(train_tokens, test_tokens)

    import pandas as pd

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({
            "PPL": "{:.1f}",
            "Top-5 accuracy": "{:.1%}",
            "Diversity": "{:.1%}",
            "Coverage": "{:.1%}",
            "Fit (s)": "{:.2f}",
        }),
        use_container_width=True,
    )

    fig = go.Figure(go.Bar(
        x=[r["Model"] for r in rows],
        y=[r["Top-5 accuracy"] * 100 for r in rows],
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(rows)],
        text=[f"{r['Top-5 accuracy']:.1%}" for r in rows],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top-5 Accuracy by Model (higher is better ↑)",
        paper_bgcolor="#0e1117", plot_bgcolor="#262730",
        font_color="white",
        yaxis_title="Top-5 accuracy (%)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(rows) >= 3:
        best = max(rows, key=lambda r: r["Top-5 accuracy"])
        st.info(
            f"🏆 **{best['Model']}** leads on top-5 accuracy "
            f"({best['Top-5 accuracy']:.1%}). Expect n-gram to win on "
            f"this tiny corpus — the neural models need more data to "
            f"generalise."
        )


# ---------------------------------------------------------------------------
# Dynamic Benchmarks (subprocess drivers)
# ---------------------------------------------------------------------------
# Streamlit can't stream multi-minute GPU benchmarks responsively, so these
# buttons shell out to the scripts under scripts/ with a hard timeout and
# cache the last-captured stdout in session_state. A subsequent page rerun
# renders the cached result instantly instead of re-training.
st.divider()
st.header("🧪 Dynamic Benchmarks")

st.markdown(
    """
Run the project's larger benchmark scripts directly from the UI. Each
run shells out to its script with a hard timeout, captures stdout, and
caches the result in this browser session. Click the same button again
to re-run; the cache persists across page reruns within the session.

**These scripts need extra setup** — see [CLAUDE.local.md](../../CLAUDE.local.md)
for the conda env and GPU pre-reqs. Without PyTorch / transformers
installed they'll exit non-zero and print the missing dependency.
"""
)

BENCHMARKS = [
    {
        "key": "bench_real_data",
        "label": "WikiText-2 (n-gram / Markov / LSTM / Transformer)",
        "script": "scripts/bench_real_data.py",
        "timeout_s": 15 * 60,
        "caption": (
            "Trains all four model families on the full WikiText-2 train "
            "split and reports PPL / top-k / diversity / coverage on the "
            "held-out 10%. Uses GPU when available."
        ),
    },
    {
        "key": "bench_bpe",
        "label": "BPE subword (LSTM + Transformer)",
        "script": "scripts/bench_bpe.py",
        "timeout_s": 30 * 60,
        "caption": (
            "Trains both neural families on SmolLM2-tokenized WikiText-2. "
            "PPL is in subword units — not directly comparable to the "
            "word-level rows above."
        ),
    },
]

# Initialise the session cache once per browser session.
if "benchmark_runs" not in st.session_state:
    st.session_state["benchmark_runs"] = {}


def _run_benchmark(script_rel_path: str, timeout_s: int):
    """Run a benchmark script with a hard timeout. Returns a dict the UI
    can render regardless of success/failure."""
    script_path = REPO_ROOT / script_rel_path
    if not script_path.exists():
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Script not found: {script_path}",
            "duration_s": 0.0,
            "finished_at": time.time(),
            "timed_out": False,
        }
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [PYTHON_BIN, str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_s": time.perf_counter() - start,
            "finished_at": time.time(),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": (
                (exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
                + f"\n\n[timeout] killed after {timeout_s}s"
            ),
            "duration_s": time.perf_counter() - start,
            "finished_at": time.time(),
            "timed_out": True,
        }


def _render_benchmark_result(result: dict) -> None:
    """Render a captured benchmark run."""
    age_s = time.time() - result["finished_at"]
    if result["ok"]:
        st.success(
            f"✅ Finished in {result['duration_s']:.1f}s "
            f"({age_s:.0f}s ago)."
        )
    elif result["timed_out"]:
        st.error(
            f"⏱ Timed out after {result['duration_s']:.1f}s. "
            f"The script is still killable from the shell; re-run "
            f"externally if you need a longer budget."
        )
    else:
        st.error(
            f"❌ Exit code {result['returncode']} after "
            f"{result['duration_s']:.1f}s ({age_s:.0f}s ago)."
        )
    if result["stdout"]:
        with st.expander("stdout", expanded=result["ok"]):
            st.code(result["stdout"], language="text")
    if result["stderr"]:
        with st.expander("stderr", expanded=not result["ok"]):
            st.code(result["stderr"], language="text")


for bench in BENCHMARKS:
    st.subheader(bench["label"])
    st.caption(bench["caption"])
    col1, col2 = st.columns([1, 2])
    with col1:
        clicked = st.button(
            f"▶ Run (timeout {bench['timeout_s'] // 60} min)",
            key=f"run_{bench['key']}",
            use_container_width=True,
        )
    with col2:
        cached = st.session_state["benchmark_runs"].get(bench["key"])
        if cached is not None:
            age_s = time.time() - cached["finished_at"]
            st.caption(f"Last run: {age_s:.0f}s ago in this session.")
        else:
            st.caption("No run cached yet in this session.")
    if clicked:
        with st.spinner(
            f"Running {bench['script']} (up to {bench['timeout_s'] // 60} min)…"
        ):
            result = _run_benchmark(bench["script"], bench["timeout_s"])
        st.session_state["benchmark_runs"][bench["key"]] = result
        _render_benchmark_result(result)
    elif cached is not None:
        _render_benchmark_result(cached)
