"""
Metrics Page — Model Evaluation and Analysis
==============================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from collections import Counter

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
