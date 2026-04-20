"""
Overview Page — Project Introduction and Architecture
=======================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import numpy as np

from src.data_loader import tokenize, load_sample_data, get_corpus_stats, train_test_split
from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.evaluation import autocomplete_accuracy

st.title("📊 Text Autocomplete — Overview")
st.markdown("""
A **text autocomplete system** that predicts the next word in a sentence using
multiple language modeling approaches. Built for learning and portfolio demonstration.
""")

# ---------------------------------------------------------------------------
# Project Description
# ---------------------------------------------------------------------------
st.header("📖 What Does This Project Do?")

tab1, tab2, tab3 = st.tabs(["What is Autocomplete?", "How It Works", "Why It Matters"])

with tab1:
    st.markdown("""
    **Text autocomplete** predicts what word comes next in a sentence.
    You use it every day:
    - Your phone suggests the next word as you type
    - Google Search shows completions as you type
    - IDEs suggest code completions

    The challenge: language is incredibly diverse. The same word can be
    followed by hundreds of different words. A good model needs to:
    1. **Understand context** — "bank" (river) vs "bank" (money)
    2. **Handle unseen inputs** — new word combinations
    3. **Rank candidates** — suggest the MOST likely words first
    """)

with tab2:
    st.markdown("""
    ### The N-gram Approach (Statistical)
    1. **Count** how often each word follows every other word in training data
    2. **Normalize** counts to probabilities: P(cat | the) = count("the cat") / count("the")
    3. **Predict** the words with highest probabilities given the current context
    4. **Fall back** to shorter contexts when exact matches aren't found

    ### The Markov Chain Approach
    1. Build a **transition graph**: each word is a node, edges carry probabilities
    2. At prediction time, look up the current word's outgoing edges
    3. Return the top-k highest-probability transitions

    ### Key Difference
    N-grams consider variable context lengths (1-4 words).
    Markov chains explicitly model the transition structure as a graph.
    Both use the same underlying principle: word co-occurrence statistics.
    """)

with tab3:
    st.markdown("""
    ### Real-World Applications
    - **Search engines**: Google processes ~8.5 billion searches/day with autocomplete
    - **Mobile keyboards**: Gboard, SwiftKey use language models for suggestions
    - **Code editors**: VS Code, Copilot suggest code completions
    - **Email**: Gmail's Smart Compose writes entire sentences

    ### Learning Value
    This project teaches you:
    - **Probability**: Conditional probability, Bayes' theorem in action
    - **Smoothing**: Handling the "unseen data" problem
    - **Evaluation**: Perplexity, accuracy, and when each metric matters
    - **API design**: How to serve ML models as web services
    - **Visualization**: Communicating model behavior to stakeholders
    """)

# ---------------------------------------------------------------------------
# Models Overview
# ---------------------------------------------------------------------------
st.header("🧠 Language Models")

col1, col2 = st.columns(2)

with col1:
    st.subheader("N-gram Model")
    st.markdown("""
    - **Approach**: Statistical (count-based)
    - **Context**: 1-4 previous words
    - **Smoothing**: Backoff + frequency filtering
    - **Speed**: ⚡ Very fast (dictionary lookup)
    - **Interpretability**: ✅ Highly interpretable

    *Best for: Fast predictions, understanding WHY a word is suggested*
    """)

with col2:
    st.subheader("Markov Chain Model")
    st.markdown("""
    - **Approach**: Probabilistic transition graph
    - **Context**: Previous word only
    - **Smoothing**: Laplace (add-k)
    - **Speed**: ⚡ Very fast (matrix lookup)
    - **Interpretability**: ✅ Visualizable as a graph

    *Best for: Simple predictions, text generation, educational demos*
    """)

# ---------------------------------------------------------------------------
# Model Accuracy Comparison Chart
# ---------------------------------------------------------------------------
st.header("📊 Model Comparison")

@st.cache_data(show_spinner=False)
def _compute_top1_accuracies() -> dict:
    """Run the same 80/20 split evaluation used elsewhere and return
    real top-1 accuracies for each model so this chart isn't illustrative."""
    corpus_tokens = tokenize(load_sample_data())
    train_tokens, test_tokens = train_test_split(corpus_tokens, test_ratio=0.2, seed=42)

    models_to_eval = {
        "Unigram\n(baseline)": NGramModel(n=1, min_freq=1).fit(train_tokens),
        "Bigram": NGramModel(n=2, min_freq=1).fit(train_tokens),
        "Trigram": NGramModel(n=3, min_freq=2).fit(train_tokens),
        "4-gram": NGramModel(n=4, min_freq=2).fit(train_tokens),
        "Markov\nChain": MarkovChainModel().fit(train_tokens),
    }

    out: dict = {}
    max_probe = min(len(test_tokens) - 1, 500)
    for label, mdl in models_to_eval.items():
        preds_list, truth_list = [], []
        ctx_len = (mdl.n - 1) if isinstance(mdl, NGramModel) else 1
        start = max(ctx_len, 1)
        for i in range(start, max_probe):
            ctx = test_tokens[i - ctx_len:i]
            truth = test_tokens[i]
            preds = mdl.predict_next(ctx, top_k=1)
            preds_list.append([w for w, _ in preds])
            truth_list.append(truth)
        out[label] = round(autocomplete_accuracy(preds_list, truth_list, top_k=1) * 100, 1)
    return out

acc_map = _compute_top1_accuracies()
models = list(acc_map.keys())
accuracies = list(acc_map.values())

fig = px.bar(
    x=models,
    y=accuracies,
    labels={"x": "Model", "y": "Top-1 Accuracy (%)"},
    title="Top-1 Accuracy on Held-Out Test Split (real numbers, computed at load)",
    color=accuracies,
    color_continuous_scale="Blues",
    text=accuracies,
)
fig.update_traces(textposition="outside")
fig.update_layout(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#262730",
    font_color="white",
    showlegend=False,
    yaxis_range=[0, max(accuracies) * 1.4 if accuracies else 60],
)
fig.update_xaxes(tickfont={"size": 11})
st.plotly_chart(fig, use_container_width=True)

st.caption("""
💡 **Why does trigram beat 4-gram?** The "sparse data problem" — with 4-grams,
there are many more possible combinations but not enough training data to estimate
their probabilities reliably. Trigrams hit the sweet spot between context length
and statistical reliability.
""")

# ---------------------------------------------------------------------------
# Corpus Statistics
# ---------------------------------------------------------------------------
st.header("📈 Corpus Statistics")

tokens = tokenize(load_sample_data())
stats = get_corpus_stats(tokens)

stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    st.metric("Total Tokens", f"{stats['total_tokens']:,}")
with stat_col2:
    st.metric("Unique Words", f"{stats['unique_tokens']:,}")
with stat_col3:
    st.metric("Type-Token Ratio", f"{stats['unique_tokens']/stats['total_tokens']:.2%}")

# Word frequency distribution
from collections import Counter
word_freq = Counter(tokens).most_common(20)
words, counts = zip(*word_freq)

fig2 = px.bar(
    x=words,
    y=counts,
    labels={"x": "Word", "y": "Frequency"},
    title="Top 20 Most Common Words in Training Corpus",
    color=counts,
    color_continuous_scale="Blues",
)
fig2.update_layout(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#262730",
    font_color="white",
    showlegend=False,
    xaxis_tickangle=45,
)
st.plotly_chart(fig2, use_container_width=True)

st.caption("""
💡 **Zipf's Law** — The most common word appears ~2x as often as the 2nd most
common, which appears ~2x as often as the 3rd, etc. This power-law distribution
is found in virtually all natural language corpora and has important implications
for language modeling.
""")
