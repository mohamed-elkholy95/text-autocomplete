"""
Autocomplete Page — Interactive Text Completion Demo
======================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
from collections import Counter

from src.data_loader import tokenize, load_sample_data
from src.ngram_model import NGramModel
from src.markov_model import MarkovChainModel
from src.beam_search import BeamSearchDecoder
from src.transformer_model import HAS_TRANSFORMERS, TransformerModel
from src.config import TOP_K

st.title("✍️ Interactive Autocomplete")

# ---------------------------------------------------------------------------
# Load data and models (cached for performance)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load and cache trained models. Cached so they persist across reruns."""
    tokens = tokenize(load_sample_data())
    ngram = NGramModel(n=3)
    ngram.fit(tokens)
    markov = MarkovChainModel()
    markov.fit(tokens)
    return tokens, ngram, markov


@st.cache_resource(show_spinner="Loading SmolLM2-135M (first call only)...")
def load_transformer():
    """Lazily load the SmolLM2-135M baseline. Cached across reruns.

    Returns None when transformers isn't installed so the UI can degrade
    gracefully rather than crash.
    """
    if not HAS_TRANSFORMERS:
        return None
    model = TransformerModel()
    model.fit(tokenize(load_sample_data()))
    return model

all_tokens, ngram_model, markov_model = load_models()

# ---------------------------------------------------------------------------
# Main Autocomplete Interface
# ---------------------------------------------------------------------------
st.markdown("Type some text and see what each model predicts as the next word!")

text = st.text_area(
    "Type your text...",
    value="Machine learning is a subset of",
    height=100,
    help="The last few words will be used as context for prediction.",
)

top_k = st.slider("Number of suggestions (top-k)", min_value=1, max_value=15, value=TOP_K)

choices = ["N-gram (Trigram)", "Markov Chain"]
if HAS_TRANSFORMERS:
    choices.append("Transformer (SmolLM2-135M)")
model_choice = st.radio(
    "Choose a model",
    choices,
    horizontal=True,
    help=(
        "N-gram uses 2-word context, Markov uses 1-word context, "
        "Transformer uses up to 1024 BPE subwords from a pretrained causal LM."
    ),
)

if "Transformer" in model_choice:
    st.info(
        "ℹ️ SmolLM2 is a BPE-tokenized pretrained model. Suggestions may be "
        "**subword pieces** (e.g. `ligence`) rather than whole words — this is "
        "faithful to how the model sees text. First load downloads ~300MB."
    )

if st.button("🔮 Predict", type="primary", use_container_width=True):
    # Tokenize the input
    input_tokens = tokenize(text)

    if not input_tokens:
        st.error("Please enter some text to get predictions!")
    else:
        # Select model
        if "Transformer" in model_choice:
            model = load_transformer()
            if model is None:
                st.error("Transformer unavailable — `transformers` package not installed.")
                st.stop()
        elif "Markov" in model_choice:
            model = markov_model
        else:
            model = ngram_model

        # Get predictions
        preds = model.predict_next(input_tokens, top_k=top_k)

        # Display results in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏆 Predictions")
            if preds:
                for i, (word, prob) in enumerate(preds):
                    # Visual probability bar
                    st.markdown(
                        f"**{i+1}. {word}** — {prob:.2%}\n\n"
                        f"<progress value='{prob*100}' max='100' "
                        f"style='width:100%;height:8px;border-radius:4px;'></progress>",
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No predictions available for this context.")

        with col2:
            st.subheader("📝 Context Used")
            st.code(" ".join(input_tokens[-10:]), language=None)
            st.caption(f"Context length: {min(len(input_tokens), 10)} words")

            # Show the completed sentence
            if preds:
                st.subheader("✨ Completed Sentence")
                completed = text + " **" + preds[0][0] + "**"
                st.markdown(completed)

        # Probability distribution chart
        if preds:
            st.subheader("📊 Probability Distribution")
            fig = go.Figure(go.Bar(
                x=[w for w, _ in preds],
                y=[p * 100 for _, p in preds],
                marker_color="#1f77b4",
                text=[f"{p:.1%}" for _, p in preds],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Next Word Probabilities ({model_choice})",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#262730",
                font_color="white",
                yaxis_title="Probability (%)",
                xaxis_title="Predicted Word",
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------
st.divider()
st.header("⚖️ Model Comparison")

st.markdown("See how both models predict the next word for the SAME input:")

comparison_text = st.text_input(
    "Enter text to compare models:",
    value="The attention mechanism",
    help="Both models will predict from this input.",
)

if st.button("Compare Models", use_container_width=True):
    comp_tokens = tokenize(comparison_text)
    if not comp_tokens:
        st.error("Please enter some text!")
    else:
        ngram_preds = ngram_model.predict_next(comp_tokens, top_k=5)
        markov_preds = markov_model.predict_next(comp_tokens, top_k=5)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("N-gram Predictions")
            for i, (w, p) in enumerate(ngram_preds):
                st.markdown(f"{i+1}. **{w}** ({p:.2%})")

        with col2:
            st.subheader("Markov Predictions")
            for i, (w, p) in enumerate(markov_preds):
                st.markdown(f"{i+1}. **{w}** ({p:.2%})")

        # Show agreement
        ngram_words = {w for w, _ in ngram_preds}
        markov_words = {w for w, _ in markov_preds}
        agreement = ngram_words & markov_words

        if agreement:
            st.success(f"✅ Both models agree on: {', '.join(agreement)}")
        else:
            st.info("ℹ️ Models disagree — different approaches capture different patterns!")

# ---------------------------------------------------------------------------
# Text Generation with Markov Chain
# ---------------------------------------------------------------------------
st.divider()
st.header("🎲 Text Generation")

st.markdown("""
Generate text by walking through the Markov chain's transition graph.
This demonstrates how the model creates sequences one word at a time.
""")

gen_col1, gen_col2 = st.columns(2)

with gen_col1:
    start_word = st.text_input("Start word (optional):", value="machine")
    max_length = st.slider("Max words to generate:", 5, 50, 15)
    temperature = st.slider("Temperature (randomness):", 0.1, 3.0, 1.0, 0.1,
                            help="Lower = more predictable, Higher = more creative")

with gen_col2:
    st.markdown("""
    **Temperature Explained:**
    - **0.1-0.5**: Very focused — repeats common patterns
    - **1.0**: Natural — uses the learned distribution as-is
    - **1.5-3.0**: Creative — gives rare words more chance
    """)

if st.button("✨ Generate Text", use_container_width=True):
    if start_word.strip():
        generated = markov_model.generate_text(
            start_word=start_word.strip().lower(),
            max_length=max_length,
            temperature=temperature,
        )
    else:
        generated = markov_model.generate_text(max_length=max_length, temperature=temperature)

    if generated:
        st.subheader("Generated Text")
        st.info(generated)
    else:
        st.warning("Could not generate text. Try a different start word.")

# ---------------------------------------------------------------------------
# Beam Search Demo
# ---------------------------------------------------------------------------
st.divider()
st.header("🔍 Beam Search Demo")

st.markdown("""
**Beam search** explores multiple prediction paths simultaneously instead of
greedily picking the single best word at each step. This often produces better
multi-word predictions.
""")

beam_text = st.text_input("Input for beam search:", value="Deep learning")
beam_width = st.slider("Beam width:", 1, 10, 3, help="Number of parallel paths to explore")
beam_steps = st.slider("Steps:", 1, 8, 3, help="Number of words to predict ahead")

if st.button("🔍 Run Beam Search", use_container_width=True):
    beam_tokens = tokenize(beam_text)
    if not beam_tokens:
        st.error("Please enter some text!")
    else:
        decoder = BeamSearchDecoder(beam_width=beam_width, max_length=beam_steps)
        results = decoder.search(ngram_model, beam_tokens, steps=beam_steps)

        st.subheader("Top Beam Search Results")

        for i, result in enumerate(results[:5]):
            full_sentence = beam_text + " " + " ".join(result["tokens"])
            st.markdown(
                f"**Beam {i+1}** (score: {result['score']:.4f}, "
                f"{result['length']} words): *{full_sentence}*"
            )

        st.caption("""
        💡 Higher beam width = more thorough search but slower.
        Beam width 1 = greedy (always picks the single best word).
        """)
