import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from src.data_loader import tokenize, load_sample_data
from src.ngram_model import NGramModel
st.title("✍️ Autocomplete")
text = st.text_area("Type text...", "Machine learning is a subset of", height=100)
top_k = st.slider("Top K", 1, 10, 5)
if st.button("Predict", type="primary"):
    all_tokens = tokenize(load_sample_data())
    model = NGramModel(n=3); model.fit(all_tokens)
    input_tokens = tokenize(text.split())
    preds = model.predict_next(input_tokens, top_k=top_k)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predictions")
        for i, (word, prob) in enumerate(preds):
            st.markdown(f"{i+1}. **{word}** ({prob:.2%})")
    with col2:
        st.subheader("Context")
        st.code(" ".join(input_tokens[-10:]))
