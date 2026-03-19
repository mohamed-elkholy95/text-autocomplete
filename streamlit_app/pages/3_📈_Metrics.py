import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st; import plotly.graph_objects as go
st.title("📈 Model Metrics")
from src.data_loader import tokenize, load_sample_data
from src.ngram_model import NGramModel
all_tokens = tokenize(load_sample_data())
train = all_tokens[:int(len(all_tokens)*0.8)]
test = all_tokens[int(len(all_tokens)*0.8):]
model = NGramModel(n=3); model.fit(train)
ppl = model.perplexity(test)
st.metric("Trigram Perplexity", f"{ppl:.1f}")
preds = model.predict_next(train[-5:], top_k=5)
fig = go.Figure(go.Bar(x=[w for w,_ in preds], y=[p for _,p in preds], marker_color="#1f77b4"))
fig.update_layout(title="Next Word Probabilities", paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white")
st.plotly_chart(fig, use_container_width=True)
