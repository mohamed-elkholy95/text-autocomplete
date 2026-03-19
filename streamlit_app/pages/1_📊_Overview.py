import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st; import plotly.express as px
st.title("📊 Text Autocomplete — Overview")
st.markdown("N-gram and LSTM-based text prediction system.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Models"); st.markdown("- **N-gram** (1-4 grams)\n- **LSTM** (PyTorch, GPU)\n- Laplace smoothing\n- Kneser-Ney backoff")
with col2:
    st.subheader("Features"); st.markdown("- Real-time predictions\n- Multiple context window\n- Perplexity evaluation\n- REST API")
fig = px.bar(x=["Bigram", "Trigram", "4-gram", "LSTM"], y=[45, 38, 30, 22], labels={"x":"Model","y":"Top-1 Accuracy (%)"},
    title="Model Accuracy Comparison", color=[45,38,30,22], color_continuous_scale="Blues")
fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#262730", font_color="white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)
