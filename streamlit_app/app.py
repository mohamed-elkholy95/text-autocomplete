"""
Text Autocomplete — Streamlit Multi-Page Application
======================================================

Main entry point that sets up the navigation between pages.

The app has three pages:
1. Overview — Project description, model comparison, and architecture
2. Autocomplete — Interactive text completion with multiple models
3. Metrics — Model evaluation, perplexity, and performance analysis
"""

import sys
from pathlib import Path

# Add project root to Python path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Text Autocomplete",
    layout="wide",
    page_icon="✍️",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark Theme
# ---------------------------------------------------------------------------
# We override Streamlit's default styling with a dark theme that looks
# professional and is easier on the eyes for data-heavy dashboards.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stMetric {
        background-color: #262730;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navigation — Multi-Page App
# ---------------------------------------------------------------------------
# Streamlit's st.navigation() creates a sidebar menu with links to each page.
# Each page is defined in the pages/ directory.
pg = st.navigation([
    st.Page("pages/1_📊_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_✍️_Autocomplete.py", title="Autocomplete", icon="✍️"),
    st.Page("pages/3_📈_Metrics.py", title="Metrics", icon="📈"),
])
pg.run()
