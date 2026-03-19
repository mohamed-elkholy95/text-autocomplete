import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
st.set_page_config(page_title="Text Autocomplete", layout="wide", page_icon="✍️")
st.markdown('<style>[data-testid="stSidebar"]{background-color:#262730}.stApp{background-color:#0e1117;color:#fff}h1,h2,h3{color:#1f77b4}</style>', unsafe_allow_html=True)
pg = st.navigation([st.Page("pages/1_📊_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_✍️_Autocomplete.py", title="Autocomplete", icon="✍️"),
    st.Page("pages/3_📈_Metrics.py", title="Metrics", icon="📈")])
pg.run()
