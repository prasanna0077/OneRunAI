"""OneRun AI - M&A Intelligence Platform (Streamlit entry point).

PyCharm / terminal:  streamlit run app.py
"""
import streamlit as st

from onerun.config import PRODUCT_NAME, SLOGAN
from onerun.data_generation.generator import data_ready, generate_all
from onerun.modules import registry
from onerun.ui import our_bank_page, pages, summary_page
from onerun.ui import theme as th

st.set_page_config(page_title=f"{PRODUCT_NAME} | M&A Intelligence", page_icon="🔷", layout="wide")
th.apply_theme()

if not data_ready():
    with st.spinner("First run: generating synthetic datasets (about 20 seconds) ..."):
        generate_all()

# ---- navigation ---------------------------------------------------------------------------
NAV = {"🏠 Home": pages.home, "🏛 Our Bank": our_bank_page.render}
for m in registry.MVP_MODULES:
    NAV[f"{m.icon} {m.title}"] = m.render
NAV["📊 Executive Intelligence"] = registry.by_key("executive").render
NAV["📚 Dataset & Model Summary"] = summary_page.render
NAV["🏗 Platform Architecture"] = pages.architecture
NAV["🗺 Roadmap"] = pages.roadmap
for m in registry.PHASE_MODULES:
    if m.key != "executive":
        NAV[f"· {m.title}"] = m.render
NAV["🧪 Data Lab"] = pages.data_lab
NAV["🙏 Credits"] = pages.credits

with st.sidebar:
    st.markdown(th.brand_block(compact=True), unsafe_allow_html=True)
    st.caption(SLOGAN)
    choice = st.radio("Navigate", list(NAV), label_visibility="collapsed")

NAV[choice]()
