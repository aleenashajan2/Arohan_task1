"""AROHAN Clinical AI - Streamlit entry point.  Run:  python -m streamlit run app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="AROHAN Clinical AI", page_icon=":material/monitor_heart:", layout="wide",
                   initial_sidebar_state="expanded")

from arohan_clinical import ui, views  # noqa: E402

ui.inject_css()
logo = ROOT / "assets" / "arohan_logo.svg"
if logo.exists():
    st.logo(str(logo), size="large")

pages = [
    st.Page(views.render_overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(views.render_dataset, title="Dataset and Data Quality", icon=":material/table_chart:", url_path="dataset"),
    st.Page(views.render_supervised, title="Supervised Learning", icon=":material/model_training:", url_path="supervised"),
    st.Page(views.render_unsupervised, title="Unsupervised Learning", icon=":material/bubble_chart:", url_path="unsupervised"),
    st.Page(views.render_labs_notes_reports, title="Labs, Notes and Reports", icon=":material/biotech:", url_path="labs-notes-reports"),
]
nav = st.navigation(pages)
with st.sidebar:
    ui.sidebar_brand(views._status_text())
nav.run()
