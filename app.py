# app.py
import streamlit as st
import altair as alt
import plotly.io as pio
from utils import load_index_from_bytes


st.set_page_config(page_title="Biosecurity Vigilance Index — Dashboard", page_icon="🛡️", layout="wide")
pio.templates.default = "plotly_white"
alt.themes.enable("quartz")


st.title("🛡️ Biosecurity Vigilance Index — Dashboard")
st.caption("Use the **Pages** in the left sidebar to explore the 🌍 Country & Map, 🔗 Compare with External, and 📈 Global trends.")

with st.sidebar:
    st.header("📥 Data")
    up = st.file_uploader("Upload index_by_country_year.csv (optional)", type=["csv"])
    if up is not None:
        st.session_state["df_bvi"] = load_index_from_bytes(up.getvalue())
        st.success("Index loaded from upload.")
    elif "df_bvi" not in st.session_state:
        st.session_state["df_bvi"] = load_index_from_bytes(None)
        st.info("Loaded default CSV from repository. Upload a file to override.")

st.write("Pick a page from the left sidebar.")


