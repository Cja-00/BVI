# pages/03_📈 Global trends.py
# -*- coding: utf-8 -*-
import streamlit as st
try:
    pass  # import your libs below normally
except Exception as e:
    st.sidebar.error(f"Page import error: {e}")
    raise

import altair as alt

from utils import YEAR_COL, global_vector_timeseries, global_tier_timeseries


st.title("📈 Global trends")

if "df_bvi" not in st.session_state:
    st.error("No index loaded. Go to Home and upload or use the default CSV.")
    st.stop()

df = st.session_state["df_bvi"]
has_total = "n_total" in df.columns

vec_ts = global_vector_timeseries(df)
tier_ts = global_tier_timeseries(df)

st.subheader("Notifications by risk vector (global)")
if vec_ts.empty:
    st.info("No vector columns found to build this chart.")
else:
    area_vec = alt.Chart(vec_ts).mark_area().encode(
        x=alt.X(f"{YEAR_COL}:O", title="Year"),
        y=alt.Y("value:Q", title="Estimated notifications" if has_total else "Aggregate level (index units)"),
        color=alt.Color("series:N", title="Vector"),
        tooltip=[alt.Tooltip(f"{YEAR_COL}:O"), "series:N", alt.Tooltip("value:Q", format=".0f")]
    ).properties(height=320)
    st.altair_chart(area_vec, use_container_width=True)

st.subheader("Notifications by measure tier (global)")
if tier_ts.empty:
    st.info("Bucket columns not found (`n_regular`, `n_emergency`, `n_regular_addenda`, `n_emergency_addenda`).")
else:
    area_tier = alt.Chart(tier_ts).mark_area().encode(
        x=alt.X(f"{YEAR_COL}:O", title="Year"),
        y=alt.Y("value:Q", title="Notifications"),
        color=alt.Color("series:N", title="Tier", sort=["Emergency","Emergency Addenda","Regular","Regular Addenda"]),
        tooltip=[alt.Tooltip(f"{YEAR_COL}:O"), "series:N", alt.Tooltip("value:Q", format=".0f")]
    ).properties(height=320)
    st.altair_chart(area_tier, use_container_width=True)

# Optional downloads
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    if not vec_ts.empty:
        st.download_button(
            "Download risk-vector time series (CSV)",
            data=vec_ts.rename(columns={YEAR_COL:"year"}).to_csv(index=False).encode("utf-8"),
            file_name="global_risk_vector_timeseries.csv",
            mime="text/csv"
        )
with c2:
    if not tier_ts.empty:
        st.download_button(
            "Download measure-tier time series (CSV)",
            data=tier_ts.rename(columns={YEAR_COL:"year"}).to_csv(index=False).encode("utf-8"),
            file_name="global_measure_tier_timeseries.csv",
            mime="text/csv"
        )


