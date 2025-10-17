# pages/01_🌍 Country & Map.py
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

from utils import (
    COUNTRY_COL, YEAR_COL, CORE_COLS, TIER_COLS, VECTOR_COLS, TARGET_COLS,
    BUCKET_YEARLY, BUCKET_CUM, available_columns, percentile_rank
)

st.title("🌍 Country & Map")

# ---- get data ----
if "df_bvi" not in st.session_state:
    st.error("No index loaded. Go to Home and upload or use the default CSV.")
    st.stop()

df = st.session_state["df_bvi"]
if df.empty or YEAR_COL not in df.columns:
    st.error("Index file is empty or missing a 'year' column.")
    st.stop()

min_year = int(df[YEAR_COL].dropna().min())
max_year = int(df[YEAR_COL].dropna().max())

# ---- metric menus (map) ----
yearly_internal = [
    ("BVI (vigilance_index)", "vigilance_index"),
    ("BVI (3y MA)", "vigilance_index_3y_ma"),
    ("Intensity (scaled)", "intensity_scaled"),
    ("Ban Flow (scaled)", "banflow_scaled"),
    ("Breadth (0–100)", "breadth"),
    ("Tier 1 proportion", "tier1_prop"),
    ("Tier 2 proportion", "tier2_prop"),
    ("Tier 3 proportion", "tier3_prop"),
    ("SPS engagement (Tier3 count)", "sps_engagement"),
    ("Notices — total", "n_total"),
    ("Notices — Regular", "n_regular"),
    ("Notices — Emergency", "n_emergency"),
    ("Notices — Regular Addenda", "n_regular_addenda"),
    ("Notices — Emergency Addenda", "n_emergency_addenda"),
    ("Notices per 1M", "n_total_per_1m"),
    ("Notices per 10M", "n_total_per_10m"),
]
yearly_internal = [(l,c) for (l,c) in yearly_internal if c in df.columns]

cumulative_internal = [
    ("Cumulative notices — total", "cum_n_total"),
    ("Cumulative notices — Regular", "cum_n_regular"),
    ("Cumulative notices — Emergency", "cum_n_emergency"),
    ("Cumulative notices — Regular Addenda", "cum_n_regular_addenda"),
    ("Cumulative notices — Emergency Addenda", "cum_n_emergency_addenda"),
    ("Cumulative notices (alias)", "cum_total_notices"),
    ("Cumulative per 1M", "cum_total_per_1m"),
    ("Cumulative per 10M", "cum_total_per_10m"),
]
cumulative_internal = [(l,c) for (l,c) in cumulative_internal if c in df.columns]

with st.sidebar:
    st.markdown("---")
    map_year = st.slider("Map year", min_value=min_year, max_value=max_year, value=max_year)
    mode = st.radio("Map metric type", ["Yearly", "Cumulative"], horizontal=True)
    options = yearly_internal if mode == "Yearly" else (cumulative_internal or yearly_internal)
    color_label, color_col = st.selectbox(
        "Color by",
        options,
        index=0,
        format_func=lambda t: t[0] if isinstance(t, tuple) else str(t),
    )

# ---- build year slice for map ----
dy0 = df[df[YEAR_COL] == map_year].copy()
if color_col not in dy0.columns:
    st.error(f"Selected metric column `{color_col}` not found for {map_year}.")
    st.stop()

dy0["__color__"] = pd.to_numeric(dy0[color_col], errors="coerce")
dy = dy0.dropna(subset=["iso3","__color__"]).copy()
dy = dy[(dy["iso3"]!="EUU") & (dy["iso3"].str.len()==3)]

st.info(
    f"Map-year diagnostic — rows in year: {int((df[YEAR_COL]==map_year).sum())} | "
    f"usable on map: {len(dy)} | unique {color_label} values: {dy['__color__'].nunique()}"
)

if dy.empty:
    st.error("No usable data for this year/metric.")
    st.stop()

# ---- choropleth (clickable) ----
fig = go.Figure(
    data=go.Choropleth(
        locations=dy["iso3"].tolist(),
        z=dy["__color__"].astype(float).tolist(),
        locationmode="ISO-3",
        colorscale="Viridis",
        colorbar_title=color_label,
        marker_line_color="white",
        marker_line_width=0.3,
        hovertext=dy[COUNTRY_COL],
        hovertemplate=f"<b>%{{hovertext}}</b><br>{color_label}: %{{z:.2f}}<extra></extra>",
    )
)
fig.update_layout(
    margin=dict(l=0,r=0,t=8,b=0),
    geo=dict(showframe=False, showcoastlines=True, projection_type="equirectangular"),
)
clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False,
                        override_height=520, override_width="100%")

# ---- resolve selected ISO3 (fallback selector) ----
selected_iso = None
if clicked:
    idx = clicked[0].get("pointNumber", None)
    if isinstance(idx, int) and 0 <= idx < len(dy):
        selected_iso = str(dy.iloc[idx]["iso3"]).upper()

if not selected_iso or selected_iso not in set(dy["iso3"]):
    st.warning("Click didn’t register (or ocean click). Choose a country below.")
    picked = st.selectbox("Country", sorted(dy[COUNTRY_COL].dropna().unique().tolist()))
    selected_iso = str(dy.loc[dy[COUNTRY_COL]==picked, "iso3"].iloc[0]).upper()

sel = dy[dy["iso3"]==selected_iso].head(1)
country = sel[COUNTRY_COL].iloc[0]
base_row = df[(df["iso3"]==selected_iso) & (df[YEAR_COL]==map_year)].head(1)

def safe_fmt(col: str, fmt: str="{:.1f}") -> str:
    try:
        v = float(base_row[col].iloc[0])
        return "—" if np.isnan(v) else fmt.format(v)
    except Exception:
        return "—"

# ---- scorecards ----
st.markdown(f"## 🇺🇳 {country} — {map_year}")
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("BVI (0–100)", safe_fmt("vigilance_index"))
with k2:
    pctl = percentile_rank(dy["__color__"], float(sel["__color__"].iloc[0]))
    st.metric(f"Percentile — {color_label}", f"{pctl:.1f}%")
with k3: st.metric("Intensity (scaled)", safe_fmt("intensity_scaled"))
with k4: st.metric("Ban Flow (scaled)", safe_fmt("banflow_scaled"))
st.metric("Breadth (0–100)", safe_fmt("breadth"))

st.markdown("---")

# ---- Sub-indicators ----
st.header("📊 Sub-indicators")
hist = df[df["iso3"] == selected_iso].sort_values(YEAR_COL).copy()

# SPS Engagement
st.subheader("📨 SPS Engagement (Tier 3 notifications)")
if "sps_engagement" not in hist.columns:
    st.info("No 'sps_engagement' column in the index file.")
else:
    eng = hist[[YEAR_COL, "sps_engagement"]].copy()
    if "sps_engagement_3y_ma" in hist.columns:
        eng["sps_engagement_3y_ma"] = pd.to_numeric(hist["sps_engagement_3y_ma"], errors="coerce")

    if eng["sps_engagement"].notna().any():
        ld = eng.melt(id_vars=[YEAR_COL], var_name="Series", value_name="Count")
        ld["Series"] = ld["Series"].map({
            "sps_engagement": "Engagement (count)",
            "sps_engagement_3y_ma": "Engagement (3y MA)"
        }).fillna(ld["Series"])

        line_eng = alt.Chart(ld).mark_line(point=True).encode(
            x=alt.X(f"{YEAR_COL}:O", title="Year"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Series:N", title="", sort=[s for s in ["Engagement (count)", "Engagement (3y MA)"] if s in ld["Series"].unique()])
        ).properties(height=260)
        st.altair_chart(line_eng, use_container_width=True)
    else:
        st.info("No SPS engagement values for this country.")
st.markdown("---")

# Risk Vector Concerns
st.subheader("🧬 Risk Vector Concerns (intensity-weighted)")
VECTOR_COLS_LOCAL = [c for c in ["contaminants", "pathogens", "pests", "biotechnology", "toxins"] if c in hist.columns]
if VECTOR_COLS_LOCAL and any(hist[v].notna().any() for v in VECTOR_COLS_LOCAL):
    lv_long = hist[[YEAR_COL] + VECTOR_COLS_LOCAL].melt(id_vars=[YEAR_COL], var_name="Vector", value_name="Level").dropna(subset=["Level"])
    level_chart = alt.Chart(lv_long).mark_line(point=True).encode(
        x=alt.X(f"{YEAR_COL}:O", title="Year"),
        y=alt.Y("Level:Q", title="Intensity-weighted level"),
        color=alt.Color("Vector:N", title="", sort=VECTOR_COLS_LOCAL)
    ).properties(height=280)
    st.altair_chart(level_chart, use_container_width=True)

    st.caption(f"Selected-year snapshot ({map_year})")
    snap = hist[hist[YEAR_COL] == map_year][VECTOR_COLS_LOCAL].head(1)
    if not snap.empty:
        total = float(snap[VECTOR_COLS_LOCAL].sum(axis=1).iloc[0])
        snap_shares = pd.DataFrame({
            "Vector": VECTOR_COLS_LOCAL,
            "Share": [(float(snap[v].iloc[0]) / total if total > 0 else 0.0) for v in VECTOR_COLS_LOCAL]
        }).sort_values("Share", ascending=False)
        st.bar_chart(snap_shares.set_index("Vector"))
    else:
        st.info("No rows for the selected year for this country.")
else:
    st.info("No vector columns present (contaminants/pathogens/pests/biotechnology/toxins).")

st.markdown("---")

# Risk Target Concerns
st.subheader("🎯 Risk Target Concerns (intensity-weighted)")
TARGET_COLS_LOCAL = [c for c in ["food_safety", "plant_health", "animal_health", "human_health", "environment"] if c in hist.columns]
TARGET_LABELS = {
    "food_safety": "Food safety",
    "plant_health": "Plant health",
    "animal_health": "Animal health",
    "human_health": "Human health",
    "environment": "Environment",
}
if TARGET_COLS_LOCAL and any(hist[t].notna().any() for t in TARGET_COLS_LOCAL):
    rt_long = hist[[YEAR_COL] + TARGET_COLS_LOCAL].melt(id_vars=[YEAR_COL], var_name="Target", value_name="Level").dropna(subset=["Level"])
    rt_long["Target"] = rt_long["Target"].map(TARGET_LABELS).fillna(rt_long["Target"])
    target_order = [TARGET_LABELS.get(t, t) for t in TARGET_COLS_LOCAL]
    target_level_chart = alt.Chart(rt_long).mark_line(point=True).encode(
        x=alt.X(f"{YEAR_COL}:O", title="Year"),
        y=alt.Y("Level:Q", title="Intensity-weighted level"),
        color=alt.Color("Target:N", title="", sort=target_order)
    ).properties(height=280)
    st.altair_chart(target_level_chart, use_container_width=True)
else:
    st.info("No risk target columns present (food_safety/plant_health/animal_health/human_health/environment).")

st.markdown("---")

# Vigilance trend
hist2 = df[df["iso3"]==selected_iso].sort_values(YEAR_COL)
if not hist2.empty:
    st.subheader("📈 Vigilance trend")
    ld = hist2[[YEAR_COL, "vigilance_index", "vigilance_index_3y_ma"]].melt(id_vars=[YEAR_COL], var_name="Series", value_name="Score")
    line = alt.Chart(ld).mark_line(point=True).encode(
        x=alt.X(f"{YEAR_COL}:O", title="Year"),
        y=alt.Y("Score:Q", title="BVI (0–100)"),
        color=alt.Color("Series:N", legend=alt.Legend(title=""), scale=alt.Scale(
            domain=["vigilance_index","vigilance_index_3y_ma"], range=["#1f77b4","#1f77b4"]
        )),
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(
            domain=["vigilance_index","vigilance_index_3y_ma"], range=[[0],[5,5]]
        )),
        tooltip=[alt.Tooltip(f"{YEAR_COL}:O"), alt.Tooltip("Score:Q", format=".1f")]
    ).properties(height=260)
    st.altair_chart(line, use_container_width=True)

# Four-bucket stacked bar (year-on-year)
st.subheader("🧱 Notification composition — four buckets (year-on-year)")
bucket_cols_present = available_columns(df, BUCKET_YEARLY)
if len(bucket_cols_present) >= 5:
    s = df[df["iso3"]==selected_iso][[YEAR_COL]+bucket_cols_present].copy()
    value_cols = [c for c in ["n_regular","n_emergency","n_regular_addenda","n_emergency_addenda"] if c in s.columns]
    long = s.melt(id_vars=[YEAR_COL], value_vars=value_cols, var_name="bucket", value_name="count")
    order = ["Emergency","Emergency Addenda","Regular","Regular Addenda"]
    name_map = {"n_regular": "Regular","n_emergency": "Emergency","n_regular_addenda": "Regular Addenda","n_emergency_addenda": "Emergency Addenda"}
    long["bucket"] = long["bucket"].map(name_map)
    long["bucket"] = pd.Categorical(long["bucket"], categories=order, ordered=True)
    chart = alt.Chart(long).mark_bar().encode(
        x=alt.X(f"{YEAR_COL}:O", title="Year"),
        y=alt.Y("count:Q", title="Notifications"),
        color=alt.Color("bucket:N", title="Type", sort=order, scale=alt.Scale(scheme="tableau10")),
        tooltip=[alt.Tooltip(f"{YEAR_COL}:O"), "bucket:N", alt.Tooltip("count:Q", format=".0f")]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Four-bucket yearly columns not found in the index file. Rebuild the index with bucket counts.")

# Optional: cumulative quick view
cum_cols_present = available_columns(df, BUCKET_CUM)
if len(cum_cols_present) >= 1:
    st.subheader("📦 Cumulative notices (quick view)")
    sc = df[df["iso3"]==selected_iso][[YEAR_COL]+cum_cols_present].copy()
    picks = [c for c in ["cum_n_total","cum_total_per_1m"] if c in sc.columns]
    if picks:
        m = sc.melt(id_vars=[YEAR_COL], value_vars=picks, var_name="Series", value_name="Value")
        cum_line = alt.Chart(m).mark_line(point=True).encode(
            x=alt.X(f"{YEAR_COL}:O", title="Year"),
            y=alt.Y("Value:Q"),
            color=alt.Color("Series:N", title=""),
            tooltip=[alt.Tooltip(f"{YEAR_COL}:O"), "Series:N", alt.Tooltip("Value:Q", format=".1f")]
        ).properties(height=220)
        st.altair_chart(cum_line, use_container_width=True)

# Download current map slice
with st.expander("⬇️ Download current map slice"):
    dl = dy[[COUNTRY_COL, "iso3", color_col]].rename(columns={color_col: color_label})
    st.dataframe(dl.sort_values(color_label, ascending=False), use_container_width=True)
    st.download_button(
        "Download CSV",
        data=dl.to_csv(index=False).encode("utf-8"),
        file_name=f"map_slice_{map_year}_{color_label.replace(' ','_')}.csv",
        mime="text/csv"
    )

with st.expander("ℹ️ Notes"):
    st.markdown("""
- The **map** uses internal metrics (choose yearly or cumulative).
- Use the **Compare with External** page for scatter comparisons.
- If an external file lacks a `year` column, the app tries to infer it or uses the provided default on that page.
- Toggle **log scales** for GDP-like skews; color by **region** for quick structure.
""")
