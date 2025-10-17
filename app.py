# app.py
import io
import re
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pycountry
import plotly.io as pio
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# =========================================================
# Page / theme
# =========================================================
st.set_page_config(page_title="Biosecurity Vigilance Index — Dashboard", page_icon="🛡️", layout="wide")
pio.templates.default = "plotly_white"
alt.themes.enable("quartz")

# =========================================================
# Constants
# =========================================================
CSV_INDEX = "index_by_country_year.csv"  # output from your builder
COUNTRY_COL = "notifying_member"
YEAR_COL = "year"

CORE_COLS = ["vigilance_index", "vigilance_index_3y_ma", "intensity_scaled", "banflow_scaled", "breadth"]
TIER_COLS = ["total_notices", "tier1_prop", "tier2_prop", "tier3_prop", "sps_engagement"]
VECTOR_COLS = ["contaminants", "pathogens", "pests", "biotechnology"]
TARGET_COLS = ["food_safety", "plant_health", "animal_health", "human_health", "environment"]

# Four-bucket columns (yearly + cumulative + per-capita if present)
BUCKET_YEARLY = ["n_regular", "n_emergency", "n_regular_addenda", "n_emergency_addenda", "n_total"]
BUCKET_CUM = ["cum_n_regular", "cum_n_emergency", "cum_n_regular_addenda", "cum_n_emergency_addenda", "cum_n_total", "cum_total_notices"]
BUCKET_PC = ["n_total_per_1m", "n_total_per_10m", "cum_total_per_1m", "cum_total_per_10m"]

# Auto-load these external CSVs if present in the working directory
DEFAULT_EXTERNAL = [
    ("GDP per capita (2025)", "GDP_per_Capita_2025.csv", 2025),
    ("Democracy Index (2024)", "Democracy_index_2024.csv", 2024),
    ("Development Index (2023)", "development_index_2023.csv", 2023),
    ("WWF Physical Risk (2024)", "wwf_physical_risk.csv", 2024),
    ("Agriculture timeseries", "agriculture_timeseries_countries.csv", None),
]

REGION_OVERRIDES = {
    # Oceania
    "AUS":"Oceania","NZL":"Oceania","FJI":"Oceania","PNG":"Oceania","SLB":"Oceania","VUT":"Oceania",
    "WSM":"Oceania","TON":"Oceania","KIR":"Oceania","TUV":"Oceania","NRU":"Oceania","MHL":"Oceania",
    "PLW":"Oceania",

    # North America (incl. Central America & Caribbean split below)
    "USA":"North America","CAN":"North America","MEX":"North America","GRL":"North America","BMU":"North America",

    # Central America
    "GTM":"Central America","SLV":"Central America","HND":"Central America","NIC":"Central America",
    "CRI":"Central America","PAN":"Central America","BLZ":"Central America",

    # Caribbean
    "CUB":"Caribbean","DOM":"Caribbean","HTI":"Caribbean","JAM":"Caribbean","BHS":"Caribbean","BRB":"Caribbean",
    "TTO":"Caribbean","DMA":"Caribbean","ATG":"Caribbean","KNA":"Caribbean","LCA":"Caribbean","VCT":"Caribbean",
    "GRD":"Caribbean","BRB":"Caribbean","GUY":"Caribbean",  # Guyana often grouped with Caribbean community
    "SUR":"Caribbean",

    # South America
    "BRA":"South America","ARG":"South America","CHL":"South America","COL":"South America","PER":"South America",
    "ECU":"South America","BOL":"South America","PRY":"South America","URY":"South America","VEN":"South America",

    # Europe (incl. non-EU, Balkans, Caucasus in Europe bucket unless noted)
    "GBR":"Europe","IRL":"Europe","ISL":"Europe","FRA":"Europe","DEU":"Europe","ITA":"Europe","ESP":"Europe",
    "PRT":"Europe","NLD":"Europe","BEL":"Europe","LUX":"Europe","DNK":"Europe","SWE":"Europe","FIN":"Europe",
    "NOR":"Europe","CHE":"Europe","AUT":"Europe","POL":"Europe","CZE":"Europe","SVK":"Europe","HUN":"Europe",
    "SVN":"Europe","HRV":"Europe","ROU":"Europe","BGR":"Europe","GRC":"Europe","MLT":"Europe","CYP":"Europe",
    "EST":"Europe","LVA":"Europe","LTU":"Europe","ALB":"Europe","MKD":"Europe","MNE":"Europe","SRB":"Europe",
    "BIH":"Europe","MDA":"Europe","UKR":"Europe","BLR":"Europe","AND":"Europe","MCO":"Europe","SMR":"Europe",
    "VAT":"Europe","LIE":"Europe","KOS":"Europe",  # Kosovo (custom code sometimes used)

    # Russia / Transcontinental notes
    "RUS":"Europe/Asia","TUR":"Europe/Asia","AZE":"Europe/Asia","GEO":"Europe/Asia","ARM":"Europe/Asia",
    "KAZ":"Europe/Asia",

    # Africa (by subregion is optional; we keep a single "Africa" bucket for color clarity)
    "ZAF":"Africa","EGY":"Africa","NGA":"Africa","ETH":"Africa","KEN":"Africa","MAR":"Africa","DZA":"Africa",
    "TUN":"Africa","LBY":"Africa","SDN":"Africa","SSD":"Africa","TCD":"Africa","NER":"Africa","MLI":"Africa",
    "BFA":"Africa","SEN":"Africa","GMB":"Africa","GIN":"Africa","GNB":"Africa","SLE":"Africa","LBR":"Africa",
    "CIV":"Africa","GHA":"Africa","TGO":"Africa","BEN":"Africa","CMR":"Africa","GAB":"Africa","GNQ":"Africa",
    "COG":"Africa","COD":"Africa","CAF":"Africa","AGO":"Africa","ZMB":"Africa","MOZ":"Africa","MWI":"Africa",
    "TZA":"Africa","UGA":"Africa","RWA":"Africa","BDI":"Africa","SOM":"Africa","ETH":"Africa","DJI":"Africa",
    "ERI":"Africa","MDG":"Africa","COM":"Africa","SYC":"Africa","MUS":"Africa","CPV":"Africa","STP":"Africa",
    "ZWE":"Africa","BWA":"Africa","NAM":"Africa","LSO":"Africa","SWZ":"Africa","SOM":"Africa","GNB":"Africa",
    "SLE":"Africa","MRT":"Africa",

    # Asia (East, South, Southeast grouped into "Asia")
    "CHN":"Asia","JPN":"Asia","KOR":"Asia","PRK":"Asia","MNG":"Asia",
    "IND":"Asia","PAK":"Asia","BGD":"Asia","LKA":"Asia","NPL":"Asia","BTN":"Asia","MDV":"Asia",
    "IDN":"Asia","VNM":"Asia","THA":"Asia","MMR":"Asia","KHM":"Asia","LAO":"Asia","MYS":"Asia","SGP":"Asia",
    "PHL":"Asia","BRN":"Asia","TLS":"Asia",

    # Middle East (West Asia) — giving its own bucket for analysis/visual clarity
    "SAU":"Middle East","ARE":"Middle East","QAT":"Middle East","BHR":"Middle East","KWT":"Middle East","OMN":"Middle East",
    "IRN":"Middle East","IRQ":"Middle East","SYR":"Middle East","JOR":"Middle East","LBN":"Middle East","ISR":"Middle East",
    "PSE":"Middle East","YEM":"Middle East",

    # Central Asia
    "KAZ":"Central Asia","KGZ":"Central Asia","TJK":"Central Asia","TKM":"Central Asia","UZB":"Central Asia",

    # Caucasus (you can keep these in Europe/Asia above; here we give optional separate bucket)
    # "ARM":"Caucasus","AZE":"Caucasus","GEO":"Caucasus",

    # Additional Americas details
    "CAN":"North America","USA":"North America","MEX":"North America",
    "BHS":"Caribbean","BRB":"Caribbean","ATG":"Caribbean","KNA":"Caribbean","LCA":"Caribbean","VCT":"Caribbean",
    "TTO":"Caribbean","BRB":"Caribbean","CUB":"Caribbean","DOM":"Caribbean","HTI":"Caribbean","JAM":"Caribbean",
    "PRY":"South America","URY":"South America","GUY":"South America","SUR":"South America","BOL":"South America",
    "ECU":"South America","PER":"South America","COL":"South America","VEN":"South America","ARG":"South America","BRA":"South America","CHL":"South America",

    # Microstates / special cases
    "HKG":"Asia","MAC":"Asia","TWN":"Asia",  # for dashboard use; political status varies
    "GIB":"Europe","GGY":"Europe","JEY":"Europe","IMN":"Europe",
}


# =========================================================
# Helpers
# =========================================================
def to_iso3_series(names: pd.Series) -> pd.Series:
    overrides = {
        "United States":"USA","United States of America":"USA",
        "Russian Federation":"RUS","Viet Nam":"VNM",
        "Iran, Islamic Republic of":"IRN","Korea, Republic of":"KOR",
        "Côte d’Ivoire":"CIV","Cote d'Ivoire":"CIV",
        "Congo":"COG","Congo, The Democratic Republic of the":"COD",
        "Tanzania, United Republic of":"TZA","Lao People's Democratic Republic":"LAO",
        "Bolivia, Plurinational State of":"BOL","Venezuela, Bolivarian Republic of":"VEN",
        "Micronesia, Federated States of":"FSM","North Macedonia":"MKD",
        "Cabo Verde":"CPV","Eswatini":"SWZ","Myanmar":"MMR",
        "Macao":"MAC","Macau, China":"MAC","Hong Kong, China":"HKG",
        "Taiwan, Province of China":"TWN","United Kingdom":"GBR","UK":"GBR",
        "European Union":"EUU","EU":"EUU"
    }
    out=[]
    for n in names.fillna(""):
        if n in overrides: out.append(overrides[n]); continue
        try:
            out.append(pycountry.countries.lookup(n).alpha_3)  # type: ignore
        except Exception:
            try:
                nn=n.replace("’","'").replace("–","-")
                out.append(pycountry.countries.lookup(nn).alpha_3)  # type: ignore
            except Exception:
                out.append(None)
    return pd.Series(out, index=names.index)

def percentile_rank(series: pd.Series, value: float) -> float:
    s=pd.to_numeric(series, errors="coerce").dropna()
    return float((s<=value).mean()*100) if (not s.empty and pd.notna(value)) else np.nan

def add_region(df_iso: pd.DataFrame) -> pd.DataFrame:
    df_iso = df_iso.copy()
    df_iso["region"] = df_iso["iso3"].map(REGION_OVERRIDES).fillna("Other/Unknown")
    return df_iso

def available_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def infer_country_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if re.search(r"(country|economy|nation|state|member)", c, re.I)]
    return candidates[0] if candidates else df.columns[0]

def tidy_external_csv(raw: pd.DataFrame, indicator_name: str, default_year: int | None = None) -> pd.DataFrame:
    """
    Returns tidy long: [country, iso3, year, indicator, value]
    Accepts wide (year columns) or long (country, year, value[|score|index]) shapes.
    """
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    ctry_col = infer_country_col(df)
    df.rename(columns={ctry_col: "country"}, inplace=True)

    year_col = None
    if any(c.lower()=="year" for c in df.columns):
        year_col = [c for c in df.columns if c.lower()=="year"][0]

    year_like = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    val_candidates = [c for c in df.columns if c.lower() in ["value","score","index"]]

    if year_col and val_candidates:
        valc = val_candidates[0]
        longdf = df[["country", year_col, valc]].rename(columns={year_col:"year", valc:"value"})
    elif year_like:
        longdf = df.melt(id_vars=["country"], value_vars=year_like, var_name="year", value_name="value")
        longdf["year"] = pd.to_numeric(longdf["year"], errors="coerce").astype("Int64")
    else:
        # fallback: take last numeric column as value, assign default_year
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            last = df.columns[-1]
            df[last] = pd.to_numeric(df[last], errors="coerce")
            num_cols = [last]
        valc = num_cols[-1]
        longdf = df[["country", valc]].rename(columns={valc:"value"})
        longdf["year"] = default_year

    longdf["iso3"] = to_iso3_series(longdf["country"])
    longdf["indicator"] = indicator_name
    longdf["value"] = pd.to_numeric(longdf["value"], errors="coerce")
    longdf = longdf.dropna(subset=["iso3","value"]).copy()
    if default_year and longdf["year"].isna().all():
        longdf["year"] = default_year
    return longdf[["country","iso3","year","indicator","value"]]

# =========================================================
# Sidebar — load core index
# =========================================================
st.sidebar.header("📥 Data")
up_index = st.sidebar.file_uploader("Upload index_by_country_year.csv (optional)", type=["csv"])

@st.cache_data
def load_index(upload: bytes | None) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(upload)) if upload else pd.read_csv(CSV_INDEX)
    df.columns = [c.strip() for c in df.columns]
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
    for c in set(CORE_COLS + TIER_COLS + VECTOR_COLS + TARGET_COLS + BUCKET_YEARLY + BUCKET_CUM + BUCKET_PC + ["ban_flow","population"]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["iso3"] = to_iso3_series(df[COUNTRY_COL]).astype(str).str.upper()
    df = df[df["iso3"].str.len()==3].copy()
    df = add_region(df)
    return df

df = load_index(up_index.getvalue() if up_index else None)
if df.empty or YEAR_COL not in df.columns:
    st.error("Index file is empty or missing a 'year' column.")
    st.stop()

min_year = int(df[YEAR_COL].dropna().min())
max_year = int(df[YEAR_COL].dropna().max())

# =========================================================
# Sidebar — load external CSVs
# =========================================================
st.sidebar.subheader("➕ External datasets")
ext_frames: list[pd.DataFrame] = []

# Auto-load common names if available
for label, fname, year_hint in DEFAULT_EXTERNAL:
    try:
        raw = pd.read_csv(fname)
        ext_frames.append(tidy_external_csv(raw, label, default_year=year_hint))
    except Exception:
        pass  # fine if missing

# Upload extra CSVs
more = st.sidebar.file_uploader("Upload more CSVs (multi)", type=["csv"], accept_multiple_files=True)
if more:
    for f in more:
        label = re.sub(r"\.csv$", "", f.name)
        year_guess = None
        m = re.search(r"(19|20)\d{2}", f.name)
        if m: year_guess = int(m.group(0))
        raw = pd.read_csv(f)
        ext_frames.append(tidy_external_csv(raw, label, default_year=year_guess))

external_long = pd.concat(ext_frames, ignore_index=True) if ext_frames else pd.DataFrame(columns=["country","iso3","year","indicator","value"])
external_long["year"] = pd.to_numeric(external_long["year"], errors="coerce").astype("Int64")

# =========================================================
# Metric menus (yearly & cumulative for MAP)
# =========================================================
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

# =========================================================
# Controls
# =========================================================
st.title("🛡️ Biosecurity Vigilance Index — Dashboard")
st.caption("Map: yearly or cumulative internal metrics. Country panel: four-bucket breakdown. External comparisons and global trends live in their own tabs.")

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

# =========================================================
# Tabs
# =========================================================
tab_country, tab_compare, tab_global = st.tabs(["🌍 Country & Map", "🔗 Compare with External", "📈 Global trends"])

# =========================================================
# TAB 1 — Country & Map
# =========================================================
with tab_country:
    # Build year slice for the map
    dy0 = df[df[YEAR_COL]==map_year].copy()
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

    # Choropleth
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

    # Resolve selected ISO3 (fallback to selector)
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

    # Scorecards
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

    # Sub-indicators — Engagement & Risk Vector Concerns
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

    # Risk Vector Concerns (intensity-weighted)
    st.subheader("🧬 Risk Vector Concerns (intensity-weighted)")
    VECTOR_COLS_LOCAL = [c for c in ["contaminants", "pathogens", "pests", "biotechnology", "toxins"] if c in hist.columns]
    if VECTOR_COLS_LOCAL and any(hist[v].notna().any() for v in VECTOR_COLS_LOCAL):
        # Levels time series
        lv_long = hist[[YEAR_COL] + VECTOR_COLS_LOCAL].melt(id_vars=[YEAR_COL], var_name="Vector", value_name="Level").dropna(subset=["Level"])
        level_chart = alt.Chart(lv_long).mark_line(point=True).encode(
            x=alt.X(f"{YEAR_COL}:O", title="Year"),
            y=alt.Y("Level:Q", title="Intensity-weighted level"),
            color=alt.Color("Vector:N", title="", sort=VECTOR_COLS_LOCAL)
        ).properties(height=280)
        st.altair_chart(level_chart, use_container_width=True)

        # Selected-year snapshot
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

    # --- Risk Target Concerns ---
st.subheader("🎯 Risk Target Concerns (intensity-weighted)")

# Use whatever target columns are actually present
TARGET_COLS_LOCAL = [c for c in ["food_safety", "plant_health", "animal_health", "human_health", "environment"] if c in hist.columns]

# Pretty labels for legend/tooltips
TARGET_LABELS = {
    "food_safety": "Food safety",
    "plant_health": "Plant health",
    "animal_health": "Animal health",
    "human_health": "Human health",
    "environment": "Environment",
}

if TARGET_COLS_LOCAL and any(hist[t].notna().any() for t in TARGET_COLS_LOCAL):
    # (A) Levels time series
    rt_long = hist[[YEAR_COL] + TARGET_COLS_LOCAL].melt(
        id_vars=[YEAR_COL], var_name="Target", value_name="Level"
    ).dropna(subset=["Level"])
    rt_long["Target"] = rt_long["Target"].map(TARGET_LABELS).fillna(rt_long["Target"])

    target_order = [TARGET_LABELS.get(t, t) for t in TARGET_COLS_LOCAL]

    target_level_chart = alt.Chart(rt_long).mark_line(point=True).encode(
        x=alt.X(f"{YEAR_COL}:O", title="Year"),
        y=alt.Y("Level:Q", title="Intensity-weighted level"),
        color=alt.Color("Target:N", title="", sort=target_order)
    ).properties(height=280)
    st.altair_chart(target_level_chart, use_container_width=True)

    st.markdown("---")

    # Vigilance trend
    hist = df[df["iso3"]==selected_iso].sort_values(YEAR_COL)
    if not hist.empty:
        st.subheader("📈 Vigilance trend")
        ld = hist[[YEAR_COL, "vigilance_index", "vigilance_index_3y_ma"]].melt(id_vars=[YEAR_COL], var_name="Series", value_name="Score")
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
        name_map = {
            "n_regular": "Regular",
            "n_emergency": "Emergency",
            "n_regular_addenda": "Regular Addenda",
            "n_emergency_addenda": "Emergency Addenda",
        }
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

    # Download current map slice (keep inside country tab)
    with st.expander("⬇️ Download current map slice"):
        dl = dy[[COUNTRY_COL, "iso3", color_col]].rename(columns={color_col: color_label})
        st.dataframe(dl.sort_values(color_label, ascending=False), use_container_width=True)
        st.download_button(
            "Download CSV",
            data=dl.to_csv(index=False).encode("utf-8"),
            file_name=f"map_slice_{map_year}_{re.sub(r'[^A-Za-z0-9]+','_',color_label)}.csv",
            mime="text/csv"
        )

    with st.expander("ℹ️ Notes"):
        st.markdown("""
- The **map** uses internal metrics (choose yearly or cumulative).
- Load external CSVs (sidebar) — wide or long — to enable the **external comparison scatter** (see tab).
- If an external file lacks a `year` column, the app tries to infer it from the filename or uses the provided default.
- Toggle **log scales** for GDP-like skews; color by **region** for quick structure.
""")


# =========================================================
# TAB 2 — Compare with External (moved here)
# =========================================================
with tab_compare:
    st.header("🔗 Compare BVI (or any internal metric) with external data")

    # Build internal metric menu (numeric columns in df)
    internal_numeric_cols = sorted([
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()
    ])
    defX = "vigilance_index" if "vigilance_index" in internal_numeric_cols else internal_numeric_cols[0]

    # Build external menu
    ext_menu = []
    if not external_long.empty:
        for ind in sorted(external_long["indicator"].dropna().unique()):
            ext_menu.append(ind)

    cA, cB, cC = st.columns([1.2, 1.2, 1])
    with cA:
        x_metric = st.selectbox("X axis (internal metric)", internal_numeric_cols, index=internal_numeric_cols.index(defX))
    with cB:
        if ext_menu:
            y_indicator = st.selectbox("Y axis (external indicator)", ext_menu, index=0)
        else:
            y_indicator = None
            st.info("Load external CSVs (sidebar) to enable comparisons.")
    with cC:
        year_cmp = st.selectbox("Year for comparison", list(range(min_year, max_year+1))[::-1], index=0)
        log_x = st.checkbox("Log X", value=False)
        log_y = st.checkbox("Log Y", value=False)
        color_by_region = st.checkbox("Color by region", value=True)
        show_trend = st.checkbox("Show OLS trendline (linear scales only)", value=True)

    if y_indicator:
        # Internal slice for X
        cmp = df[df[YEAR_COL] == year_cmp][["iso3", COUNTRY_COL, "region", x_metric]].rename(columns={x_metric: "__x__"}).copy()
        cmp["__x__"] = pd.to_numeric(cmp["__x__"], errors="coerce")

        # External Y for chosen year/indicator
        ext_y = external_long[(external_long["indicator"] == y_indicator) & (external_long["year"] == year_cmp)]
        if ext_y.empty:
            st.warning(f"No external data for **{y_indicator}** in **{year_cmp}**. Try a different year or indicator.")
        else:
            cmp = cmp.merge(ext_y[["iso3", "value"]].rename(columns={"value": "__y__"}), on="iso3", how="inner")
            cmp["__y__"] = pd.to_numeric(cmp["__y__"], errors="coerce")
            cmp = cmp.dropna(subset=["__x__", "__y__"]).copy()

            # Log filtering
            if log_x:
                cmp = cmp[cmp["__x__"] > 0]
            if log_y:
                cmp = cmp[cmp["__y__"] > 0]

            if cmp.empty:
                st.info("No overlapping rows after joining and log filtering. Switch year/indicator or disable log scales.")
            else:
                import plotly.express as px
                pcolor = "region" if color_by_region else None
                fig_sc = px.scatter(
                    cmp,
                    x="__x__", y="__y__",
                    hover_name=COUNTRY_COL,
                    color=pcolor,
                    labels={"__x__": x_metric, "__y__": y_indicator, "region": "Region"},
                )
                fig_sc.update_xaxes(type="log" if log_x else "linear", title=x_metric)
                fig_sc.update_yaxes(type="log" if log_y else "linear", title=y_indicator)
                fig_sc.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)

                # Optional OLS trendline (linear-linear only)
                if show_trend and (not log_x) and (not log_y) and len(cmp) >= 3:
                    m, b = np.polyfit(cmp["__x__"], cmp["__y__"], 1)
                    xline = np.linspace(cmp["__x__"].min(), cmp["__x__"].max(), 50)
                    yline = m * xline + b
                    fig_sc.add_trace(
                        go.Scatter(x=xline, y=yline, mode="lines", name="OLS fit", hoverinfo="skip")
                    )

                st.subheader(f"Scatter — {x_metric} vs {y_indicator} — {year_cmp}")
                st.plotly_chart(fig_sc, use_container_width=True)

                # Correlations
                pearson = cmp["__x__"].corr(cmp["__y__"], method="pearson") if len(cmp) >= 3 else np.nan
                spearman = cmp["__x__"].corr(cmp["__y__"], method="spearman") if len(cmp) >= 3 else np.nan
                st.caption(f"Pearson r = {round(pearson,3) if pd.notna(pearson) else 'NA'} | "
                           f"Spearman ρ = {round(spearman,3) if pd.notna(spearman) else 'NA'} | n = {len(cmp)}")

                # Residuals (linear-linear only)
                if show_trend and (not log_x) and (not log_y) and len(cmp) >= 3:
                    cmp = cmp.copy()
                    cmp["pred"] = m * cmp["__x__"] + b
                    cmp["residual"] = cmp["__y__"] - cmp["pred"]
                else:
                    cmp["residual"] = np.nan

                out_tbl = cmp[[COUNTRY_COL, "region", "__x__", "__y__", "residual"]].rename(
                    columns={"__x__": x_metric, "__y__": y_indicator}
                ).sort_values(y_indicator, ascending=False).reset_index(drop=True)

                st.dataframe(out_tbl.head(25), use_container_width=True)
                st.download_button(
                    "Download compare slice (CSV)",
                    data=out_tbl.to_csv(index=False).encode("utf-8"),
                    file_name=f"compare_{year_cmp}_{re.sub(r'[^A-Za-z0-9]+','_',y_indicator)}.csv",
                    mime="text/csv"
                )

# =========================================================
# TAB 3 — Global trends
# =========================================================
with tab_global:
    st.header("📈 Global trends across countries")

    # ---------- Helper to estimate vector counts ----------
    VECTORS = [c for c in ["contaminants","pathogens","pests","biotechnology","toxins"] if c in df.columns]
    has_total = "n_total" in df.columns

    def global_vector_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
        """
        Returns long df: [year, series, value]
        If n_total exists, estimate notifications by vector as share_of_vector * n_total per country-year then sum.
        Else, fall back to summing vector 'levels' across countries (units: index-units).
        """
        g = []
        if not VECTORS:
            return pd.DataFrame(columns=["year","series","value"])
        grp = frame.groupby(YEAR_COL, dropna=True)
        for y, sub in grp:
            sub = sub.copy()
            if has_total:
                row_tot = sub[VECTORS].sum(axis=1).replace(0, np.nan)
                shares = sub[VECTORS].div(row_tot, axis=0).fillna(0.0)
                for v in VECTORS:
                    est = (shares[v] * pd.to_numeric(sub.get("n_total", 0), errors="coerce")).fillna(0.0)
                    g.append((int(y), v, float(est.sum())))
            else:
                # Fallback: sum of levels (comparable within-app, not absolute counts)
                sums = pd.to_numeric(sub[VECTORS], errors="coerce").fillna(0.0).sum(axis=0)
                for v in VECTORS:
                    g.append((int(y), v, float(sums.get(v, 0.0))))
        out = pd.DataFrame(g, columns=[YEAR_COL, "series", "value"]).sort_values([YEAR_COL,"series"])
        return out

    def global_tier_timeseries(frame: pd.DataFrame) -> pd.DataFrame:
        """Long df: [year, series, value] for Regular, Emergency, Regular Addenda, Emergency Addenda."""
        cols = [c for c in ["n_regular","n_emergency","n_regular_addenda","n_emergency_addenda"] if c in frame.columns]
        if not cols:
            return pd.DataFrame(columns=["year","series","value"])
        sums = frame.groupby(YEAR_COL, dropna=True)[cols].sum(min_count=1).reset_index()
        rename = {
            "n_regular":"Regular",
            "n_emergency":"Emergency",
            "n_regular_addenda":"Regular Addenda",
            "n_emergency_addenda":"Emergency Addenda",
        }
        long = sums.melt(id_vars=[YEAR_COL], var_name="series", value_name="value")
        long["series"] = long["series"].map(rename).fillna(long["series"])
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        return long.sort_values([YEAR_COL,"series"])

    # ---------- Build data ----------
    vec_ts = global_vector_timeseries(df)

 
    st.subheader("Notifications by risk vector (global)")
    if vec_ts.empty:
        st.info("No vector columns found to build this chart.")
    else:
        # area chart (stacked)
        area_vec = alt.Chart(vec_ts).mark_area().encode(
            x=alt.X(f"{YEAR_COL}:O", title="Year"),
            y=alt.Y("value:Q", title="Estimated notifications" if has_total else "Aggregate level (index units)"),
            color=alt.Color("series:N", title="Vector"),
            tooltip=[alt.Tooltip(f"{YEAR_COL}:O"), "series:N", alt.Tooltip("value:Q", format=".0f")]
        ).properties(height=600)
        st.altair_chart(area_vec, use_container_width=True)


