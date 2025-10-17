# pages/02_🔗 Compare with External.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    COUNTRY_COL, YEAR_COL, DEFAULT_EXTERNAL, tidy_external_csv
)

try:
    import numpy as np, pandas as pd, altair as alt
    import plotly.graph_objects as go
    # only on the Country & Map page if used:
    from streamlit_plotly_events import plotly_events
    from utils import (...)  # your actual utils imports
except Exception as e:
    st.sidebar.error(f"Page import error: {e}")
    raise

st.title("🔗 Compare BVI with External Data")

# ---- get core df ----
if "df_bvi" not in st.session_state:
    st.error("No index loaded. Go to Home and upload or use the default CSV.")
    st.stop()
df = st.session_state["df_bvi"]
min_year = int(df[YEAR_COL].dropna().min())
max_year = int(df[YEAR_COL].dropna().max())

# ---- external datasets loader (page-local) ----
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

# ---- metric pickers ----
internal_numeric_cols = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()])
if not internal_numeric_cols:
    st.error("No numeric internal columns available.")
    st.stop()
defX = "vigilance_index" if "vigilance_index" in internal_numeric_cols else internal_numeric_cols[0]

cA, cB, cC = st.columns([1.2, 1.2, 1])
with cA:
    x_metric = st.selectbox("X axis (internal metric)", internal_numeric_cols, index=internal_numeric_cols.index(defX))
with cB:
    if not external_long.empty:
        ext_menu = sorted(external_long["indicator"].dropna().unique().tolist())
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

# ---- plot ----
if y_indicator:
    cmp = df[df[YEAR_COL] == year_cmp][["iso3", COUNTRY_COL, "region", x_metric]].rename(columns={x_metric: "__x__"}).copy()
    cmp["__x__"] = pd.to_numeric(cmp["__x__"], errors="coerce")

    ext_y = external_long[(external_long["indicator"] == y_indicator) & (external_long["year"] == year_cmp)]
    if ext_y.empty:
        st.warning(f"No external data for **{y_indicator}** in **{year_cmp}**. Try a different year or indicator.")
        st.stop()

    cmp = cmp.merge(ext_y[["iso3", "value"]].rename(columns={"value": "__y__"}), on="iso3", how="inner")
    cmp["__y__"] = pd.to_numeric(cmp["__y__"], errors="coerce")
    cmp = cmp.dropna(subset=["__x__", "__y__"]).copy()

    if log_x: cmp = cmp[cmp["__x__"] > 0]
    if log_y: cmp = cmp[cmp["__y__"] > 0]

    if cmp.empty:
        st.info("No overlapping rows after join/log filtering. Switch year/indicator or disable log scales.")
    else:
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

        if show_trend and (not log_x) and (not log_y) and len(cmp) >= 3:
            m, b = np.polyfit(cmp["__x__"], cmp["__y__"], 1)
            xline = np.linspace(cmp["__x__"].min(), cmp["__x__"].max(), 50)
            yline = m * xline + b
            fig_sc.add_trace(
                go.Scatter(x=xline, y=yline, mode="lines", name="OLS fit", hoverinfo="skip")
            )

        st.subheader(f"Scatter — {x_metric} vs {y_indicator} — {year_cmp}")
        st.plotly_chart(fig_sc, use_container_width=True)

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

