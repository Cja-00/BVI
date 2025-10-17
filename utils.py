# utils.py
import io, re
from typing import List
import numpy as np
import pandas as pd
import pycountry
import streamlit as st

# =========================
# Constants
# =========================
CSV_INDEX = "index_by_country_year.csv"  # default demo path
COUNTRY_COL = "notifying_member"
YEAR_COL = "year"

CORE_COLS   = ["vigilance_index", "vigilance_index_3y_ma", "intensity_scaled", "banflow_scaled", "breadth"]
TIER_COLS   = ["total_notices", "tier1_prop", "tier2_prop", "tier3_prop", "sps_engagement"]
VECTOR_COLS = ["contaminants", "pathogens", "pests", "biotechnology", "toxins"]
TARGET_COLS = ["food_safety", "plant_health", "animal_health", "human_health", "environment"]

# Four-bucket columns (yearly + cumulative + per-capita if present)
BUCKET_YEARLY = ["n_regular", "n_emergency", "n_regular_addenda", "n_emergency_addenda", "n_total"]
BUCKET_CUM    = ["cum_n_regular", "cum_n_emergency", "cum_n_regular_addenda", "cum_n_emergency_addenda", "cum_n_total", "cum_total_notices"]
BUCKET_PC     = ["n_total_per_1m", "n_total_per_10m", "cum_total_per_1m", "cum_total_per_10m"]

# Auto-load these external CSVs if present in CWD (used on Compare page)
DEFAULT_EXTERNAL = [
    ("GDP per capita (2025)", "GDP_per_Capita_2025.csv", 2025),
    ("Democracy Index (2024)", "Democracy_index_2024.csv", 2024),
    ("Development Index (2023)", "development_index_2023.csv", 2023),
    ("WWF Physical Risk (2024)", "wwf_physical_risk.csv", 2024),
    ("Agriculture timeseries", "agriculture_timeseries_countries.csv", None),
]

REGION_OVERRIDES = {
    # (same content as your original dict)
    "AUS":"Oceania","NZL":"Oceania","FJI":"Oceania","PNG":"Oceania","SLB":"Oceania","VUT":"Oceania",
    "WSM":"Oceania","TON":"Oceania","KIR":"Oceania","TUV":"Oceania","NRU":"Oceania","MHL":"Oceania","PLW":"Oceania",
    "USA":"North America","CAN":"North America","MEX":"North America","GRL":"North America","BMU":"North America",
    "GTM":"Central America","SLV":"Central America","HND":"Central America","NIC":"Central America","CRI":"Central America","PAN":"Central America","BLZ":"Central America",
    "CUB":"Caribbean","DOM":"Caribbean","HTI":"Caribbean","JAM":"Caribbean","BHS":"Caribbean","BRB":"Caribbean","TTO":"Caribbean","DMA":"Caribbean","ATG":"Caribbean","KNA":"Caribbean","LCA":"Caribbean","VCT":"Caribbean","GRD":"Caribbean","GUY":"Caribbean","SUR":"Caribbean",
    "BRA":"South America","ARG":"South America","CHL":"South America","COL":"South America","PER":"South America","ECU":"South America","BOL":"South America","PRY":"South America","URY":"South America","VEN":"South America",
    "GBR":"Europe","IRL":"Europe","ISL":"Europe","FRA":"Europe","DEU":"Europe","ITA":"Europe","ESP":"Europe","PRT":"Europe","NLD":"Europe","BEL":"Europe","LUX":"Europe","DNK":"Europe","SWE":"Europe","FIN":"Europe","NOR":"Europe","CHE":"Europe","AUT":"Europe","POL":"Europe","CZE":"Europe","SVK":"Europe","HUN":"Europe","SVN":"Europe","HRV":"Europe","ROU":"Europe","BGR":"Europe","GRC":"Europe","MLT":"Europe","CYP":"Europe","EST":"Europe","LVA":"Europe","LTU":"Europe","ALB":"Europe","MKD":"Europe","MNE":"Europe","SRB":"Europe","BIH":"Europe","MDA":"Europe","UKR":"Europe","BLR":"Europe","AND":"Europe","MCO":"Europe","SMR":"Europe","VAT":"Europe","LIE":"Europe","KOS":"Europe",
    "RUS":"Europe/Asia","TUR":"Europe/Asia","AZE":"Europe/Asia","GEO":"Europe/Asia","ARM":"Europe/Asia","KAZ":"Europe/Asia",
    "ZAF":"Africa","EGY":"Africa","NGA":"Africa","ETH":"Africa","KEN":"Africa","MAR":"Africa","DZA":"Africa","TUN":"Africa","LBY":"Africa","SDN":"Africa","SSD":"Africa","TCD":"Africa","NER":"Africa","MLI":"Africa","BFA":"Africa","SEN":"Africa","GMB":"Africa","GIN":"Africa","GNB":"Africa","SLE":"Africa","LBR":"Africa","CIV":"Africa","GHA":"Africa","TGO":"Africa","BEN":"Africa","CMR":"Africa","GAB":"Africa","GNQ":"Africa","COG":"Africa","COD":"Africa","CAF":"Africa","AGO":"Africa","ZMB":"Africa","MOZ":"Africa","MWI":"Africa","TZA":"Africa","UGA":"Africa","RWA":"Africa","BDI":"Africa","SOM":"Africa","DJI":"Africa","ERI":"Africa","MDG":"Africa","COM":"Africa","SYC":"Africa","MUS":"Africa","CPV":"Africa","STP":"Africa","ZWE":"Africa","BWA":"Africa","NAM":"Africa","LSO":"Africa","SWZ":"Africa","MRT":"Africa",
    "CHN":"Asia","JPN":"Asia","KOR":"Asia","PRK":"Asia","MNG":"Asia","IND":"Asia","PAK":"Asia","BGD":"Asia","LKA":"Asia","NPL":"Asia","BTN":"Asia","MDV":"Asia","IDN":"Asia","VNM":"Asia","THA":"Asia","MMR":"Asia","KHM":"Asia","LAO":"Asia","MYS":"Asia","SGP":"Asia","PHL":"Asia","BRN":"Asia","TLS":"Asia",
    "SAU":"Middle East","ARE":"Middle East","QAT":"Middle East","BHR":"Middle East","KWT":"Middle East","OMN":"Middle East","IRN":"Middle East","IRQ":"Middle East","SYR":"Middle East","JOR":"Middle East","LBN":"Middle East","ISR":"Middle East","PSE":"Middle East","YEM":"Middle East",
    "KAZ":"Central Asia","KGZ":"Central Asia","TJK":"Central Asia","TKM":"Central Asia","UZB":"Central Asia",
    "HKG":"Asia","MAC":"Asia","TWN":"Asia","GIB":"Europe","GGY":"Europe","JEY":"Europe","IMN":"Europe",
}

# =========================
# Helpers
# =========================
def to_iso3_series(names: pd.Series) -> pd.Series:
    overrides = {
        "United States":"USA","United States of America":"USA",
        "Russian Federation":"RUS","Viet Nam":"VNM","Iran, Islamic Republic of":"IRN",
        "Korea, Republic of":"KOR","Côte d’Ivoire":"CIV","Cote d'Ivoire":"CIV",
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
        if n in overrides:
            out.append(overrides[n]); continue
        try:
            out.append(pycountry.countries.lookup(n).alpha_3)  # type: ignore
        except Exception:
            try:
                nn = n.replace("’","'").replace("–","-")
                out.append(pycountry.countries.lookup(nn).alpha_3)  # type: ignore
            except Exception:
                out.append(None)
    return pd.Series(out, index=names.index)

def percentile_rank(series: pd.Series, value: float) -> float:
    s=pd.to_numeric(series, errors="coerce").dropna()
    return float((s<=value).mean()*100) if (not s.empty and pd.notna(value)) else np.nan

# utils.py

def add_region(df_iso: pd.DataFrame) -> pd.DataFrame:
    df = df_iso.copy()
    # Preserve real nulls but coerce others to uppercase strings
    iso = df["iso3"]
    # Use pandas' nullable string dtype to avoid weird NA coercions
    iso_str = iso.astype("string").str.upper()
    # Keep NA where original was NA
    iso_str = iso_str.where(iso.notna(), other=pd.NA)
    df["region"] = iso_str.map(REGION_OVERRIDES)
    df["region"] = df["region"].fillna("Other/Unknown")
    return df

@st.cache_data(show_spinner=False)
def load_index_from_bytes(upload: bytes | None) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(upload)) if upload else pd.read_csv(CSV_INDEX)
    df.columns = [c.strip() for c in df.columns]
    # Guard: ensure the country column exists
    if COUNTRY_COL not in df.columns:
        raise ValueError(f"Expected column '{COUNTRY_COL}' not found in uploaded CSV. Got: {list(df.columns)}")

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
    for c in set(CORE_COLS + TIER_COLS + VECTOR_COLS + TARGET_COLS + BUCKET_YEARLY + BUCKET_CUM + BUCKET_PC + ["ban_flow","population"]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build iso3 cleanly, then uppercase string dtype
    df["iso3"] = to_iso3_series(df[COUNTRY_COL]).astype("string").str.upper()
    # Drop rows where iso3 is missing or malformed
    df = df[df["iso3"].notna() & (df["iso3"].str.len() == 3)].copy()

    df = add_region(df)
    return df


def available_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def infer_country_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if re.search(r"(country|economy|nation|state|member)", c, re.I)]
    return candidates[0] if candidates else df.columns[0]

def tidy_external_csv(raw: pd.DataFrame, indicator_name: str, default_year: int | None = None) -> pd.DataFrame:
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


# -------- Global trend helpers (cached) --------
@st.cache_data(show_spinner=False)
def global_vector_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    vecs = [c for c in VECTOR_COLS if c in df.columns]
    if not vecs:
        return pd.DataFrame(columns=[YEAR_COL,"series","value"])
    has_total = "n_total" in df.columns
    g=[]
    for y, sub in df.groupby(YEAR_COL, dropna=True):
        sub=sub.copy()
        if has_total:
            row_tot = sub[vecs].sum(axis=1).replace(0, np.nan)
            shares  = sub[vecs].div(row_tot, axis=0).fillna(0.0)
            for v in vecs:
                est = (shares[v]*pd.to_numeric(sub["n_total"], errors="coerce")).fillna(0.0)
                g.append((int(y), v, float(est.sum())))
        else:
            sums = pd.to_numeric(sub[vecs], errors="coerce").fillna(0.0).sum(axis=0)
            for v in vecs:
                g.append((int(y), v, float(sums.get(v,0.0))))
    return pd.DataFrame(g, columns=[YEAR_COL,"series","value"]).sort_values([YEAR_COL,"series"])

@st.cache_data(show_spinner=False)
def global_tier_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["n_regular","n_emergency","n_regular_addenda","n_emergency_addenda"] if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=[YEAR_COL,"series","value"])
    sums = df.groupby(YEAR_COL, dropna=True)[cols].sum(min_count=1).reset_index()
    rename = {"n_regular":"Regular","n_emergency":"Emergency","n_regular_addenda":"Regular Addenda","n_emergency_addenda":"Emergency Addenda"}
    long = sums.melt(id_vars=[YEAR_COL], var_name="series", value_name="value")
    long["series"] = long["series"].map(rename).fillna(long["series"])
    long["value"]  = pd.to_numeric(long["value"], errors="coerce")
    return long.sort_values([YEAR_COL,"series"])
