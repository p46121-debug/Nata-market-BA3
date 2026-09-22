"""Shared Streamlit helpers: loading models, theme, reusable widgets."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import nata_core as core

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
LOCAL_DATA = ROOT / "data" / "nata_supermarket_data.xlsx"
MODEL_FILES = {"segmentation": "segmentation_model.sav", "response": "response_model.sav",
               "demand": "demand_models.sav", "results": "analytics_results.sav"}

# Colour-blind-checked categorical palette (fixed order) - same as the notebook
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEG_COLORS = dict(zip(core.PERSONA_NAMES, C[:4]))
CAT_COLORS = dict(zip(core.PNAMES, C[:6]))
SEQ = "Blues"

pio.templates["nata"] = go.layout.Template(layout=go.Layout(
    colorway=C, font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    margin=dict(l=10, r=10, t=50, b=10), hoverlabel=dict(font_size=12),
    xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(gridcolor="rgba(128,128,128,0.18)", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="")))
pio.templates.default = "plotly+nata"


def chart(fig, height=380, key=None):
    title = fig.layout.title.text
    if title:  # render the title as text so it never collides with the legend
        st.markdown(f"**{title}**".replace("$", "\\$"))
        fig.update_layout(title_text=None)
    fig.update_layout(height=height, legend_title_text="", margin=dict(t=30))
    st.plotly_chart(fig, key=key)


# --------------------------------------------------------------------------------------------------
# Artefact loading: saved .sav files first, retrain from data as a fallback
# --------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved models...")
def _load_saved():
    return {k: joblib.load(MODEL_DIR / f) for k, f in MODEL_FILES.items()}


@st.cache_resource(show_spinner=False)
def _train_from_bytes(data: bytes):
    bar = st.progress(0.0, text="Training models from the uploaded data (about 1 minute)...")
    art = core.train_all(core.load_raw(data), progress=lambda m, p: bar.progress(min(p, 1.0), text=m))
    bar.empty()
    art.pop("_clean_df", None)
    return art


def data_bytes() -> bytes | None:
    """Excel data from the sidebar upload, or from data/ when running locally."""
    if st.session_state.get("raw_bytes"):
        return st.session_state["raw_bytes"]
    if LOCAL_DATA.exists():
        return LOCAL_DATA.read_bytes()
    return None


def get_artifacts() -> dict:
    try:
        art = _load_saved()
        st.session_state["model_source"] = "saved .sav files"
        return art
    except Exception as e:  # missing files or library-version mismatch
        err = e
    data = data_bytes()
    if data is None:
        st.error("The saved models in `models/` could not be loaded "
                 f"(`{type(err).__name__}: {err}`). Upload `nata_supermarket_data.xlsx` in the sidebar "
                 "and the app will retrain every model automatically.")
        st.stop()
    st.session_state["model_source"] = "retrained from data (saved models could not be loaded)"
    return _train_from_bytes(data)


@st.cache_data(show_spinner="Preparing live data...")
def live_dataset(data: bytes) -> pd.DataFrame:
    """Cleaned, feature-engineered customer table scored with the deployed models (only when data is available)."""
    df, _ = core.clean(core.load_raw(data))
    art = get_artifacts()
    df["Segment"] = core.predict_segment(art["segmentation"], df)["Segment"].values
    df["Response_Score"] = core.predict_response(art["response"], df)["Response_Probability"].values
    return df


def sidebar():
    with st.sidebar:
        st.markdown("### Data (optional)")
        up = st.file_uploader("Upload `nata_supermarket_data.xlsx`", type=["xlsx"],
                              help="Unlocks the Live Data Explorer. It is also used to retrain the models if the "
                                   "saved .sav files cannot be loaded. The file stays in your session and is never stored.")
        if up is not None:
            st.session_state["raw_bytes"] = up.getvalue()
        src = st.session_state.get("model_source", "saved .sav files")
        st.caption(f"Models: {src}")
        if data_bytes() is None:
            st.caption("Live data: not loaded")
        else:
            st.caption("Live data: loaded")


# --------------------------------------------------------------------------------------------------
# Customer input form (shared by predictor pages)
# --------------------------------------------------------------------------------------------------
LABELS = {"MntWines": "Wine", "MntFruits": "Fruit", "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
          "MntSweetProducts": "Sweets", "MntGoldProds": "Gold", "NumDealsPurchases": "Deal purchases",
          "NumWebPurchases": "Web purchases", "NumCatalogPurchases": "Catalogue purchases",
          "NumStorePurchases": "Store purchases", "NumWebVisitsMonth": "Web visits / month"}


def customer_form(art: dict, key: str = "cf") -> pd.DataFrame | None:
    R = art["results"]
    presets = R["presets"]
    names = list(presets.keys())
    pick = st.selectbox("Start from a preset profile (then edit any field)", names, key=f"{key}_preset")
    p = presets[pick]
    k = f"{key}_{names.index(pick)}"
    with st.form(f"{key}_form"):
        st.markdown("**1. Demographics**")
        c = st.columns(4)
        age = c[0].number_input("Age (years)", 18, 90, int(p["Age"]), key=f"{k}_age")
        income = c[1].number_input("Annual income ($)", 1_000, 200_000, int(p["Income"]), step=1_000, key=f"{k}_inc")
        edu = c[2].selectbox("Education", core.EDUCATION_LEVELS, core.EDUCATION_LEVELS.index(p["Education"]), key=f"{k}_edu")
        mar = c[3].selectbox("Marital status", core.MARITAL_LEVELS, core.MARITAL_LEVELS.index(p["Marital_Status"]), key=f"{k}_mar")
        c = st.columns(4)
        kid = c[0].number_input("Kids at home", 0, 3, int(p["Kidhome"]), key=f"{k}_kid")
        teen = c[1].number_input("Teens at home", 0, 3, int(p["Teenhome"]), key=f"{k}_teen")
        ten = c[2].number_input("Months since enrolment", 0, 60, int(p["Tenure_Months"]), key=f"{k}_ten")
        rec = c[3].number_input("Days since last purchase", 0, 365, int(p["Recency"]), key=f"{k}_rec")
        st.markdown("**2. Spend in the last two years ($)**")
        c = st.columns(6)
        spend = {col: c[i].number_input(LABELS[col], 0, 5_000, int(p[col]), key=f"{k}_{col}") for i, col in enumerate(core.PRODUCTS)}
        st.markdown("**3. Purchases and web activity**")
        c = st.columns(5)
        chans = {col: c[i].number_input(LABELS[col], 0, 40, int(p[col]), key=f"{k}_{col}")
                 for i, col in enumerate(["NumDealsPurchases"] + core.CHANNELS + ["NumWebVisitsMonth"])}
        st.markdown("**4. Campaign history & service**")
        c = st.columns(6)
        cmps = {col: int(c[i].checkbox(f"Accepted Cmp{i + 1}", bool(p[col]), key=f"{k}_{col}")) for i, col in enumerate(core.CMPS)}
        comp = int(c[5].checkbox("Complained", bool(p["Complain"]), key=f"{k}_comp"))
        go_ = st.form_submit_button("Run the models", type="primary", width="stretch")
    if not go_ and f"{key}_last" not in st.session_state:
        return None
    if go_:
        row = {"Age": age, "Income": income, "Education": edu, "Marital_Status": mar, "Kidhome": kid, "Teenhome": teen,
               "Tenure_Months": ten, "Recency": rec, **spend, **chans, **cmps, "Complain": comp}
        st.session_state[f"{key}_last"] = row
    row = st.session_state[f"{key}_last"]
    if row["NumDealsPurchases"] > row["NumWebPurchases"] + row["NumCatalogPurchases"] + row["NumStorePurchases"]:
        st.warning("Deal purchases are higher than total purchases. Deal share is capped at 100%.")
    return core.add_features(pd.DataFrame([row]))


def fmt_money(x, dp=0):
    return f"{'-' if x < 0 else ''}${abs(x):,.{dp}f}"


def insight(text: str):
    st.info(text.replace("$", "\\$"), icon=":material/lightbulb:")
