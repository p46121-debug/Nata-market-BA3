import pandas as pd
import plotly.express as px
import streamlit as st

import nata_core as core
from app_utils import C, SEG_COLORS, chart, get_artifacts

art = get_artifacts()

st.title("Batch scoring")
st.caption("Upload a list of customers (CSV or Excel) and get segment, response score, contact decision and expected demand for each one.")

tpl = core.batch_template()
st.download_button("Download CSV template", tpl.to_csv(index=False).encode(), "nata_batch_template.csv", "text/csv",
                   icon=":material/download:")
with st.expander("Required columns"):
    st.markdown(", ".join(f"`{c}`" for c in core.REQUIRED_BATCH) +
                "\n\nOptional: `AcceptedCmp1`-`AcceptedCmp5`, `Complain` (default 0), and any ID column. "
                "Raw-format files work too: `Year_Birth` instead of `Age` and `Dt_Customer` instead of `Tenure_Months`. "
                f"Education must be one of {core.EDUCATION_LEVELS}; marital status one of {core.MARITAL_LEVELS}.")

up = st.file_uploader("Upload customers", type=["csv", "xlsx"])
use_demo = st.toggle("Or score the 3 example customers from the template")
if up is None and not use_demo:
    st.stop()

if up is not None:
    df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
else:
    df = tpl.copy()

if "Age" not in df.columns and "Year_Birth" in df.columns:
    df["Age"] = 2014 - df["Year_Birth"]
if "Tenure_Months" not in df.columns and "Dt_Customer" in df.columns:
    df["Tenure_Months"] = ((core.DATA_END - pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")).dt.days / 30.44).clip(lower=0)
missing = [c for c in core.REQUIRED_BATCH if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()
df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single", "Absurd": "Single", "YOLO": "Single"})
bad = ~df["Education"].isin(core.EDUCATION_LEVELS)
if bad.any():
    st.warning(f"{bad.sum()} rows have an unknown education level. They are treated as the base level in the response model.")
num = [c for c in core.REQUIRED_BATCH if c not in ("Education", "Marital_Status")]
df[num] = df[num].apply(pd.to_numeric, errors="coerce")
if df[num].isna().any().any():
    n0 = len(df); df = df.dropna(subset=num)
    st.warning(f"Dropped {n0 - len(df)} rows with missing or non-numeric values.")

X = core.add_features(df)
seg = core.predict_segment(art["segmentation"], X)
rsp = core.predict_response(art["response"], X)
dem = core.predict_demand(art["demand"], X)
out = df.copy()
out["Segment"] = seg["Segment"].values
out["Response_Probability"] = rsp["Response_Probability"].round(4).values
out["Score_Decile"] = rsp["Decile"].values
out["Contact"] = rsp["Contact"].values
out["Expected_Profit_$"] = rsp["Expected_Profit_$"].round(2).values
for nm in core.PNAMES + ["Total"]:
    out[f"Expected_{nm}_2yr_$"] = dem[nm].round(0).values
out = out.sort_values("Response_Probability", ascending=False)

c = st.columns(4)
c[0].metric("Customers scored", f"{len(out):,}")
c[1].metric("Recommended contacts", f"{int(out.Contact.sum()):,}")
c[2].metric("Expected responders (contacted)", f"{out.loc[out.Contact, 'Response_Probability'].sum():,.0f}")
c[3].metric("Expected campaign profit", f"${out.loc[out.Contact, 'Expected_Profit_$'].sum():,.0f}")

a, b = st.columns(2)
with a:
    s = out.Segment.value_counts().reindex(core.PERSONA_NAMES).fillna(0)
    chart(px.bar(x=s.index, y=s.values, color=s.index, color_discrete_map=SEG_COLORS, title="Customers by segment",
                 labels={"x": "", "y": "Customers"}).update_layout(showlegend=False), 320)
with b:
    chart(px.histogram(out, x="Response_Probability", nbins=20, title="Distribution of response scores",
                       color_discrete_sequence=[C[0]]).add_vline(x=art["response"]["cutoff"], line_color=C[1],
                                                                 annotation_text="Contact cut-off"), 320)
st.dataframe(out, hide_index=True)
st.download_button("Download scored file", out.to_csv(index=False).encode(), "nata_scored_output.csv", "text/csv",
                   type="primary", icon=":material/download:")
