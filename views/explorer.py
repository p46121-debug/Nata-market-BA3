import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nata_core as core
from app_utils import C, CAT_COLORS, SEG_COLORS, chart, data_bytes, live_dataset

st.title("Live data explorer")
st.caption("Filter the customer base and see how the metrics change. This page needs the raw data file.")

data = data_bytes()
if data is None:
    st.info("To keep the case data private, the repository does not include it. Upload **nata_supermarket_data.xlsx** in the "
            "sidebar to unlock this page. The file stays only in your browser session.", icon=":material/lock:")
    st.stop()

df = live_dataset(data)

with st.container(border=True):
    c = st.columns(3)
    seg = c[0].multiselect("Segment", core.PERSONA_NAMES, default=core.PERSONA_NAMES)
    edu = c[1].multiselect("Education", core.EDUCATION_LEVELS, default=core.EDUCATION_LEVELS)
    mar = c[2].multiselect("Marital status", core.MARITAL_LEVELS, default=core.MARITAL_LEVELS)
    c = st.columns(3)
    age = c[0].slider("Age", int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())))
    inc = c[1].slider("Income ($)", 0, int(df.Income.max()) + 1000, (0, int(df.Income.max()) + 1000), 1000)
    kids = c[2].multiselect("Children at home", sorted(df.Children.unique()), default=sorted(df.Children.unique()))

f = df[df.Segment.isin(seg) & df.Education.isin(edu) & df.Marital_Status.isin(mar) & df.Age.between(*age) &
       df.Income.between(*inc) & df.Children.isin(kids)]
if f.empty:
    st.warning("No customers match these filters."); st.stop()

c = st.columns(6)
c[0].metric("Customers", f"{len(f):,}", f"{len(f) / len(df):.0%} of base", delta_color="off")
c[1].metric("Avg 2-yr spend", f"${f.Total_Spend.mean():,.0f}", f"{f.Total_Spend.mean() / df.Total_Spend.mean() - 1:+.0%} vs all")
c[2].metric("Share of revenue", f"{f.Total_Spend.sum() / df.Total_Spend.sum():.0%}")
c[3].metric("Deal share", f"{f.Deal_Share.mean():.0%}")
c[4].metric("Latest response", f"{f.Response.mean():.1%}", f"{100 * (f.Response.mean() - df.Response.mean()):+.1f} pts", delta_color="off")
c[5].metric("Avg model score", f"{f.Response_Score.mean():.0%}")

a, b = st.columns(2)
with a:
    s = f[core.PRODUCTS].sum().set_axis(core.PNAMES)
    fig = px.pie(values=s.values, names=s.index, hole=0.55, color=s.index, color_discrete_map=CAT_COLORS, title="Category mix of spend")
    chart(fig, 360)
with b:
    ch = f[core.CHANNELS].sum().set_axis(["Web", "Catalogue", "Store"])
    fig = px.bar(x=ch.index, y=100 * ch.values / ch.sum(), text=[f"{v:.0f}%" for v in 100 * ch.values / ch.sum()],
                 labels={"x": "", "y": "% of purchases"}, title="Channel mix", color_discrete_sequence=[C[0]])
    chart(fig, 360)

fig = px.scatter(f, x="Income", y="Total_Spend", color="Segment", color_discrete_map=SEG_COLORS, opacity=0.65,
                 hover_data={"Age": True, "Children": True, "Response_Score": ":.2f"},
                 category_orders={"Segment": core.PERSONA_NAMES}, labels={"Total_Spend": "2-yr spend ($)"},
                 title="Income vs spend (each dot is a customer)")
fig.update_traces(marker=dict(size=6, line=dict(width=0)))
chart(fig, 460)

xvar = st.selectbox("Break down spend and response by", ["Age_Group", "Education", "Marital_Status", "Children", "Segment"])
g = f.groupby(xvar, observed=True).agg(Customers=("ID", "size"), Avg_Spend=("Total_Spend", "mean"),
                                       Response_Rate=("Response", "mean"), Deal_Share=("Deal_Share", "mean"))
a, b = st.columns(2)
with a:
    chart(px.bar(g, x=g.index.astype(str), y="Avg_Spend", title=f"Avg 2-yr spend by {xvar}", labels={"x": xvar},
                 color_discrete_sequence=[C[0]]), 320)
with b:
    chart(px.bar(g, x=g.index.astype(str), y=100 * g.Response_Rate, title=f"Response rate % by {xvar}",
                 labels={"x": xvar, "y": "Response %"}, color_discrete_sequence=[C[1]]), 320)

with st.expander(f"Filtered customer table ({len(f):,} rows)"):
    cols = ["ID", "Segment", "Response_Score", "Age", "Education", "Marital_Status", "Income", "Children", "Total_Spend",
            "Deal_Share", "Recency", "Prev_Accepted", "Response"]
    st.dataframe(f[cols].sort_values("Response_Score", ascending=False), hide_index=True)
    st.download_button("Download filtered list", f[cols].to_csv(index=False).encode(), "nata_filtered.csv", "text/csv")
