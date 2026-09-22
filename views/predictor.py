import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nata_core as core
from app_utils import C, CAT_COLORS, SEG_COLORS, chart, customer_form, get_artifacts

art = get_artifacts()
R = art["results"]

st.title("Customer predictor")
st.caption("Enter one customer's details. All three models run together: **segment** (K-Means), "
           "**campaign response** (Logistic Regression) and **expected category demand** (Gradient Boosting).")

cust = customer_form(art, key="pred")
if cust is None:
    st.info("Pick a preset or type in a customer's details, then click **Run the models**.", icon=":material/arrow_upward:")
    st.stop()

seg = core.predict_segment(art["segmentation"], cust).iloc[0]
rsp = core.predict_response(art["response"], cust).iloc[0]
dem = core.predict_demand(art["demand"], cust).iloc[0]
name = seg["Segment"]
pb = core.PERSONA_PLAYBOOK[name]

st.divider()
c = st.columns([1.3, 0.9, 1.1, 1])
with c[0]:
    st.caption("Segment")
    st.markdown(f"<div style='font-size:1.45rem;line-height:1.3;color:{SEG_COLORS[name]};font-weight:600'>{name}</div>",
                unsafe_allow_html=True)
c[1].metric("Response probability", f"{rsp['Response_Probability']:.0%}")
c[2].metric("Campaign decision", "Contact" if rsp["Contact"] else "Don't contact",
            delta=f"Top {rsp['Top_%_of_base']:.0f}% of customer base", delta_color="off")
c[3].metric("Expected profit per contact", f"{'-' if rsp['Expected_Profit_$'] < 0 else '+'}${abs(rsp['Expected_Profit_$']):.2f}", help="p × \\$11 − \\$3")

tab1, tab2, tab3 = st.tabs(["Segment", "Campaign response", "Demand forecast"])

with tab1:
    a, b = st.columns([1.3, 1])
    with a:
        pts = R["pca_points"]
        fig = px.scatter(pts, x="PC1", y="PC2", color="Segment", color_discrete_map=SEG_COLORS, opacity=0.35,
                         category_orders={"Segment": core.PERSONA_NAMES}, title="Where this customer sits")
        fig.update_traces(marker=dict(size=5, line=dict(width=0)))
        fig.add_scatter(x=[seg["PC1"]], y=[seg["PC2"]], mode="markers+text", text=["This customer"], textposition="top center",
                        marker=dict(symbol="star", size=22, color="#e34948", line=dict(color="white", width=2)), name="This customer")
        chart(fig, 430)
    with b:
        dist = pd.Series({n: seg[f"dist::{n}"] for n in core.PERSONA_NAMES})
        sim = np.exp(-dist ** 2 / 2); sim = 100 * sim / sim.sum()
        fig = px.bar(x=sim.values, y=sim.index, orientation="h", color=sim.index, color_discrete_map=SEG_COLORS,
                     labels={"x": "Relative closeness to segment centre (%)", "y": ""}, title="Segment fit")
        fig.update_layout(showlegend=False)
        chart(fig, 250)
        with st.container(border=True):
            st.markdown(f"**{name}**: {pb['profile']}")
            st.markdown(f"**Offer:** {pb['offer']}  \n**Channel:** {pb['channel']}  \n**Campaign style:** {pb['campaign']}")

with tab2:
    a, b = st.columns([1, 1.4])
    with a:
        p = rsp["Response_Probability"]
        fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number={"suffix": "%"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": C[0]},
                                            "threshold": {"line": {"color": C[1], "width": 4}, "value": 100 * art["response"]["cutoff"]},
                                            "steps": [{"range": [0, 27.3], "color": "rgba(128,128,128,0.15)"}]},
                                     title={"text": "Probability of accepting the next campaign"}))
        chart(fig, 300)
        st.caption(f"Orange line = contact cut-off ({art['response']['cutoff']:.0%}), set to maximise campaign profit. "
                   "Grey band = below the 27.3% break-even rate.")
    with b:
        contrib = core.response_contributions(art["response"], cust)
        if len(contrib):
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
            fig = px.bar(x=top.values, y=top.index, orientation="h",
                         color=np.where(top.values > 0, "Raises probability", "Lowers probability"),
                         color_discrete_map={"Raises probability": C[0], "Lowers probability": C[1]},
                         labels={"x": "Contribution to log-odds vs an average customer", "y": "", "color": ""},
                         title="Why? Top 10 factors for this customer")
            chart(fig, 380)

with tab3:
    actual = pd.Series({nm: float(cust[col].iloc[0]) for col, nm in zip(core.PRODUCTS, core.PNAMES)})
    pred = dem[core.PNAMES]
    fig = go.Figure()
    fig.add_bar(x=core.PNAMES, y=pred.values, name="Expected (model, from demographics)", marker_color=C[0])
    fig.add_bar(x=core.PNAMES, y=actual.values, name="Actual (entered)", marker_color=C[3])
    fig.update_layout(barmode="group", title="2-year category spend: expected vs actual", yaxis_title="$ over 2 years")
    chart(fig, 380)
    gap = (pred - actual)
    c = st.columns(3)
    c[0].metric("Expected 2-yr spend", f"${dem['Total']:,.0f}")
    c[1].metric("Expected monthly spend", f"${dem['Total'] / 24:,.0f}")
    c[2].metric("Biggest cross-sell gap", f"{gap.idxmax()}" if gap.max() > 0 else "None",
                delta=f"${gap.max():,.0f} below expected" if gap.max() > 0 else None, delta_color="off")
    st.caption("The demand model uses only attributes known at sign-up: age, income, children, partner, education and tenure. "
               "Categories where actual spend is well below expected are cross-sell opportunities.")
