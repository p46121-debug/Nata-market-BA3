import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import C, CAT_COLORS, SEQ, chart, get_artifacts, insight

R = get_artifacts()["results"]
T = R["tests"]

st.title("Customer insights")
st.caption("Who Nata's customers are, what they buy, and what drives their spending")

tabs = st.tabs(["Demographics", "Shopping habits", "Spend drivers", "Channels & deals", "Correlations", "Complaints"])


def hist_fig(h, title, xt, fmt=None):
    fig = go.Figure(go.Bar(x=(h.left + h.right) / 2, y=h["count"], width=(h.right - h.left) * 0.95,
                           marker_color=C[0], customdata=np.stack([h.left, h.right], axis=1),
                           hovertemplate="%{customdata[0]:,.0f} to %{customdata[1]:,.0f}<br>%{y} customers<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title="Customers", bargap=0)
    return fig


with tabs[0]:
    K = R["kpis"]
    c = st.columns(5)
    c[0].metric("Median age (2014)", f"{K['median_age']:.0f}")
    c[1].metric("Median income", f"${K['median_income']:,.0f}")
    c[2].metric("Hold a degree", f"{K['pct_degree']:.0f}%")
    c[3].metric("Partnered", f"{K['pct_partnered']:.0f}%")
    c[4].metric("Parents", f"{K['pct_parents']:.0f}%")
    a, b = st.columns(2)
    with a:
        chart(hist_fig(R["age_hist"], "Age distribution", "Age"), 320)
        e = R["education_counts"]
        chart(px.bar(x=e.values, y=e.index, orientation="h", title="Education", labels={"x": "Customers", "y": ""},
                     color_discrete_sequence=[C[0]]), 300)
    with b:
        chart(hist_fig(R["income_hist"], "Annual household income", "Income ($)"), 320)
        m = R["marital_counts"]
        chart(px.bar(x=m.values, y=m.index, orientation="h", title="Marital status", labels={"x": "Customers", "y": ""},
                     color_discrete_sequence=[C[0]]), 300)
    insight("A mature, educated, mostly partnered customer base: 89% hold a degree and 72% have children at home.")

with tabs[1]:
    w = R["wallet"].sort_values("Share %", ascending=False)
    a, b = st.columns(2)
    with a:
        fig = px.bar(w, x=w.index, y="Share %", text=w["Share %"].map("{:.0f}%".format), title="Share of wallet by category",
                     color=w.index, color_discrete_map=CAT_COLORS)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="% of 2-yr spend")
        chart(fig)
    with b:
        L = R["lorenz"]
        fig = go.Figure()
        fig.add_scatter(x=L.pct_customers, y=L.pct_spend, mode="lines", line=dict(color=C[0], width=3), name="Nata customers",
                        hovertemplate="Top %{x:.0f}% of customers → %{y:.0f}% of spend<extra></extra>")
        fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="grey", dash="dash"), name="Equal spend")
        fig.add_vline(x=20, line_color=C[1], annotation_text=f"Top 20% = {R['top20_share']:.0f}% of spend")
        fig.update_layout(title="Spend concentration (Lorenz curve)", xaxis_title="% of customers (ranked by spend)",
                          yaxis_title="% of total spend")
        chart(fig)
    st.dataframe(w.style.format({"Spend": "${:,.0f}", "Share %": "{:.1f}", "Customers buying %": "{:.1f}"}))
    insight("Wine and meat make up 78% of spend and nearly every customer buys them. Fruit, fish and sweets together are about 15%. "
            "The top 20% of customers generate more than half of all revenue.")

with tabs[2]:
    d = R["income_spend_2d"]
    xc, yc = (d["xe"][:-1] + d["xe"][1:]) / 2, (d["ye"][:-1] + d["ye"][1:]) / 2
    fig = go.Figure(go.Heatmap(x=xc, y=yc, z=np.where(d["H"].T == 0, np.nan, d["H"].T), colorscale=SEQ,
                               hovertemplate="Income ≈ $%{x:,.0f}<br>Spend ≈ $%{y:,.0f}<br>%{z} customers<extra></extra>",
                               colorbar=dict(title="Customers")))
    fig.update_layout(title=f"Income vs 2-yr spend (density) · Spearman rho = {T['income_spend_rho']:.2f}",
                      xaxis_title="Income ($)", yaxis_title="Total 2-yr spend ($)")
    chart(fig, 420)
    dim = st.segmented_control("Average 2-yr spend by", list(R["spend_by"].keys()), default="Children")
    s = R["spend_by"][dim or "Children"]
    fig = px.bar(x=s.index.astype(str), y=s.values, text=[f"${v:,.0f}" for v in s.values], labels={"x": dim, "y": "Mean 2-yr spend ($)"},
                 color_discrete_sequence=[C[0]])
    chart(fig, 340)
    prof_dim = st.selectbox("Category spend profile by", list(R["cat_profile"].keys()))
    P = R["cat_profile"][prof_dim]
    fig = px.imshow(P.round(0), text_auto=True, color_continuous_scale=SEQ, aspect="auto",
                    labels=dict(color="Mean $"), title=f"Mean 2-yr spend per category by {prof_dim.lower()}")
    chart(fig, 330)
    insight(f"Income is the strongest driver of spend. Spend falls sharply with each child (Kruskal-Wallis p = {T['kw_children_p']:.1e}). "
            "PhD holders spend about 50 times more on wine than Basic-education customers, and customers aged 65+ spend the most.")

with tabs[3]:
    a, b = st.columns(2)
    with a:
        ch = R["channels"]
        fig = px.bar(ch, x=ch.index, y="Share %", text=ch["Share %"].map("{:.0f}%".format), title="Share of purchases by channel",
                     color_discrete_sequence=[C[0]])
        fig.update_layout(xaxis_title="", yaxis_title="% of purchases")
        chart(fig)
    with b:
        cb = R["channel_by_income"] * 100
        fig = go.Figure()
        for i, (col, nm) in enumerate(zip(["Web_Share", "Catalog_Share", "Store_Share"], ["Web", "Catalogue", "Store"])):
            fig.add_bar(x=cb.index.astype(str), y=cb[col], name=nm, marker_color=C[i])
        fig.update_layout(barmode="stack", title="Channel mix by income quartile", yaxis_title="% of purchases",
                          xaxis_title="Income quartile")
        chart(fig)
    dc = R["deal_by_children"]
    a, b = st.columns(2)
    with a:
        fig = px.bar(x=dc.index.astype(str), y=100 * dc.Deal_Share, text=[f"{v:.0f}%" for v in 100 * dc.Deal_Share],
                     labels={"x": "Children at home", "y": "% of purchases on deal"}, title="Deal reliance by number of children",
                     color_discrete_sequence=[C[1]])
        chart(fig, 330)
    with b:
        fig = px.bar(x=dc.index.astype(str), y=dc.NumWebVisitsMonth, text=[f"{v:.1f}" for v in dc.NumWebVisitsMonth],
                     labels={"x": "Children at home", "y": "Web visits per month"}, title="Web browsing by number of children",
                     color_discrete_sequence=[C[0]])
        chart(fig, 330)
    insight(f"The store handles 46% of purchases. Catalogue use rises with income. Deal share rises from 9% to 43% as the number of children grows "
            f"(income vs deal share rho = {T['income_deal_rho']:.2f}). Web visits are negatively correlated with spend "
            f"(rho = {T['webvisits_spend_rho']:.2f}), so heavy browsers buy little. This is a web-conversion gap.")

with tabs[4]:
    corr = R["corr"]
    fig = px.imshow(corr.round(2), text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="Spearman correlation matrix")
    chart(fig, 650)

with tabs[5]:
    cp = R["complaints"].copy()
    cp.index = ["No complaint", "Complained"]
    c = st.columns(3)
    c[0].metric("Customers who complained", f"{int(cp.loc['Complained', 'Customers'])}", f"{R['kpis']['pct_complain']:.1f}% of base",
                delta_color="off")
    c[1].metric("Spend difference p-value", f"{T['mw_complain_spend_p']:.2f}", "Not significant", delta_color="off")
    c[2].metric("Response difference p-value", f"{T['fisher_complain_response_p']:.2f}", "Not significant", delta_color="off")
    st.dataframe(cp.style.format({"Mean_Spend": "${:,.0f}", "Response_Rate": "{:.1%}", "Mean_Income": "${:,.0f}",
                                  "Mean_Tenure": "{:.1f}"}))
    a, b = st.columns(2)
    for col, (k, s) in zip([a, b], R["complaints_by"].items()):
        with col:
            chart(px.bar(x=s.index.astype(str), y=s.values, labels={"x": k, "y": "% complaining"}, title=f"Complaint rate by {k.lower()}",
                         color_discrete_sequence=[C[7]]), 300)
    insight("Complaints are rare and are not linked to lower spend or response. Nata should collect NPS/CSAT data "
            "so it can connect service quality to customer loyalty.")
