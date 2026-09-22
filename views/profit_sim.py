import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_utils import C, chart, fmt_money, get_artifacts

M = get_artifacts()["results"]["response"]
oof, y = M["oof"], M["y"]
order = np.argsort(-oof)
ys = y[order]
n = len(y)

st.title("Campaign profit simulator")
st.caption("Uses honest out-of-fold response scores for every customer. Change the economics and see who should be contacted.")

c = st.columns(3)
cost = c[0].number_input("Cost per contact ($)", 0.5, 50.0, 3.0, 0.5)
rev = c[1].number_input("Revenue per response ($)", 1.0, 500.0, 11.0, 1.0,
                        help="The data file uses \\$11. Try a margin-based customer lifetime value instead.")
base_n = c[2].number_input("Customers in the campaign base", 100, 1_000_000, n, 100,
                           help="Scales the results from the sample to Nata's full customer base")
scale = base_n / n

k = np.arange(1, n + 1)
profit = (rev * np.cumsum(ys) - cost * k) * scale
kbest = int(profit.argmax()) + 1
mass = (rev * y.sum() - cost * n) * scale

st.divider()
c = st.columns(4)
c[0].metric("Optimal share to contact", f"{100 * kbest / n:.0f}%", help=f"Score cut-off ≈ {oof[order][kbest - 1]:.2f}")
c[1].metric("Profit at optimum", fmt_money(profit.max()))
c[2].metric("Mass-mailing profit", fmt_money(mass))
c[3].metric("Gain from targeting", f"${profit.max() - mass:,.0f}")

fig = go.Figure()
fig.add_scatter(x=100 * k / n, y=profit, mode="lines", line=dict(color=C[0], width=3), name="Model-ranked targeting",
                hovertemplate="Contact top %{x:.0f}% → profit $%{y:,.0f}<extra></extra>")
rng = np.random.default_rng(0)
rand = (rev * np.cumsum(y[rng.permutation(n)]) - cost * k) * scale
fig.add_scatter(x=100 * k / n, y=rand, mode="lines", line=dict(color="grey", dash="dot"), name="Random targeting")
fig.add_hline(y=mass, line_dash="dash", line_color=C[1], annotation_text="Mass mailing")
fig.add_scatter(x=[100 * kbest / n], y=[profit.max()], mode="markers", marker=dict(size=14, color=C[0], line=dict(color="white", width=2)),
                name="Optimum")
fig.update_layout(title="Profit vs share of customers contacted", xaxis_title="% of customers contacted (highest score first)",
                  yaxis_title="Profit ($)")
chart(fig, 440)

st.subheader("What if the budget is fixed?")
pct = st.slider("Share of customers you can afford to contact (%)", 1, 100, int(round(100 * kbest / n)))
kk = max(1, int(n * pct / 100))
reached = ys[:kk].sum()
tbl = pd.DataFrame({
    "Metric": ["Contacts", "Responders reached", "Share of all responders", "Response rate", "Cost", "Revenue", "Profit", "ROI"],
    "Model-targeted": [f"{kk * scale:,.0f}", f"{reached * scale:,.0f}", f"{reached / y.sum():.0%}", f"{reached / kk:.1%}",
                       f"${cost * kk * scale:,.0f}", f"${rev * reached * scale:,.0f}", fmt_money((rev * reached - cost * kk) * scale),
                       f"{(rev * reached - cost * kk) / (cost * kk):.0%}"],
    "Random (same budget)": [f"{kk * scale:,.0f}", f"{y.mean() * kk * scale:,.0f}", f"{kk / n:.0%}", f"{y.mean():.1%}",
                             f"${cost * kk * scale:,.0f}", f"${rev * y.mean() * kk * scale:,.0f}",
                             fmt_money((rev * y.mean() - cost) * kk * scale), f"{(rev * y.mean() - cost) / cost:.0%}"],
})
st.dataframe(tbl, hide_index=True)
st.caption(f"Break-even response rate at these economics = cost / revenue = {cost / rev:.1%}.")
