import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_utils import C, SEG_COLORS, chart, get_artifacts

art = get_artifacts()
R = art["results"]
K, resp, seg = R["kpis"], R["response"], R["seg_profile"]

st.title("Nata Supermarkets: Customer Analytics")
st.caption("Business Analytics case study · 37 stores across Canada · customer data 2012-2014 · 3 machine-learning models")

st.markdown("""
**The problem.** In 2021 Nata underperformed internally and against peers. VP Technology Vina Verago traced the gap to
**(1) poor promotion targeting** and **(2) weak demand and inventory forecasting**, while competitors such as Walmart
were already using analytics. This app shows what Nata's customer data reveals and lets you use the models directly.
""")

mass = R["response"]["y"].sum() * 11 - len(R["response"]["y"]) * 3
ys = R["response"]["y"][(-R["response"]["oof"]).argsort()]
best_profit = (11 * ys.cumsum() - 3 * (1 + np.arange(len(ys)))).max()
c = st.columns(5)
c[0].metric("Customers analysed", f"{K['customers']:,}", help="After cleaning 2,240 raw records")
c[1].metric("Top 20% share of spend", f"{R['top20_share']:.0f}%")
c[2].metric("Wine + meat share of wallet", f"{R['wallet'].loc[['Wine', 'Meat'], 'Share %'].sum():.0f}%")
c[3].metric("Response model ROC-AUC", f"{resp['table'].iloc[0]['Test ROC-AUC']:.2f}", help=resp["best"])
c[4].metric("Campaign profit", f"${best_profit:,.0f}", delta=f"+${best_profit - mass:,.0f} vs mass mailing",
            help="Contacting only the model's top-scored customers (\\$3 cost, \\$11 revenue per response)")

st.divider()
left, right = st.columns([1.1, 1])
with left:
    st.subheader("Headline findings")
    st.markdown(f"""
1. **Revenue is concentrated.** The top 20% of customers produce **{R['top20_share']:.0f}%** of spend, and wine and meat make up **78%** of the wallet.
2. **Income and children drive spend.** Spend rises strongly with income (rho = {R['tests']['income_spend_rho']:.2f}) and falls with each child. Parents rely on deals.
3. **Four segments.** *Affluent Premium Shoppers* are {seg.loc['Affluent Premium Shoppers', 'Share of customers %']:.0f}% of customers but **{seg.loc['Affluent Premium Shoppers', 'Share of revenue %']:.0f}% of revenue**.
4. **Mass promotions lose money.** The latest campaign converted {K['response_rate']:.1f}%, below the 27.3% break-even rate. Model-based targeting turns a **-\\${abs(mass):,.0f}** loss into about **+\\${best_profit:,.0f}**.
5. **Demand is predictable.** Demographics explain about **{R['demand']['table'].loc['Total', 'Gradient Boosting R²']:.0%}** of the variance in a customer's spend, which supports planning stock by each store's segment mix.
6. **Complaints are rare** ({K['pct_complain']:.1f}%) and are not linked to spend or response, so they are not the cause of the underperformance.
""")
with right:
    st.subheader("Customer share vs revenue share")
    fig = go.Figure()
    fig.add_bar(y=seg.index, x=seg["Share of customers %"], name="% of customers", orientation="h", marker_color=C[0])
    fig.add_bar(y=seg.index, x=seg["Share of revenue %"], name="% of revenue", orientation="h", marker_color=C[1])
    fig.update_layout(barmode="group", xaxis_title="%", yaxis=dict(autorange="reversed"), bargap=0.25)
    chart(fig, 360)

st.divider()
st.subheader("How to use this app")
g = st.columns(4)
g[0].markdown("**Data analytics**  \nCustomer insights, promotion performance and a live explorer (upload the data file in the sidebar).")
g[1].markdown("**ML models**  \nSegmentation (K-Means), response prediction (4 classifiers) and demand forecasting (3 regressors).")
g[2].markdown("**Try the models**  \nScore a single customer, simulate campaign profit, score a CSV file, or plan a store's stock.")
g[3].markdown("**Strategy**  \nRecommendations, KPIs and business value for Nata.")
