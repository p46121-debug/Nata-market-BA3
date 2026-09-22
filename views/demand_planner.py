import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import nata_core as core
from app_utils import CAT_COLORS, chart, get_artifacts

art = get_artifacts()
R = art["results"]
D = R["demand"]

st.title("Demand planner")
st.caption("Forecast monthly category demand for a store's customer mix, or for a planned new cohort of customers")

tab1, tab2 = st.tabs(["Store catchment by segment", "New-customer cohort (ML forecast)"])

with tab1:
    st.markdown("Enter how many active customers of each segment a store serves. Demand is based on each segment's average monthly spend.")
    mix = D["segment_mix"]
    c = st.columns(4)
    counts = {n: c[i].number_input(n, 0, 1_000_000, int(round(2000 * mix[n])), 50, key=f"cnt_{i}")
              for i, n in enumerate(core.PERSONA_NAMES)}
    buf = st.slider("Safety-stock buffer (%)", 0, 50, 10, help="Extra stock on top of expected demand")
    per = D["monthly_per_customer"]
    dem = per.mul(pd.Series(counts), axis=0)
    total = dem.sum()
    c = st.columns(3)
    c[0].metric("Expected monthly demand", f"${total.sum():,.0f}")
    c[1].metric("Recommended stock value (with buffer)", f"${total.sum() * (1 + buf / 100):,.0f}")
    c[2].metric("Wine + meat share", f"{(total['Wine'] + total['Meat']) / total.sum():.0%}")
    fig = go.Figure()
    for nm in core.PNAMES:
        fig.add_bar(x=dem.index, y=dem[nm], name=nm, marker_color=CAT_COLORS[nm])
    fig.update_layout(barmode="stack", title="Monthly demand by segment and category", yaxis_title="$ per month")
    chart(fig, 400)
    plan = pd.DataFrame({"Expected $/month": total, "Share %": 100 * total / total.sum(),
                         f"Stock target $ (+{buf}%)": total * (1 + buf / 100)})
    st.dataframe(plan.style.format({"Expected $/month": "${:,.0f}", "Share %": "{:.1f}", f"Stock target $ (+{buf}%)": "${:,.0f}"}))

with tab2:
    st.markdown("Describe a planned group of new customers (for example, a new store's catchment or an acquisition campaign). "
                "The app generates a synthetic cohort and forecasts each customer's demand with the **Gradient Boosting** demand models.")
    ref = R["cohort_ref"]
    c = st.columns(4)
    n = c[0].number_input("Number of new customers", 10, 20_000, 500, 50)
    ten = c[3].slider("Months as a customer", 1, 24, 12, help="Tenure strongly affects spend. 12 months is a typical established customer.")
    inc = c[1].slider("Average income ($)", 10_000, 120_000, int(round(ref["income_mean"], -3)), 1_000)
    inc_sd = c[2].slider("Income spread (sd, $)", 2_000, 40_000, 15_000, 1_000)
    c = st.columns(4)
    age = c[0].slider("Age range", 18, 80, (30, 60))
    p_kid = c[1].slider("% with young kids", 0, 100, int(100 * ref["p_kid"]))
    p_teen = c[2].slider("% with teenagers", 0, 100, int(100 * ref["p_teen"]))
    p_par = c[3].slider("% partnered", 0, 100, int(100 * ref["p_partnered"]))
    with st.expander("Education mix (%)"):
        cols = st.columns(5)
        edu = {e: cols[i].number_input(e, 0, 100, int(round(100 * ref["education_mix"][e])), key=f"edu_{e}")
               for i, e in enumerate(core.EDUCATION_LEVELS)}
    if sum(edu.values()) == 0:
        st.error("The education mix must add up to more than 0."); st.stop()
    cohort = core.generate_cohort(int(n), inc, inc_sd, age[0], age[1], p_kid / 100, p_teen / 100, p_par / 100, edu, tenure=ten)
    pred = core.predict_demand(art["demand"], cohort)
    monthly = pred[core.PNAMES].sum() / 24
    avg = D["monthly_per_customer"].mul(D["segment_mix"], axis=0).sum()
    c = st.columns(3)
    c[0].metric("Forecast monthly demand", f"${monthly.sum():,.0f}")
    c[1].metric("Per customer per month", f"${monthly.sum() / n:,.1f}",
                delta=f"{monthly.sum() / n / avg.sum() - 1:+.0%} vs current average customer")
    c[2].metric("First-year demand", f"${12 * monthly.sum():,.0f}")
    fig = go.Figure()
    fig.add_bar(x=core.PNAMES, y=monthly / n, name="New cohort", marker_color=[CAT_COLORS[p] for p in core.PNAMES])
    fig.add_scatter(x=core.PNAMES, y=avg.values, mode="markers", name="Current average customer",
                    marker=dict(symbol="line-ew", size=40, line=dict(width=3, color="#7f7f7f")))
    fig.update_layout(title="Monthly demand per customer by category", yaxis_title="$ per customer per month")
    chart(fig, 400)
    st.dataframe(pd.DataFrame({"Monthly $ (cohort)": monthly, "Share %": 100 * monthly / monthly.sum()})
                 .style.format({"Monthly $ (cohort)": "${:,.0f}", "Share %": "{:.1f}"}))
    st.caption("A simulated cohort is used because only aggregate reference distributions are stored in the app, not raw customer records.")
