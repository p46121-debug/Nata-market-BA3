import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nata_core as core
from app_utils import CAT_COLORS, C, chart, get_artifacts, insight

R = get_artifacts()["results"]
D = R["demand"]

st.title("Demand forecasting")
st.caption("ML model 3: regressors that predict each customer's category spend from attributes known at sign-up, "
           "rolled up to segment-level demand for inventory planning")

st.info("The data contains **no transaction dates**, so a classical time-series forecast (ARIMA/Prophet) is not possible. "
        "The forecastable question is: *given who a customer is, how much of each category will they buy?*", icon=":material/info:")

t = D["table"]
c = st.columns(4)
c[0].metric("Total-spend R² (Gradient Boosting)", f"{t.loc['Total', 'Gradient Boosting R²']:.2f}")
c[1].metric("Wine R²", f"{t.loc['Wine', 'Gradient Boosting R²']:.2f}")
c[2].metric("Meat R²", f"{t.loc['Meat', 'Gradient Boosting R²']:.2f}")
c[3].metric("Linear baseline (Ridge) total R²", f"{t.loc['Total', 'Ridge R²']:.2f}")

a, b = st.columns([1.3, 1])
with a:
    st.subheader("Out-of-fold accuracy by category")
    st.dataframe(t.style.format({"Mean 2-yr $": "${:,.0f}", "Ridge R²": "{:.2f}", "Random Forest R²": "{:.2f}",
                                 "Gradient Boosting R²": "{:.2f}", "GB MAE $": "${:,.0f}"}))
with b:
    fi = D["fi"].head(8)[::-1]
    chart(px.bar(x=fi.values, y=fi.index, orientation="h", title="What predicts spend (GB importance)",
                 labels={"x": "Importance", "y": ""}, color_discrete_sequence=[C[0]]), 330)

st.subheader("Segment-level forecast error (%)")
st.caption("Sum of out-of-fold predictions vs actual total demand, per segment and category")
chart(px.imshow(D["seg_err"].round(1), text_auto=True, color_continuous_scale="RdBu_r", zmin=-60, zmax=60, aspect="auto"), 300)

st.subheader("Average monthly demand by segment and category (sample base)")
m = D["monthly_total"]
fig = go.Figure()
for nm in core.PNAMES:
    fig.add_bar(x=m.index, y=m[nm], name=nm, marker_color=CAT_COLORS[nm])
fig.update_layout(barmode="stack", yaxis_title="$ per month")
chart(fig, 400)
insight("Demographics explain about 79% of the variance in total spend, and the relationship is non-linear. For the two segments that "
        "generate 90% of revenue, segment-level demand is forecast mostly within 10%. Wine and meat are the most predictable; gold and "
        "sweets are driven by impulse. Try **Demand planner** to forecast demand for a store's customer mix.")
