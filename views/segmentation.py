import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nata_core as core
from app_utils import C, SEG_COLORS, SEQ, chart, get_artifacts

R = get_artifacts()["results"]
P, q = R["seg_profile"], R["seg_quality"]

st.title("Customer segments")
st.caption("ML model 1: K-Means clustering on 15 standardised behavioural and demographic features")

c = st.columns(4)
c[0].metric("Segments (k)", "4")
c[1].metric("Silhouette score", f"{q['silhouette']:.3f}", help="Soft boundaries are typical for retail behaviour data")
c[2].metric("Agreement with Ward clustering (ARI)", f"{q['ari_ward']:.2f}")
c[3].metric("Top-2 segments' revenue", f"{P['Share of revenue %'].iloc[:2].sum():.0f}%")

with st.expander("How was k chosen?"):
    e = R["elbow"]
    a, b = st.columns(2)
    with a:
        chart(px.line(e, x="k", y="inertia", markers=True, title="Elbow method", color_discrete_sequence=[C[0]]), 300)
    with b:
        chart(px.line(e, x="k", y="silhouette", markers=True, title="Silhouette score", color_discrete_sequence=[C[0]]), 300)
    st.markdown("The elbow bends at k = 3-4. k = 2 only separates high spenders from low spenders, so k = 4 was chosen "
                "because it gives distinct, actionable personas.")

a, b = st.columns([1.2, 1])
with a:
    pts = R["pca_points"]
    fig = px.scatter(pts, x="PC1", y="PC2", color="Segment", color_discrete_map=SEG_COLORS, opacity=0.6,
                     category_orders={"Segment": core.PERSONA_NAMES},
                     title=f"Segments in PCA space ({100 * R['pca_var'].sum():.0f}% of variance shown)")
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    chart(fig, 460)
with b:
    fig = go.Figure()
    fig.add_bar(x=P.index, y=P["Share of customers %"], name="% of customers", marker_color=C[0])
    fig.add_bar(x=P.index, y=P["Share of revenue %"], name="% of revenue", marker_color=C[1])
    fig.update_layout(barmode="group", title="Customer share vs revenue share", yaxis_title="%")
    chart(fig, 460)

st.subheader("Segment profiles")
show = P[["Customers", "Share of customers %", "Share of revenue %", "Income", "Age", "Children", "Total_Spend",
          "Avg_Basket", "Deal_Share", "Web_Visits", "Prev_Accepted", "Response"]]
st.dataframe(show.style.format({"Customers": "{:,.0f}", "Share of customers %": "{:.1f}", "Share of revenue %": "{:.1f}",
                                "Income": "${:,.0f}", "Age": "{:.0f}", "Children": "{:.2f}", "Total_Spend": "${:,.0f}",
                                "Avg_Basket": "${:,.0f}", "Deal_Share": "{:.0%}", "Web_Visits": "{:.1f}",
                                "Prev_Accepted": "{:.2f}", "Response": "{:.1%}"}))

a, b = st.columns(2)
with a:
    chart(px.imshow(R["seg_cat_index"].round(0), text_auto=True, color_continuous_scale=SEQ, aspect="auto",
                    title="Category spend index (100 = average customer)"), 330)
with b:
    chart(px.imshow(R["seg_channel_index"].round(0), text_auto=True, color_continuous_scale=SEQ, aspect="auto",
                    title="Channel & deal index (100 = average customer)"), 330)

st.subheader("Playbook by segment")
cols = st.columns(4)
for col, name in zip(cols, core.PERSONA_NAMES):
    pb = core.PERSONA_PLAYBOOK[name]
    with col.container(border=True):
        st.markdown(f"<span style='color:{SEG_COLORS[name]};font-size:1.3em'>●</span> **{name}**", unsafe_allow_html=True)
        st.caption(f"{P.loc[name, 'Share of customers %']:.0f}% of customers · {P.loc[name, 'Share of revenue %']:.0f}% of revenue · "
                   f"{100 * P.loc[name, 'Response']:.0f}% response")
        st.markdown(f"{pb['profile']}\n\n**Offer:** {pb['offer']}\n\n**Channel:** {pb['channel']}\n\n**Campaign:** {pb['campaign']}")
st.caption("To find the segment of a new customer, use **Try the models → Customer predictor**.")
