import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import C, SEQ, chart, get_artifacts, insight

R = get_artifacts()["results"]
camp = R["campaigns"]
y = R["response"]["y"]

st.title("Promotion performance")
st.caption("The five historical campaigns and the latest one (`Response`), plus their economics")

n, r = len(y), int(y.sum())
c = st.columns(4)
c[0].metric("Latest campaign response", f"{100 * r / n:.1f}%")
c[1].metric("Break-even response rate", "27.3%", help="\\$3 cost per contact / \\$11 revenue per response")
c[2].metric("Mass-mailing cost", f"${3 * n:,}")
c[3].metric("Mass-mailing profit", f"-${abs(11 * r - 3 * n):,}", delta="Loss-making", delta_color="inverse")

a, b = st.columns(2)
with a:
    fig = px.bar(camp, x=camp.index, y="Acceptance %", text=camp["Acceptance %"].map("{:.1f}%".format),
                 color=["Past"] * 5 + ["Latest"], color_discrete_map={"Past": C[0], "Latest": C[1]},
                 title="Acceptance rate by campaign")
    fig.add_hline(y=27.3, line_dash="dash", line_color="grey", annotation_text="Break-even 27.3%")
    fig.update_layout(xaxis_title="", showlegend=False)
    chart(fig)
with b:
    rp = R["resp_by_prev"]
    fig = px.bar(x=rp.index.astype(str), y=100 * rp["mean"], text=[f"{100 * v:.0f}% (n={s})" for v, s in zip(rp["mean"], rp["size"])],
                 labels={"x": "Past campaigns accepted", "y": "Latest-campaign response %"},
                 title="Past acceptors respond far more", color_discrete_sequence=[C[0]])
    chart(fig)

st.subheader("Who accepted each campaign?")
show = camp.copy()
st.dataframe(show.style.format({"Acceptance %": "{:.1f}", "Mean income of accepters": "${:,.0f}",
                                "Mean spend of accepters": "${:,.0f}", "Mean children of accepters": "{:.2f}"}))

st.subheader("Latest-campaign response rate by customer attribute")
cols = st.columns(len(R["resp_by"]))
for col, (k, s) in zip(cols, R["resp_by"].items()):
    with col:
        fig = px.bar(x=s.index.astype(str), y=s.values, labels={"x": k, "y": "Response %"}, color_discrete_sequence=[C[0]])
        fig.update_layout(title=k, margin=dict(t=40))
        chart(fig, 300, key=f"rb_{k}")

st.subheader("Campaign acceptance by segment (%)")
fig = px.imshow(R["seg_campaign"].round(1), text_auto=True, color_continuous_scale=SEQ, aspect="auto")
chart(fig, 300)

insight("Campaigns 1 and 5 appealed to affluent customers, Campaign 4 to families, and Campaign 3 was the only one to reach budget customers. "
        "Response is highest for recent shoppers, households without children, singles, PhDs and the top income quartile. "
        "One mailing list for every campaign wastes money, so targeting should be model-based and the offer should match the segment.")
