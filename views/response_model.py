import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_utils import C, chart, fmt_money, get_artifacts, insight

R = get_artifacts()["results"]
M = R["response"]
T = M["table"]

st.title("Response model results")
st.caption("ML model 2: predicts who will accept a campaign. 75/25 stratified split with 5-fold cross-validation on the training set.")

c = st.columns([1.5, 1, 1, 1.2])
c[0].metric("Selected model", M["best"])
c[1].metric("Test ROC-AUC", f"{T.iloc[0]['Test ROC-AUC']:.3f}")
c[2].metric("Top-10% lift", f"{T.iloc[0]['Top-10% lift']:.1f}x")
c[3].metric("Out-of-fold AUC (all customers)", f"{M['oof_auc']:.3f}")

st.subheader("Model comparison")
st.dataframe(T.style.format("{:.3f}").highlight_max(axis=0, subset=["CV ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Top-10% lift"],
                                                    color="rgba(42,120,214,0.25)"))

a, b = st.columns(2)
with a:
    fig = go.Figure()
    for i, (name, df) in enumerate(M["roc"].items()):
        fig.add_scatter(x=df.fpr, y=df.tpr, mode="lines", name=f"{name} ({T.loc[name, 'Test ROC-AUC']:.2f})",
                        line=dict(color=C[i], width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="grey", dash="dash"), name="Random", showlegend=False)
    fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    chart(fig, 420)
with b:
    g = M["gains"]
    fig = go.Figure()
    fig.add_scatter(x=g.pct_contacted, y=g.pct_responders, mode="lines", line=dict(color=C[0], width=3), name=M["best"],
                    hovertemplate="Contact top %{x:.0f}% → reach %{y:.0f}% of responders<extra></extra>")
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color="grey", dash="dash"), name="Random targeting")
    fig.update_layout(title="Cumulative gains (test set)", xaxis_title="% of customers contacted",
                      yaxis_title="% of responders captured")
    chart(fig, 420)

a, b = st.columns([1, 2])
with a:
    cm = M["cm"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["Pred: no", "Pred: yes"], y=["Actual: no", "Actual: yes"],
                    title="Confusion matrix @ 0.5")
    fig.update_coloraxes(showscale=False)
    chart(fig, 340)
    tn, fp, fn, tp = cm.ravel()
    st.caption(f"Recall {tp / (tp + fn):.0%} · Precision {tp / (tp + fp):.0%} · Accuracy {(tp + tn) / cm.sum():.0%}")
with b:
    dec = M["deciles"]
    fig = px.bar(x=dec.index.astype(str), y=dec.Response_Rate, text=[f"{v:.0f}%" for v in dec.Response_Rate],
                 labels={"x": "Score decile (D1 = highest)", "y": "Actual response %"}, title="Response rate by score decile",
                 color_discrete_sequence=[C[0]])
    fig.add_hline(y=27.3, line_dash="dash", line_color=C[1], annotation_text="Break-even 27.3%")
    chart(fig, 340)

st.subheader("What drives response?")
a, b = st.columns(2)
with a:
    p = M["perm"].head(12)[::-1]
    fig = px.bar(x=p.values, y=p.index, orientation="h", labels={"x": "Mean drop in AUC when shuffled", "y": ""},
                 title="Permutation importance", color_discrete_sequence=[C[0]])
    chart(fig, 430)
with b:
    o = M["odds"]
    o = pd.concat([o.head(6), o.tail(8)])
    fig = px.bar(x=o.values, y=o.index, orientation="h", log_x=True, color=np.where(o.values < 1, "Lowers response", "Raises response"),
                 color_discrete_map={"Lowers response": C[1], "Raises response": C[0]},
                 labels={"x": "Odds ratio per 1 SD (log scale)", "y": "", "color": ""}, title="Logistic-regression odds ratios")
    fig.add_vline(x=1, line_color="grey")
    chart(fig, 430)

with st.expander("Explainable rule set (depth-3 decision tree) for campaign managers"):
    st.caption(f"Test ROC-AUC {M['tree_auc']:.2f}. It is weaker than the main model but can be applied without software.")
    st.code(M["tree_text"], language="text")

st.subheader("Does the targeting rule hold on unseen data?")
h = M["holdout"].copy()
h["Response rate %"] = 100 * h.Responders / h.Contacts
st.dataframe(h.style.format({"Profit $": fmt_money, "Response rate %": "{:.1f}"}), hide_index=True)
insight(f"The cut-off ({M['cut_tr']:.2f}) was chosen on training data only. On the untouched test set, profit moves from "
        f"-${abs(h['Profit $'].iloc[0]):,.0f} to +${h['Profit $'].iloc[1]:,.0f}, with {h.Responders.iloc[1] / h.Responders.iloc[0]:.0%} of responders still reached.")
