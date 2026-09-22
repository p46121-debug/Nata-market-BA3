import pandas as pd
import streamlit as st

st.title("Recommendations & business value")
st.caption("Each recommendation is tied to evidence in this app and to a KPI Nata can track")

recs = {
    "A. Promotion targeting": [
        ("Replace mass mailings with model-scored targeting. Contact only customers above the break-even score (about the top 20%).",
         "Mass campaign lost $2.7k; targeted campaign earns about $1.2k; 4.7x lift in the top decile", "Campaign ROI > 0; response > 27%"),
        ("Match the offer to the segment: premium and loyalty for Affluent Premium Shoppers, wine and family bundles for Mid-Income Family Regulars, "
         "digital coupons for the budget segments.", "Segment x campaign acceptance heat-map", "Acceptance by segment"),
        ("Trigger campaigns by recency: contact within 3-4 weeks of the last purchase, and run a win-back offer after 60 days.",
         "Response is 27% at recency of 24 days or less and 7% at 75 days or more", "Response by recency band"),
        ("Use past acceptors as the priority list and test cell for new offers.", "Response rises from 9% (no past acceptances) to 90% (four)", "Test-cell response"),
        ("Contact affluent customers by catalogue and personal outreach, and families by web and email.", "Catalogue index 234; web index 161", "Channel conversion"),
    ],
    "B. Demand forecasting & inventory": [
        ("Plan inventory by each store's segment mix using the demand planner.", "R² 0.79 for total spend; segment-level error mostly under 10%",
         "Stock-outs, markdowns, inventory turns"),
        ("Protect wine and meat availability with shelf space, supplier terms and safety stock.", "78% of the wallet; bought by more than 99% of customers",
         "Category availability %"),
        ("Rationalise fruit, fish and sweets SKUs, and treat gold as an impulse or gifting line.", "About 15% of spend; low predictability", "SKU productivity"),
        ("Capture dated POS transactions (basket, SKU, store, date).", "The data has no transaction dates", "Weekly SKU forecast MAPE"),
    ],
    "C. Customer growth & experience": [
        ("Fix web conversion with personalised offers, an app and click-and-collect.", "Web visits correlated -0.47 with spend",
         "Visit-to-purchase conversion"),
        ("Build a tiered loyalty programme for the top 20% of customers.", "The top 20% produce 52% of spend", "Top-quintile retention and spend"),
        ("Improve feedback capture (NPS/CSAT).", "Complaints (0.9%) are too rare to guide decisions", "NPS, complaint resolution time"),
    ],
}
for section, rows in recs.items():
    st.subheader(section)
    st.dataframe(pd.DataFrame(rows, columns=["Recommendation", "Evidence", "KPI"]), hide_index=True,
                 column_config={"Recommendation": st.column_config.TextColumn(width="large"),
                                "Evidence": st.column_config.TextColumn(width="medium")})

st.subheader("Business value: how this helps Nata compete")
c = st.columns(4)
items = [("Better decisions", "Every customer has a score (whom to contact), a segment (what to offer) and a forecast (what to stock). "
          "This is the analytics toolkit the case credits for Walmart's growth."),
         ("Immediate P&L impact", "Campaign ROI moves from about -44% to about +100%, and contact cost falls by about 80%. The savings can go into "
          "loyalty for the 22% of customers who drive 51% of revenue."),
         ("Leaner inventory", "Stock aligned to the wine- and meat-heavy demand of the top two segments means less waste and fewer stock-outs."),
         ("Repeatable capability", "The clean → segment → score → forecast pipeline can be refreshed each quarter (`train_models.py`) and "
          "extended to basket and churn analysis.")]
for col, (h, t) in zip(c, items):
    with col.container(border=True):
        st.markdown(f"**{h}**\n\n{t}")

st.subheader("Limitations")
st.markdown("""
- The data covers 2012-2014, while the case is set in 2022. Patterns should be re-validated on current data.
- The response model is trained on a single campaign (313 responders).
- Segment boundaries are soft (silhouette about 0.15).
- The profit analysis uses the dataset's fixed \\$3 and \\$11 values, not real margins. Use the profit simulator to test other values.
""")
