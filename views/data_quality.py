import streamlit as st

from app_utils import get_artifacts

R = get_artifacts()["results"]
m = R["meta"]

st.title("Data & cleaning")
st.caption("Source: nata_supermarket_data.xlsx, sheet 'marketing' (Exhibit 1 of the case plus the three omitted fields)")

c = st.columns(4)
c[0].metric("Raw records", f"{m['n_raw']:,}")
c[1].metric("Clean records", f"{m['n_clean']:,}")
c[2].metric("Removed", f"{m['n_raw'] - m['n_clean']:,}")
c[3].metric("Raw fields", "29")

st.subheader("Data-quality issues and treatment")
st.dataframe(R["cleaning_log"], hide_index=True)

st.subheader("The three fields omitted from Exhibit 1")
st.markdown("""
| Field | Meaning | How it is used |
|---|---|---|
| `Z_CostContact` | Cost of contacting one customer in a campaign (= **\\$3** for all rows) | Campaign unit economics |
| `Z_Revenue` | Revenue earned per responding customer (= **\\$11** for all rows) | Campaign unit economics |
| `Response` | 1 if the customer accepted the **latest** campaign | Target variable of the response model |
""")

st.subheader("Engineered features")
st.markdown("""
- **Age** = 2014 - Year_Birth, calculated at the end of the data window rather than the 2022 case date.
- **Tenure_Months**: months from enrolment to 30-Jun-2014.
- **Children**, **Is_Parent** and **Partnered** (Married or Together).
- **Total_Spend**: the sum of the six categories. **Total_Purchases**: web + catalogue + store purchases. **Avg_Basket** = spend / purchases.
- **Deal_Share**, and each channel's share of purchases and each category's share of spend.
- **Prev_Accepted**: the number of the five earlier campaigns the customer accepted.
""")

with st.expander("Raw column audit (types, missing values, unique values)"):
    st.dataframe(R["raw_audit"])
st.caption(f"Models built with scikit-learn {m['sklearn']}, pandas {m['pandas']}, numpy {m['numpy']}.")
