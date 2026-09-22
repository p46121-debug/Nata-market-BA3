"""
Nata Supermarkets: Customer Analytics app
Entry point for Streamlit:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Nata Supermarkets Analytics", page_icon=":material/shopping_cart:", layout="wide")

from app_utils import get_artifacts, sidebar  # noqa: E402

sidebar()
get_artifacts()   # load (or rebuild) models once; cached for every page

pages = {
    "Overview": [
        st.Page("views/home.py", title="Executive summary", icon=":material/dashboard:", default=True),
        st.Page("views/data_quality.py", title="Data & cleaning", icon=":material/cleaning_services:"),
    ],
    "Data analytics": [
        st.Page("views/insights.py", title="Customer insights", icon=":material/insights:"),
        st.Page("views/campaigns.py", title="Promotion performance", icon=":material/campaign:"),
        st.Page("views/explorer.py", title="Live data explorer", icon=":material/table_view:"),
    ],
    "ML models": [
        st.Page("views/segmentation.py", title="Customer segments", icon=":material/groups:"),
        st.Page("views/response_model.py", title="Response model results", icon=":material/model_training:"),
        st.Page("views/demand.py", title="Demand forecasting", icon=":material/inventory_2:"),
    ],
    "Try the models": [
        st.Page("views/predictor.py", title="Customer predictor", icon=":material/person_search:"),
        st.Page("views/profit_sim.py", title="Campaign profit simulator", icon=":material/calculate:"),
        st.Page("views/batch.py", title="Batch scoring (CSV)", icon=":material/upload_file:"),
        st.Page("views/demand_planner.py", title="Demand planner", icon=":material/store:"),
    ],
    "Strategy": [
        st.Page("views/recommendations.py", title="Recommendations", icon=":material/flag:"),
    ],
}
st.navigation(pages, expanded=True).run()
