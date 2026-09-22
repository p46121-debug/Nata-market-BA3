# Nata Supermarkets: Customer Analytics App

An interactive Streamlit app for the *Nata Supermarkets* Business Analytics case. It presents the full data analysis and lets anyone use the three machine-learning models by entering inputs.

| Section | Pages |
|---|---|
| **Overview** | Executive summary · Data & cleaning |
| **Data analytics** | Customer insights (demographics, shopping habits, spend drivers, channels & deals, correlations, complaints) · Promotion performance · Live data explorer* |
| **ML models** | Customer segments (K-Means + PCA) · Response model results (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) · Demand forecasting (Ridge, Random Forest, Gradient Boosting) |
| **Try the models** | Customer predictor (all 3 models from one form) · Campaign profit simulator · Batch scoring (CSV upload/download) · Demand planner |
| **Strategy** | Recommendations, KPIs and business value |

\*The case data is copyrighted, so it is **not** stored in this repository. Upload `nata_supermarket_data.xlsx` in the app's sidebar to unlock the live explorer. Everything else runs from the saved models.

## Repository structure
```
app.py                 # entry point (navigation + sidebar)
app_utils.py           # model loading, chart theme, shared input form
nata_core.py           # cleaning, feature engineering, training, prediction
train_models.py        # rebuilds models/*.sav from the Excel file
views/                 # one file per page
models/                # trained models + pre-computed analytics (.sav, joblib)
requirements.txt       # pinned library versions (must match the .sav files)
.streamlit/config.toml # theme and upload limit
```

## How the models load
1. On start-up the app loads the four `.sav` files in `models/`.
2. If they cannot be loaded (missing, or a library-version mismatch), the app asks for the Excel file in the sidebar, retrains every model (about 1 minute) and caches the result.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain the models, put the Excel file at `data/nata_supermarket_data.xlsx`, then run `python train_models.py`. The `data/` folder is git-ignored.

## Key results
- The top 20% of customers produce 52% of spend; wine and meat make up 78% of the wallet.
- 4 segments: Affluent Premium Shoppers are 22% of customers and 51% of revenue.
- Response model: ROC-AUC 0.89 and 4.7x lift in the top decile. Targeting the top 19% of customers turns a -$2,719 mass mailing into +$1,160.
- Demand model: R² 0.79 for total spend from demographic data alone.
