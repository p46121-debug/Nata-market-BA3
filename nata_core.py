"""
nata_core.py - shared logic for the Nata Supermarkets analytics app.

Cleaning, feature engineering, model training and prediction helpers.
Every result reproduces the analysis notebook (same cleaning rules, features, hyper-parameters and random seeds).
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (adjusted_rand_score, average_precision_score, confusion_matrix,
                             mean_absolute_error, r2_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

RANDOM_STATE = 42
COST, REV = 3, 11                      # Z_CostContact, Z_Revenue from the data file
DATA_END = pd.Timestamp("2014-06-30")  # end of data window (tenure reference)

PRODUCTS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PNAMES = ["Wine", "Fruit", "Meat", "Fish", "Sweets", "Gold"]
CHANNELS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CMPS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
EDUCATION_LEVELS = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
MARITAL_LEVELS = ["Married", "Together", "Single", "Divorced", "Widow"]

SEG_FEATS = ["Income", "Total_Spend", "Total_Purchases", "Avg_Basket", "Deal_Share", "Web_Share",
             "Catalog_Share", "Store_Share", "NumWebVisitsMonth", "Wine_Share", "Meat_Share", "Children",
             "Age", "Recency", "Tenure_Months"]
RESP_NUM = ["Age", "Income", "Kidhome", "Teenhome", "Tenure_Months", "Recency"] + PRODUCTS + \
           ["NumDealsPurchases"] + CHANNELS + ["NumWebVisitsMonth"] + CMPS + ["Complain", "Deal_Share"]
RESP_CAT = ["Education", "Marital_Status"]
DEM_FEATS = ["Age", "Income", "Kidhome", "Teenhome", "Tenure_Months", "Partnered", "Education"]

PERSONA_NAMES = ["Affluent Premium Shoppers", "Mid-Income Family Regulars",
                 "Price-Sensitive Families", "Young Budget Browsers"]
PERSONA_PLAYBOOK = {
    "Affluent Premium Shoppers": {
        "profile": "High income, few or no children, large baskets, heavy catalogue and meat/wine buyers, rarely use deals.",
        "offer": "Premium and loyalty rewards (early access to fine wine, premium meat). Avoid discounts, which only erode margin.",
        "channel": "Catalogue and personal outreach",
        "campaign": "Campaigns 1 and 5 style (22-26% acceptance in this segment)"},
    "Mid-Income Family Regulars": {
        "profile": "Mid income, about 1 child, wine-led baskets, most active web buyers, moderate deal use.",
        "offer": "Wine bundles, family-meal packs, points on repeat purchases.",
        "channel": "Web and email",
        "campaign": "Campaign 4 style (15% acceptance in this segment)"},
    "Price-Sensitive Families": {
        "profile": "Lower-mid income, about 2 children, 40% of purchases on deal, browse more than they buy.",
        "offer": "Low-cost digital coupons on staples only. Keep them out of paid mailings.",
        "channel": "Web and app (zero marginal cost)",
        "campaign": "Digital-only; score-based exceptions"},
    "Young Budget Browsers": {
        "profile": "Youngest and lowest-income group, highest web visits, gold-leaning, small baskets.",
        "offer": "Web conversion nudges: personalised web offers, click-and-collect, app onboarding.",
        "channel": "Web and app",
        "campaign": "Campaign 3 style (the only campaign that reached them)"},
}

# --------------------------------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------------------------------
def load_raw(src) -> pd.DataFrame:
    """Read the 'marketing' sheet from a path, bytes or file-like object."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    return pd.read_excel(src, sheet_name="marketing")


def _parse_date(v):
    if isinstance(v, str):
        return pd.to_datetime(v, format="%d-%m-%Y")
    return pd.Timestamp(year=v.year, month=v.day, day=v.month)   # Excel swapped day/month


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the notebook's cleaning rules. Returns (clean_df, cleaning_log)."""
    df = raw.copy()
    log = []
    df["Dt_Customer"] = df["Dt_Customer"].map(_parse_date)
    n_swapped = int((raw["Dt_Customer"].map(lambda v: not isinstance(v, str))).sum())
    log.append(("Mixed date formats in Dt_Customer", f"{n_swapped} Excel dates with day/month swapped",
                "Parsed and corrected (range 30-Jul-2012 to 29-Jun-2014)", 0))
    dup = df.drop(columns="ID").duplicated()
    df = df[~dup]
    log.append(("Duplicate customers", "Identical on every field except ID", "Dropped", int(dup.sum())))
    bad_age = df.Year_Birth < 1920
    df = df[~bad_age]
    log.append(("Impossible birth years", "1893, 1899, 1900", "Dropped", int(bad_age.sum())))
    bad_inc = df.Income >= 600_000
    df = df[~bad_inc.fillna(False)]
    log.append(("Placeholder income", "666,666", "Dropped", int(bad_inc.fillna(False).sum())))
    odd = df.Marital_Status.isin(["Alone", "Absurd", "YOLO"])
    df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single", "Absurd": "Single", "YOLO": "Single"})
    log.append(("Invalid marital labels", "Alone / Absurd / YOLO", "Recoded to Single", 0))
    miss = int(df.Income.isna().sum())
    df["Income"] = df.groupby("Education")["Income"].transform(lambda s: s.fillna(s.median()))
    log.append(("Missing income", f"{miss} rows", "Imputed with education-group median", 0))
    df = df.drop(columns=[c for c in ["Z_CostContact", "Z_Revenue"] if c in df.columns]).reset_index(drop=True)
    log.append(("Constant columns", "Z_CostContact = 3, Z_Revenue = 11", "Used as campaign unit economics", 0))
    log = pd.DataFrame(log, columns=["Issue", "Evidence", "Treatment", "Rows removed"])
    return add_features(df), log


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering. Works on the cleaned dataset and on user-entered customers.
    Accepts either Year_Birth or Age, and either Dt_Customer or Tenure_Months."""
    df = df.copy()
    if "Age" not in df.columns:
        df["Age"] = 2014 - df["Year_Birth"]
    if "Tenure_Months" not in df.columns:
        df["Tenure_Months"] = ((DATA_END - pd.to_datetime(df.Dt_Customer)).dt.days / 30.44).round(1)
    for c in CMPS + ["Complain"]:
        if c not in df.columns:
            df[c] = 0
    df["Age_Group"] = pd.cut(df.Age, [0, 34, 44, 54, 64, 200], labels=["<35", "35-44", "45-54", "55-64", "65+"])
    df["Children"] = df.Kidhome + df.Teenhome
    df["Is_Parent"] = (df.Children > 0).astype(int)
    df["Partnered"] = df.Marital_Status.isin(["Married", "Together"]).astype(int)
    df["Total_Spend"] = df[PRODUCTS].sum(axis=1)
    df["Total_Purchases"] = df[CHANNELS].sum(axis=1)
    tp = df.Total_Purchases.replace(0, np.nan)
    ts = df.Total_Spend.replace(0, np.nan)
    df["Avg_Basket"] = (df.Total_Spend / tp).fillna(0)
    df["Deal_Share"] = (df.NumDealsPurchases / tp).fillna(0).clip(0, 1)
    for ch, nm in zip(CHANNELS, ["Web", "Catalog", "Store"]):
        df[f"{nm}_Share"] = (df[ch] / tp).fillna(0)
    for p, nm in zip(PRODUCTS, PNAMES):
        df[f"{nm}_Share"] = (df[p] / ts).fillna(0)
    df["Prev_Accepted"] = df[CMPS].sum(axis=1)
    return df


# --------------------------------------------------------------------------------------------------
# Training - builds every artefact the app needs
# --------------------------------------------------------------------------------------------------
def _hist(x, bins):
    c, e = np.histogram(x, bins=bins)
    return pd.DataFrame({"left": e[:-1], "right": e[1:], "count": c})


def train_all(raw: pd.DataFrame, progress=lambda msg, pct: None) -> dict:
    """Clean the raw data, train all models and pre-compute every analytics table.
    Returns a dict with keys: results, segmentation, response, demand."""
    progress("Cleaning data", 0.03)
    df, log = clean(raw)
    R = {"meta": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                  "n_raw": len(raw), "n_clean": len(df)}}
    R["cleaning_log"] = log
    R["raw_audit"] = pd.DataFrame({"dtype": raw.dtypes.astype(str), "missing": raw.isna().sum(),
                                   "unique": raw.nunique()})

    # ---------------- Descriptive analytics ----------------
    progress("Descriptive analytics", 0.08)
    R["kpis"] = {
        "customers": len(df), "median_age": float(df.Age.median()), "median_income": float(df.Income.median()),
        "pct_degree": 100 * df.Education.isin(["Graduation", "Master", "PhD"]).mean(),
        "pct_partnered": 100 * df.Partnered.mean(), "pct_parents": 100 * df.Is_Parent.mean(),
        "pct_complain": 100 * df.Complain.mean(), "response_rate": 100 * df.Response.mean(),
        "total_spend": float(df.Total_Spend.sum()), "mean_spend": float(df.Total_Spend.mean()),
    }
    R["age_hist"] = _hist(df.Age, 25)
    R["income_hist"] = _hist(df.Income, 30)
    R["spend_hist"] = _hist(df.Total_Spend, 30)
    R["education_counts"] = df.Education.value_counts()
    R["marital_counts"] = df.Marital_Status.value_counts()
    R["kids_counts"] = df.Children.value_counts().sort_index()
    wallet = df[PRODUCTS].sum().set_axis(PNAMES)
    R["wallet"] = pd.DataFrame({"Spend": wallet, "Share %": 100 * wallet / wallet.sum(),
                                "Customers buying %": 100 * (df[PRODUCTS] > 0).mean().set_axis(PNAMES).values})
    ch = df[CHANNELS].sum().set_axis(["Web", "Catalog", "Store"])
    R["channels"] = pd.DataFrame({"Purchases": ch, "Share %": 100 * ch / ch.sum()})
    s = df.Total_Spend.sort_values(ascending=False).values
    cum = s.cumsum() / s.sum()
    idx = np.linspace(0, len(s) - 1, 101).astype(int)
    R["lorenz"] = pd.DataFrame({"pct_customers": 100 * (idx + 1) / len(s), "pct_spend": 100 * cum[idx]})
    R["top20_share"] = 100 * cum[int(0.2 * len(s)) - 1]
    H, xe, ye = np.histogram2d(df.Income, df.Total_Spend, bins=[40, 40])
    R["income_spend_2d"] = {"H": H, "xe": xe, "ye": ye}
    R["spend_by"] = {
        "Children": df.groupby("Children").Total_Spend.mean(),
        "Age group": df.groupby("Age_Group", observed=True).Total_Spend.mean(),
        "Education": df.groupby("Education").Total_Spend.mean().reindex(EDUCATION_LEVELS),
        "Marital status": df.groupby("Marital_Status").Total_Spend.mean(),
    }
    df["Income_Band"] = pd.qcut(df.Income, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    R["cat_profile"] = {
        "Education": df.groupby("Education")[PRODUCTS].mean().reindex(EDUCATION_LEVELS).set_axis(PNAMES, axis=1),
        "Children": df.groupby("Children")[PRODUCTS].mean().set_axis(PNAMES, axis=1),
        "Income band": df.groupby("Income_Band", observed=True)[PRODUCTS].mean().set_axis(PNAMES, axis=1),
        "Age group": df.groupby("Age_Group", observed=True)[PRODUCTS].mean().set_axis(PNAMES, axis=1),
    }
    R["deal_by_children"] = df.groupby("Children")[["Deal_Share", "NumWebVisitsMonth", "Avg_Basket"]].mean()
    R["channel_by_income"] = df.groupby("Income_Band", observed=True)[["Web_Share", "Catalog_Share", "Store_Share", "Deal_Share"]].mean()
    corr_cols = ["Age", "Income", "Children", "Tenure_Months", "Recency"] + PRODUCTS + ["NumDealsPurchases"] + \
                CHANNELS + ["NumWebVisitsMonth", "Prev_Accepted", "Response"]
    R["corr"] = df[corr_cols].corr(method="spearman")
    R["tests"] = {
        "income_spend_rho": stats.spearmanr(df.Income, df.Total_Spend)[0],
        "income_deal_rho": stats.spearmanr(df.Income, df.Deal_Share)[0],
        "webvisits_spend_rho": stats.spearmanr(df.NumWebVisitsMonth, df.Total_Spend)[0],
        "kw_children_p": stats.kruskal(*[g.Total_Spend for _, g in df.groupby("Children")]).pvalue,
        "kw_age_p": stats.kruskal(*[g.Total_Spend for _, g in df.groupby("Age_Group", observed=True)]).pvalue,
        "kw_edu_p": stats.kruskal(*[g.Total_Spend for _, g in df.groupby("Education")]).pvalue,
    }
    # complaints
    R["complaints"] = df.groupby("Complain").agg(Customers=("ID", "size"), Mean_Spend=("Total_Spend", "mean"),
                                                 Response_Rate=("Response", "mean"), Mean_Income=("Income", "mean"),
                                                 Mean_Tenure=("Tenure_Months", "mean"))
    R["complaints_by"] = {
        "Education": 100 * df.groupby("Education").Complain.mean().reindex(EDUCATION_LEVELS),
        "Age group": 100 * df.groupby("Age_Group", observed=True).Complain.mean(),
    }
    R["tests"]["fisher_complain_response_p"] = stats.fisher_exact(pd.crosstab(df.Complain, df.Response))[1]
    R["tests"]["mw_complain_spend_p"] = stats.mannwhitneyu(df.loc[df.Complain == 1, "Total_Spend"],
                                                           df.loc[df.Complain == 0, "Total_Spend"]).pvalue
    # campaigns
    camp = pd.DataFrame({"Acceptance %": 100 * df[CMPS + ["Response"]].mean().values},
                        index=["Cmp1", "Cmp2", "Cmp3", "Cmp4", "Cmp5", "Latest"])
    camp["Accepters"] = df[CMPS + ["Response"]].sum().values
    camp["Mean income of accepters"] = [df.loc[df[c] == 1, "Income"].mean() for c in CMPS + ["Response"]]
    camp["Mean spend of accepters"] = [df.loc[df[c] == 1, "Total_Spend"].mean() for c in CMPS + ["Response"]]
    camp["Mean children of accepters"] = [df.loc[df[c] == 1, "Children"].mean() for c in CMPS + ["Response"]]
    R["campaigns"] = camp
    R["resp_by_prev"] = df.groupby("Prev_Accepted").Response.agg(["mean", "size"])
    R["resp_by"] = {
        "Children": 100 * df.groupby("Children").Response.mean(),
        "Marital status": 100 * df.groupby("Marital_Status").Response.mean(),
        "Education": 100 * df.groupby("Education").Response.mean().reindex(EDUCATION_LEVELS),
        "Income band": 100 * df.groupby("Income_Band", observed=True).Response.mean(),
        "Recency (days)": 100 * df.groupby(pd.cut(df.Recency, [-1, 24, 49, 74, 99],
                                                  labels=["0-24", "25-49", "50-74", "75-99"]), observed=True).Response.mean(),
    }
    enrol = df.set_index("Dt_Customer").resample("MS").size()
    R["enrolment"] = enrol

    # ---------------- ML 1: segmentation ----------------
    progress("Segmentation (K-Means)", 0.18)
    seg_scaler = StandardScaler().fit(df[SEG_FEATS])
    Xs = seg_scaler.transform(df[SEG_FEATS])
    elbow = []
    for k in range(2, 9):
        km_k = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(Xs)
        elbow.append({"k": k, "inertia": km_k.inertia_, "silhouette": silhouette_score(Xs, km_k.labels_)})
    R["elbow"] = pd.DataFrame(elbow)
    km = KMeans(n_clusters=4, n_init=50, random_state=RANDOM_STATE).fit(Xs)
    df["Cluster"] = km.labels_
    ward = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Xs)
    R["seg_quality"] = {"silhouette": silhouette_score(Xs, km.labels_), "ari_ward": adjusted_rand_score(km.labels_, ward)}
    order = df.groupby("Cluster").Total_Spend.mean().sort_values(ascending=False).index.tolist()
    persona = {int(c): PERSONA_NAMES[i] for i, c in enumerate(order)}
    df["Segment"] = df.Cluster.map(persona)
    cprof = df.groupby("Segment").agg(
        Customers=("ID", "size"), Income=("Income", "mean"), Age=("Age", "mean"), Children=("Children", "mean"),
        Total_Spend=("Total_Spend", "mean"), Purchases=("Total_Purchases", "mean"), Avg_Basket=("Avg_Basket", "mean"),
        Deal_Share=("Deal_Share", "mean"), Web_Visits=("NumWebVisitsMonth", "mean"), Web=("Web_Share", "mean"),
        Catalog=("Catalog_Share", "mean"), Store=("Store_Share", "mean"), Wine=("Wine_Share", "mean"),
        Meat=("Meat_Share", "mean"), Gold=("Gold_Share", "mean"), Prev_Accepted=("Prev_Accepted", "mean"),
        Response=("Response", "mean"), Complain=("Complain", "mean")).loc[PERSONA_NAMES]
    cprof["Share of customers %"] = 100 * cprof.Customers / cprof.Customers.sum()
    cprof["Share of revenue %"] = 100 * df.groupby("Segment").Total_Spend.sum().loc[PERSONA_NAMES] / df.Total_Spend.sum()
    R["seg_profile"] = cprof
    cat_idx = df.groupby("Segment")[PRODUCTS].mean().loc[PERSONA_NAMES].set_axis(PNAMES, axis=1)
    R["seg_cat_index"] = 100 * cat_idx / df[PRODUCTS].mean().values
    chc = CHANNELS + ["NumDealsPurchases", "NumWebVisitsMonth"]
    ch_idx = df.groupby("Segment")[chc].mean().loc[PERSONA_NAMES]
    ch_idx = 100 * ch_idx / df[chc].mean().values
    ch_idx.columns = ["Web buys", "Catalog buys", "Store buys", "Deal buys", "Web visits"]
    R["seg_channel_index"] = ch_idx
    sc = 100 * df.groupby("Segment")[CMPS + ["Response"]].mean().loc[PERSONA_NAMES]
    sc.columns = ["Cmp1", "Cmp2", "Cmp3", "Cmp4", "Cmp5", "Latest"]
    R["seg_campaign"] = sc
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs)
    P = pca.transform(Xs)
    R["pca_points"] = pd.DataFrame({"PC1": P[:, 0].round(3), "PC2": P[:, 1].round(3), "Segment": df.Segment.values})
    R["pca_var"] = pca.explained_variance_ratio_
    # typical (median) customer of each segment -> presets for the input forms
    input_cols = ["Age", "Income", "Kidhome", "Teenhome", "Tenure_Months", "Recency"] + PRODUCTS + \
                 ["NumDealsPurchases"] + CHANNELS + ["NumWebVisitsMonth"]
    presets = {}
    for sname, g in df.groupby("Segment"):
        p = g[input_cols].median().round(0).to_dict()
        p["Education"] = g.Education.mode()[0]
        p["Marital_Status"] = g.Marital_Status.mode()[0]
        p.update({c: 0 for c in CMPS}); p["Complain"] = 0
        presets[sname] = p
    overall = df[input_cols].median().round(0).to_dict()
    overall.update({"Education": "Graduation", "Marital_Status": "Married", **{c: 0 for c in CMPS}, "Complain": 0})
    presets = {"Typical customer (overall median)": overall, **{k: presets[k] for k in PERSONA_NAMES}}
    input_ranges = {c: (float(df[c].min()), float(df[c].max())) for c in input_cols}
    SEG = {"scaler": seg_scaler, "kmeans": km, "pca": pca, "persona_map": persona, "features": SEG_FEATS}
    R["presets"], R["input_ranges"] = presets, input_ranges

    # ---------------- ML 2: response model ----------------
    progress("Campaign-response models", 0.30)
    X = df[RESP_NUM + RESP_CAT]; y = df.Response
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    def pre():
        return ColumnTransformer([("num", StandardScaler(), RESP_NUM),
                                  ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), RESP_CAT)])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                                                        random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, roc, fitted, probs = [], {}, {}, {}
    for i, (name, mdl) in enumerate(models.items()):
        progress(f"Training {name}", 0.30 + 0.07 * i)
        pipe = Pipeline([("pre", pre()), ("clf", mdl)])
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr); p = pipe.predict_proba(X_te)[:, 1]
        fitted[name], probs[name] = pipe, p
        top = np.argsort(-p)[: int(0.1 * len(p))]
        rows.append({"Model": name, "CV ROC-AUC": cv_auc.mean(), "CV sd": cv_auc.std(),
                     "Test ROC-AUC": roc_auc_score(y_te, p), "Test PR-AUC": average_precision_score(y_te, p),
                     "Top-10% lift": y_te.iloc[top].mean() / y_te.mean()})
        fpr, tpr, _ = roc_curve(y_te, p)
        roc[name] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    table = pd.DataFrame(rows).set_index("Model").sort_values("Test ROC-AUC", ascending=False)
    best_name = table.index[0]
    p_best = probs[best_name]
    o = np.argsort(-p_best)
    gains = pd.DataFrame({"pct_contacted": 100 * np.arange(1, len(o) + 1) / len(o),
                          "pct_responders": 100 * np.cumsum(y_te.values[o]) / y_te.sum()})
    progress("Explaining the model", 0.60)
    pi = permutation_importance(fitted[best_name], X_te, y_te, n_repeats=20, random_state=RANDOM_STATE,
                                scoring="roc_auc", n_jobs=-1)
    perm = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
    lr_fit = fitted["Logistic Regression"]
    fn = [n.split("__", 1)[1] for n in lr_fit.named_steps["pre"].get_feature_names_out()]
    odds = pd.Series(np.exp(lr_fit.named_steps["clf"].coef_[0]), index=fn).sort_values()
    # rule tree
    extra = df[["Prev_Accepted", "Children", "Total_Spend", "Partnered"]]
    tree_feats = ["Prev_Accepted", "Recency", "Tenure_Months", "Income", "Children", "Teenhome", "NumCatalogPurchases",
                  "NumStorePurchases", "MntMeatProducts", "MntWines", "Total_Spend", "Partnered"]
    rule_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, class_weight="balanced", random_state=RANDOM_STATE)
    rule_tree.fit(X_tr.join(extra)[tree_feats], y_tr)
    tree_auc = roc_auc_score(y_te, rule_tree.predict_proba(X_te.join(extra)[tree_feats])[:, 1])
    tree_text = export_text(rule_tree, feature_names=tree_feats, decimals=0)
    cm = confusion_matrix(y_te, (p_best >= 0.5).astype(int))
    # out-of-fold scores for the whole base -> profit curve & deciles
    progress("Out-of-fold scoring & profit curve", 0.68)
    oof = cross_val_predict(Pipeline([("pre", pre()), ("clf", models[best_name])]), X, y, cv=cv, method="predict_proba")[:, 1]
    oo = np.argsort(-oof); ys = y.values[oo]
    k = np.arange(1, len(oo) + 1)
    profit = REV * np.cumsum(ys) - COST * k
    kbest = int(profit.argmax()) + 1
    cutoff = float(oof[oo][kbest - 1])
    # hold-out check of the cut-off rule
    oof_tr = cross_val_predict(Pipeline([("pre", pre()), ("clf", models[best_name])]), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    otr = np.argsort(-oof_tr); ptr = REV * np.cumsum(y_tr.values[otr]) - COST * np.arange(1, len(otr) + 1)
    cut_tr = oof_tr[otr][ptr.argmax()]
    sel = p_best >= cut_tr
    holdout = pd.DataFrame({"Strategy": ["Mass mailing (test set)", "Model-targeted (test set)"],
                            "Contacts": [len(y_te), int(sel.sum())], "Responders": [int(y_te.sum()), int(y_te[sel].sum())],
                            "Profit $": [REV * y_te.sum() - COST * len(y_te), REV * y_te[sel].sum() - COST * sel.sum()]})
    dec = pd.qcut(pd.Series(oof).rank(method="first", ascending=False), 10, labels=[f"D{i}" for i in range(1, 11)])
    deciles = pd.DataFrame({"Decile": dec, "y": y.values, "score": oof}).groupby("Decile", observed=True).agg(
        Customers=("y", "size"), Response_Rate=("y", "mean"), Min_Score=("score", "min"), Max_Score=("score", "max"))
    deciles["Lift"] = deciles.Response_Rate / y.mean()
    deciles["Response_Rate"] *= 100
    seg_target = pd.DataFrame({"Segment": df.Segment, "Targeted": oof >= cutoff, "Response": y}).groupby("Segment").mean().loc[PERSONA_NAMES] * 100
    # deployment model: best model refit on ALL customers
    progress("Fitting deployment model", 0.74)
    deploy = Pipeline([("pre", pre()), ("clf", models[best_name])]).fit(X, y)
    RESP = {"model": deploy, "model_name": best_name, "num": RESP_NUM, "cat": RESP_CAT, "cutoff": cutoff,
            "score_distribution": np.sort(oof)}
    R["response"] = {"table": table, "roc": roc, "gains": gains, "perm": perm, "odds": odds, "tree_text": tree_text,
                     "tree_auc": tree_auc, "cm": cm, "best": best_name, "oof": oof, "y": y.values,
                     "cutoff": cutoff, "k_best": kbest, "holdout": holdout, "cut_tr": cut_tr, "deciles": deciles,
                     "seg_target": seg_target, "oof_auc": roc_auc_score(y, oof)}

    # ---------------- ML 3: demand forecasting ----------------
    progress("Demand-forecasting models", 0.78)
    Xd = pd.get_dummies(df[DEM_FEATS], columns=["Education"], drop_first=True).astype(float)
    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    dem_rows, dem_pred, dem_models = [], {}, {}
    targets = list(zip(PRODUCTS + ["Total_Spend"], PNAMES + ["Total"]))
    for j, (col, nm) in enumerate(targets):
        progress(f"Demand model: {nm}", 0.78 + 0.03 * j)
        yv = df[col]
        r = {"Category": nm, "Mean 2-yr $": yv.mean()}
        cands = [("Ridge", Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])),
                 ("Random Forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)),
                 ("Gradient Boosting", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                                 subsample=0.8, random_state=RANDOM_STATE))]
        for mn, m_ in cands:
            pr = cross_val_predict(m_, Xd, yv, cv=kf)
            r[f"{mn} R²"] = r2_score(yv, pr)
            if mn == "Gradient Boosting":
                r["GB MAE $"] = mean_absolute_error(yv, pr); dem_pred[nm] = pr
        dem_rows.append(r)
        dem_models[nm] = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                                                   random_state=RANDOM_STATE).fit(Xd, yv)
    agg = pd.DataFrame({nm: dem_pred[nm] for nm in PNAMES}).assign(Segment=df.Segment.values).groupby("Segment").sum().loc[PERSONA_NAMES]
    act = df.groupby("Segment")[PRODUCTS].sum().loc[PERSONA_NAMES].set_axis(PNAMES, axis=1)
    fi = pd.Series(dem_models["Total"].feature_importances_, index=Xd.columns).sort_values(ascending=False)
    monthly_total = (df.groupby("Segment")[PRODUCTS].sum().loc[PERSONA_NAMES] / 24).set_axis(PNAMES, axis=1)
    monthly_per_cust = (df.groupby("Segment")[PRODUCTS].mean().loc[PERSONA_NAMES] / 24).set_axis(PNAMES, axis=1)
    DEM = {"models": dem_models, "columns": list(Xd.columns), "features": DEM_FEATS}
    R["demand"] = {"table": pd.DataFrame(dem_rows).set_index("Category"), "seg_err": 100 * (agg - act) / act,
                   "fi": fi, "monthly_total": monthly_total, "monthly_per_customer": monthly_per_cust,
                   "segment_mix": df.Segment.value_counts(normalize=True).loc[PERSONA_NAMES]}
    # cohort-generator reference distributions (aggregate only)
    R["cohort_ref"] = {"income_mean": float(df.Income.mean()), "income_sd": float(df.Income.std()),
                       "age_mean": float(df.Age.mean()), "age_sd": float(df.Age.std()),
                       "education_mix": df.Education.value_counts(normalize=True).reindex(EDUCATION_LEVELS).fillna(0).to_dict(),
                       "p_kid": float((df.Kidhome > 0).mean()), "p_teen": float((df.Teenhome > 0).mean()),
                       "p_partnered": float(df.Partnered.mean())}
    progress("Done", 1.0)
    return {"results": R, "segmentation": SEG, "response": RESP, "demand": DEM, "_clean_df": df}


# --------------------------------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------------------------------
def predict_segment(seg: dict, customers: pd.DataFrame) -> pd.DataFrame:
    X = seg["scaler"].transform(customers[seg["features"]])
    d = seg["kmeans"].transform(X)                       # distance to each centroid
    lab = seg["kmeans"].predict(X)
    out = pd.DataFrame({"Segment": [seg["persona_map"][int(l)] for l in lab]}, index=customers.index)
    for c, name in seg["persona_map"].items():
        out[f"dist::{name}"] = d[:, c]
    P = seg["pca"].transform(X)
    out["PC1"], out["PC2"] = P[:, 0], P[:, 1]
    return out


def predict_response(resp: dict, customers: pd.DataFrame) -> pd.DataFrame:
    X = customers[resp["num"] + resp["cat"]]
    p = resp["model"].predict_proba(X)[:, 1]
    dist = resp["score_distribution"]
    pct_above = 100 * (1 - np.searchsorted(dist, p, side="right") / len(dist))
    out = pd.DataFrame({"Response_Probability": p,
                        "Top_%_of_base": pct_above.clip(0.1, 100),
                        "Decile": np.clip(np.ceil(pct_above / 10), 1, 10).astype(int),
                        "Contact": p >= resp["cutoff"],
                        "Expected_Profit_$": p * REV - COST}, index=customers.index)
    return out


def response_contributions(resp: dict, customer: pd.DataFrame) -> pd.Series:
    """Per-feature log-odds contribution for one customer (logistic regression only)."""
    clf = resp["model"].named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.Series(dtype=float)
    pre = resp["model"].named_steps["pre"]
    z = pre.transform(customer[resp["num"] + resp["cat"]])
    z = z.toarray() if hasattr(z, "toarray") else z
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    return pd.Series(z[0] * clf.coef_[0], index=names)


def demand_matrix(dem: dict, customers: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(customers[dem["features"]], columns=["Education"]).reindex(columns=dem["columns"], fill_value=0).astype(float)
    return X


def predict_demand(dem: dict, customers: pd.DataFrame) -> pd.DataFrame:
    X = demand_matrix(dem, customers)
    out = pd.DataFrame({nm: np.clip(dem["models"][nm].predict(X), 0, None) for nm in PNAMES + ["Total"]}, index=customers.index)
    return out


def generate_cohort(n, income_mean, income_sd, age_min, age_max, p_kid, p_teen, p_partnered, education_mix,
                    tenure=12, seed=RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    edu = rng.choice(list(education_mix.keys()), size=n, p=np.array(list(education_mix.values())) / sum(education_mix.values()))
    return pd.DataFrame({
        "Age": rng.integers(age_min, age_max + 1, n),
        "Income": np.clip(rng.normal(income_mean, income_sd, n), 2_000, 170_000),
        "Kidhome": rng.binomial(1, p_kid, n), "Teenhome": rng.binomial(1, p_teen, n),
        "Tenure_Months": float(tenure), "Partnered": rng.binomial(1, p_partnered, n), "Education": edu,
    })


def batch_template() -> pd.DataFrame:
    cols = ["Customer_Ref", "Age", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome", "Tenure_Months",
            "Recency"] + PRODUCTS + ["NumDealsPurchases"] + CHANNELS + ["NumWebVisitsMonth"] + CMPS + ["Complain"]
    ex = [["C001", 45, "Graduation", "Married", 52000, 0, 1, 12, 30, 180, 8, 70, 12, 8, 25, 2, 4, 2, 5, 5, 0, 0, 0, 0, 0, 0],
          ["C002", 38, "PhD", "Single", 81000, 0, 0, 20, 10, 900, 60, 500, 90, 60, 60, 1, 6, 6, 9, 2, 0, 0, 0, 1, 1, 0],
          ["C003", 33, "2n Cycle", "Together", 28000, 1, 0, 6, 70, 10, 3, 12, 4, 2, 15, 2, 2, 0, 3, 8, 0, 0, 0, 0, 0, 0]]
    return pd.DataFrame(ex, columns=cols)


REQUIRED_BATCH = ["Age", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome", "Tenure_Months", "Recency"] + \
                 PRODUCTS + ["NumDealsPurchases"] + CHANNELS + ["NumWebVisitsMonth"]
