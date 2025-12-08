"""
per_group_metrics.py
--------------------
Train HGB model (same pipeline as model_training_weather.py) and
report performance:
- globally on test set
- by Origin airport (top-K)
- by Operating Airline (top-K)
using the SAME random train/test split.

This is mainly for analysis / plots; not for deployment.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "flights_with_weather_sample.parquet")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

print("📂 Loading merged flights + weather ...")
df = pd.read_parquet(DATA_PATH)
print(f"✅ Loaded {len(df):,} rows")

# meta columns for grouping
df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
df["FlightDate_date"] = df["FlightDate"].dt.date
df["Origin_str"] = df["Origin"].astype("string")
df["Dest_str"] = df["Dest"].astype("string")

# drop all-NaN origin & dest weather cols
weather_prefixes = ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "wdir", "wpgt", "pres", "tsun"]
for col in weather_prefixes:
    if col in df.columns and df[col].isna().all():
        df.drop(columns=[col], inplace=True)
dest_weather_prefixes = [f"dest_{c}" for c in weather_prefixes]
for col in dest_weather_prefixes:
    if col in df.columns and df[col].isna().all():
        df.drop(columns=[col], inplace=True)

# features (same as model_training_weather.py)
base_feats = ["IATA_Code_Operating_Airline", "Origin", "Dest", "Month", "DayOfWeek"]
if "DepHour" in df.columns:
    base_feats.append("DepHour")

if "is_weekend" not in df.columns:
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(np.int8)
if "route_freq" not in df.columns:
    route_tmp = df["Origin_str"].fillna("NA") + "_" + df["Dest_str"].fillna("NA")
    df["route_freq"] = route_tmp.map(route_tmp.value_counts()).astype(np.int32)
eng_feats = ["is_weekend", "route_freq"]

candidate_weather = [
    "tavg", "tmin", "tmax", "prcp", "snow", "wspd", "wdir", "wpgt", "pres", "tsun",
    "dest_tavg", "dest_tmin", "dest_tmax", "dest_prcp", "dest_snow",
    "dest_wspd", "dest_wdir", "dest_wpgt", "dest_pres", "dest_tsun",
]
wx_feats = [c for c in candidate_weather if c in df.columns]

if "Delayed" not in df.columns:
    raise ValueError("Column 'Delayed' not found.")

# categorical encoding
cat_cols = ["IATA_Code_Operating_Airline", "Origin", "Dest"]
for c in cat_cols:
    if df[c].dtype == "object":
        df[c] = df[c].astype("category")
    if str(df[c].dtype) == "category":
        df[c] = df[c].cat.codes.astype("int16")
    else:
        df[c] = df[c].astype("int16")

all_feats = base_feats + eng_feats + wx_feats
print("Using features:")
print(all_feats)

X = df[all_feats].copy()
y = df["Delayed"].astype(np.int8)

print(f"Delay rate: {y.mean():.2%}")

# same random split as main script
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
test_idx = X_test.index  # for grouping later

# imputer
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# origin_day_volume
train_meta = pd.DataFrame({
    "Origin": df.loc[X_train.index, "Origin_str"].values,
    "FlightDate": df.loc[X_train.index, "FlightDate_date"].values
})
counts = train_meta.groupby(["Origin", "FlightDate"]).size().rename("origin_day_volume")

train_key = pd.MultiIndex.from_arrays(
    [train_meta["Origin"].values, train_meta["FlightDate"].values]
)
X_train_imp["origin_day_volume"] = counts.reindex(train_key).values

test_meta = pd.DataFrame({
    "Origin": df.loc[test_idx, "Origin_str"].values,
    "FlightDate": df.loc[test_idx, "FlightDate_date"].values
})
test_key = pd.MultiIndex.from_arrays(
    [test_meta["Origin"].values, test_meta["FlightDate"].values]
)
X_test_imp["origin_day_volume"] = counts.reindex(test_key).values

med_vol = float(np.nanmedian(X_train_imp["origin_day_volume"]))
X_train_imp["origin_day_volume"].fillna(med_vol, inplace=True)
X_test_imp["origin_day_volume"].fillna(med_vol, inplace=True)

# target encoding
def add_target_encoding(train_df, train_y, test_df, keys, new_col):
    tmp = train_df[keys].copy()
    tmp["y"] = train_y.values
    stats = tmp.groupby(keys)["y"].mean().rename(new_col)

    train_df[new_col] = train_df[keys].merge(
        stats, left_on=keys, right_index=True, how="left"
    )[new_col].values

    test_df[new_col] = test_df[keys].merge(
        stats, left_on=keys, right_index=True, how="left"
    )[new_col].values

    global_rate = float(train_y.mean())
    train_df[new_col].fillna(global_rate, inplace=True)
    test_df[new_col].fillna(global_rate, inplace=True)

add_target_encoding(X_train_imp, y_train, X_test_imp, ["Origin", "Month"], "te_origin_month")
add_target_encoding(X_train_imp, y_train, X_test_imp, ["Origin", "Dest"],  "te_route")

# HGB model
hgb = HistGradientBoostingClassifier(
    max_depth=7,
    learning_rate=0.08,
    max_iter=300,
    random_state=42,
)
hgb.fit(X_train_imp, y_train)
y_prob = hgb.predict_proba(X_test_imp)[:, 1]
y_pred_default = (y_prob >= 0.5).astype(int)

# globally find best-F1 threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
thresholds = np.append(thresholds, 1.0)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.nanargmax(f1s)
best_thr = float(thresholds[best_idx])

print("\n🌍 Global HGB performance on test set:")
print(f"Default thr=0.50: Acc={accuracy_score(y_test, y_pred_default):.3f} "
      f"P={precision_score(y_test, y_pred_default, zero_division=0):.3f} "
      f"R={recall_score(y_test, y_pred_default, zero_division=0):.3f} "
      f"F1={f1_score(y_test, y_pred_default, zero_division=0):.3f} "
      f"AUC={roc_auc_score(y_test, y_prob):.3f}")

y_pred_best = (y_prob >= best_thr).astype(int)
print(f"Best-F1 thr={best_thr:.3f}: Acc={accuracy_score(y_test, y_pred_best):.3f} "
      f"P={precision_score(y_test, y_pred_best, zero_division=0):.3f} "
      f"R={recall_score(y_test, y_pred_best, zero_division=0):.3f} "
      f"F1={f1_score(y_test, y_pred_best, zero_division=0):.3f} "
      f"AUC={roc_auc_score(y_test, y_prob):.3f} "
      f"AP={average_precision_score(y_test, y_prob):.3f}")

# -------- helper to compute metrics on a subset --------
def compute_subset_metrics(mask, name, thr):
    idx = np.where(mask)[0]
    if len(idx) < 200:
        print(f"  - {name}: skipped (only {len(idx)} samples)")
        return None

    y_true_sub = y_test.iloc[idx]
    y_prob_sub = y_prob[idx]
    y_pred_sub = (y_prob_sub >= thr).astype(int)

    acc = accuracy_score(y_true_sub, y_pred_sub)
    p = precision_score(y_true_sub, y_pred_sub, zero_division=0)
    r = recall_score(y_true_sub, y_pred_sub, zero_division=0)
    f1 = f1_score(y_true_sub, y_pred_sub, zero_division=0)
    try:
        auc = roc_auc_score(y_true_sub, y_prob_sub)
    except ValueError:
        auc = np.nan

    print(f"  - {name}: n={len(idx)}, Acc={acc:.3f}, P={p:.3f}, R={r:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    return {
        "group": name,
        "n": len(idx),
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "auc": auc,
    }

print("\n🛫 Per-Origin metrics (top 5 origins by traffic) at best-F1 threshold:")
test_meta_sub = df.loc[test_idx, :].copy()
origin_counts = test_meta_sub["Origin_str"].value_counts().head(5)

origin_results = []
for org in origin_counts.index:
    mask = (test_meta_sub["Origin_str"].values == org)
    res = compute_subset_metrics(mask, f"Origin={org}", best_thr)
    if res is not None:
        origin_results.append(res)

print("\n🛩️ Per-Airline metrics (top 5 operating airlines) at best-F1 threshold:")
airline_counts = test_meta_sub["IATA_Code_Operating_Airline"].value_counts().head(5)

airline_results = []
for code in airline_counts.index:
    mask = (test_meta_sub["IATA_Code_Operating_Airline"].values == code)
    res = compute_subset_metrics(mask, f"Airline_code={code}", best_thr)
    if res is not None:
        airline_results.append(res)

# You can optionally save these results to CSV for tables in the report
if origin_results:
    pd.DataFrame(origin_results).to_csv(
        os.path.join(PLOT_DIR, "per_origin_hgb_metrics.csv"), index=False
    )
if airline_results:
    pd.DataFrame(airline_results).to_csv(
        os.path.join(PLOT_DIR, "per_airline_hgb_metrics.csv"), index=False
    )

print("\n✅ per_group_metrics.py finished. CSVs (if any) written to outputs/plots/")
