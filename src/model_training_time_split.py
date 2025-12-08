"""
model_training_time_split.py
----------------------------
Same feature pipeline as model_training_weather.py, but using a
TIME-BASED train/test split:

- Train: FlightDate < 2020-11-01 (roughly Jan–Oct 2020)
- Test:  FlightDate >= 2020-11-01 (Nov–Dec 2020)

This mimics "train on past, test on future".
We train Logistic Regression + HGB and compare metrics.
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "flights_with_weather_sample.parquet")

print("📂 Loading merged flights + weather ...")
df = pd.read_parquet(DATA_PATH)
print(f"✅ Loaded {len(df):,} rows")

# meta
df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
df["FlightDate_date"] = df["FlightDate"].dt.date
df["Origin_str"] = df["Origin"].astype("string")
df["Dest_str"] = df["Dest"].astype("string")

# drop all-NaN origin & dest weather columns
weather_prefixes = ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "wdir", "wpgt", "pres", "tsun"]
for col in weather_prefixes:
    if col in df.columns and df[col].isna().all():
        df.drop(columns=[col], inplace=True)
dest_weather_prefixes = [f"dest_{c}" for c in weather_prefixes]
for col in dest_weather_prefixes:
    if col in df.columns and df[col].isna().all():
        df.drop(columns=[col], inplace=True)

# features
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

# ---------- Time-based split ----------
cutoff = pd.Timestamp("2020-11-01")
train_idx = df["FlightDate"] < cutoff
test_idx = df["FlightDate"] >= cutoff

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")
print(f"Train delay rate: {y_train.mean():.2%}, Test delay rate: {y_test.mean():.2%}")

# ---------- Imputer ----------
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# origin_day_volume (train only → map to test)
train_meta = pd.DataFrame({
    "Origin": df.loc[train_idx, "Origin_str"].values,
    "FlightDate": df.loc[train_idx, "FlightDate_date"].values
})
train_meta.index = X_train.index

counts = train_meta.groupby(["Origin", "FlightDate"]).size().rename("origin_day_volume")

train_key = pd.MultiIndex.from_arrays(
    [train_meta["Origin"].values, train_meta["FlightDate"].values]
)
X_train_imp["origin_day_volume"] = counts.reindex(train_key).values

test_meta = pd.DataFrame({
    "Origin": df.loc[test_idx, "Origin_str"].values,
    "FlightDate": df.loc[test_idx, "FlightDate_date"].values
})
test_meta.index = X_test.index
test_key = pd.MultiIndex.from_arrays(
    [test_meta["Origin"].values, test_meta["FlightDate"].values]
)
X_test_imp["origin_day_volume"] = counts.reindex(test_key).values

med_vol = float(np.nanmedian(X_train_imp["origin_day_volume"]))
X_train_imp["origin_day_volume"].fillna(med_vol, inplace=True)
X_test_imp["origin_day_volume"].fillna(med_vol, inplace=True)

# target encoding (train stats only)
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

# ---------- Logistic Regression ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=None,
)
logreg.fit(X_train_scaled, y_train)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = (y_prob_lr >= 0.5).astype(int)

# ---------- HGB ----------
hgb = HistGradientBoostingClassifier(
    max_depth=7,
    learning_rate=0.08,
    max_iter=300,
    random_state=42,
)
hgb.fit(X_train_imp, y_train)
y_prob_hgb = hgb.predict_proba(X_test_imp)[:, 1]
y_pred_hgb = (y_prob_hgb >= 0.5).astype(int)

print("\n⏱️ Time-based split results (Test = Nov–Dec 2020):")

def report(name, y_true, y_pred, y_prob):
    print(f"\n📊 {name}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"AUC:       {roc_auc_score(y_true, y_prob):.3f}")
    print(f"AP:        {average_precision_score(y_true, y_prob):.3f}")

report("Logistic Regression (time split)", y_test, y_pred_lr, y_prob_lr)
report("HGB (time split)", y_test, y_pred_hgb, y_prob_hgb)

# Best-F1 thr for HGB (optional)
prec, rec, thr = precision_recall_curve(y_test, y_prob_hgb)
thr = np.append(thr, 1.0)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.nanargmax(f1s)
best_thr = float(thr[best_idx])
print(f"\n🔍 HGB best-F1 on time-split test: F1={f1s[best_idx]:.3f} at thr={best_thr:.3f} "
      f"(P={prec[best_idx]:.3f}, R={rec[best_idx]:.3f})")

print("\n✅ model_training_time_split.py finished.")
