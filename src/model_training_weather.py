"""
model_training_weather.py
-------------------------
Pre-flight + daily weather modeling with:
- Robust categorical encoding
- Drop all-NaN weather columns (origin + dest)
- Train-only median imputation
- Engineered features:
    * is_weekend, route_freq
    * origin_day_volume (origin x day congestion proxy)
    * target encoding: te_origin_month, te_route
- Models:
    * Logistic Regression (balanced)
    * RandomForest (balanced)
    * HistGradientBoosting (HGB)
- Threshold sweep (best-F1 & high-recall point) + PR curves
- ROC curves + confusion matrix heatmaps
- Saves models, scaler, imputer

Input:
  ../outputs/flights_with_weather_sample.parquet
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)

# enable HGB
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier

# seaborn for heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

warnings.filterwarnings("ignore")

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "flights_with_weather_sample.parquet")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("📂 Loading merged flights + weather ...")
df = pd.read_parquet(DATA_PATH)
print(f"✅ Loaded {len(df):,} rows")

# Keep meta for engineered features
df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
df["FlightDate_date"] = df["FlightDate"].dt.date
df["Origin_str"] = df["Origin"].astype("string")
df["Dest_str"] = df["Dest"].astype("string")

# --------- Drop all-null weather columns ----------
weather_prefixes = ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "wdir", "wpgt", "pres", "tsun"]
for col in weather_prefixes:
    if col in df.columns and df[col].isna().all():
        print(f"⚠️ Dropping all-NaN weather column: {col}")
        df.drop(columns=[col], inplace=True)

# --------- Drop all-null DESTINATION weather columns ----------
dest_weather_prefixes = [f"dest_{c}" for c in weather_prefixes]
for col in dest_weather_prefixes:
    if col in df.columns and df[col].isna().all():
        print(f"⚠️ Dropping all-NaN destination weather column: {col}")
        df.drop(columns=[col], inplace=True)

# ---------- Base & engineered (basic) ----------
base_feats = ["IATA_Code_Operating_Airline", "Origin", "Dest", "Month", "DayOfWeek"]
# 如果之后重新跑 data_cleaning.py 生成了 DepHour，这里自动加上
if "DepHour" in df.columns:
    base_feats.append("DepHour")

if "is_weekend" not in df.columns:
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(np.int8)
if "route_freq" not in df.columns:
    route_tmp = df["Origin_str"].fillna("NA") + "_" + df["Dest_str"].fillna("NA")
    df["route_freq"] = route_tmp.map(route_tmp.value_counts()).astype(np.int32)
eng_feats = ["is_weekend", "route_freq"]

# Origin weather + Destination weather
candidate_weather = [
    # origin weather
    "tavg", "tmin", "tmax", "prcp", "snow", "wspd", "wdir", "wpgt", "pres", "tsun",
    # dest weather (from weather_merge.py)
    "dest_tavg", "dest_tmin", "dest_tmax", "dest_prcp", "dest_snow",
    "dest_wspd", "dest_wdir", "dest_wpgt", "dest_pres", "dest_tsun",
]
wx_feats = [c for c in candidate_weather if c in df.columns]

if "Delayed" not in df.columns:
    raise ValueError("Column 'Delayed' not found. Make sure merged data includes it.")

# ---------- Robust categorical encoding ----------
cat_cols = ["IATA_Code_Operating_Airline", "Origin", "Dest"]
for c in cat_cols:
    if df[c].dtype == "object":
        df[c] = df[c].astype("category")
    if str(df[c].dtype) == "category":
        df[c] = df[c].cat.codes.astype("int16")
    else:
        df[c] = df[c].astype("int16")

all_feats = base_feats + eng_feats + wx_feats
print("Using feature set:")
print(all_feats)

X = df[all_feats].copy()
y = df["Delayed"].astype(np.int8)

print(f"Delay rate: {y.mean():.2%}")

# ---------- Train/Test Split (random, stratified) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Imputation ----------
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)
X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)

# ---------- Additional engineered features AFTER imputation ----------
# A) origin_day_volume: flights per origin+date in train
train_meta = pd.DataFrame({
    "Origin": df.loc[X_train.index, "Origin_str"].values,
    "FlightDate": df.loc[X_train.index, "FlightDate_date"].values
})
counts = train_meta.groupby(["Origin", "FlightDate"]).size().rename("origin_day_volume")

# map to train
train_key = pd.MultiIndex.from_arrays(
    [train_meta["Origin"].values, train_meta["FlightDate"].values]
)
X_train_imp["origin_day_volume"] = counts.reindex(train_key).values

# map to test
test_meta = pd.DataFrame({
    "Origin": df.loc[X_test.index, "Origin_str"].values,
    "FlightDate": df.loc[X_test.index, "FlightDate_date"].values
})
test_key = pd.MultiIndex.from_arrays(
    [test_meta["Origin"].values, test_meta["FlightDate"].values]
)
X_test_imp["origin_day_volume"] = counts.reindex(test_key).values

# fill NaN with median (train only)
med_vol = float(np.nanmedian(X_train_imp["origin_day_volume"]))
X_train_imp["origin_day_volume"].fillna(med_vol, inplace=True)
X_test_imp["origin_day_volume"].fillna(med_vol, inplace=True)

# B) Target encoding (train stats → map to train/test; fallback to global mean)
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
y_pred_lr = logreg.predict(X_test_scaled)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]

# ---------- Random Forest ----------
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    max_features="sqrt",
)
rf.fit(X_train_imp, y_train)
y_pred_rf = rf.predict(X_test_imp)
y_prob_rf = rf.predict_proba(X_test_imp)[:, 1]

# ---------- HistGradientBoosting ----------
hgb = HistGradientBoostingClassifier(
    max_depth=7,
    learning_rate=0.08,
    max_iter=300,
    random_state=42,
)
hgb.fit(X_train_imp, y_train)
y_pred_hgb = hgb.predict(X_test_imp)
y_prob_hgb = hgb.predict_proba(X_test_imp)[:, 1]

# ---------- Evaluation ----------
def evaluate(y_true, y_pred, y_prob, name):
    print(f"\n📊 {name} Results:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"AUC:       {roc_auc_score(y_true, y_prob):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

evaluate(y_test, y_pred_lr,  y_prob_lr,  "Logistic Regression (weather + eng)")
evaluate(y_test, y_pred_rf,  y_prob_rf,  "Random Forest (weather + eng)")
evaluate(y_test, y_pred_hgb, y_prob_hgb, "HistGradientBoosting (weather + eng)")

# ---------- Threshold sweep + PR Curve ----------
def sweep_threshold(y_true, y_prob, name, target_recall=0.60):
    """Find best-F1 and a high-recall threshold; also plot PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print(f"\n🔍 Threshold sweep for {name} (AP={ap:.3f})")

    thresholds = np.append(thresholds, 1.0)  # match length

    # best F1
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.nanargmax(f1s)
    best_thr = float(thresholds[best_idx])
    print(f"   Best F1={f1s[best_idx]:.3f} at thr={best_thr:.3f} "
          f"(P={precisions[best_idx]:.3f}, R={recalls[best_idx]:.3f})")

    # high recall target
    hi_idx = None
    for i, r in enumerate(recalls):
        if r >= target_recall:
            hi_idx = i
            break
    if hi_idx is not None:
        thr_hi = float(thresholds[hi_idx])
        y_hat_hi = (y_prob >= thr_hi).astype(int)
        print(f"   Recall≥{target_recall:.2f} at thr={thr_hi:.3f}: "
              f"P={precision_score(y_true, y_hat_hi, zero_division=0):.3f} "
              f"R={recall_score(y_true, y_hat_hi):.3f} "
              f"F1={f1_score(y_true, y_hat_hi):.3f}")
    else:
        print(f"   No operating point reaches recall ≥ {target_recall:.2f}")

    # PR curve
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve - {name} (AP={ap:.3f})")
    plt.tight_layout()
    pr_path = os.path.join(PLOT_DIR, f"pr_curve_{name.replace(' ', '_').lower()}.png")
    plt.savefig(pr_path, dpi=180)
    plt.close()
    print(f"🖼️ Saved PR curve: {pr_path}")

sweep_threshold(y_test, y_prob_lr,  "logreg_weather_eng", target_recall=0.60)
sweep_threshold(y_test, y_prob_rf,  "rf_weather_eng",     target_recall=0.60)
sweep_threshold(y_test, y_prob_hgb, "hgb_weather_eng",    target_recall=0.60)

# ---------- Example calibrated RF at fixed threshold ----------
rf_thr = 0.35
y_pred_rf_cal = (y_prob_rf >= rf_thr).astype(int)
print(f"\n🎯 RF at thr={rf_thr:.2f}: "
      f"Acc={accuracy_score(y_test, y_pred_rf_cal):.3f} "
      f"P={precision_score(y_test, y_pred_rf_cal, zero_division=0):.3f} "
      f"R={recall_score(y_test, y_pred_rf_cal):.3f} "
      f"F1={f1_score(y_test, y_pred_rf_cal):.3f}")

# ---------- Feature Importances (RF) ----------
importances = rf.feature_importances_
fi = pd.Series(importances, index=X_train_imp.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
fi.plot(kind="barh")
plt.title("Random Forest Feature Importance (Weather + Engineered)")
plt.tight_layout()
fi_path = os.path.join(PLOT_DIR, "feature_importance_rf_weather.png")
plt.savefig(fi_path, dpi=180)
plt.close()
print(f"🖼️ Saved RF feature importance plot: {fi_path}")

# ---------- ROC Curves (per model) ----------
def plot_roc_single(y_true, y_prob, label, fname):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {label}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, fname)
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"🖼️ Saved ROC: {path}")

plot_roc_single(y_test, y_prob_lr,  "LogReg (weather+eng)", "roc_logreg_weather.png")
plot_roc_single(y_test, y_prob_rf,  "RF (weather+eng)",     "roc_rf_weather.png")
plot_roc_single(y_test, y_prob_hgb, "HGB (weather+eng)",    "roc_hgb_weather.png")

# ---------- Combined ROC ----------
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_hgb, tpr_hgb, _ = roc_curve(y_test, y_prob_hgb)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={roc_auc_score(y_test, y_prob_lr):.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_auc_score(y_test, y_prob_rf):.3f})")
plt.plot(fpr_hgb, tpr_hgb, label=f"HGB (AUC={roc_auc_score(y_test, y_prob_hgb):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Weather + Engineered Models")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(PLOT_DIR, "roc_curves_weather.png")
plt.savefig(roc_path, dpi=180)
plt.close()
print(f"🖼️ Saved ROC curves: {roc_path}")

# ---------- Confusion Matrix Heatmaps ----------
if HAS_SEABORN:
    for name, y_pred, fname in [
        ("LogReg (weather+eng)", y_pred_lr,  "cm_logreg_weather.png"),
        ("RF (weather+eng)",     y_pred_rf,  "cm_rf_weather.png"),
        ("HGB (weather+eng)",    y_pred_hgb, "cm_hgb_weather.png"),
    ]:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Delay", "Delay"],
            yticklabels=["No Delay", "Delay"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        cm_path = os.path.join(PLOT_DIR, fname)
        plt.savefig(cm_path, dpi=180)
        plt.close()
        print(f"🖼️ Saved confusion matrix heatmap: {cm_path}")
else:
    print("ℹ️ seaborn not installed, skipping confusion matrix heatmaps for weather models.")

# ---------- Save ----------
joblib.dump(logreg,  os.path.join(MODEL_DIR, "logreg_weather_balanced.joblib"))
joblib.dump(rf,      os.path.join(MODEL_DIR, "random_forest_weather_balanced.joblib"))
joblib.dump(hgb,     os.path.join(MODEL_DIR, "hgb_weather.joblib"))
joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler_weather.joblib"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer_weather.joblib"))
print(f"\n✅ Models, scaler & imputer saved to {MODEL_DIR}/")
