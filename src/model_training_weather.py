"""
model_training_weather.py
-------------------------
Pre-flight + daily weather modeling with:
- Robust categorical encoding
- Drop all-NaN weather columns
- Train-only median imputation
- Engineered features AFTER imputation:
    * origin_day_volume (origin x day congestion proxy)
    * target encoding: te_origin_month, te_route
- Models:
    * Logistic Regression (balanced)
    * RandomForest (balanced)
    * HistGradientBoosting (HGB)
- Threshold sweep (best-F1 & high-recall point) + PR curves
- ROC curves (individual & all models)
- Saves models, scaler, imputer

Input:
  ../outputs/flights_with_weather_sample.parquet

Outputs:
  ../outputs/plots/feature_importance_rf_weather.png
  ../outputs/plots/roc_curves_weather.png
  ../outputs/plots/roc_curves_weather_all.png
  ../outputs/plots/pr_curve_logreg_weather_eng.png
  ../outputs/plots/pr_curve_rf_weather_eng.png
  ../outputs/plots/pr_curve_hgb_weather_eng.png
  ../outputs/models/logreg_weather_balanced.joblib
  ../outputs/models/random_forest_weather_balanced.joblib
  ../outputs/models/hgb_weather.joblib
  ../outputs/models/scaler_weather.joblib
  ../outputs/models/imputer_weather.joblib
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

# Keep meta (for engineered features)
df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.date
df["Origin_str"] = df["Origin"].astype("string")
df["Dest_str"] = df["Dest"].astype("string")

# ---------- Base & engineered (basic) ----------
base_feats = ["IATA_Code_Operating_Airline", "Origin", "Dest", "Month", "DayOfWeek"]
if "is_weekend" not in df.columns:
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(np.int8)
if "route_freq" not in df.columns:
    route_tmp = df["Origin_str"].fillna("NA") + "_" + df["Dest_str"].fillna("NA")
    df["route_freq"] = route_tmp.map(route_tmp.value_counts()).astype(np.int32)
eng_feats = ["is_weekend", "route_freq"]

candidate_weather = ["tavg","tmin","tmax","prcp","snow","wspd","wdir","wpgt","pres","tsun"]
wx_feats = [c for c in candidate_weather if c in df.columns]

if "Delayed" not in df.columns:
    raise ValueError("Column 'Delayed' not found. Make sure merged data includes it.")

# ---------- Robust categorical encoding ----------
def encode_cat(series):
    s = series.astype("string")
    codes = pd.Categorical(s).codes.astype("int32")   # -1 for NaN
    codes = np.where(codes == -1, 0, codes).astype("int16")
    return codes

for col in ["IATA_Code_Operating_Airline", "Origin", "Dest"]:
    df.loc[:, col] = encode_cat(df[col])

# Weather strings -> numeric
for c in wx_feats:
    df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Assemble X / y ----------
feature_cols = base_feats + eng_feats + wx_feats
X = df[feature_cols].copy()
y = df["Delayed"].astype(np.int8)

# ensure numeric
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# drop all-NaN cols (important)
all_nan_cols = X.columns[X.isna().all()]
if len(all_nan_cols) > 0:
    print(f"🧹 Dropping all-NaN columns: {list(all_nan_cols)}")
    X = X.drop(columns=list(all_nan_cols))

print(f"Delay rate: {y.mean():.2%}")
print(f"Using features ({X.shape[1]}): {list(X.columns)}")

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# meta aligned with X indices
meta = df[["FlightDate","Origin_str","Dest_str"]].loc[X.index]
meta_train = meta.loc[X_train.index]
meta_test  = meta.loc[X_test.index]

# ---------- Impute ----------
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

X_train_imp = pd.DataFrame(X_train_imp, columns=X_train.columns, index=X_train.index)
X_test_imp  = pd.DataFrame(X_test_imp,  columns=X_test.columns,  index=X_test.index)

# ---------- Engineered features AFTER imputation ----------
# A) origin_day_volume (train-only fit)
train_pairs = meta_train.assign(_ones=1)
od_vol = (
    train_pairs.groupby(["Origin_str","FlightDate"])["_ones"]
    .sum()
    .rename("origin_day_volume")
)

def map_origin_day_volume(meta_df):
    keys = list(zip(meta_df["Origin_str"], meta_df["FlightDate"]))
    return pd.Series(keys, index=meta_df.index).map(od_vol)

X_train_imp["origin_day_volume"] = map_origin_day_volume(meta_train).astype("float64")
X_test_imp["origin_day_volume"]  = map_origin_day_volume(meta_test).astype("float64")
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

add_target_encoding(X_train_imp, y_train, X_test_imp, ["Origin","Month"], "te_origin_month")
add_target_encoding(X_train_imp, y_train, X_test_imp, ["Origin","Dest"],  "te_route")

# ---------- Logistic Regression ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train_scaled, y_train)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = (y_prob_lr >= 0.5).astype(int)

# ---------- Random Forest ----------
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    max_features="sqrt",
)
rf.fit(X_train_imp, y_train)
y_prob_rf = rf.predict_proba(X_test_imp)[:, 1]
y_pred_rf = (y_prob_rf >= 0.5).astype(int)

# ---------- HistGradientBoosting ----------
hgb = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=None,
    max_iter=400,
    l2_regularization=0.0,
    random_state=42
)
hgb.fit(X_train_imp, y_train)
y_prob_hgb = hgb.predict_proba(X_test_imp)[:, 1]
y_pred_hgb = (y_prob_hgb >= 0.5).astype(int)

# ---------- Eval helper ----------
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
# ---------- Threshold sweep for better F1 / Recall (fixed indexing) ----------
from sklearn.metrics import precision_recall_curve, average_precision_score

def sweep_threshold(y_true, y_prob, name, target_recall=0.60):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds 对应的是 precisions[:-1], recalls[:-1]
    P = precisions[:-1]
    R = recalls[:-1]
    T = thresholds

    # F1 按阈值对齐计算
    f1s = 2 * (P * R) / (P + R + 1e-9)
    best_idx = int(np.nanargmax(f1s))
    best_thr = float(T[best_idx])
    ap = average_precision_score(y_true, y_prob)

    # 在“最佳 F1 阈值”下评估
    y_hat_f1 = (y_prob >= best_thr).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print(f"\n🔧 {name} | Best-F1 thr={best_thr:.2f}  "
          f"Acc={accuracy_score(y_true, y_hat_f1):.3f} "
          f"P={precision_score(y_true, y_hat_f1, zero_division=0):.3f} "
          f"R={recall_score(y_true, y_hat_f1):.3f} "
          f"F1={f1_score(y_true, y_hat_f1):.3f}  (AP={ap:.3f})")

    # 选择一个“高召回”运行点：先在 R>=target_recall 的位置里取第一个阈值（同样用 R 对齐）
    hi_idx = np.where(R >= target_recall)[0]
    if hi_idx.size > 0:
        thr_hi = float(T[hi_idx[0]])
        y_hat_hi = (y_prob >= thr_hi).astype(int)
        print(f"   High Recall (R≥{target_recall:.2f}) thr={thr_hi:.2f}  "
              f"Acc={accuracy_score(y_true, y_hat_hi):.3f} "
              f"P={precision_score(y_true, y_hat_hi, zero_division=0):.3f} "
              f"R={recall_score(y_true, y_hat_hi):.3f} "
              f"F1={f1_score(y_true, y_hat_hi):.3f}")
    else:
        print(f"   No operating point reaches recall ≥ {target_recall:.2f}")

    # PR 曲线
    import matplotlib.pyplot as plt, os
    plt.figure(figsize=(6,5))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve - {name} (AP={ap:.3f})")
    plt.tight_layout()
    pr_path = os.path.join(PLOT_DIR, f"pr_curve_{name.replace(' ', '_').lower()}.png")
    plt.savefig(pr_path, dpi=180); plt.close()
    print(f"🖼️ Saved PR curve: {pr_path}")


# ---------- Calibrated threshold example for RF ----------
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
plt.figure(figsize=(8, 5))
fi.tail(15).plot(kind="barh")
plt.title("Top Feature Importances (RF + weather + engineered)")
plt.xlabel("Importance")
plt.tight_layout()
fi_path = os.path.join(PLOT_DIR, "feature_importance_rf_weather.png")
plt.savefig(fi_path, dpi=180)
plt.close()
print(f"🖼️ Saved feature importance: {fi_path}")

# ---------- ROC Curves ----------
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, y_prob_lr)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)
fpr_hgb, tpr_hgb, _ = roc_curve(y_test, y_prob_hgb)

plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr,  label="LogReg")
plt.plot(fpr_rf, tpr_rf,  label="RF")
plt.plot(fpr_hgb, tpr_hgb, label="HGB")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (LogReg / RF / HGB)")
plt.legend()
plt.tight_layout()
roc_all = os.path.join(PLOT_DIR, "roc_curves_weather_all.png")
plt.savefig(roc_all, dpi=180)
plt.close()
print(f"🖼️ Saved ROC curves (all): {roc_all}")

# Also keep the 2-model version for continuity
plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label="LogReg (weather+eng)")
plt.plot(fpr_rf, tpr_rf, label="RF (weather+eng)")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (with weather + engineered features)")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(PLOT_DIR, "roc_curves_weather.png")
plt.savefig(roc_path, dpi=180)
plt.close()
print(f"🖼️ Saved ROC curves: {roc_path}")

# ---------- Save ----------
joblib.dump(logreg, os.path.join(MODEL_DIR, "logreg_weather_balanced.joblib"))
joblib.dump(rf,    os.path.join(MODEL_DIR, "random_forest_weather_balanced.joblib"))
joblib.dump(hgb,   os.path.join(MODEL_DIR, "hgb_weather.joblib"))
joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler_weather.joblib"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer_weather.joblib"))
print(f"\n✅ Models, scaler & imputer saved to {MODEL_DIR}/")
