"""
model_training_weather.py
-------------------------
Train models with pre-flight features + merged daily weather (no leakage).

Input:
  ../outputs/flights_with_weather_sample.parquet   # from weather_merge.py

Outputs:
  ../outputs/plots/feature_importance_rf_weather.png
  ../outputs/plots/roc_curves_weather.png
  ../outputs/models/logreg_weather_balanced.joblib
  ../outputs/models/random_forest_weather_balanced.joblib
  ../outputs/models/scaler_weather.joblib
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

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

# ---------- Keep only pre-flight + weather features ----------
# 基础特征（起飞前可得）
base_feats = [
    "IATA_Code_Operating_Airline",
    "Origin", "Dest",
    "Month", "DayOfWeek"
]

# 工程特征（若不存在就临时构造）
if "is_weekend" not in df.columns:
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(np.int8)
if "route_freq" not in df.columns:
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    df["route_freq"] = route.map(route.value_counts()).astype(np.int32)

eng_feats = ["is_weekend", "route_freq"]

# 天气特征（按你下载到的列自动挑选可用项）
candidate_weather = ["tavg","tmin","tmax","prcp","snow","wspd","wdir","wpgt","pres","tsun"]
wx_feats = [c for c in candidate_weather if c in df.columns]

# 目标变量
if "Delayed" not in df.columns:
    raise ValueError("Column 'Delayed' not found. Make sure you merged with flights having 'Delayed'.")

# ---------- Typing & cleaning ----------
# Category → codes（避免 SettingWithCopy，用 .loc）
for col in ["IATA_Code_Operating_Airline", "Origin", "Dest"]:
    if df[col].dtype == "object":
        df.loc[:, col] = df[col].astype("category")
    if str(df[col].dtype) == "category":
        df.loc[:, col] = df[col].cat.codes.astype(np.int16)
    else:
        df.loc[:, col] = df[col].astype(np.int16)

# weather 列可能有字符串 "<NA>"，统一转数值
for c in wx_feats:
    df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")

# 缺失填充：先按 Month 中位数，再全局中位数
for c in wx_feats:
    if df[c].isna().any():
        if "Month" in df.columns:
            df.loc[:, c] = df.groupby("Month")[c].transform(lambda s: s.fillna(s.median()))
        df.loc[:, c] = df[c].fillna(df[c].median())

# ---------- Assemble X / y ----------
feature_cols = base_feats + eng_feats + wx_feats
X = df[feature_cols].copy()
y = df["Delayed"].astype(np.int8)

print(f"Delay rate: {y.mean():.2%}")
print(f"Using features ({len(feature_cols)}): {feature_cols}")

# ---------- Train/Test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------- Logistic Regression (balanced) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]

# ---------- Random Forest (balanced) ----------
rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    max_features="sqrt",
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

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

evaluate(y_test, y_pred_lr, y_prob_lr, "Logistic Regression (weather + balanced)")
evaluate(y_test, y_pred_rf, y_prob_rf, "Random Forest (weather + balanced)")

# ---------- Feature Importances (RF) ----------
importances = rf.feature_importances_
fi = pd.Series(importances, index=X_train.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
fi.tail(12).plot(kind="barh")
plt.title("Top Feature Importances (RF + weather)")
plt.xlabel("Importance")
plt.tight_layout()
fi_path = os.path.join(PLOT_DIR, "feature_importance_rf_weather.png")
plt.savefig(fi_path, dpi=180)
plt.close()
print(f"🖼️ Saved feature importance: {fi_path}")

# ---------- ROC Curves ----------
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label="LogReg (weather)")
plt.plot(fpr_rf, tpr_rf, label="RandomForest (weather)")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves with Weather")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(PLOT_DIR, "roc_curves_weather.png")
plt.savefig(roc_path, dpi=180)
plt.close()
print(f"🖼️ Saved ROC curves: {roc_path}")

# ---------- Save ----------
joblib.dump(logreg, os.path.join(MODEL_DIR, "logreg_weather_balanced.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_weather.joblib"))
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_weather_balanced.joblib"))
print(f"\n✅ Models & scaler saved to {MODEL_DIR}/")
