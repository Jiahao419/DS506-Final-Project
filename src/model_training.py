"""
model_training.py
-----------------
Baseline modeling for flight delay prediction on 1% sample.

- Uses ONLY pre-flight features (no leakage)
- Handles class imbalance with class_weight='balanced'
- Adds simple engineered features
- Saves models and plots (feature importances + ROC + confusion heatmaps)
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

# seaborn (optional, for heatmaps)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

warnings.filterwarnings("ignore")  # keep console clean

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "flights_sample_1pct.csv")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Load ----------
print("📂 Loading sample data ...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df):,} rows")

# ---------- Feature Engineering (pre-flight only) ----------
base_keep_cols = [
    "IATA_Code_Operating_Airline",
    "Origin", "Dest",
    "Month", "DayOfWeek",
    "DepHour",      # may or may not exist
    "Delayed"
]

if "DepHour" not in df.columns:
    print("⚠️ Column 'DepHour' not found in preprocessed data. "
          "If you just updated data_cleaning.py, please re-run it to regenerate flights_preprocessed.")
    keep_cols = [c for c in base_keep_cols if c in df.columns]
else:
    keep_cols = base_keep_cols

df = df[keep_cols].copy()

# is_weekend
df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(np.int8)

# route frequency
df["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
route_counts = df["route"].value_counts()
df["route_freq"] = df["route"].map(route_counts).astype(np.int32)
df.drop(columns=["route"], inplace=True)

# Encode categorical
for c in ["IATA_Code_Operating_Airline", "Origin", "Dest"]:
    if df[c].dtype == "object":
        df[c] = df[c].astype("category")
    if str(df[c].dtype) == "category":
        df[c] = df[c].cat.codes.astype(np.int16)
    else:
        df[c] = df[c].astype(np.int16)

# ---------- Train/Test Split ----------
X = df.drop(columns=["Delayed"])
y = df["Delayed"].astype(np.int8)

print(f"Delay rate in dataset: {y.mean():.2%}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- Logistic Regression (balanced) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]

# ---------- Random Forest (balanced) ----------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    max_features="sqrt",
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# ---------- Evaluation helper ----------
def eval_and_print(y_true, y_pred, y_prob, name):
    print(f"\n📊 {name}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"AUC:       {roc_auc_score(y_true, y_prob):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

eval_and_print(y_train, logreg.predict(X_train_scaled), logreg.predict_proba(X_train_scaled)[:, 1], "LogReg (train)")
eval_and_print(y_test,  y_pred_lr, y_prob_lr, "LogReg (test)")
eval_and_print(y_test,  y_pred_rf, y_prob_rf, "Random Forest (test)")

# ---------- Feature importance for RF ----------
importances = rf.feature_importances_
fi = pd.Series(importances, index=X_train.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
fi.plot(kind="barh")
plt.title("Random Forest Feature Importance (Baseline)")
plt.tight_layout()
fi_path = os.path.join(PLOT_DIR, "feature_importance_rf.png")
plt.savefig(fi_path, dpi=180)
plt.close()
print(f"🖼️ Saved feature importance plot: {fi_path}")

# ---------- ROC Curves ----------
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={roc_auc_score(y_test, y_prob_lr):.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_auc_score(y_test, y_prob_rf):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(PLOT_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=180)
plt.close()
print(f"🖼️ Saved ROC curves: {roc_path}")

# ---------- Confusion Matrix Heatmaps ----------
if HAS_SEABORN:
    for name, y_pred, fname in [
        ("LogReg (test)", y_pred_lr, "cm_logreg_baseline.png"),
        ("Random Forest (test)", y_pred_rf, "cm_rf_baseline.png"),
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
    print("ℹ️ seaborn not installed, skipping confusion matrix heatmaps for baseline models.")

# ---------- Save models ----------
joblib.dump(logreg, os.path.join(MODEL_DIR, "logreg_balanced.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_balanced.joblib"))
print(f"\n✅ Models & scaler saved to {MODEL_DIR}/")
