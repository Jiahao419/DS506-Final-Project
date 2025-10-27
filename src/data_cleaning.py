"""
data_cleaning.py  — memory-safe cleaning for 40M+ rows
"""

import os
import numpy as np
import pandas as pd

# -------- robust paths --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "data", "flights_cleaned.csv")
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "flights_preprocessed.csv")
PARQUET_OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "flights_preprocessed.parquet")

print("🚀 Starting data cleaning...")

# -------- 1) load with dtype hints to save memory --------
# 有些列用小整数类型能大幅降内存
dtype_map = {
    "Cancelled": "int8",
    "Diverted": "int8",
    "DepDelayMinutes": "float32",
    "ArrDelayMinutes": "float32",
    "CarrierDelay": "float32",
    "WeatherDelay": "float32",
    "NASDelay": "float32",
    "SecurityDelay": "float32",
    "LateAircraftDelay": "float32",
}

df = pd.read_csv(INPUT_PATH, low_memory=False, dtype=dtype_map)
print(f"✅ Loaded dataset with {len(df):,} rows and {len(df.columns)} columns")

# -------- 2) drop unused --------
drop_cols = ["OriginCityName", "DestCityName"]
df = df.drop(columns=drop_cols, errors="ignore")

# -------- 3) fillna for delay cause --------
delay_cols = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
for col in delay_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype("float32")

for col in ["DepDelayMinutes", "ArrDelayMinutes"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype("float32")

# -------- 4) remove cancelled/diverted --------
if {"Cancelled", "Diverted"}.issubset(df.columns):
    before = len(df)
    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    print(f"✈️ Removed {before - len(df):,} cancelled/diverted flights")

# -------- 5) time features --------
df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
df["Month"] = df["FlightDate"].dt.month.astype("int8")
df["DayOfWeek"] = (df["FlightDate"].dt.dayofweek + 1).astype("int8")

# -------- 6) target --------
df["Delayed"] = (df["ArrDelayMinutes"] > 15).astype("int8")

# -------- 7) encode categorical airline code --------
# IATA_Code_Operating_Airline 有时是航空公司码（如 AA, UA）
if "IATA_Code_Operating_Airline" in df.columns:
    df["IATA_Code_Operating_Airline"] = df["IATA_Code_Operating_Airline"].astype("category")
    df["IATA_Code_Operating_Airline"] = df["IATA_Code_Operating_Airline"].cat.add_categories(["__UNK__"])
    df["IATA_Code_Operating_Airline"] = df["IATA_Code_Operating_Airline"].cat.codes.astype("int16")
    df["IATA_Code_Operating_Airline"] = df["IATA_Code_Operating_Airline"].replace({-1: 0}).astype("int16")

# -------- 8) drop redundant/leaky columns --------
drop_after_filter = [
    "Cancelled", "Diverted", "CancellationCode",
    "DepartureDelayGroups", "ArrivalDelayGroups"
]
df = df.drop(columns=[c for c in drop_after_filter if c in df.columns])

# -------- 9) keep non-negative minutes & cap huge outliers --------
df = df[(df["DepDelayMinutes"] >= 0) & (df["ArrDelayMinutes"] >= 0)]
df = df[df["ArrDelayMinutes"] < 1000]

# -------- 🔟 save (parquet preferred) --------
os.makedirs(os.path.dirname(PARQUET_OUTPUT_PATH), exist_ok=True)
try:
    import pyarrow  # noqa: F401
    df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
    print(f"💾 Saved parquet to {PARQUET_OUTPUT_PATH}")
except Exception as e:
    print(f"ℹ️ Parquet save skipped ({e}). Falling back to CSV...")
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"💾 Saved CSV to {CSV_OUTPUT_PATH}")

print("📏 Final shape:", df.shape)
print(df.dtypes.head(30))
print("✅ Done.")

# 在脚本最后另存一个样本（可选）
SAMPLE_CSV = os.path.join(BASE_DIR, "data", "flights_sample_1pct.csv")
sample = df.sample(frac=0.01, random_state=42)
sample.to_csv(SAMPLE_CSV, index=False)
print(f"🧪 Saved 1% sample to {SAMPLE_CSV} (rows: {len(sample):,})")
print("🔢 Delayed rate:", float(df["Delayed"].mean()))
print("🛫 Top 10 origins by count:\n", df["Origin"].value_counts().head(10))
print("🛬 Top 10 destinations by count:\n", df["Dest"].value_counts().head(10))
