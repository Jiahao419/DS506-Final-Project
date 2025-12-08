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

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

# -------- 1) streaming read to avoid memory blow-up --------
print(f"📂 Reading from: {INPUT_PATH}")
chunks = []
chunk_size = 1_000_000

usecols = None  # if you want to restrict columns, you can set a list here

for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=chunk_size, usecols=usecols)):
    print(f"  🧩 Chunk {i+1}: {len(chunk):,} rows")
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
del chunks
print(f"✅ Loaded full dataset: {len(df):,} rows")

# -------- 2) basic cleaning: drop obvious junk --------
# drop rows with missing critical fields
critical_cols = ["FlightDate", "Origin", "Dest", "ArrDelayMinutes"]
missing_mask = df[critical_cols].isna().any(axis=1)
before = len(df)
df = df[~missing_mask]
print(f"🧹 Removed {before - len(df):,} rows with missing critical fields")

# ensure numeric delay columns are numeric
delay_cols = [
    "DepDelayMinutes", "ArrDelayMinutes",
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay"
]
for col in delay_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# fill NaNs in delay columns with 0 (no delay reported)
for col in delay_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype("float32")

# also ensure DepDelayMinutes and ArrDelayMinutes exist as float
for col in ["DepDelayMinutes", "ArrDelayMinutes"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype("float32")

# -------- 3) cast types to save memory --------
int_cols = ["Quarter", "Month", "DayOfMonth", "DayOfWeek"]
for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

for col in ["Cancelled", "Diverted"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype("int8")

# CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay already handled above
for col in ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]:
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

# planned departure hour (pre-flight information, from CRSDepTime)
if "CRSDepTime" in df.columns:
    dep_str = df["CRSDepTime"].astype(str).str.zfill(4)
    df["DepHour"] = dep_str.str.slice(0, 2).astype("int8")

# -------- 6) target --------
df["Delayed"] = (df["ArrDelayMinutes"] > 15).astype("int8")

# -------- 7) encode categorical airline code --------
# IATA_Code_Operating_Airline 
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

# optional: drop columns you never use downstream
optional_drop = [
    "Tail_Number", "Flight_Number_Operating_Airline", "OriginCityName", "DestCityName",
    "CRSDepTime", "CRSArrTime", "DepTime", "ArrTime"
]
df = df.drop(columns=[c for c in optional_drop if c in df.columns], errors="ignore")

# -------- 9) save outputs --------
print("💾 Saving cleaned data...")
df.to_parquet(PARQUET_OUTPUT_PATH, index=False)
print(f"💾 Saved parquet to {PARQUET_OUTPUT_PATH}")

# also save a CSV version if you want (smaller than original)
df.to_csv(CSV_OUTPUT_PATH, index=False)
print(f"💾 Saved CSV to {CSV_OUTPUT_PATH}")

print("📏 Final shape:", df.shape)
print(df.dtypes.head(30))
print("✅ Done.")

SAMPLE_CSV = os.path.join(BASE_DIR, "data", "flights_sample_1pct.csv")
sample = df.sample(frac=0.01, random_state=42)
sample.to_csv(SAMPLE_CSV, index=False)
print(f"🧪 Saved 1% sample to {SAMPLE_CSV} (rows: {len(sample):,})")
print("🔢 Delayed rate:", float(df["Delayed"].mean()))
print("🛫 Top 10 origins by count:\n", df["Origin"].value_counts().head(10))
print("🛬 Top 10 destinations by count:\n", df["Dest"].value_counts().head(10))
