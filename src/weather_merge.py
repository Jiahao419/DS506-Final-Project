"""
weather_merge.py
----------------
Merge daily weather features into flight dataset.
Join keys: Origin (airport code) + date (FlightDate).

Inputs (defaults):
- ../data/weather_daily.csv        # daily weather by airport (Origin)
- ../data/flights_sample_1pct.csv  # 1% sample for midterm
- ../outputs/flights_preprocessed.parquet  # full cleaned flights (optional)

Outputs:
- ../outputs/flights_with_weather_sample.parquet      # merged sample
- ../outputs/flights_with_weather/ (partitioned by Origin)  # merged full (optional)

Usage:
    python weather_merge.py                # merge sample -> parquet
    python weather_merge.py --full        # merge full dataset (partitioned)
"""

import os
import argparse
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

WEATHER_CSV = os.path.join(DATA_DIR, "weather_daily.csv")
SAMPLE_CSV  = os.path.join(DATA_DIR, "flights_sample_1pct.csv")
FULL_PARQUET = os.path.join(OUT_DIR, "flights_preprocessed.parquet")

SAMPLE_OUT = os.path.join(OUT_DIR, "flights_with_weather_sample.parquet")
FULL_OUT_DIR = os.path.join(OUT_DIR, "flights_with_weather")  # partitioned

# ---- columns we expect from weather_daily.csv (Meteostat-style) ----
# date, Origin, tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun, etc.
WEATHER_KEEP = [
    "date", "Origin",        # join keys (date as datetime, Origin as airport code)
    "tavg", "tmin", "tmax",  # temperature (°C)
    "prcp", "snow",          # precipitation / snow (mm)
    "wspd", "wdir", "wpgt",  # wind speed, direction, gust
    "pres", "tsun"           # pressure, sunshine duration (sec)
]

def normalize_weather(dfw: pd.DataFrame) -> pd.DataFrame:
    """Standardize weather columns: ensure required cols exist, types are right, and downcast dtypes."""
    # Some sources name columns slightly differently; try to harmonize
    rename_map = {
        "time": "date",
        "station": "Origin",   # if you stored station code as Origin already, this won't apply
        "wsp" : "wspd",
        "ws"  : "wspd",
        "wd"  : "wdir",
        "gust": "wpgt"
    }
    dfw = dfw.rename(columns=rename_map)

    # keep columns which exist
    keep = [c for c in WEATHER_KEEP if c in dfw.columns]
    dfw = dfw[keep].copy()

    # types
    if "date" in dfw.columns:
        dfw["date"] = pd.to_datetime(dfw["date"], errors="coerce").dt.date
    if "Origin" in dfw.columns:
        dfw["Origin"] = dfw["Origin"].astype(str)

    # downcast numerics
    for c in dfw.columns:
        if c in ("date", "Origin"):
            continue
        if pd.api.types.is_numeric_dtype(dfw[c]):
            dfw[c] = pd.to_numeric(dfw[c], errors="coerce", downcast="float")

    # deduplicate per Origin+date
    dfw = dfw.sort_values(["Origin", "date"]).drop_duplicates(["Origin", "date"], keep="last")
    return dfw


def merge_sample():
    print("📂 Loading sample flights & weather ...")
    flights = pd.read_csv(SAMPLE_CSV)
    weather = pd.read_csv(WEATHER_CSV)
    weather = normalize_weather(weather)

    print(f"✅ Flights rows: {len(flights):,} | Weather rows: {len(weather):,}")

    # prepare join keys
    flights["FlightDate"] = pd.to_datetime(flights["FlightDate"], errors="coerce").dt.date
    flights["Origin"] = flights["Origin"].astype(str)

    # merge
    merged = flights.merge(
        weather, left_on=["FlightDate", "Origin"], right_on=["date", "Origin"], how="left"
    ).drop(columns=["date"])

    # simple missing handling: fill with medians per-month or global
    for col in [c for c in WEATHER_KEEP if c not in ("date", "Origin") and c in merged.columns]:
        if merged[col].isna().any():
            # month-wise median if FlightDate present
            if "Month" in merged.columns:
                merged[col] = merged.groupby("Month")[col].transform(
                    lambda s: s.fillna(s.median())
                )
            merged[col] = merged[col].fillna(merged[col].median())

    # save parquet
    merged.to_parquet(SAMPLE_OUT, index=False)
    print(f"💾 Saved merged sample -> {SAMPLE_OUT} (rows: {len(merged):,})")


def merge_full_partitioned():
    """Merge full flights parquet with weather and write partitioned by Origin to save memory."""
    os.makedirs(FULL_OUT_DIR, exist_ok=True)

    print("📂 Loading weather (full) ...")
    weather = pd.read_csv(WEATHER_CSV)
    weather = normalize_weather(weather)
    print(f"✅ Weather rows: {len(weather):,}")

    print("📂 Reading flights parquet metadata ...")
    # read minimal columns first to get unique origins & row count
    base_cols = ["FlightDate", "Origin"]
    import pyarrow.parquet as pq
    table = pq.read_table(FULL_PARQUET, columns=base_cols)
    df_min = table.to_pandas()
    df_min["FlightDate"] = pd.to_datetime(df_min["FlightDate"], errors="coerce").dt.date
    df_min["Origin"] = df_min["Origin"].astype(str)
    origins = sorted(df_min["Origin"].unique().tolist())
    print(f"🛫 Unique origins: {len(origins)}")

    # we will merge per Origin and write a parquet file for each (partitioning)
    for i, org in enumerate(origins, 1):
        print(f"[{i}/{len(origins)}] 🔗 Merging Origin={org} ...")
        # filter flights for this origin (read only needed rows & all cols)
        mask = df_min["Origin"] == org
        rows = mask.sum()
        if rows == 0:
            continue

        # read full rows for this origin from parquet (row filter using pandas after read chunk)
        # to keep it simple, read full then filter; if memory is tight, switch to row-group filtering
        df_full = pq.read_table(FULL_PARQUET).to_pandas()
        df_org = df_full[df_full["Origin"] == org].copy()
        df_org["FlightDate"] = pd.to_datetime(df_org["FlightDate"], errors="coerce").dt.date
        df_org["Origin"] = df_org["Origin"].astype(str)

        wx_org = weather[weather["Origin"] == org]
        merged = df_org.merge(wx_org, left_on=["FlightDate", "Origin"], right_on=["date", "Origin"], how="left")
        merged = merged.drop(columns=["date"])

        # fill missing weather with per-month median if Month exists
        wx_cols = [c for c in WEATHER_KEEP if c not in ("date", "Origin") and c in merged.columns]
        for col in wx_cols:
            if merged[col].isna().any():
                if "Month" in merged.columns:
                    merged[col] = merged.groupby("Month")[col].transform(lambda s: s.fillna(s.median()))
                merged[col] = merged[col].fillna(merged[col].median())

        # write partition file
        out_path = os.path.join(FULL_OUT_DIR, f"origin={org}.parquet")
        merged.to_parquet(out_path, index=False)
        print(f"   💾 wrote {out_path} (rows: {len(merged):,})")

    print(f"✅ Done. Partitioned files are in: {FULL_OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Merge full dataset (partitioned by Origin)")
    args = parser.parse_args()

    # sanity checks
    if not os.path.exists(WEATHER_CSV):
        raise FileNotFoundError(
            f"Weather CSV not found: {WEATHER_CSV}\n"
            "→ 先用 Meteostat 生成 daily 天气到 data/weather_daily.csv"
        )

    if args.full:
        if not os.path.exists(FULL_PARQUET):
            raise FileNotFoundError(f"Full parquet not found: {FULL_PARQUET}")
        merge_full_partitioned()
    else:
        if not os.path.exists(SAMPLE_CSV):
            raise FileNotFoundError(f"Sample CSV not found: {SAMPLE_CSV}")
        merge_sample()


if __name__ == "__main__":
    main()
