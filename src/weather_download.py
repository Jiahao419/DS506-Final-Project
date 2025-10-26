"""
weather_download.py
-------------------
Fetch daily weather data for major US airports using Meteostat.
Saves output to ../data/weather_daily.csv
"""

from datetime import datetime
import pandas as pd
from meteostat import Stations, Daily
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "weather_daily.csv")

# ---- Airport coordinates (lat, lon) ----
airport_coords = {
    "ATL": (33.6367, -84.4281),
    "ORD": (41.9786, -87.9048),
    "DFW": (32.8968, -97.0380),
    "DEN": (39.8561, -104.6737),
    "CLT": (35.2140, -80.9431),
    "LAX": (33.9416, -118.4085),
    "SEA": (47.4502, -122.3088),
    "PHX": (33.4343, -112.0116),
    "LAS": (36.0840, -115.1537),
    "IAH": (29.9902, -95.3368)
}

# Time range
start = datetime(2020, 1, 1)
end = datetime(2020, 12, 31)

# ---- Download ----
all_weather = []

for code, (lat, lon) in airport_coords.items():
    print(f"🌤 Fetching weather for {code} ...")

    # find nearest weather station to the airport
    stations = Stations().nearby(lat, lon)
    station = stations.fetch(1)
    if station.empty:
        print(f"⚠️ No station found for {code}")
        continue

    station_id = station.index[0]

    # fetch daily data
    data = Daily(station_id, start, end)
    data = data.fetch()
    if data.empty:
        print(f"⚠️ No daily data for {code}")
        continue

    data["Origin"] = code
    all_weather.append(data)

# ---- Combine & Save ----
if len(all_weather) > 0:
    weather = pd.concat(all_weather)
    weather.reset_index(inplace=True)
    weather.rename(columns={"time": "date"}, inplace=True)
    weather.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved weather data to: {OUTPUT_PATH}")
    print(weather.head())
else:
    print("❌ No weather data downloaded.")
