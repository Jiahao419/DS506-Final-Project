"""
eda_visualization.py
--------------------
Generate exploratory plots for flight delay dataset (1% sample).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Paths ===
DATA_PATH = "../data/flights_sample_1pct.csv"
OUTPUT_DIR = "../outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
print("📂 Loading sample data...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df):,} rows")

# === Delay rate by month ===
month_delay = df.groupby("Month")["Delayed"].mean() * 100
plt.figure(figsize=(8,5))
sns.barplot(x=month_delay.index, y=month_delay.values, palette="Blues_d")
plt.title("Average Flight Delay Rate by Month")
plt.xlabel("Month")
plt.ylabel("Delay Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/delay_rate_by_month.png")
plt.close()

# === Delay rate by day of week ===
dow_delay = df.groupby("DayOfWeek")["Delayed"].mean() * 100
plt.figure(figsize=(8,5))
sns.barplot(x=dow_delay.index, y=dow_delay.values, palette="Greens_d")
plt.title("Average Flight Delay Rate by Day of Week")
plt.xlabel("Day of Week (1=Mon, 7=Sun)")
plt.ylabel("Delay Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/delay_rate_by_dow.png")
plt.close()

# === Delay rate by origin airport (Top 10) ===
top_orig = df["Origin"].value_counts().nlargest(10).index
origin_delay = df[df["Origin"].isin(top_orig)].groupby("Origin")["Delayed"].mean() * 100
plt.figure(figsize=(9,5))
sns.barplot(x=origin_delay.index, y=origin_delay.values, palette="Reds_d")
plt.title("Delay Rate for Top 10 Origin Airports")
plt.xlabel("Origin Airport")
plt.ylabel("Delay Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/delay_rate_by_airport.png")
plt.close()

# === Delay vs Departure Delay Minutes (scatter sample) ===
plt.figure(figsize=(7,5))
sample = df.sample(10000, random_state=42)
sns.scatterplot(data=sample, x="DepDelayMinutes", y="ArrDelayMinutes", alpha=0.3)
plt.title("Departure vs Arrival Delay (sample of 10k)")
plt.xlabel("Departure Delay (min)")
plt.ylabel("Arrival Delay (min)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dep_vs_arr_delay.png")
plt.close()

print(f"🎨 Plots saved in {OUTPUT_DIR}/")
