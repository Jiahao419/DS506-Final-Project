## ✈️ Flight Delay Prediction using US DOT & Weather Data  
**Boston University – CS506: Data Science Tools & Applications**  

---

### 📘 Project Overview
This project predicts **flight delays** using data from the **US Department of Transportation (Bureau of Transportation Statistics)** combined with **daily weather data** from **Meteostat**.

We explore how operational (e.g., routes, airlines, weekdays) and environmental (e.g., temperature, wind, pressure) factors influence the likelihood of flight delays.

---

### 🧩 Dataset

| Source | Description | Link |
|--------|--------------|------|
| **BTS On-Time Performance (2020)** | Flight-level data including scheduled and actual times, airlines, and airports. | [US DOT BTS Dataset](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) |
| **Meteostat (Daily Weather)** | Daily temperature, wind speed, and pressure data per airport. | [Meteostat API](https://dev.meteostat.net/python/) |
| **Processed Dataset (merged)** | Cleaned and merged flight + weather + engineered features. | [Google Drive Folder](https://drive.google.com/drive/folders/11Bs78yYzX7t18sY3JP_uk08K3PCpzmxg?usp=drive_link) |

> ⚠️ Due to GitHub’s 100 MB file limit, only a **1% sample dataset** (`flights_sample_1pct.csv`) is stored in this repo for testing.  
> The **full datasets** are hosted on Google Drive.

---

### 📂 Project Structure

```
CS506/
│
├── data/
│   ├── flights_sample_1pct.csv          # 1% sample for testing
│
├── src/                                 # Source code
│   ├── data_cleaning.py                 # Preprocess raw flight data
│   ├── weather_download.py              # Download weather data using Meteostat API
│   ├── weather_merge.py                 # Merge weather and flight datasets
│   ├── eda_visualization.py             # Generate EDA plots
│   ├── model_training.py                # Baseline models (LogReg / RF)
│   ├── model_training_weather.py        # Models with weather + engineered features
│
├── outputs/
│   └── plots/                           # Visualization results
│       ├── delay_rate_by_month.png
│       ├── delay_rate_by_dow.png
│       ├── delay_rate_by_airport.png
│       ├── dep_vs_arr_delay.png
│       ├── feature_importance_rf.png
│       ├── feature_importance_rf_weather.png
│       ├── roc_curves.png
│       ├── roc_curves_weather.png
│       ├── roc_curves_weather_all.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

### ⚙️ Setup & Usage

#### 1️⃣ Create and activate environment
```bash
python -m venv .venv
.\.venv\Scriptsctivate        # (Windows)
# or source .venv/bin/activate  # (Mac/Linux)

pip install -r requirements.txt
```

#### 2️⃣ Run the workflow
```bash
# Step 1: Data cleaning (requires BTS raw CSV)
python src/data_cleaning.py

# Step 2: Download weather data
python src/weather_download.py

# Step 3: Merge weather with flight data
python src/weather_merge.py

# Step 4: Exploratory Data Analysis
python src/eda_visualization.py

# Step 5: Baseline model training
python src/model_training.py

# Step 6: Weather + engineered features model
python src/model_training_weather.py
```

All generated plots and models will appear in:
```
outputs/plots/
outputs/models/
```

---

### 📊 Model Results Summary

| Model | Features | Accuracy | Precision | Recall | F1-score | AUC |
|:------|:----------|:----------|:-----------|:---------|:----------|
| Logistic Regression | Operational only | 0.784 | 0.237 | 0.082 | 0.122 | 0.560 |
| Random Forest | Operational only | 0.784 | 0.237 | 0.082 | 0.122 | 0.560 |
| Logistic Regression | + Weather + Engineered | 0.588 | 0.225 | **0.509** | 0.312 | 0.577 |
| Random Forest | + Weather + Engineered | 0.784 | 0.237 | 0.082 | 0.122 | 0.560 |
| HistGradientBoosting | + Weather + Engineered | **0.814** | 0.257 | 0.010 | 0.019 | **0.591** |

✅ *Adding weather and engineered features improved AUC from ~0.55 to 0.59.  
“Route frequency” and “origin-day volume” are the strongest predictors of delays.*

---

### 📈 Visualization Samples

| Plot | Description |
|------|--------------|
| ![](outputs/plots/roc_curves_weather_all.png) | ROC comparison of Logistic Regression, Random Forest, and HGB models |
| ![](outputs/plots/feature_importance_rf_weather.png) | Top feature importances — route frequency, origin-day volume, temperature, etc. |
| ![](outputs/plots/delay_rate_by_month.png) | Average flight delay rate by month |
| ![](outputs/plots/delay_rate_by_airport.png) | Delay rates for top 10 origin airports |

---

### 📤 Full Dataset Access

You can access all cleaned and merged datasets from Google Drive:  
📂 [**CS506 Project Data (Google Drive)**](https://drive.google.com/drive/folders/YOUR-FOLDER-ID-HERE)

Contents:
- `flights_cleaned.csv` – full cleaned flight dataset (~3.8 GB)  
- `weather_daily.csv` – daily weather data (~185 KB)  
- `flights_with_weather_sample.parquet` – merged dataset (sample for testing)

---

### 🧠 Future Work
- Improve **class imbalance handling** and **probability calibration**.  
- Add **temporal validation** (train on Jan–Oct, test on Nov–Dec).  
- Deploy as an **interactive web dashboard (Flask / Streamlit)** for predictions.  

---

### 🪪 License
This project is for academic use only — **Boston University CS506**.  
All datasets belong to their original providers:  
- *U.S. Department of Transportation (Bureau of Transportation Statistics)*  
- *Meteostat Weather Service*

---

### 🙌 Acknowledgements
Special thanks to **Divya Appapogu**, **Nathan Djunaedi**, **Eric Wang**, and **Hamdi Abdulaleh** for their collaboration and feedback during the project.

---

### ✅ Quick Summary
> **Goal:** Predict flight delays using weather and operational data  
> **Key Result:** AUC improved from 0.55 → 0.59 with weather integration  
> **Tools:** Python, Scikit-learn, Meteostat API, Matplotlib, Pandas  
> **Outcome:** Reproducible pipeline + visual insights into delay causes  
