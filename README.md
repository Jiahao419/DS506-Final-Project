# DS506-Final-Project
Predicting flight delays using US Department of Transportation and weather data.

# Flight Delay Prediction

## Project Description
Flight delays are a common issue in air travel, causing inconvenience for passengers and operational challenges for airlines. In this project, we aim to **predict whether a flight will be delayed** based on historical flight data and weather information. By identifying key factors that contribute to delays, our project can provide insights that may improve scheduling, operations, and passenger experience.

## Goals
- Build a predictive model to classify whether a flight will be delayed.
- Identify which features (e.g., airline, departure time, weather conditions) most strongly influence delays.
- Visualize patterns and trends in delays across different airports, airlines, and times of day.

## Data Collection
We will collect data from publicly available sources:
- **Bureau of Transportation Statistics (BTS)**: On-time performance data, including scheduled and actual departure/arrival times, delays, airline, and airport information.
- **NOAA / National Weather Service**: Weather conditions (temperature, precipitation, wind speed, visibility) for major airports.
- **Federal Aviation Administartion**: airport status, airspace system performance metrics, and airport operational statistics.

## Data Cleaning
- Handle missing and inconsistent data (e.g. missing weather values).
- Convert time fields into features such as *hour of day, day of week, month*.
- Encode categorical variables (e.g., airline, airport codes).
- Standardize numerical features for modeling.

## Feature Extraction
We plan to extract features such as:
- **Flight information**: Airline, origin, destination, flight distance.
- **Temporal features**: Departure hour, day of week, month.
- **Weather features**: Temperature, precipitation, wind speed, visibility.

## Data Visualization
We will use visualizations to explore and communicate insights, such as:
- Heatmaps of average delay rates by airport and airline.
- Time series plots of delays by month or season.
- Scatter plots of weather conditions vs. delay likelihood.
- Interactive dashboards (if time allows).

## Modeling Plan
We will experiment with multiple models, including:
- Logistic Regression (baseline).
- Decision Trees and Random Forest.
- (Optional) Neural networks for more complex modeling.
Model performance will be evaluated using accuracy, precision, recall, and AUC.

## Test Plan
- Split the dataset into **80% training and 20% testing**.
- Optionally use a **time-based split** (train on earlier months, test on later months).
- Compare baseline and advanced models.
- Ensure reproducibility with fixed random seeds and GitHub workflow tests.

## Timeline
- **Proposal (Sep 22)**: Define goals, data sources, and methods.
- **Midterm Report (Oct 27)**: Collect and clean data, initial visualizations, baseline models.
- **Final Report (Dec 10)**: Complete modeling, produce final visualizations, evaluate results, and record final presentation.

## Team Members
- Jiahao Liu 
- Xinyi Wang
- Abdulelah (Abdul) Hamdi
- Nathan Arlan Djunaedi