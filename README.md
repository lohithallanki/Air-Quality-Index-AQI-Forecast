# Air-Quality-Index-AQI-Forecast
# 🌫️ Air Quality Index (AQI) Forecast — ML Pipeline

A complete end-to-end machine learning pipeline for forecasting the **Air Quality Index (AQI)** using classical ML models (Random Forest, XGBoost, Gradient Boosting). Built on a UCI-style hourly dataset with pollutant concentrations, meteorological features, and engineered lag/rolling features.

---

## 📊 Results

| Model | MAE | RMSE | R² | MAPE (%) |
|---|---|---|---|---|
| Random Forest | 6.58 | 8.37 | 0.9634 | 21.25 |
| XGBoost | 4.55 | 5.72 | 0.9829 | 9.62 |
| **Gradient Boosting** ★ | **4.13** | **5.25** | **0.9856** | **8.14** |

> ★ Best model by RMSE. All models evaluated on a **temporally split** test set (no data leakage).

---

## 🗂️ Project Structure

```
aqi-forecast/
├── aqi_pipeline.py          # Full end-to-end ML pipeline
├── outputs/
│   ├── aqi_dataset.csv      # Generated dataset (5000 hourly records)
│   ├── eda_plots.png        # Exploratory Data Analysis visualizations
│   └── model_evaluation.png # Model comparison & evaluation plots
├── models/
│   ├── random_forest.pkl    # Trained Random Forest model
│   ├── xgboost.pkl          # Trained XGBoost model
│   └── preprocessor.pkl     # Fitted imputer + scaler + feature list
└── README.md
```

---

## ⚙️ Pipeline Stages

1. **Dataset Generation** — 5000 hourly records with realistic seasonal/diurnal patterns and ~3% missing values injected
2. **Exploratory Data Analysis** — AQI distribution, hourly/monthly trends, correlation heatmap, pollutant scatter plots, category breakdown
3. **Feature Engineering** — Lag features (1h, 3h, 6h, 24h), 6h & 24h rolling means, weekend flag, time features
4. **Preprocessing** — Median imputation, Standard scaling, temporal 80/20 train-test split
5. **Model Training** — Random Forest, XGBoost, Gradient Boosting with tuned hyperparameters
6. **Evaluation** — Actual vs Predicted, residual distributions, feature importances, time-series overlay
7. **Model Persistence** — All models and preprocessor saved via `joblib`
8. **Inference Demo** — Load saved models and predict on new unseen input

---

## 🧪 Features Used (22 total)

| Category | Features |
|---|---|
| Time | `hour`, `month`, `day_of_week`, `is_weekend` |
| Pollutants | `PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3` |
| Meteorology | `temperature`, `humidity`, `wind_speed`, `wind_direction`, `pressure` |
| Lag features | `AQI_lag1`, `AQI_lag3`, `AQI_lag6`, `AQI_lag24`, `PM25_lag1` |
| Rolling features | `AQI_roll6`, `AQI_roll24` |

### Top-5 Features by Importance (Random Forest)
| Feature | Importance |
|---|---|
| PM2.5 | 0.4510 |
| PM10 | 0.3782 |
| Hour of Day | 0.0360 |
| Temperature | 0.0255 |
| PM2.5 Lag-1 | 0.0102 |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/aqi-forecast.git
cd aqi-forecast
```

### 2. Install dependencies
```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn joblib
```

### 3. Run the full pipeline
```bash
python aqi_pipeline.py
```

This will generate the dataset, train all models, save evaluation plots, and run an inference demo — all in one go.

---

## 🔮 Inference on New Data

```python
import joblib
import pandas as pd

# Load saved artifacts
pp  = joblib.load("models/preprocessor.pkl")
rf  = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")

# Prepare a new sample
sample = pd.DataFrame([{
    "hour": 8, "month": 3, "day_of_week": 1, "is_weekend": 0,
    "PM2.5": 45.2, "PM10": 72.1, "NO2": 55.3, "CO": 1.8,
    "SO2": 12.4, "O3": 38.6, "temperature": 18.5, "humidity": 65.2,
    "wind_speed": 3.1, "wind_direction": 215, "pressure": 1010.5,
    "AQI_lag1": 98.0, "AQI_lag3": 95.5, "AQI_lag6": 90.2,
    "AQI_lag24": 88.4, "AQI_roll6": 93.7, "AQI_roll24": 91.0,
    "PM25_lag1": 42.0
}])

X = pp["imputer"].transform(sample[pp["features"]])
print("RF  Prediction:", rf.predict(X)[0].round(1))
print("XGB Prediction:", xgb.predict(X)[0].round(1))
```

---

## 📈 AQI Scale Reference (US EPA)

| AQI Range | Category | Health Implication |
|---|---|---|
| 0 – 50 | 🟢 Good | Air quality is satisfactory |
| 51 – 100 | 🟡 Moderate | Acceptable for most people |
| 101 – 150 | 🟠 Unhealthy for Sensitive Groups | Sensitive groups may be affected |
| 151 – 200 | 🔴 Unhealthy | Everyone may begin to experience effects |
| 201 – 300 | 🟣 Very Unhealthy | Health alert — serious effects |
| 301 – 500 | 🟤 Hazardous | Emergency conditions |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- [scikit-learn](https://scikit-learn.org/) — Random Forest, Gradient Boosting, preprocessing
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient boosted trees
- [pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) — Data manipulation
- [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) — Visualizations
- [joblib](https://joblib.readthedocs.io/) — Model persistence

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
