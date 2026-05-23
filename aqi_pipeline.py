"""
AQI Forecast - End-to-End ML Pipeline
Dataset: Based on UCI Air Quality Dataset structure (real-world features)
Models: Random Forest + XGBoost with full pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

np.random.seed(42)
OUTPUT = "/home/claude/aqi_project/outputs"
MODELS = "/home/claude/aqi_project/models"
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET  (UCI AQI-style features)
# ─────────────────────────────────────────────
print("=" * 60)
print("  AIR QUALITY INDEX FORECAST — ML PIPELINE")
print("=" * 60)
print("\n[1/7] Generating dataset …")

N = 5000
dates = pd.date_range("2019-01-01", periods=N, freq="h")

# Time features
hour  = dates.hour.to_numpy()
month = dates.month.to_numpy()
dow   = dates.dayofweek.to_numpy()          # 0=Mon

# Pollutant concentrations (µg/m³ / ppm)
PM25  = np.abs(np.random.normal(35, 20, N) + 15*np.sin(2*np.pi*hour/24)
               + 10*np.sin(2*np.pi*month/12) + np.random.normal(0, 5, N))
PM10  = PM25 * np.random.uniform(1.3, 2.0, N) + np.random.normal(0, 8, N)
NO2   = np.abs(np.random.normal(40, 15, N) + 8*np.sin(2*np.pi*hour/24))
CO    = np.abs(np.random.normal(1.2, 0.6, N) + 0.3*np.sin(2*np.pi*hour/24))
SO2   = np.abs(np.random.normal(10, 5, N))
O3    = np.abs(np.random.normal(50, 20, N) - 10*np.sin(2*np.pi*hour/24)
               + 5*np.sin(2*np.pi*month/12))

# Meteorological features
temp     = 22 + 8*np.sin(2*np.pi*(hour-14)/24) + 6*np.sin(2*np.pi*(month-7)/12) + np.random.normal(0,2,N)
humidity = np.clip(60 - 0.5*temp + np.random.normal(0,10,N), 10, 100)
wind_spd = np.abs(np.random.weibull(2, N)*5 + 1)
wind_dir = np.random.uniform(0, 360, N)
pressure = np.random.normal(1013, 8, N)

# Inject missing values (~3%)
for arr in [PM25, PM10, NO2, CO, SO2, O3, temp, humidity, wind_spd]:
    idx = np.random.choice(N, int(0.03*N), replace=False)
    arr[idx] = np.nan

# AQI target  (US EPA breakpoints simplified)
def compute_aqi(pm25):
    aqi = np.where(pm25 < 12,   pm25/12*50,
          np.where(pm25 < 35.4, 51 + (pm25-12)/23.4*49,
          np.where(pm25 < 55.4, 101+ (pm25-35.4)/19.9*49,
          np.where(pm25 < 150.4,151+ (pm25-55.4)/94.9*49,
          200 + pm25/10))))
    return np.clip(aqi, 0, 500)

pm25_filled = np.where(np.isnan(PM25), np.nanmean(PM25), PM25)
AQI = compute_aqi(pm25_filled) + np.random.normal(0, 5, N)
AQI = np.clip(AQI, 0, 500)

df = pd.DataFrame({
    "datetime": dates, "hour": hour, "month": month, "day_of_week": dow,
    "PM2.5": PM25, "PM10": PM10, "NO2": NO2, "CO": CO, "SO2": SO2, "O3": O3,
    "temperature": temp, "humidity": humidity, "wind_speed": wind_spd,
    "wind_direction": wind_dir, "pressure": pressure, "AQI": AQI
})

# Lag + rolling features
df = df.sort_values("datetime").reset_index(drop=True)
df["AQI_lag1"]  = df["AQI"].shift(1)
df["AQI_lag3"]  = df["AQI"].shift(3)
df["AQI_lag6"]  = df["AQI"].shift(6)
df["AQI_lag24"] = df["AQI"].shift(24)
df["AQI_roll6"] = df["AQI"].shift(1).rolling(6).mean()
df["AQI_roll24"]= df["AQI"].shift(1).rolling(24).mean()
df["PM25_lag1"] = df["PM2.5"].shift(1)
df["is_weekend"]= (df["day_of_week"] >= 5).astype(int)
df = df.dropna().reset_index(drop=True)

print(f"   Dataset shape : {df.shape}")
print(f"   AQI range     : {df['AQI'].min():.1f} – {df['AQI'].max():.1f}")
print(f"   Missing values: {df.isnull().sum().sum()}")

df.to_csv(f"{OUTPUT}/aqi_dataset.csv", index=False)

# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
print("\n[2/7] Exploratory Data Analysis …")

plt.rcParams.update({"font.family": "monospace", "axes.spines.top": False,
                      "axes.spines.right": False})
PALETTE = ["#0F4C81","#E05C2A","#2E9E6E","#C4A000","#8B2FC9","#D63D6F"]

fig = plt.figure(figsize=(20, 16), facecolor="#0D1117")
fig.suptitle("AQI Dataset — Exploratory Data Analysis",
             fontsize=20, color="white", fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

def dark_ax(ax):
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="#8B949E", labelsize=9)
    ax.xaxis.label.set_color("#8B949E")
    ax.yaxis.label.set_color("#8B949E")
    ax.title.set_color("#E6EDF3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

# AQI distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["AQI"], bins=50, color=PALETTE[0], edgecolor="#0D1117", alpha=0.9)
ax1.axvline(df["AQI"].mean(), color=PALETTE[1], lw=1.5, ls="--", label=f"Mean={df['AQI'].mean():.1f}")
ax1.legend(fontsize=8, labelcolor="white", facecolor="#161B22")
ax1.set_title("AQI Distribution"); ax1.set_xlabel("AQI"); ax1.set_ylabel("Count")
dark_ax(ax1)

# AQI by hour
ax2 = fig.add_subplot(gs[0, 1])
hourly = df.groupby("hour")["AQI"].mean()
ax2.plot(hourly.index, hourly.values, color=PALETTE[2], lw=2, marker="o", ms=3)
ax2.fill_between(hourly.index, hourly.values, alpha=0.2, color=PALETTE[2])
ax2.set_title("Avg AQI by Hour"); ax2.set_xlabel("Hour"); ax2.set_ylabel("AQI")
dark_ax(ax2)

# AQI by month
ax3 = fig.add_subplot(gs[0, 2])
monthly = df.groupby("month")["AQI"].mean()
bars = ax3.bar(monthly.index, monthly.values, color=PALETTE[3], edgecolor="#0D1117")
ax3.set_title("Avg AQI by Month"); ax3.set_xlabel("Month"); ax3.set_ylabel("AQI")
dark_ax(ax3)

# Correlation heatmap
ax4 = fig.add_subplot(gs[1, :2])
cols = ["PM2.5","PM10","NO2","CO","SO2","O3","temperature","humidity","wind_speed","AQI"]
corr = df[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, ax=ax4, cmap=cmap, annot=True, fmt=".2f",
            annot_kws={"size":7}, linewidths=0.5, linecolor="#0D1117",
            cbar_kws={"shrink":0.7})
ax4.set_title("Feature Correlation Matrix")
ax4.set_facecolor("#161B22"); ax4.tick_params(colors="#8B949E", labelsize=8)
ax4.title.set_color("#E6EDF3")

# Pollutant scatter vs AQI
ax5 = fig.add_subplot(gs[1, 2])
sample = df.sample(800)
sc = ax5.scatter(sample["PM2.5"], sample["AQI"], c=sample["NO2"],
                  cmap="plasma", alpha=0.6, s=12)
plt.colorbar(sc, ax=ax5, label="NO2").ax.yaxis.label.set_color("#8B949E")
ax5.set_title("PM2.5 vs AQI (colored by NO2)")
ax5.set_xlabel("PM2.5"); ax5.set_ylabel("AQI")
dark_ax(ax5)

# AQI category pie
ax6 = fig.add_subplot(gs[2, 0])
bins   = [0,50,100,150,200,300,500]
labels = ["Good","Moderate","USG","Unhealthy","Very Unhealthy","Hazardous"]
cats   = pd.cut(df["AQI"], bins=bins, labels=labels)
sizes  = cats.value_counts()
ax6.pie(sizes, labels=sizes.index, colors=PALETTE[:len(sizes)],
        autopct="%1.1f%%", textprops={"color":"white","fontsize":8})
ax6.set_title("AQI Category Split", color="#E6EDF3")
ax6.set_facecolor("#161B22")

# Time series snippet
ax7 = fig.add_subplot(gs[2, 1:])
sample_ts = df.head(7*24)
ax7.plot(range(len(sample_ts)), sample_ts["AQI"], color=PALETTE[5], lw=1.2)
ax7.fill_between(range(len(sample_ts)), sample_ts["AQI"], alpha=0.15, color=PALETTE[5])
ax7.set_title("AQI Time Series — First 7 Days")
ax7.set_xlabel("Hours"); ax7.set_ylabel("AQI")
dark_ax(ax7)

plt.savefig(f"{OUTPUT}/eda_plots.png", dpi=150, bbox_inches="tight",
            facecolor="#0D1117")
plt.close()
print("   EDA plots saved.")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────
print("\n[3/7] Feature Engineering & Preprocessing …")

FEATURES = [
    "hour","month","day_of_week","is_weekend",
    "PM2.5","PM10","NO2","CO","SO2","O3",
    "temperature","humidity","wind_speed","wind_direction","pressure",
    "AQI_lag1","AQI_lag3","AQI_lag6","AQI_lag24",
    "AQI_roll6","AQI_roll24","PM25_lag1"
]
TARGET = "AQI"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)   # temporal split

print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)
X_train_sc  = scaler.fit_transform(X_train_imp)
X_test_sc   = scaler.transform(X_test_imp)

# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n[4/7] Training Models …")

results = {}

# --- Random Forest ---
print("   → Random Forest …")
rf = RandomForestRegressor(
    n_estimators=200, max_depth=20, min_samples_split=5,
    min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)
rf.fit(X_train_imp, y_train)
rf_pred = rf.predict(X_test_imp)
results["Random Forest"] = {
    "model": rf, "pred": rf_pred,
    "MAE" : mean_absolute_error(y_test, rf_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
    "R2"  : r2_score(y_test, rf_pred),
    "MAPE": np.mean(np.abs((y_test.values - rf_pred) / np.clip(y_test.values, 5, None))) * 100
}
print(f"     MAE={results['Random Forest']['MAE']:.2f}  "
      f"RMSE={results['Random Forest']['RMSE']:.2f}  "
      f"R²={results['Random Forest']['R2']:.4f}")

# --- XGBoost ---
print("   → XGBoost …")
xgb = XGBRegressor(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    eval_metric="rmse", verbosity=0)
xgb.fit(X_train_imp, y_train,
        eval_set=[(X_test_imp, y_test)], verbose=False)
xgb_pred = xgb.predict(X_test_imp)
results["XGBoost"] = {
    "model": xgb, "pred": xgb_pred,
    "MAE" : mean_absolute_error(y_test, xgb_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, xgb_pred)),
    "R2"  : r2_score(y_test, xgb_pred),
    "MAPE": np.mean(np.abs((y_test.values - xgb_pred) / np.clip(y_test.values, 5, None))) * 100
}
print(f"     MAE={results['XGBoost']['MAE']:.2f}  "
      f"RMSE={results['XGBoost']['RMSE']:.2f}  "
      f"R²={results['XGBoost']['R2']:.4f}")

# --- Gradient Boosting (baseline comparison) ---
print("   → Gradient Boosting (baseline) …")
gb = GradientBoostingRegressor(
    n_estimators=150, max_depth=5, learning_rate=0.08,
    subsample=0.8, random_state=42)
gb.fit(X_train_imp, y_train)
gb_pred = gb.predict(X_test_imp)
results["Gradient Boosting"] = {
    "model": gb, "pred": gb_pred,
    "MAE" : mean_absolute_error(y_test, gb_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, gb_pred)),
    "R2"  : r2_score(y_test, gb_pred),
    "MAPE": np.mean(np.abs((y_test.values - gb_pred) / np.clip(y_test.values, 5, None))) * 100
}
print(f"     MAE={results['Gradient Boosting']['MAE']:.2f}  "
      f"RMSE={results['Gradient Boosting']['RMSE']:.2f}  "
      f"R²={results['Gradient Boosting']['R2']:.4f}")

# Save models
joblib.dump(rf,  f"{MODELS}/random_forest.pkl")
joblib.dump(xgb, f"{MODELS}/xgboost.pkl")
joblib.dump({"imputer": imputer, "scaler": scaler, "features": FEATURES},
            f"{MODELS}/preprocessor.pkl")
print("\n   Models saved to /models/")

# ─────────────────────────────────────────────
# 5. EVALUATION PLOTS
# ─────────────────────────────────────────────
print("\n[5/7] Generating Evaluation Plots …")

fig, axes = plt.subplots(3, 3, figsize=(20, 16), facecolor="#0D1117")
fig.suptitle("Model Evaluation — AQI Forecast",
             fontsize=20, color="white", fontweight="bold", y=0.99)
plt.subplots_adjust(hspace=0.45, wspace=0.35)

model_colors = {"Random Forest": PALETTE[0],
                "XGBoost": PALETTE[1],
                "Gradient Boosting": PALETTE[2]}

# Row 0: Actual vs Predicted (RF + XGB)
for i, mname in enumerate(["Random Forest","XGBoost"]):
    ax = axes[0][i]
    ax.set_facecolor("#161B22")
    pred = results[mname]["pred"]
    ax.scatter(y_test, pred, alpha=0.3, s=8, color=model_colors[mname])
    lim = [min(y_test.min(), pred.min())-5, max(y_test.max(), pred.max())+5]
    ax.plot(lim, lim, "w--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"{mname}\nActual vs Predicted", color="#E6EDF3", fontsize=10)
    ax.set_xlabel("Actual AQI", color="#8B949E", fontsize=9)
    ax.set_ylabel("Predicted AQI", color="#8B949E", fontsize=9)
    ax.tick_params(colors="#8B949E")
    for s in ax.spines.values(): s.set_edgecolor("#30363D")
    r2 = results[mname]["R2"]
    ax.text(0.05, 0.92, f"R²={r2:.4f}", transform=ax.transAxes,
            color="white", fontsize=9, bbox=dict(fc="#0D1117", alpha=0.7))

# Metrics bar chart
ax = axes[0][2]; ax.set_facecolor("#161B22")
metrics = ["MAE","RMSE","R2","MAPE"]
x = np.arange(len(metrics)); w = 0.25
for j, (mname, col) in enumerate(model_colors.items()):
    vals = [results[mname][m] for m in metrics]
    ax.bar(x + j*w, vals, w, label=mname, color=col, edgecolor="#0D1117", alpha=0.9)
ax.set_xticks(x + w); ax.set_xticklabels(metrics, color="#8B949E", fontsize=9)
ax.set_title("Metrics Comparison", color="#E6EDF3", fontsize=10)
ax.legend(fontsize=7, labelcolor="white", facecolor="#161B22")
ax.tick_params(colors="#8B949E")
for s in ax.spines.values(): s.set_edgecolor("#30363D")

# Row 1: Time series forecast (RF)
ax = axes[1][0:2].reshape(-1)
ax_ts = fig.add_subplot(3,3,(4,5))  # span 2 cols
n_show = 200
idx = np.arange(n_show)
ax_ts.plot(idx, y_test.values[:n_show], color="white", lw=1.2, label="Actual", alpha=0.9)
ax_ts.plot(idx, results["Random Forest"]["pred"][:n_show],
          color=PALETTE[0], lw=1.2, ls="--", label="RF Pred", alpha=0.85)
ax_ts.plot(idx, results["XGBoost"]["pred"][:n_show],
          color=PALETTE[1], lw=1.2, ls=":", label="XGB Pred", alpha=0.85)
ax_ts.set_facecolor("#161B22")
ax_ts.set_title("Forecast vs Actual (first 200 test samples)",
               color="#E6EDF3", fontsize=10)
ax_ts.set_xlabel("Sample", color="#8B949E"); ax_ts.set_ylabel("AQI", color="#8B949E")
ax_ts.tick_params(colors="#8B949E")
ax_ts.legend(fontsize=8, labelcolor="white", facecolor="#161B22")
for s in ax_ts.spines.values(): s.set_edgecolor("#30363D")
# hide the two slots used
axes[1][0].set_visible(False); axes[1][1].set_visible(False)

# Residual distribution
ax = axes[1][2]; ax.set_facecolor("#161B22")
for mname, col in model_colors.items():
    resid = y_test.values - results[mname]["pred"]
    ax.hist(resid, bins=40, alpha=0.5, color=col, label=mname, edgecolor="#0D1117")
ax.axvline(0, color="white", lw=1.2, ls="--")
ax.set_title("Residual Distribution", color="#E6EDF3", fontsize=10)
ax.set_xlabel("Residual", color="#8B949E"); ax.tick_params(colors="#8B949E")
ax.legend(fontsize=7, labelcolor="white", facecolor="#161B22")
for s in ax.spines.values(): s.set_edgecolor("#30363D")

# Row 2: Feature Importance (RF)
ax = axes[2][0:2].reshape(-1)
ax_fi = fig.add_subplot(3,3,(7,8))
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(15)
colors_fi = plt.cm.YlOrRd(np.linspace(0.3, 1, len(fi)))
ax_fi.barh(fi.index, fi.values, color=colors_fi, edgecolor="#0D1117")
ax_fi.set_facecolor("#161B22")
ax_fi.set_title("Random Forest — Top 15 Feature Importances",
               color="#E6EDF3", fontsize=10)
ax_fi.tick_params(colors="#8B949E", labelsize=8)
for s in ax_fi.spines.values(): s.set_edgecolor("#30363D")
axes[2][0].set_visible(False); axes[2][1].set_visible(False)

# XGB feature importance
ax_xfi = axes[2][2]; ax_xfi.set_facecolor("#161B22")
xfi = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(15)
colors_xfi = plt.cm.Blues(np.linspace(0.3, 1, len(xfi)))
ax_xfi.barh(xfi.index, xfi.values, color=colors_xfi, edgecolor="#0D1117")
ax_xfi.set_title("XGBoost — Top 15 Feature Importances",
                color="#E6EDF3", fontsize=10)
ax_xfi.tick_params(colors="#8B949E", labelsize=8)
for s in ax_xfi.spines.values(): s.set_edgecolor("#30363D")

plt.savefig(f"{OUTPUT}/model_evaluation.png", dpi=150,
            bbox_inches="tight", facecolor="#0D1117")
plt.close()
print("   Evaluation plots saved.")

# ─────────────────────────────────────────────
# 6. SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n[6/7] Writing Summary Report …")

best = min(results, key=lambda m: results[m]["RMSE"])
report = f"""
╔══════════════════════════════════════════════════════════════╗
║          AQI FORECAST — RESULTS SUMMARY                      ║
╚══════════════════════════════════════════════════════════════╝

Dataset         : 5000 hourly records (synthetic, UCI-style features)
Train / Test    : 80% / 20% (temporal split — no data leakage)
Target          : Air Quality Index (AQI, 0-500 EPA scale)
Features used   : {len(FEATURES)} (pollutants + meteorology + lag/rolling)

───────────────────────────────────────────────────────────────
 MODEL              MAE      RMSE     R²       MAPE (%)
───────────────────────────────────────────────────────────────
"""
for mname, r in results.items():
    star = " ★" if mname == best else "  "
    report += f" {mname:<20}{r['MAE']:>6.2f}  {r['RMSE']:>7.2f}  {r['R2']:>6.4f}  {r['MAPE']:>8.2f}{star}\n"

report += f"""───────────────────────────────────────────────────────────────
 ★ Best model by RMSE: {best}

AQI CATEGORY BREAKDOWN (test set — actual):
"""
bins_test   = [0,50,100,150,200,300,500]
labels_test = ["Good","Moderate","USG","Unhealthy","Very Unhealthy","Hazardous"]
cats_test   = pd.cut(y_test, bins=bins_test, labels=labels_test)
for cat, cnt in cats_test.value_counts().sort_index().items():
    report += f"  {cat:<20}: {cnt:>4} samples\n"

report += f"""
TOP-5 FEATURES (Random Forest):
"""
top5 = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(5)
for feat, imp in top5.items():
    report += f"  {feat:<20}: {imp:.4f}\n"

report += """
OUTPUT FILES:
  outputs/aqi_dataset.csv        — Full dataset with engineered features
  outputs/eda_plots.png          — EDA visualizations
  outputs/model_evaluation.png   — Model evaluation plots
  models/random_forest.pkl       — Trained Random Forest model
  models/xgboost.pkl             — Trained XGBoost model
  models/preprocessor.pkl        — Imputer + Scaler + feature list
"""
print(report)
with open(f"{OUTPUT}/summary_report.txt", "w") as f:
    f.write(report)

# ─────────────────────────────────────────────
# 7. INFERENCE DEMO
# ─────────────────────────────────────────────
print("[7/7] Inference Demo — predicting on a new sample …\n")

pp = joblib.load(f"{MODELS}/preprocessor.pkl")
sample_input = pd.DataFrame([{
    "hour": 8, "month": 3, "day_of_week": 1, "is_weekend": 0,
    "PM2.5": 45.2, "PM10": 72.1, "NO2": 55.3, "CO": 1.8,
    "SO2": 12.4, "O3": 38.6,
    "temperature": 18.5, "humidity": 65.2, "wind_speed": 3.1,
    "wind_direction": 215, "pressure": 1010.5,
    "AQI_lag1": 98.0, "AQI_lag3": 95.5, "AQI_lag6": 90.2,
    "AQI_lag24": 88.4, "AQI_roll6": 93.7, "AQI_roll24": 91.0,
    "PM25_lag1": 42.0
}])
inp = pp["imputer"].transform(sample_input[pp["features"]])
rf_loaded  = joblib.load(f"{MODELS}/random_forest.pkl")
xgb_loaded = joblib.load(f"{MODELS}/xgboost.pkl")
rf_out  = rf_loaded.predict(inp)[0]
xgb_out = xgb_loaded.predict(inp)[0]

def aqi_label(v):
    if v<=50: return "Good 🟢"
    if v<=100: return "Moderate 🟡"
    if v<=150: return "Unhealthy for Sensitive Groups 🟠"
    if v<=200: return "Unhealthy 🔴"
    return "Very Unhealthy / Hazardous 🟣"

print(f"  Input: PM2.5=45.2 µg/m³, NO2=55.3 ppb, Temp=18.5°C, AQI_lag1=98")
print(f"  Random Forest prediction : AQI = {rf_out:.1f}  → {aqi_label(rf_out)}")
print(f"  XGBoost prediction       : AQI = {xgb_out:.1f}  → {aqi_label(xgb_out)}")
print("\n✅ Pipeline complete! Check /outputs/ for all artifacts.\n")
