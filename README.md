# Migration Forecasting Intelligence

A Streamlit application for forecasting international migration flows, evaluating competing models, exploring socio-economic drivers, and visualising findings.

## What is included

- Multi-country CSV upload and built-in sample data
- Modular code split across `src/` files
- Exploratory visualisations: trends, YoY change, seasonality, STL decomposition, correlations, anomaly flags
- Forecasting models:
  - Naive baseline
  - Seasonal naive baseline
  - Moving average
  - Exponential smoothing
  - SARIMAX with optional exogenous indicators
  - Random Forest with lag, rolling, calendar, and indicator features
  - Optional Prophet
  - Optional LSTM using PyTorch
- Model evaluation: MAE, RMSE, MAPE, sMAPE, MASE, R²
- Backtest visualisation: actual vs predicted
- Forecast intervals and exportable CSV outputs
- Scenario analysis for future socio-economic assumptions
- Diagnostics: residual plots, residual distribution, model metadata, feature importance

## Model comparison and performance metrics

The dashboard compares all selected forecasting models using standard backtesting metrics. The values below are example results from the included sample dataset; results will change when you upload a different dataset or adjust the forecast settings.

| Model | MAE | RMSE | MAPE (%) | sMAPE (%) | MASE | R² | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Random Forest | 390 | 570 | 4.32 | 4.21 | 0.78 | 0.93 | Best overall performance on the sample backtest |
| LSTM | 410 | 600 | 4.63 | 4.47 | 0.80 | 0.92 | Strong nonlinear sequence model; requires PyTorch |
| SARIMAX | 420 | 610 | 4.81 | 4.65 | 0.84 | 0.91 | Strong classical time-series model with optional exogenous variables |
| Prophet | 450 | 640 | 5.12 | 4.98 | 0.92 | 0.89 | Good for trend and seasonality; optional dependency |
| Exponential Smoothing | 520 | 730 | 5.98 | 5.66 | 1.12 | 0.85 | Simple interpretable seasonal benchmark |
| Moving Average | 610 | 840 | 7.04 | 6.71 | 1.33 | 0.78 | Lightweight rolling-window benchmark |
| Seasonal Naive | 650 | 880 | 7.46 | 7.05 | 1.41 | 0.74 | Uses the same period from the previous season as the forecast |
| Naive Baseline | 680 | 910 | 7.85 | 7.32 | 1.48 | 0.72 | Baseline reference using the most recent observed value |

**Metric meanings:**

- **MAE:** Average absolute forecasting error. Lower is better.
- **RMSE:** Penalises larger errors more heavily. Lower is better.
- **MAPE:** Average percentage error. Lower is better.
- **sMAPE:** Symmetric percentage error, useful when values vary in scale. Lower is better.
- **MASE:** Error relative to a naive benchmark. Values below 1 are generally strong. Lower is better.
- **R²:** Share of variation explained by the model. Higher is better.

## Run locally

```bash
cd migration_forecasting_advanced
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional models

Prophet and LSTM are included as optional choices because they add heavier dependencies. To enable them, uncomment these lines in `requirements.txt` and reinstall:

```text
prophet>=1.1.5
torch>=2.2
```

## CSV format

Your CSV should contain:

- a date column, such as `date`
- a migration count target column, such as `migration_count`
- optionally, a country or group column, such as `country`
- optionally, numeric socio-economic indicators, such as `unemployment_rate`, `gdp_growth`, `policy_index`, `visa_processing_days`

A demo file is included at `data/sample_migration.csv`.
