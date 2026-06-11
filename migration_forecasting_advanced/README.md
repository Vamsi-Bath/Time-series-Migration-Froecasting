# International Migration Forecasting Intelligence

A modular Streamlit application for forecasting international migration flows, evaluating competing models, exploring socio-economic drivers, and visualising findings.

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
