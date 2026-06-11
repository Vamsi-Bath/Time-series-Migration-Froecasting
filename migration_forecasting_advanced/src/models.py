from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import DEFAULT_TARGET_COLUMN
from .data import infer_frequency
from .features import future_exog_frame, make_ml_frame
from .metrics import metric_dict

try:
    from prophet import Prophet  # type: ignore
except Exception:  # pragma: no cover
    Prophet = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass
class ForecastResult:
    name: str
    fitted: pd.DataFrame
    forecast: pd.DataFrame
    metrics: dict[str, float]
    residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_info: dict[str, str | float | int] = field(default_factory=dict)
    feature_importance: Optional[pd.DataFrame] = None


def _future_dates(df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    freq = infer_frequency(df["date"])
    offset = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(df["date"].max() + offset, periods=horizon, freq=freq)


def _train_test(df: pd.DataFrame, test_periods: int):
    test_periods = int(max(1, min(test_periods, max(1, len(df) // 3))))
    return df.iloc[:-test_periods].copy(), df.iloc[-test_periods:].copy()


def _package(name: str, fitted: pd.DataFrame, forecast: pd.DataFrame, y_train, model_info=None, feature_importance=None) -> ForecastResult:
    fitted = fitted.copy()
    fitted["residual"] = fitted["actual"] - fitted["prediction"]
    residuals = fitted[["date", "actual", "prediction", "residual"]].copy()
    return ForecastResult(
        name=name,
        fitted=fitted,
        forecast=forecast,
        metrics=metric_dict(fitted["actual"], fitted["prediction"], y_train=y_train),
        residuals=residuals,
        model_info=model_info or {},
        feature_importance=feature_importance,
    )


def fit_naive(df: pd.DataFrame, horizon: int, test_periods: int, exog_cols=None, scenario=None) -> ForecastResult:
    train, test = _train_test(df, test_periods)
    pred = np.repeat(train[DEFAULT_TARGET_COLUMN].iloc[-1], len(test))
    future_pred = np.repeat(df[DEFAULT_TARGET_COLUMN].iloc[-1], horizon)
    resid_std = float(np.std(test[DEFAULT_TARGET_COLUMN].values - pred))
    fitted = pd.DataFrame({"date": test["date"], "actual": test[DEFAULT_TARGET_COLUMN], "prediction": pred})
    forecast = pd.DataFrame({"date": _future_dates(df, horizon), "prediction": future_pred})
    forecast["lower"] = forecast["prediction"] - 1.64 * resid_std
    forecast["upper"] = forecast["prediction"] + 1.64 * resid_std
    return _package("Naive", fitted, forecast, train[DEFAULT_TARGET_COLUMN], {"type": "last observed value baseline"})


def fit_seasonal_naive(df: pd.DataFrame, horizon: int, test_periods: int, seasonality: int = 12, exog_cols=None, scenario=None) -> ForecastResult:
    train, test = _train_test(df, test_periods)
    history = train[DEFAULT_TARGET_COLUMN].values
    preds = []
    for i in range(len(test)):
        idx = len(history) - seasonality + i
        preds.append(history[idx] if idx >= 0 and idx < len(history) else history[-1])
    seasonal_values = df[DEFAULT_TARGET_COLUMN].tail(seasonality).values
    future_pred = [seasonal_values[i % len(seasonal_values)] for i in range(horizon)]
    resid_std = float(np.std(test[DEFAULT_TARGET_COLUMN].values - np.asarray(preds)))
    fitted = pd.DataFrame({"date": test["date"], "actual": test[DEFAULT_TARGET_COLUMN], "prediction": preds})
    forecast = pd.DataFrame({"date": _future_dates(df, horizon), "prediction": future_pred})
    forecast["lower"] = forecast["prediction"] - 1.64 * resid_std
    forecast["upper"] = forecast["prediction"] + 1.64 * resid_std
    return _package("Seasonal Naive", fitted, forecast, train[DEFAULT_TARGET_COLUMN], {"seasonality": seasonality})


def fit_moving_average(df: pd.DataFrame, horizon: int, test_periods: int, window: int = 6, exog_cols=None, scenario=None) -> ForecastResult:
    train, test = _train_test(df, test_periods)
    history = train[DEFAULT_TARGET_COLUMN].tolist()
    preds = []
    for actual in test[DEFAULT_TARGET_COLUMN].tolist():
        preds.append(float(np.mean(history[-window:])))
        history.append(actual)
    full_history = df[DEFAULT_TARGET_COLUMN].tolist()
    future_pred = []
    for _ in range(horizon):
        nxt = float(np.mean(full_history[-window:]))
        future_pred.append(nxt)
        full_history.append(nxt)
    resid_std = float(np.std(test[DEFAULT_TARGET_COLUMN].values - np.asarray(preds)))
    fitted = pd.DataFrame({"date": test["date"], "actual": test[DEFAULT_TARGET_COLUMN], "prediction": preds})
    forecast = pd.DataFrame({"date": _future_dates(df, horizon), "prediction": future_pred})
    forecast["lower"] = forecast["prediction"] - 1.64 * resid_std
    forecast["upper"] = forecast["prediction"] + 1.64 * resid_std
    return _package("Moving Average", fitted, forecast, train[DEFAULT_TARGET_COLUMN], {"window": window})


def fit_exponential_smoothing(df: pd.DataFrame, horizon: int, test_periods: int, exog_cols=None, scenario=None) -> ForecastResult:
    train, test = _train_test(df, test_periods)
    seasonal = "add" if len(train) >= 24 else None
    model = ExponentialSmoothing(train[DEFAULT_TARGET_COLUMN], trend="add", seasonal=seasonal, seasonal_periods=12 if seasonal else None)
    fitted_model = model.fit(optimized=True)
    pred = fitted_model.forecast(len(test))

    full_model = ExponentialSmoothing(df[DEFAULT_TARGET_COLUMN], trend="add", seasonal="add" if len(df) >= 24 else None, seasonal_periods=12 if len(df) >= 24 else None).fit(optimized=True)
    future_pred = full_model.forecast(horizon)
    resid_std = float(np.std(test[DEFAULT_TARGET_COLUMN].values - np.asarray(pred)))
    fitted = pd.DataFrame({"date": test["date"].values, "actual": test[DEFAULT_TARGET_COLUMN].values, "prediction": np.asarray(pred)})
    forecast = pd.DataFrame({"date": _future_dates(df, horizon), "prediction": np.asarray(future_pred)})
    forecast["lower"] = forecast["prediction"] - 1.64 * resid_std
    forecast["upper"] = forecast["prediction"] + 1.64 * resid_std
    return _package("Exponential Smoothing", fitted, forecast, train[DEFAULT_TARGET_COLUMN], {"trend": "add", "seasonal": seasonal or "none"})


def fit_sarimax(df: pd.DataFrame, horizon: int, test_periods: int, exog_cols: list[str], scenario=None) -> ForecastResult:
    data = df.set_index("date").asfreq(infer_frequency(df["date"]))
    y = data[DEFAULT_TARGET_COLUMN].interpolate().ffill().bfill()
    exog = data[exog_cols].apply(pd.to_numeric, errors="coerce").interpolate().ffill().bfill() if exog_cols else None
    train_y, test_y = y.iloc[:-test_periods], y.iloc[-test_periods:]
    train_exog = exog.iloc[:-test_periods] if exog is not None else None
    test_exog = exog.iloc[-test_periods:] if exog is not None else None

    model = SARIMAX(train_y, exog=train_exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)
    test_pred = fitted_model.get_forecast(steps=len(test_y), exog=test_exog).predicted_mean

    future_dates = _future_dates(df, horizon)
    future_exog = None
    if exog_cols:
        future_exog = future_exog_frame(df.reset_index(drop=True), future_dates, exog_cols, scenario).set_index("date")

    full_model = SARIMAX(y, exog=exog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    future = full_model.get_forecast(steps=horizon, exog=future_exog)
    conf = future.conf_int(alpha=0.10)
    fitted = pd.DataFrame({"date": test_y.index, "actual": test_y.values, "prediction": test_pred.values})
    forecast = pd.DataFrame({"date": future_dates, "prediction": future.predicted_mean.values, "lower": conf.iloc[:, 0].values, "upper": conf.iloc[:, 1].values})
    return _package("SARIMAX", fitted, forecast, train_y, {"order": "(1,1,1)", "seasonal_order": "(1,1,1,12)", "exogenous_variables": ", ".join(exog_cols) or "none"})


def fit_random_forest(df: pd.DataFrame, horizon: int, test_periods: int, exog_cols: list[str], scenario=None) -> ForecastResult:
    frame, feature_cols = make_ml_frame(df, exog_cols)
    frame = frame.dropna(subset=feature_cols + [DEFAULT_TARGET_COLUMN]).copy()
    if len(frame) < max(18, test_periods + 8):
        raise ValueError("Random Forest needs more complete rows after creating lag features. Try fewer lags or more history.")
    train, test = _train_test(frame, test_periods)
    model = RandomForestRegressor(n_estimators=450, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[feature_cols], train[DEFAULT_TARGET_COLUMN])
    pred = model.predict(test[feature_cols])

    # Refit on all data for future iterative forecasting.
    model.fit(frame[feature_cols], frame[DEFAULT_TARGET_COLUMN])
    history = df.sort_values("date").copy().reset_index(drop=True)
    future_dates = _future_dates(df, horizon)
    future_exog = future_exog_frame(history, future_dates, exog_cols, scenario)
    future_rows = []
    for i, date in enumerate(future_dates):
        row = {"date": date, DEFAULT_TARGET_COLUMN: np.nan}
        for col in exog_cols:
            row[col] = future_exog.loc[future_exog["date"] == date, col].iloc[0] if col in future_exog.columns else np.nan
        temp = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        temp_frame, _ = make_ml_frame(temp, exog_cols)
        features = temp_frame.iloc[[-1]][feature_cols].ffill(axis=0).fillna(method="ffill")
        features = features.fillna(frame[feature_cols].median(numeric_only=True))
        yhat = float(model.predict(features)[0])
        row[DEFAULT_TARGET_COLUMN] = yhat
        future_rows.append({"date": date, "prediction": yhat})
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    resid_std = float(np.std(test[DEFAULT_TARGET_COLUMN].values - pred))
    fitted = pd.DataFrame({"date": test["date"].values, "actual": test[DEFAULT_TARGET_COLUMN].values, "prediction": pred})
    forecast = pd.DataFrame(future_rows)
    forecast["lower"] = forecast["prediction"] - 1.64 * resid_std
    forecast["upper"] = forecast["prediction"] + 1.64 * resid_std
    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    return _package("Random Forest", fitted, forecast, train[DEFAULT_TARGET_COLUMN], {"trees": 450, "features": len(feature_cols)}, importance)


def fit_prophet(df: pd.DataFrame, horizon: int, test_periods: int, exog_cols: list[str], scenario=None) -> ForecastResult:
    if Prophet is None:
        raise ImportError("Prophet is not installed. Install prophet to enable this model.")
    data = df[["date", DEFAULT_TARGET_COLUMN] + exog_cols].rename(columns={"date": "ds", DEFAULT_TARGET_COLUMN: "y"}).copy()
    train, test = _train_test(data, test_periods)
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.90)
    for col in exog_cols:
        model.add_regressor(col)
    model.fit(train)
    pred_test = model.predict(test[["ds"] + exog_cols])

    future_dates = _future_dates(df, horizon)
    future = future_exog_frame(df, future_dates, exog_cols, scenario).rename(columns={"date": "ds"})
    future_pred = model.predict(future)
    fitted = pd.DataFrame({"date": test["ds"].values, "actual": test["y"].values, "prediction": pred_test["yhat"].values})
    forecast = future_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date", "yhat": "prediction", "yhat_lower": "lower", "yhat_upper": "upper"})
    return _package("Prophet", fitted, forecast, train["y"], {"yearly_seasonality": True, "exogenous_variables": ", ".join(exog_cols) or "none"})


if nn is not None:
    class LSTMRegressor(nn.Module):
        def __init__(self, n_features: int, hidden_size: int = 48):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.linear(out[:, -1, :])
else:
    LSTMRegressor = None


def fit_lstm(df: pd.DataFrame, horizon: int, test_periods: int, exog_cols: list[str], scenario=None, lookback: int = 12, epochs: int = 80) -> ForecastResult:
    if torch is None or LSTMRegressor is None:
        raise ImportError("PyTorch is not installed. Install torch to enable LSTM.")
    from sklearn.preprocessing import MinMaxScaler

    data = df[["date", DEFAULT_TARGET_COLUMN] + exog_cols].copy().sort_values("date")
    feature_cols = [DEFAULT_TARGET_COLUMN] + exog_cols
    if len(data) < lookback + test_periods + 12:
        raise ValueError("LSTM needs more history. Use at least ~36 observations.")
    scaler = MinMaxScaler()
    values = scaler.fit_transform(data[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill())

    x, y = [], []
    for i in range(lookback, len(values)):
        x.append(values[i - lookback : i])
        y.append(values[i, 0])
    X, Y = np.asarray(x), np.asarray(y)
    test_n = min(test_periods, len(Y) // 4)
    X_train, y_train = X[:-test_n], Y[:-test_n]
    X_test, y_test = X[-test_n:], Y[-test_n:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(n_features=X_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1)), batch_size=16, shuffle=False)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy().ravel()

    holder = np.zeros((len(pred_scaled), len(feature_cols)))
    holder[:, 0] = pred_scaled
    pred = scaler.inverse_transform(holder)[:, 0]
    actual_holder = np.zeros_like(holder)
    actual_holder[:, 0] = y_test
    actual = scaler.inverse_transform(actual_holder)[:, 0]
    fitted_dates = data["date"].iloc[-len(actual):].values

    history_scaled = values.copy()
    future_preds = []
    for _ in range(horizon):
        seq = history_scaled[-lookback:].reshape(1, lookback, len(feature_cols))
        with torch.no_grad():
            next_scaled_y = float(model(torch.tensor(seq, dtype=torch.float32).to(device)).cpu().numpy().ravel()[0])
        next_row = history_scaled[-min(12, len(history_scaled)):].mean(axis=0)
        next_row[0] = next_scaled_y
        history_scaled = np.vstack([history_scaled, next_row])
        future_preds.append(next_scaled_y)
    fholder = np.zeros((horizon, len(feature_cols)))
    fholder[:, 0] = future_preds
    future_actual = scaler.inverse_transform(fholder)[:, 0]
    resid_std = float(np.std(actual - pred))
    forecast = pd.DataFrame({"date": _future_dates(df, horizon), "prediction": future_actual})
    forecast["lower"] = forecast["prediction"] - 1.64 * resid_std
    forecast["upper"] = forecast["prediction"] + 1.64 * resid_std
    fitted = pd.DataFrame({"date": fitted_dates, "actual": actual, "prediction": pred})
    return _package("LSTM", fitted, forecast, data[DEFAULT_TARGET_COLUMN].iloc[: -len(actual)], {"lookback": lookback, "epochs": epochs})


MODEL_REGISTRY = {
    "Naive": fit_naive,
    "Seasonal Naive": fit_seasonal_naive,
    "Moving Average": fit_moving_average,
    "Exponential Smoothing": fit_exponential_smoothing,
    "SARIMAX": fit_sarimax,
    "Random Forest": fit_random_forest,
    "Prophet (optional)": fit_prophet,
    "LSTM (optional)": fit_lstm,
}
