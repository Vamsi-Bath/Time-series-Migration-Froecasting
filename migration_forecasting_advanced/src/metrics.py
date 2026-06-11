from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, 1e-9)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100)


def mase(y_true, y_pred, y_train, seasonality: int = 12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= seasonality:
        naive_error = np.mean(np.abs(np.diff(y_train))) if len(y_train) > 1 else 1
    else:
        naive_error = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    naive_error = max(float(naive_error), 1e-9)
    return float(np.mean(np.abs(y_true - y_pred)) / naive_error)


def metric_dict(y_true, y_pred, y_train=None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE %": mape(y_true, y_pred),
        "sMAPE %": smape(y_true, y_pred),
    }
    if len(y_true) >= 2:
        metrics["R2"] = float(r2_score(y_true, y_pred))
    if y_train is not None:
        metrics["MASE"] = mase(y_true, y_pred, y_train)
    return metrics


def metrics_table(results) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({"Model": r.name, **r.metrics})
    if not rows:
        return pd.DataFrame()
    preferred = ["Model", "MAE", "RMSE", "MAPE %", "sMAPE %", "MASE", "R2"]
    table = pd.DataFrame(rows)
    cols = [c for c in preferred if c in table.columns]
    return table[cols].sort_values("RMSE")
