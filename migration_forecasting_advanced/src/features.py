from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_TARGET_COLUMN


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["year"] = out["date"].dt.year
    out["time_idx"] = np.arange(len(out))
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
    lags: Iterable[int] = (1, 2, 3, 6, 12),
    rolling_windows: Iterable[int] = (3, 6, 12),
) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    for window in rolling_windows:
        out[f"rolling_mean_{window}"] = out[target_col].shift(1).rolling(window).mean()
        out[f"rolling_std_{window}"] = out[target_col].shift(1).rolling(window).std()
    out["yoy_change"] = out[target_col].pct_change(12)
    return out


def make_ml_frame(
    df: pd.DataFrame,
    exog_cols: list[str],
    target_col: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, list[str]]:
    out = add_calendar_features(df)
    out = add_lag_features(out, target_col=target_col)
    feature_cols = [
        "month",
        "quarter",
        "year",
        "time_idx",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_mean_6",
        "rolling_std_6",
        "rolling_mean_12",
        "rolling_std_12",
    ] + [c for c in exog_cols if c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, feature_cols


def future_exog_frame(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    exog_cols: list[str],
    adjustments: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build simple future exogenous assumptions using recent averages and scenario adjustments."""
    adjustments = adjustments or {}
    rows = []
    recent = history.tail(min(12, len(history)))
    for date in future_dates:
        row = {"date": date}
        for col in exog_cols:
            if col not in history.columns:
                continue
            value = float(pd.to_numeric(recent[col], errors="coerce").mean())
            # Adjustment is interpreted as percentage change, e.g. +10 means +10%.
            value *= 1 + adjustments.get(col, 0.0) / 100.0
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)
