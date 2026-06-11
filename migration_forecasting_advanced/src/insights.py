from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .config import DEFAULT_TARGET_COLUMN


def kpi_summary(df: pd.DataFrame) -> dict[str, str]:
    latest = float(df[DEFAULT_TARGET_COLUMN].iloc[-1])
    avg = float(df[DEFAULT_TARGET_COLUMN].mean())
    yoy = df[DEFAULT_TARGET_COLUMN].pct_change(12).iloc[-1] * 100 if len(df) > 12 else np.nan
    volatility = df[DEFAULT_TARGET_COLUMN].pct_change().std() * 100
    return {
        "Latest migration": f"{latest:,.0f}",
        "Average migration": f"{avg:,.0f}",
        "YoY change": "—" if pd.isna(yoy) else f"{yoy:.1f}%",
        "Monthly volatility": "—" if pd.isna(volatility) else f"{volatility:.1f}%",
    }


def decompose_series(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    y = df.set_index("date")[DEFAULT_TARGET_COLUMN].astype(float).asfreq(pd.infer_freq(df["date"]) or "MS").interpolate()
    if len(y) < period * 2:
        return pd.DataFrame()
    stl = STL(y, period=period, robust=True).fit()
    return pd.DataFrame({"date": y.index, "observed": y.values, "trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid})


def anomaly_table(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    out = df[["date", DEFAULT_TARGET_COLUMN]].copy()
    out["rolling_mean_12"] = out[DEFAULT_TARGET_COLUMN].rolling(12, min_periods=6).mean()
    out["rolling_std_12"] = out[DEFAULT_TARGET_COLUMN].rolling(12, min_periods=6).std()
    out["z_score"] = (out[DEFAULT_TARGET_COLUMN] - out["rolling_mean_12"]) / out["rolling_std_12"].replace(0, np.nan)
    out = out[out["z_score"].abs() >= z_threshold].sort_values("z_score", key=lambda s: s.abs(), ascending=False)
    return out.head(20)


def textual_findings(df: pd.DataFrame, corr_cols: list[str]) -> list[str]:
    findings = []
    if len(df) > 12:
        yoy = df[DEFAULT_TARGET_COLUMN].pct_change(12).iloc[-1] * 100
        findings.append(f"The latest year-over-year change is {yoy:.1f}%.")
    if corr_cols:
        corr = df[[DEFAULT_TARGET_COLUMN] + corr_cols].corr(numeric_only=True)[DEFAULT_TARGET_COLUMN].drop(DEFAULT_TARGET_COLUMN).sort_values(key=lambda s: s.abs(), ascending=False)
        if not corr.empty:
            top = corr.index[0]
            findings.append(f"The strongest linear association with migration is {top.replace('_', ' ')} (correlation {corr.iloc[0]:.2f}).")
    change = (df[DEFAULT_TARGET_COLUMN].iloc[-1] - df[DEFAULT_TARGET_COLUMN].iloc[0]) / max(df[DEFAULT_TARGET_COLUMN].iloc[0], 1) * 100
    findings.append(f"Across the selected history, migration changed by {change:.1f}% from first to latest observation.")
    return findings
