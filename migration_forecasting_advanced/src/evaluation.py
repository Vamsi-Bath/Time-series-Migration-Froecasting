from __future__ import annotations

import pandas as pd

from .metrics import metrics_table
from .models import MODEL_REGISTRY, ForecastResult


def run_selected_models(
    df: pd.DataFrame,
    model_names: list[str],
    horizon: int,
    test_periods: int,
    exog_cols: list[str],
    scenario: dict[str, float] | None = None,
) -> tuple[list[ForecastResult], list[str]]:
    """Fit selected models and return successful results plus readable warnings."""
    results: list[ForecastResult] = []
    warnings: list[str] = []
    for name in model_names:
        fitter = MODEL_REGISTRY[name]
        try:
            results.append(fitter(df, horizon=horizon, test_periods=test_periods, exog_cols=exog_cols, scenario=scenario))
        except Exception as exc:  # keep dashboard resilient
            warnings.append(f"{name} failed: {exc}")
    return results, warnings


def best_model(results: list[ForecastResult]) -> ForecastResult | None:
    if not results:
        return None
    return min(results, key=lambda r: r.metrics.get("RMSE", float("inf")))


def combined_forecast(results: list[ForecastResult]) -> pd.DataFrame:
    frames = []
    for r in results:
        tmp = r.forecast.copy()
        tmp["model"] = r.name
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def combined_backtest(results: list[ForecastResult]) -> pd.DataFrame:
    frames = []
    for r in results:
        tmp = r.fitted.copy()
        tmp["model"] = r.name
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def model_scorecard(results: list[ForecastResult]) -> pd.DataFrame:
    table = metrics_table(results)
    if table.empty:
        return table
    table = table.copy()
    table["Rank"] = table["RMSE"].rank(method="min").astype(int)
    return table.sort_values(["Rank", "RMSE"])
