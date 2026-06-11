from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import DEFAULT_TARGET_COLUMN


def plot_trend(df: pd.DataFrame, title: str):
    fig = px.line(df, x="date", y=DEFAULT_TARGET_COLUMN, markers=True, title=title)
    fig.update_layout(yaxis_title="Migration count", xaxis_title="Date")
    return fig


def plot_yoy(df: pd.DataFrame):
    out = df[["date", DEFAULT_TARGET_COLUMN]].copy()
    out["YoY %"] = out[DEFAULT_TARGET_COLUMN].pct_change(12) * 100
    fig = px.line(out, x="date", y="YoY %", title="Year-over-year migration change")
    fig.add_hline(y=0, line_dash="dash")
    return fig


def plot_seasonality(df: pd.DataFrame):
    out = df.copy()
    out["month"] = out["date"].dt.month_name()
    order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    fig = px.box(out, x="month", y=DEFAULT_TARGET_COLUMN, category_orders={"month": order}, title="Monthly seasonality distribution")
    fig.update_layout(xaxis_title="Month", yaxis_title="Migration count")
    return fig


def plot_decomposition(decomp: pd.DataFrame):
    fig = go.Figure()
    for col in ["observed", "trend", "seasonal", "resid"]:
        fig.add_trace(go.Scatter(x=decomp["date"], y=decomp[col], mode="lines", name=col.title()))
    fig.update_layout(title="STL decomposition: observed, trend, seasonal, residual", xaxis_title="Date")
    return fig


def plot_correlation(df: pd.DataFrame, cols: list[str]):
    corr = df[[DEFAULT_TARGET_COLUMN] + cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
    return fig


def plot_metric_bars(scorecard: pd.DataFrame, metric: str = "RMSE"):
    fig = px.bar(scorecard.sort_values(metric), x="Model", y=metric, title=f"Model comparison by {metric}", text_auto=".2s")
    return fig


def plot_backtest(results):
    fig = go.Figure()
    for r in results:
        fig.add_trace(go.Scatter(x=r.fitted["date"], y=r.fitted["actual"], name="Actual", mode="lines+markers", showlegend=(r == results[0])))
        fig.add_trace(go.Scatter(x=r.fitted["date"], y=r.fitted["prediction"], name=f"{r.name} backtest", mode="lines+markers"))
    fig.update_layout(title="Backtest: actual vs predicted", xaxis_title="Date", yaxis_title="Migration count")
    return fig


def plot_forecasts(history: pd.DataFrame, results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history[DEFAULT_TARGET_COLUMN], name="Actual history", mode="lines+markers"))
    for r in results:
        fig.add_trace(go.Scatter(x=r.forecast["date"], y=r.forecast["prediction"], name=f"{r.name} forecast", mode="lines+markers"))
        if {"lower", "upper"}.issubset(r.forecast.columns):
            fig.add_trace(
                go.Scatter(
                    x=list(r.forecast["date"]) + list(r.forecast["date"])[::-1],
                    y=list(r.forecast["upper"]) + list(r.forecast["lower"])[::-1],
                    fill="toself",
                    line=dict(width=0),
                    opacity=0.18,
                    name=f"{r.name} 90% interval",
                    showlegend=True,
                )
            )
    fig.update_layout(title="Forecast comparison", xaxis_title="Date", yaxis_title="Migration count")
    return fig


def plot_residuals(result):
    fig = px.scatter(result.residuals, x="date", y="residual", title=f"Residual diagnostics: {result.name}")
    fig.add_hline(y=0, line_dash="dash")
    return fig


def plot_residual_hist(result):
    return px.histogram(result.residuals, x="residual", nbins=25, title=f"Residual distribution: {result.name}")


def plot_feature_importance(importance: pd.DataFrame):
    top = importance.head(15).sort_values("importance")
    return px.bar(top, x="importance", y="feature", orientation="h", title="Random Forest feature importance")


def plot_scenario(base_forecast: pd.DataFrame, scenario_forecast: pd.DataFrame, model_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_forecast["date"], y=base_forecast["prediction"], mode="lines+markers", name="Base forecast"))
    fig.add_trace(go.Scatter(x=scenario_forecast["date"], y=scenario_forecast["prediction"], mode="lines+markers", name="Scenario forecast"))
    fig.update_layout(title=f"Scenario comparison using {model_name}", xaxis_title="Date", yaxis_title="Migration count")
    return fig
