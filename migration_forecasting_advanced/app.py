from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import APP_TITLE, DEFAULT_GROUP_COLUMN, DEFAULT_TARGET_COLUMN, SUPPORTED_MODELS
from src.data import aggregate_country, make_sample_data, numeric_indicator_columns, standardize_data
from src.evaluation import best_model, combined_backtest, combined_forecast, model_scorecard, run_selected_models
from src.insights import anomaly_table, decompose_series, kpi_summary, textual_findings
from src.visualizations import (
    plot_backtest,
    plot_correlation,
    plot_decomposition,
    plot_feature_importance,
    plot_forecasts,
    plot_metric_bars,
    plot_residual_hist,
    plot_residuals,
    plot_scenario,
    plot_seasonality,
    plot_trend,
    plot_yoy,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🌍", layout="wide")


def download_df(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return make_sample_data()


def read_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return load_sample()
    return pd.read_csv(uploaded_file)


def sidebar_controls(raw_df: pd.DataFrame):
    columns = list(raw_df.columns)
    numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()

    with st.sidebar:
        st.header("1. Dataset")
        date_default = columns.index("date") if "date" in columns else 0
        date_col = st.selectbox("Date column", columns, index=date_default)

        target_options = numeric_cols or columns
        target_default = target_options.index(DEFAULT_TARGET_COLUMN) if DEFAULT_TARGET_COLUMN in target_options else 0
        target_col = st.selectbox("Migration count / target column", target_options, index=target_default)

        group_options = ["None"] + columns
        group_default = group_options.index(DEFAULT_GROUP_COLUMN) if DEFAULT_GROUP_COLUMN in group_options else 0
        group_choice = st.selectbox("Country/group column", group_options, index=group_default)
        group_col = None if group_choice == "None" else group_choice

        df = standardize_data(raw_df, date_col, target_col, group_col)
        country_options = ["All countries combined"] + sorted(df[DEFAULT_GROUP_COLUMN].astype(str).unique())
        country = st.selectbox("Country/group to analyze", country_options)
        data = aggregate_country(df, country)

        st.header("2. Model lab")
        horizon = st.slider("Forecast horizon", min_value=3, max_value=48, value=18)
        max_test = min(36, max(3, len(data) // 3))
        test_periods = st.slider("Backtest periods", min_value=3, max_value=max_test, value=min(12, max_test))
        possible_exog = numeric_indicator_columns(data)
        default_exog = possible_exog[: min(4, len(possible_exog))]
        exog_cols = st.multiselect("Socio-economic indicators / regressors", possible_exog, default=default_exog)
        default_models = ["Seasonal Naive", "Exponential Smoothing", "SARIMAX", "Random Forest"]
        model_choices = st.multiselect("Models to train", SUPPORTED_MODELS, default=[m for m in default_models if m in SUPPORTED_MODELS])

        st.header("3. Scenario assumptions")
        st.caption("Adjust future indicator assumptions as percentage changes from recent averages.")
        scenario = {}
        for col in exog_cols[:6]:
            scenario[col] = st.slider(f"{col.replace('_', ' ')} adjustment", -30, 30, 0, key=f"scenario_{col}")
        run = st.button("Run analysis", type="primary")

    return df, data, country, horizon, test_periods, exog_cols, model_choices, scenario, run


def main() -> None:
    st.title("🌍 International Migration Forecasting Intelligence")
    st.caption("A modular forecasting dashboard with EDA, multiple models, backtesting, uncertainty intervals, diagnostics, scenario analysis, and exportable outputs.")

    uploaded = st.sidebar.file_uploader("Upload migration CSV", type=["csv"])
    raw_df = read_upload(uploaded)
    all_df, data, country, horizon, test_periods, exog_cols, model_choices, scenario, run = sidebar_controls(raw_df)

    if len(data) < 18:
        st.error("The selected series is too short for a meaningful forecasting analysis. Use at least 18 observations, preferably 36+.")
        st.stop()

    summary = kpi_summary(data)
    kpi_cols = st.columns(4)
    for col, (label, value) in zip(kpi_cols, summary.items()):
        col.metric(label, value)

    tabs = st.tabs([
        "Executive summary",
        "Exploratory analysis",
        "Model evaluation",
        "Forecasts",
        "Drivers & scenarios",
        "Diagnostics",
        "Data & exports",
    ])

    results = []
    warnings = []
    scenario_results = []

    if run and model_choices:
        with st.spinner("Training and evaluating selected models..."):
            results, warnings = run_selected_models(data, model_choices, horizon, test_periods, exog_cols, scenario=None)
            if any(v != 0 for v in scenario.values()):
                scenario_results, _ = run_selected_models(data, model_choices, horizon, test_periods, exog_cols, scenario=scenario)

    with tabs[0]:
        st.subheader(f"Executive findings for {country}")
        for finding in textual_findings(data, exog_cols):
            st.markdown(f"- {finding}")
        anomalies = anomaly_table(data)
        if not anomalies.empty:
            st.markdown("- Potential outlier periods were detected using a rolling z-score screen.")
            st.dataframe(anomalies, use_container_width=True)
        else:
            st.markdown("- No large rolling z-score outliers were detected with the current threshold.")
        if results:
            scorecard = model_scorecard(results)
            winning = best_model(results)
            st.success(f"Best backtest model: {winning.name} with RMSE {winning.metrics['RMSE']:,.2f}.")
            st.dataframe(scorecard, use_container_width=True)
        else:
            st.info("Use the sidebar to run the model lab and generate forecast/evaluation results.")

    with tabs[1]:
        st.subheader("Exploratory analysis and visualisations")
        st.plotly_chart(plot_trend(data, f"Migration trend: {country}"), use_container_width=True)
        if len(data) > 12:
            st.plotly_chart(plot_yoy(data), use_container_width=True)
        st.plotly_chart(plot_seasonality(data), use_container_width=True)
        decomp = decompose_series(data)
        if not decomp.empty:
            st.plotly_chart(plot_decomposition(decomp), use_container_width=True)
        if exog_cols:
            st.plotly_chart(plot_correlation(data, exog_cols), use_container_width=True)

    with tabs[2]:
        st.subheader("Model evaluation and backtesting")
        if warnings:
            for warning in warnings:
                st.warning(warning)
        if results:
            scorecard = model_scorecard(results)
            st.dataframe(scorecard, use_container_width=True)
            st.plotly_chart(plot_metric_bars(scorecard, "RMSE"), use_container_width=True)
            st.plotly_chart(plot_backtest(results), use_container_width=True)
            backtest = combined_backtest(results)
            download_df(backtest, "Download backtest predictions", "migration_backtest_predictions.csv")
        else:
            st.info("Run models from the sidebar to compare MAE, RMSE, MAPE, sMAPE, MASE, and R².")

    with tabs[3]:
        st.subheader("Forecasts and uncertainty intervals")
        if results:
            st.plotly_chart(plot_forecasts(data, results), use_container_width=True)
            forecasts = combined_forecast(results)
            st.dataframe(forecasts, use_container_width=True)
            download_df(forecasts, "Download forecasts", "migration_forecasts.csv")
        else:
            st.info("Run the model lab to view forecast curves and export future predictions.")

    with tabs[4]:
        st.subheader("Drivers and scenario analysis")
        if exog_cols:
            for col in exog_cols:
                driver_df = data[["date", col]].rename(columns={col: DEFAULT_TARGET_COLUMN})
                st.plotly_chart(plot_trend(driver_df, col.replace("_", " ").title()), use_container_width=True)
        else:
            st.info("Select numeric socio-economic indicators in the sidebar to analyze drivers.")

        if results and scenario_results:
            base = best_model(results)
            matched = next((r for r in scenario_results if r.name == base.name), None)
            if matched is not None:
                st.plotly_chart(plot_scenario(base.forecast, matched.forecast, base.name), use_container_width=True)
                comparison = base.forecast[["date", "prediction"]].rename(columns={"prediction": "base_prediction"}).merge(
                    matched.forecast[["date", "prediction"]].rename(columns={"prediction": "scenario_prediction"}), on="date"
                )
                comparison["difference"] = comparison["scenario_prediction"] - comparison["base_prediction"]
                comparison["difference_pct"] = comparison["difference"] / comparison["base_prediction"] * 100
                st.dataframe(comparison, use_container_width=True)
        elif any(v != 0 for v in scenario.values()):
            st.info("Run analysis to see the impact of your scenario assumptions.")

    with tabs[5]:
        st.subheader("Model diagnostics")
        if results:
            selected_name = st.selectbox("Diagnostic model", [r.name for r in results])
            selected = next(r for r in results if r.name == selected_name)
            left, right = st.columns(2)
            with left:
                st.plotly_chart(plot_residuals(selected), use_container_width=True)
            with right:
                st.plotly_chart(plot_residual_hist(selected), use_container_width=True)
            st.json(selected.model_info)
            if selected.feature_importance is not None:
                st.plotly_chart(plot_feature_importance(selected.feature_importance), use_container_width=True)
                st.dataframe(selected.feature_importance, use_container_width=True)
        else:
            st.info("Run models to inspect residuals, intervals, model metadata, and feature importance.")

    with tabs[6]:
        st.subheader("Data and exports")
        st.markdown("Expected columns: one date column, one migration target, optional country/group column, and optional numeric indicators.")
        st.dataframe(data, use_container_width=True)
        download_df(data, "Download selected dataset", "selected_migration_data.csv")
        download_df(all_df, "Download standardized full dataset", "standardized_migration_data.csv")

        with st.expander("Project structure"):
            st.code(
                """
migration_forecasting_advanced/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_migration.csv
└── src/
    ├── config.py
    ├── data.py
    ├── features.py
    ├── metrics.py
    ├── models.py
    ├── evaluation.py
    ├── insights.py
    └── visualizations.py
                """.strip()
            )


if __name__ == "__main__":
    main()
