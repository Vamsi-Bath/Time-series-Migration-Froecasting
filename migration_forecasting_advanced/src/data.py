from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_GROUP_COLUMN, DEFAULT_TARGET_COLUMN, SAMPLE_COUNTRIES


def make_sample_data(seed: int = 42) -> pd.DataFrame:
    """Create a realistic monthly multi-country migration dataset for demo use."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-01", "2024-12-01", freq="MS")
    rows: list[dict] = []
    n = len(dates)

    for i, country in enumerate(SAMPLE_COUNTRIES):
        base = 7200 + i * 780
        trend = np.linspace(0, 3100 + i * 320, n)
        season = 950 * np.sin(2 * np.pi * np.arange(n) / 12 + i / 2.5)
        secondary_season = 300 * np.cos(2 * np.pi * np.arange(n) / 6 + i)
        covid_shock = np.where((dates >= "2020-04-01") & (dates <= "2020-12-01"), -2500 - 140 * i, 0)
        recovery = np.where(dates >= "2021-06-01", 760 + 120 * i, 0)
        policy_shift = np.where(dates >= "2023-01-01", 420 + 90 * i, 0)

        unemployment = 5.0 + 0.7 * np.sin(2 * np.pi * np.arange(n) / 19 + i) + rng.normal(0, 0.22, n)
        gdp_growth = 2.2 + 0.8 * np.sin(2 * np.pi * np.arange(n) / 31 + i / 3) + rng.normal(0, 0.32, n)
        exchange_rate = 1.0 + 0.09 * np.sin(2 * np.pi * np.arange(n) / 25 + i) + rng.normal(0, 0.025, n)
        policy_index = np.clip(55 + 8 * np.sin(2 * np.pi * np.arange(n) / 38 + i) + rng.normal(0, 2.8, n), 20, 90)
        visa_processing_days = np.clip(42 - 0.18 * policy_index + 4 * unemployment + rng.normal(0, 2.8, n), 12, 80)
        conflict_pressure = np.clip(30 + 7 * np.sin(2 * np.pi * np.arange(n) / 48 + i / 2) + rng.normal(0, 3, n), 5, 70)
        student_visa_share = np.clip(0.18 + 0.03 * np.sin(2 * np.pi * np.arange(n) / 12 - i) + rng.normal(0, 0.008, n), 0.08, 0.35)

        noise = rng.normal(0, 420, n)
        migration = (
            base
            + trend
            + season
            + secondary_season
            + covid_shock
            + recovery
            + policy_shift
            - unemployment * 165
            + gdp_growth * 290
            + policy_index * 33
            - visa_processing_days * 19
            + conflict_pressure * 22
            + student_visa_share * 2100
            + noise
        )
        migration = np.maximum(np.round(migration), 500)

        for j, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "country": country,
                    "migration_count": int(migration[j]),
                    "unemployment_rate": round(float(unemployment[j]), 2),
                    "gdp_growth": round(float(gdp_growth[j]), 2),
                    "exchange_rate": round(float(exchange_rate[j]), 3),
                    "policy_index": round(float(policy_index[j]), 1),
                    "visa_processing_days": round(float(visa_processing_days[j]), 1),
                    "conflict_pressure_index": round(float(conflict_pressure[j]), 1),
                    "student_visa_share": round(float(student_visa_share[j]), 3),
                }
            )
    return pd.DataFrame(rows)


def standardize_data(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize arbitrary uploaded CSV columns to the app's expected names."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[date_col, target_col])
    out = out.rename(columns={date_col: "date", target_col: DEFAULT_TARGET_COLUMN})

    if group_col and group_col in out.columns:
        out = out.rename(columns={group_col: DEFAULT_GROUP_COLUMN})
    elif DEFAULT_GROUP_COLUMN not in out.columns:
        out[DEFAULT_GROUP_COLUMN] = "All"

    # Convert numeric-looking context columns while leaving date/group untouched.
    for col in out.columns:
        if col not in {"date", DEFAULT_GROUP_COLUMN}:
            out[col] = pd.to_numeric(out[col], errors="ignore")

    out = out.sort_values([DEFAULT_GROUP_COLUMN, "date"]).reset_index(drop=True)
    return out


def aggregate_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    if country == "All countries combined":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg = df.groupby("date", as_index=False)[numeric_cols].sum()
        for col in numeric_cols:
            if col != DEFAULT_TARGET_COLUMN and col in df.columns:
                agg[col] = df.groupby("date")[col].mean().values
        agg[DEFAULT_GROUP_COLUMN] = country
        return agg.sort_values("date")
    return df[df[DEFAULT_GROUP_COLUMN] == country].sort_values("date").copy()


def infer_frequency(dates: pd.Series) -> str:
    dates = pd.to_datetime(dates).sort_values()
    freq = pd.infer_freq(dates)
    if freq:
        return freq
    # Monthly is the intended default for migration data.
    return "MS"


def numeric_indicator_columns(df: pd.DataFrame, target_col: str = DEFAULT_TARGET_COLUMN) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]


def save_sample_csv(path: str) -> None:
    make_sample_data().to_csv(path, index=False)
