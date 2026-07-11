"""Forecast filtering and aggregation helpers."""

from __future__ import annotations

import pandas as pd


def add_state_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "state_id" not in out.columns and "store_id" in out.columns:
        out["state_id"] = out["store_id"].astype(str).str.split("_").str[0]
    if "cat_id" not in out.columns and "dept_id" in out.columns:
        out["cat_id"] = out["dept_id"].astype(str).str.rsplit("_", n=1).str[0]
    return out


def filter_frame(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = add_state_id(df)
    for col, value in filters.items():
        if value in (None, "Todos", "Todas", "") or col not in out.columns:
            continue
        if isinstance(value, (list, tuple, set)):
            if value:
                out = out.loc[out[col].isin(value)]
        else:
            out = out.loc[out[col].eq(value)]
    return out


def forecast_by_day(forecast: pd.DataFrame) -> pd.DataFrame:
    if forecast.empty:
        return pd.DataFrame(columns=["forecast_date", "forecast_units"])
    return forecast.groupby("forecast_date", observed=True)["forecast_units"].sum().reset_index()


def forecast_by_dimension(forecast: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if forecast.empty or dimension not in forecast.columns:
        return pd.DataFrame(columns=[dimension, "forecast_units"])
    return forecast.groupby(dimension, observed=True)["forecast_units"].sum().reset_index().sort_values("forecast_units", ascending=False)


def series_options(forecast: pd.DataFrame, store_id: str | None = None, dept_id: str | None = None) -> list[str]:
    df = filter_frame(forecast, {"store_id": store_id, "dept_id": dept_id})
    return sorted(df["unique_id"].dropna().unique().tolist()) if "unique_id" in df else []


def comparison_forecasts(candidate_forecasts: pd.DataFrame, unique_id: str) -> pd.DataFrame:
    if candidate_forecasts.empty or "unique_id" not in candidate_forecasts:
        return pd.DataFrame()
    return candidate_forecasts.loc[candidate_forecasts["unique_id"].eq(unique_id)].copy()
