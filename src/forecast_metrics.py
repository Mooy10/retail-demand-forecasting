"""Forecast accuracy metrics for baseline backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float, zero_value: float = np.nan) -> float:
    if denominator == 0 or np.isnan(denominator):
        return zero_value
    return numerator / denominator


def calculate_metric_row(group: pd.DataFrame) -> dict[str, float]:
    actual = group["actual"].to_numpy(dtype="float64")
    pred = group["prediction"].to_numpy(dtype="float64")
    errors = actual - pred
    abs_errors = np.abs(errors)
    squared_errors = np.square(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(squared_errors)))
    actual_sum = float(np.sum(np.abs(actual)))
    error_sum = float(np.sum(abs_errors))
    wape = safe_divide(error_sum, actual_sum, 0.0 if error_sum == 0 else np.nan)

    denominator = np.abs(actual) + np.abs(pred)
    smape_terms = np.divide(
        2 * abs_errors,
        denominator,
        out=np.zeros_like(abs_errors, dtype="float64"),
        where=denominator > 0,
    )
    smape = float(np.mean(smape_terms) * 100)

    mape_terms = np.divide(
        abs_errors,
        np.abs(actual),
        out=np.full_like(abs_errors, np.nan, dtype="float64"),
        where=np.abs(actual) > 0,
    )
    mape = float(np.nanmean(mape_terms) * 100) if not np.isnan(mape_terms).all() else np.nan

    scale = float(group["rmsse_scale"].iloc[0]) if "rmsse_scale" in group.columns else np.nan
    mse = float(np.mean(squared_errors))
    if scale > 0:
        rmsse = float(np.sqrt(mse / scale))
    else:
        rmsse = 0.0 if mse == 0 else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "rmsse": rmsse,
        "smape": smape,
        "mape": mape,
        "actual_volume": actual_sum,
        "prediction_volume": float(np.sum(pred)),
        "observations": int(len(group)),
    }


def metrics_by_series(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "unique_id", "cutoff", "model"]
    metadata_cols = [
        "demand_pattern",
        "abc_class",
        "store_id",
        "dept_id",
        "item_id",
    ]
    rows = []
    for keys, group in predictions.groupby(group_cols, observed=True, sort=False):
        row = dict(zip(group_cols, keys, strict=False))
        for col in metadata_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        row.update(calculate_metric_row(group))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    numeric_cols = ["mae", "rmse", "wape", "rmsse", "smape", "mape"]
    metrics[numeric_cols] = metrics[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return metrics


def summarize_metrics(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=False))
        weights = group["actual_volume"].to_numpy(dtype="float64")
        for metric in ["mae", "rmse", "wape", "rmsse", "smape"]:
            values = group[metric].to_numpy(dtype="float64")
            valid = ~np.isnan(values)
            if valid.any():
                row[f"mean_{metric}"] = float(np.mean(values[valid]))
                row[f"median_{metric}"] = float(np.median(values[valid]))
                if weights[valid].sum() > 0:
                    row[f"weighted_{metric}"] = float(np.average(values[valid], weights=weights[valid]))
                else:
                    row[f"weighted_{metric}"] = float(np.mean(values[valid]))
            else:
                row[f"mean_{metric}"] = np.nan
                row[f"median_{metric}"] = np.nan
                row[f"weighted_{metric}"] = np.nan
        row["series_count"] = int(group["unique_id"].nunique())
        row["actual_volume"] = float(group["actual_volume"].sum())
        rows.append(row)
    return pd.DataFrame(rows)