"""Metric helpers for dashboard pages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or pd.isna(denominator):
        return default
    return float(numerator) / float(denominator)


def format_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/D"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value/1_000_000:.{decimals}f}M"
    if abs_value >= 1_000:
        return f"{value/1_000:.{decimals}f}K"
    return f"{value:.{decimals}f}"


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "N/D"
    return "$" + format_number(float(value), 2)


def overview_kpis(forecast: pd.DataFrame, inventory: pd.DataFrame, holdout: pd.DataFrame, policy: pd.DataFrame, scenario: str = "base") -> dict[str, float | str]:
    scenario_inventory = inventory.loc[inventory.get("scenario", pd.Series(dtype=str)).eq(scenario)] if not inventory.empty else pd.DataFrame()
    scenario_policy = policy.loc[policy.get("scenario", pd.Series(dtype=str)).eq(scenario)] if not policy.empty else pd.DataFrame()
    hybrid = holdout.loc[holdout.get("model", pd.Series(dtype=str)).eq("hybrid_holdout_w3")] if not holdout.empty else pd.DataFrame()
    policy_hybrid = scenario_policy.loc[scenario_policy.get("policy", pd.Series(dtype=str)).eq("hybrid_official")] if not scenario_policy.empty else pd.DataFrame()
    fallback_series = forecast.loc[forecast.get("fallback_used", False).astype(bool), "unique_id"].nunique() if not forecast.empty and "fallback_used" in forecast else 0
    confidence = forecast[["unique_id", "selector_confidence"]].drop_duplicates()["selector_confidence"].value_counts(normalize=True).mul(100).to_dict() if not forecast.empty and "selector_confidence" in forecast else {}
    return {
        "forecast_28d": float(forecast["forecast_units"].sum()) if not forecast.empty else 0.0,
        "wape": float(hybrid["weighted_wape"].iloc[0]) if not hybrid.empty else np.nan,
        "rmsse": float(hybrid["weighted_rmsse"].iloc[0]) if not hybrid.empty else np.nan,
        "fallback_series": int(fallback_series),
        "recommended_orders": int((scenario_inventory.get("recommended_order_quantity", pd.Series(dtype=float)) > 0).sum()) if not scenario_inventory.empty else 0,
        "simulated_total_cost": float(policy_hybrid["total_cost"].sum()) if not policy_hybrid.empty else 0.0,
        "simulated_savings_vs_baseline": float(policy_hybrid.get("simulated_savings_vs_baseline", pd.Series([0.0])).sum()) if not policy_hybrid.empty else 0.0,
        "confidence_distribution": confidence,
    }


def historical_daily_average(history: pd.DataFrame) -> float:
    if history.empty:
        return 0.0
    daily = history.groupby("date", observed=True)["demand"].sum()
    return float(daily.mean()) if len(daily) else 0.0


def model_label(model: str) -> str:
    labels = {
        "seasonal_naive_28": "Seasonal Naive 28",
        "seasonal_average_weekday": "Seasonal Avg Weekday",
        "xgboost_phase6": "XGBoost Fase 6",
        "xgboost_phase7": "XGBoost Fase 7",
        "lightgbm_phase7": "LightGBM Fase 7",
        "hybrid_holdout_w3": "Híbrido Holdout W3",
        "hybrid_official": "Híbrido oficial",
    }
    return labels.get(str(model), str(model))


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
