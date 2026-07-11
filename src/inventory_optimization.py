"""Simulated inventory optimization for official store-department forecasts."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_DATA_DIR
    from inventory_utils import assumptions_for_series, load_inventory_assumptions, risk_level, round_order_quantity, safe_divide, z_score
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR
    from src.inventory_utils import assumptions_for_series, load_inventory_assumptions, risk_level, round_order_quantity, safe_divide, z_score

FORECAST_FILE = PROCESSED_DATA_DIR / "official_forecast_with_uncertainty.parquet"
SCENARIOS_FILE = PROCESSED_DATA_DIR / "inventory_initial_scenarios.parquet"
ADVANCED_FEATURES = PROCESSED_DATA_DIR / "ml_store_department_advanced_features.parquet"
RECOMMENDATIONS = PROCESSED_DATA_DIR / "inventory_recommendations.parquet"
DAILY_PROJECTION = PROCESSED_DATA_DIR / "inventory_daily_projection.parquet"
RUN_INFO = PROCESSED_DATA_DIR / "inventory_optimization_run_metrics.json"


def print_step(message: str) -> None:
    print(f"[inventory_optimization] {message}")


def representative_prices() -> pd.DataFrame:
    features = pd.read_parquet(ADVANCED_FEATURES, columns=["unique_id", "date", "mean_sell_price"])
    features["date"] = pd.to_datetime(features["date"])
    latest = features.sort_values(["unique_id", "date"]).groupby("unique_id", observed=True).tail(28)
    prices = latest.groupby("unique_id", observed=True).agg(representative_price=("mean_sell_price", "mean")).reset_index()
    prices["representative_price"] = prices["representative_price"].fillna(5.0).clip(lower=0.01)
    return prices


def forecast_summary(forecast: pd.DataFrame) -> pd.DataFrame:
    ordered = forecast.sort_values(["unique_id", "horizon"])
    summary = ordered.groupby(["unique_id", "store_id", "dept_id"], observed=True).agg(
        forecast_demand_7d=("forecast_units", lambda s: float(s.iloc[:7].sum())),
        forecast_demand_14d=("forecast_units", lambda s: float(s.iloc[:14].sum())),
        forecast_demand_28d=("forecast_units", "sum"),
        average_daily_forecast=("forecast_units", "mean"),
        peak_daily_forecast=("forecast_units", "max"),
        sigma_error=("error_std", "first"),
        percentile_error_95=("error_percentile_95", "first"),
        selected_forecast_model=("source_model_used", "first"),
        selector_confidence=("selector_confidence", "first"),
    ).reset_index()
    return summary


def build_daily_projection_for_series(forecast_rows: pd.DataFrame, initial_inventory: float, reorder_point: float, recommended_order: float, lead_time_days: int) -> tuple[pd.DataFrame, str | None, str | None, float, float]:
    inventory = float(initial_inventory)
    arrivals: dict[pd.Timestamp, float] = {}
    order_placed = False
    suggested_order_date = None
    expected_arrival_date = None
    cumulative_stockout = 0.0
    inventory_values = []
    rows = []
    for _, row in forecast_rows.sort_values("horizon").iterrows():
        date = pd.Timestamp(row["forecast_date"])
        if date in arrivals:
            inventory += arrivals[date]
        before = inventory
        reorder_flag = False
        arrival = pd.NaT
        if (not order_placed) and before <= reorder_point and recommended_order > 0:
            reorder_flag = True
            order_placed = True
            suggested_order_date = date.date().isoformat()
            arrival_date = date + pd.Timedelta(days=int(lead_time_days))
            expected_arrival_date = arrival_date.date().isoformat()
            arrivals[arrival_date] = arrivals.get(arrival_date, 0.0) + recommended_order
            arrival = arrival_date
        demand = float(row["forecast_units"])
        after = before - demand
        stockout_units = max(0.0, -after)
        if stockout_units > 0:
            cumulative_stockout += stockout_units
            after = 0.0
        inventory = after
        inventory_values.append(after)
        rows.append({
            "unique_id": row["unique_id"],
            "scenario": row["scenario"],
            "date": date,
            "forecast_demand": demand,
            "projected_inventory_before_order": max(0.0, before),
            "recommended_order": recommended_order if reorder_flag else 0.0,
            "projected_inventory_after_order": max(0.0, after),
            "stockout_units": stockout_units,
            "stockout_flag": bool(stockout_units > 0),
            "reorder_flag": bool(reorder_flag),
            "expected_arrival_date": arrival,
        })
    avg_inventory = float(np.mean(inventory_values)) if inventory_values else 0.0
    return pd.DataFrame(rows), suggested_order_date, expected_arrival_date, cumulative_stockout, avg_inventory


def optimize_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_inventory_assumptions()
    forecast = pd.read_parquet(FORECAST_FILE)
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"])
    scenarios = pd.read_parquet(SCENARIOS_FILE)
    prices = representative_prices()
    summary = forecast_summary(forecast).merge(prices, on="unique_id", how="left")
    rows = []
    daily_rows = []
    for _, scenario in scenarios.iterrows():
        unique_id = scenario["unique_id"]
        row = summary.loc[summary["unique_id"].eq(unique_id)].iloc[0]
        meta = {**row.to_dict(), **scenario.to_dict()}
        assumptions = assumptions_for_series(meta, config)
        service_level = float(assumptions["service_level_default"])
        lead_time_days = int(assumptions["lead_time_days_default"])
        review_period_days = int(assumptions["review_period_days_default"])
        ordering_cost = float(assumptions["ordering_cost_default"])
        holding_rate = float(assumptions["holding_cost_rate_annual_default"])
        stockout_cost = float(assumptions["stockout_cost_per_unit_default"])
        days_per_year = float(assumptions["days_per_year"])
        unit_price = float(row.get("representative_price", assumptions.get("simulated_unit_cost_default", 5.0)) or assumptions.get("simulated_unit_cost_default", 5.0))
        holding_cost_per_unit = max(unit_price * holding_rate, 0.01)
        avg_daily = float(row["average_daily_forecast"])
        annual_demand = max(avg_daily * days_per_year, 0.0)
        sigma = max(float(row["sigma_error"]), 0.0)
        safety_stock = z_score(service_level) * sigma * math.sqrt(max(lead_time_days, 0))
        percentile_based_safety_stock = max(float(row["percentile_error_95"]), 0.0) * math.sqrt(max(lead_time_days, 1))
        demand_during_lead = avg_daily * lead_time_days
        reorder_point = demand_during_lead + safety_stock
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit) if annual_demand > 0 and holding_cost_per_unit > 0 else 0.0
        order_up_to = avg_daily * (lead_time_days + review_period_days) + safety_stock
        initial_inventory = float(scenario["initial_inventory"])
        raw_order = max(0.0, order_up_to - initial_inventory)
        recommended_order = round_order_quantity(raw_order, float(assumptions["minimum_order_quantity_default"]), float(assumptions["order_rounding_multiple_default"]))
        forecast_rows = forecast.loc[forecast["unique_id"].eq(unique_id)].copy()
        forecast_rows["scenario"] = scenario["scenario"]
        daily, suggested_order_date, expected_arrival_date, projected_stockout_units, avg_inventory = build_daily_projection_for_series(forecast_rows, initial_inventory, reorder_point, recommended_order, lead_time_days)
        daily_rows.append(daily)
        final_inventory = float(daily["projected_inventory_after_order"].iloc[-1]) if not daily.empty else initial_inventory
        overstock_units = max(0.0, final_inventory - safety_stock)
        inventory_coverage_days = safe_divide(initial_inventory, avg_daily, 999.0 if initial_inventory > 0 else 0.0)
        excess_inventory_days = max(0.0, inventory_coverage_days - 28.0)
        stockout_risk_score = min(1.0, safe_divide(projected_stockout_units, float(row["forecast_demand_28d"]) + safety_stock, 0.0))
        estimated_holding_cost = max(0.0, avg_inventory * unit_price * holding_rate * 28.0 / days_per_year)
        estimated_stockout_cost = max(0.0, projected_stockout_units * stockout_cost)
        estimated_ordering_cost = ordering_cost if recommended_order > 0 else 0.0
        rows.append({
            "unique_id": unique_id,
            "store_id": row["store_id"],
            "dept_id": row["dept_id"],
            "scenario": scenario["scenario"],
            "forecast_demand_7d": float(row["forecast_demand_7d"]),
            "forecast_demand_14d": float(row["forecast_demand_14d"]),
            "forecast_demand_28d": float(row["forecast_demand_28d"]),
            "average_daily_forecast": avg_daily,
            "peak_daily_forecast": float(row["peak_daily_forecast"]),
            "initial_inventory": initial_inventory,
            "safety_stock": max(0.0, safety_stock),
            "percentile_based_safety_stock": percentile_based_safety_stock,
            "expected_demand_during_lead_time": demand_during_lead,
            "reorder_point": reorder_point,
            "eoq": max(0.0, eoq),
            "order_up_to_level": order_up_to,
            "recommended_order_quantity": recommended_order,
            "inventory_coverage_days": inventory_coverage_days,
            "projected_stockout_units": projected_stockout_units,
            "stockout_risk_score": stockout_risk_score,
            "stockout_risk_level": risk_level(stockout_risk_score),
            "overstock_units": overstock_units,
            "excess_inventory_days": excess_inventory_days,
            "estimated_holding_cost": estimated_holding_cost,
            "estimated_stockout_cost": estimated_stockout_cost,
            "estimated_ordering_cost": estimated_ordering_cost,
            "estimated_total_inventory_cost": estimated_holding_cost + estimated_stockout_cost + estimated_ordering_cost,
            "suggested_order_date": suggested_order_date,
            "expected_stockout_date": daily.loc[daily["stockout_flag"], "date"].min() if daily["stockout_flag"].any() else pd.NaT,
            "expected_arrival_date": expected_arrival_date,
            "service_level_target": service_level,
            "lead_time_days": lead_time_days,
            "review_period_days": review_period_days,
            "selected_forecast_model": row["selected_forecast_model"],
            "selector_confidence": row["selector_confidence"],
            "unit_cost_method": "representative_m5_sell_price_as_simulated_unit_value",
            "representative_unit_price": unit_price,
            "assumptions_used": json.dumps({k: assumptions[k] for k in ["service_level_default", "lead_time_days_default", "review_period_days_default", "ordering_cost_default", "holding_cost_rate_annual_default", "stockout_cost_per_unit_default", "minimum_order_quantity_default", "order_rounding_multiple_default"]}),
            "simulation_label": "simulated_inventory_optimization",
        })
    recommendations = pd.DataFrame(rows).sort_values(["unique_id", "scenario"]).reset_index(drop=True)
    daily_projection = pd.concat(daily_rows, ignore_index=True).sort_values(["unique_id", "scenario", "date"]).reset_index(drop=True)
    return recommendations, daily_projection


def main() -> int:
    start = time.perf_counter()
    recommendations, daily_projection = optimize_inventory()
    recommendations.to_parquet(RECOMMENDATIONS, index=False)
    daily_projection.to_parquet(DAILY_PROJECTION, index=False)
    info = {
        "execution_seconds": time.perf_counter() - start,
        "recommendation_rows": int(len(recommendations)),
        "daily_projection_rows": int(len(daily_projection)),
        "series": int(recommendations["unique_id"].nunique()),
        "scenarios": sorted(recommendations["scenario"].unique()),
        "avg_safety_stock": float(recommendations["safety_stock"].mean()),
        "avg_reorder_point": float(recommendations["reorder_point"].mean()),
        "high_or_critical_series_scenarios": int(recommendations["stockout_risk_level"].isin(["High", "Critical"]).sum()),
    }
    RUN_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print_step(f"Saved inventory recommendations: {len(recommendations):,} rows")
    print_step(f"Saved daily projection: {len(daily_projection):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
