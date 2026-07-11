"""Simulated economic comparison of inventory policies."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_DATA_DIR, REPORTS_DIR
    from inventory_utils import assumptions_for_series, load_inventory_assumptions, round_order_quantity, safe_divide, z_score
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
    from src.inventory_utils import assumptions_for_series, load_inventory_assumptions, round_order_quantity, safe_divide, z_score

OFFICIAL_FORECAST = PROCESSED_DATA_DIR / "official_forecast_with_uncertainty.parquet"
SCENARIOS = PROCESSED_DATA_DIR / "inventory_initial_scenarios.parquet"
RECOMMENDATIONS = PROCESSED_DATA_DIR / "inventory_recommendations.parquet"
POLICY_COMPARISON = REPORTS_DIR / "inventory_policy_comparison.csv"
COST_BY_STORE = REPORTS_DIR / "inventory_cost_by_store.csv"
COST_BY_DEPT = REPORTS_DIR / "inventory_cost_by_department.csv"
SCENARIO_COMPARISON = REPORTS_DIR / "inventory_scenario_comparison.csv"
RUN_INFO = PROCESSED_DATA_DIR / "inventory_economic_analysis_run_metrics.json"


def print_step(message: str) -> None:
    print(f"[inventory_economic_analysis] {message}")


def policy_forecasts() -> pd.DataFrame:
    official = pd.read_parquet(OFFICIAL_FORECAST)
    scenarios = pd.read_parquet(SCENARIOS)
    recent = scenarios[["unique_id", "recent_daily_demand"]].drop_duplicates()
    frames = []
    hybrid = official[["unique_id", "store_id", "dept_id", "forecast_date", "horizon", "forecast_units", "error_std"]].copy()
    hybrid["policy"] = "hybrid_official"
    hybrid["policy_forecast_units"] = hybrid["forecast_units"]
    frames.append(hybrid)
    baseline = official[["unique_id", "store_id", "dept_id", "forecast_date", "horizon", "baseline_forecast", "error_std"]].copy()
    baseline["policy"] = "baseline_seasonal_naive_28"
    baseline["policy_forecast_units"] = baseline["baseline_forecast"].clip(lower=0)
    frames.append(baseline)
    simple = official[["unique_id", "store_id", "dept_id", "forecast_date", "horizon", "error_std"]].merge(recent, on="unique_id", how="left")
    simple["policy"] = "no_forecast_historical_average"
    simple["policy_forecast_units"] = simple["recent_daily_demand"].fillna(0).clip(lower=0)
    frames.append(simple)
    return pd.concat(frames, ignore_index=True, sort=False)


def build_policy_comparison() -> pd.DataFrame:
    config = load_inventory_assumptions()
    policy_daily = policy_forecasts()
    scenarios = pd.read_parquet(SCENARIOS)
    rec = pd.read_parquet(RECOMMENDATIONS)
    rec_base = rec[["unique_id", "scenario", "representative_unit_price", "service_level_target"]].drop_duplicates()
    summary = policy_daily.groupby(["policy", "unique_id", "store_id", "dept_id"], observed=True).agg(
        forecast_demand_28d=("policy_forecast_units", "sum"),
        average_daily_forecast=("policy_forecast_units", "mean"),
        sigma_error=("error_std", "first"),
    ).reset_index()
    rows = []
    for _, scenario in scenarios.iterrows():
        series_summary = summary.loc[summary["unique_id"].eq(scenario["unique_id"])]
        for _, row in series_summary.iterrows():
            meta = {**row.to_dict(), **scenario.to_dict()}
            assumptions = assumptions_for_series(meta, config)
            unit_info = rec_base.loc[(rec_base["unique_id"].eq(row["unique_id"])) & (rec_base["scenario"].eq(scenario["scenario"]))]
            unit_price = float(unit_info["representative_unit_price"].iloc[0]) if not unit_info.empty else float(assumptions.get("simulated_unit_cost_default", 5.0))
            service_level = float(assumptions["service_level_default"])
            lead_time = int(assumptions["lead_time_days_default"])
            review_period = int(assumptions["review_period_days_default"])
            ordering_cost = float(assumptions["ordering_cost_default"])
            holding_rate = float(assumptions["holding_cost_rate_annual_default"])
            stockout_cost = float(assumptions["stockout_cost_per_unit_default"])
            avg_daily = float(row["average_daily_forecast"])
            safety_stock = z_score(service_level) * max(float(row["sigma_error"]), 0.0) * np.sqrt(max(lead_time, 1))
            order_up_to = avg_daily * (lead_time + review_period) + safety_stock
            initial_inventory = float(scenario["initial_inventory"])
            recommended_order = round_order_quantity(max(0.0, order_up_to - initial_inventory), float(assumptions["minimum_order_quantity_default"]), float(assumptions["order_rounding_multiple_default"]))
            demand = float(row["forecast_demand_28d"])
            ending = initial_inventory + recommended_order - demand
            stockout_units = max(0.0, -ending)
            ending_inventory = max(0.0, ending)
            excess_inventory = max(0.0, ending_inventory - safety_stock)
            avg_inventory = max(0.0, (initial_inventory + ending_inventory) / 2)
            holding_cost = avg_inventory * unit_price * holding_rate * 28 / float(assumptions["days_per_year"])
            ordering_cost_total = ordering_cost if recommended_order > 0 else 0.0
            stockout_cost_total = stockout_units * stockout_cost
            service_level_approx = max(0.0, 1.0 - safe_divide(stockout_units, demand, 0.0))
            rows.append({
                "policy": row["policy"],
                "unique_id": row["unique_id"],
                "store_id": row["store_id"],
                "dept_id": row["dept_id"],
                "scenario": scenario["scenario"],
                "forecast_demand_28d": demand,
                "recommended_order_quantity": recommended_order,
                "holding_cost": max(0.0, holding_cost),
                "stockout_cost": max(0.0, stockout_cost_total),
                "ordering_cost": max(0.0, ordering_cost_total),
                "total_cost": max(0.0, holding_cost + stockout_cost_total + ordering_cost_total),
                "stockout_units": stockout_units,
                "excess_inventory": excess_inventory,
                "coverage_days": safe_divide(initial_inventory + recommended_order, avg_daily, 0.0),
                "service_level_approx": service_level_approx,
                "simulation_label": "simulated_policy_economic_comparison",
            })
    return pd.DataFrame(rows)


def add_savings(policy: pd.DataFrame) -> pd.DataFrame:
    keys = ["unique_id", "scenario"]
    baseline = policy.loc[policy["policy"].eq("baseline_seasonal_naive_28"), keys + ["total_cost", "stockout_units", "excess_inventory", "coverage_days"]].rename(columns={"total_cost": "baseline_total_cost", "stockout_units": "baseline_stockout_units", "excess_inventory": "baseline_excess_inventory", "coverage_days": "baseline_coverage_days"})
    simple = policy.loc[policy["policy"].eq("no_forecast_historical_average"), keys + ["total_cost", "stockout_units", "excess_inventory", "coverage_days"]].rename(columns={"total_cost": "simple_total_cost", "stockout_units": "simple_stockout_units", "excess_inventory": "simple_excess_inventory", "coverage_days": "simple_coverage_days"})
    out = policy.merge(baseline, on=keys, how="left").merge(simple, on=keys, how="left")
    out["simulated_savings_vs_baseline"] = out["baseline_total_cost"] - out["total_cost"]
    out["simulated_savings_vs_simple"] = out["simple_total_cost"] - out["total_cost"]
    out["stockout_reduction_vs_baseline"] = out["baseline_stockout_units"] - out["stockout_units"]
    out["stockout_reduction_vs_simple"] = out["simple_stockout_units"] - out["stockout_units"]
    out["excess_inventory_change_vs_baseline"] = out["excess_inventory"] - out["baseline_excess_inventory"]
    out["coverage_change_vs_baseline"] = out["coverage_days"] - out["baseline_coverage_days"]
    return out


def main() -> int:
    start = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    policy = add_savings(build_policy_comparison())
    policy.to_csv(POLICY_COMPARISON, index=False)
    policy.groupby(["store_id", "policy", "scenario"], observed=True).agg(total_cost=("total_cost", "sum"), holding_cost=("holding_cost", "sum"), stockout_cost=("stockout_cost", "sum"), ordering_cost=("ordering_cost", "sum"), stockout_units=("stockout_units", "sum"), excess_inventory=("excess_inventory", "sum")).reset_index().to_csv(COST_BY_STORE, index=False)
    policy.groupby(["dept_id", "policy", "scenario"], observed=True).agg(total_cost=("total_cost", "sum"), holding_cost=("holding_cost", "sum"), stockout_cost=("stockout_cost", "sum"), ordering_cost=("ordering_cost", "sum"), stockout_units=("stockout_units", "sum"), excess_inventory=("excess_inventory", "sum")).reset_index().to_csv(COST_BY_DEPT, index=False)
    policy.groupby(["scenario", "policy"], observed=True).agg(total_cost=("total_cost", "sum"), holding_cost=("holding_cost", "sum"), stockout_cost=("stockout_cost", "sum"), ordering_cost=("ordering_cost", "sum"), stockout_units=("stockout_units", "sum"), excess_inventory=("excess_inventory", "sum"), service_level_approx=("service_level_approx", "mean"), simulated_savings_vs_baseline=("simulated_savings_vs_baseline", "sum"), simulated_savings_vs_simple=("simulated_savings_vs_simple", "sum")).reset_index().to_csv(SCENARIO_COMPARISON, index=False)
    RUN_INFO.write_text(json.dumps({"execution_seconds": time.perf_counter() - start, "rows": int(len(policy)), "policies": sorted(policy["policy"].unique()), "scenarios": sorted(policy["scenario"].unique())}, indent=2), encoding="utf-8")
    print_step(f"Saved policy comparison: {len(policy):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
