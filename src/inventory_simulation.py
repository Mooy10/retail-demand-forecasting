"""Simulate initial inventory scenarios for store-department planning."""

from __future__ import annotations

import json
import time

import pandas as pd

try:
    from config import PROCESSED_DATA_DIR
    from inventory_utils import assumptions_for_series, derive_state_id, load_inventory_assumptions
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR
    from src.inventory_utils import assumptions_for_series, derive_state_id, load_inventory_assumptions

OFFICIAL_FORECAST = PROCESSED_DATA_DIR / "official_forecast_store_department.parquet"
FORECAST_DATASET = PROCESSED_DATA_DIR / "forecast_store_department.parquet"
OUTPUT = PROCESSED_DATA_DIR / "inventory_initial_scenarios.parquet"
RUN_INFO = PROCESSED_DATA_DIR / "inventory_initial_scenarios_run_metrics.json"

SCENARIO_DAYS = {
    "lean": 7,
    "base": None,
    "conservative": 21,
}


def print_step(message: str) -> None:
    print(f"[inventory_simulation] {message}")


def recent_daily_demand() -> pd.DataFrame:
    official = pd.read_parquet(OFFICIAL_FORECAST)
    cutoff_date = pd.to_datetime(official["forecast_date"]).min() - pd.Timedelta(days=1)
    history = pd.read_parquet(FORECAST_DATASET)
    history["date"] = pd.to_datetime(history["date"])
    recent = history.loc[(history["date"] <= cutoff_date) & (history["date"] > cutoff_date - pd.Timedelta(days=28))].copy()
    demand = recent.groupby(["unique_id", "store_id", "dept_id"], observed=True).agg(
        recent_daily_demand=("demand", "mean"),
        recent_total_demand_28d=("demand", "sum"),
        recent_days=("date", "nunique"),
    ).reset_index()
    demand["state_id"] = demand["store_id"].map(derive_state_id)
    return demand


def build_initial_scenarios() -> pd.DataFrame:
    config = load_inventory_assumptions()
    demand = recent_daily_demand()
    rows = []
    for _, row in demand.iterrows():
        assumptions = assumptions_for_series(row, config)
        for scenario, days in SCENARIO_DAYS.items():
            initial_days = float(days if days is not None else assumptions["initial_inventory_days_default"])
            initial_inventory = max(0.0, float(row["recent_daily_demand"]) * initial_days)
            rows.append({
                "unique_id": row["unique_id"],
                "store_id": row["store_id"],
                "dept_id": row["dept_id"],
                "state_id": row["state_id"],
                "scenario": scenario,
                "initial_inventory": initial_inventory,
                "initial_inventory_days": initial_days,
                "recent_daily_demand": float(row["recent_daily_demand"]),
                "recent_total_demand_28d": float(row["recent_total_demand_28d"]),
                "assumption_source": assumptions["assumption_source"],
                "simulation_label": "simulated_initial_inventory",
            })
    return pd.DataFrame(rows).sort_values(["unique_id", "scenario"]).reset_index(drop=True)


def main() -> int:
    start = time.perf_counter()
    scenarios = build_initial_scenarios()
    scenarios.to_parquet(OUTPUT, index=False)
    RUN_INFO.write_text(json.dumps({"execution_seconds": time.perf_counter() - start, "rows": int(len(scenarios)), "series": int(scenarios["unique_id"].nunique()), "scenarios": sorted(scenarios["scenario"].unique())}, indent=2), encoding="utf-8")
    print_step(f"Saved inventory scenarios: {len(scenarios):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
