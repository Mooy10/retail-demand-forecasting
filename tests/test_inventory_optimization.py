import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.inventory_utils import round_order_quantity, z_score


def test_inventory_recommendations_contract_if_available():
    path = PROCESSED_DATA_DIR / "inventory_recommendations.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert df["unique_id"].nunique() == 70
    assert set(df["scenario"].unique()) == {"lean", "base", "conservative"}
    assert (df["safety_stock"] >= 0).all()
    assert (df["reorder_point"] >= df["expected_demand_during_lead_time"]).all()
    assert (df["eoq"] >= 0).all()
    assert (df["recommended_order_quantity"] >= 0).all()
    assert (df["inventory_coverage_days"] >= 0).all()
    assert (df["estimated_total_inventory_cost"] >= 0).all()
    assert df["simulation_label"].eq("simulated_inventory_optimization").all()
    numeric = df.select_dtypes(include=["number"])
    assert not np.isinf(numeric.to_numpy()).any()


def test_inventory_policy_comparison_if_available():
    path = REPORTS_DIR / "inventory_policy_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert {"hybrid_official", "baseline_seasonal_naive_28", "no_forecast_historical_average"}.issubset(set(df["policy"].unique()))
    assert df["simulation_label"].eq("simulated_policy_economic_comparison").all()
    assert (df[["holding_cost", "stockout_cost", "ordering_cost", "total_cost", "stockout_units", "excess_inventory"]] >= 0).all().all()


def test_inventory_math_synthetic_zero_and_high_demand():
    assert z_score(0.95) > 1.0
    assert round_order_quantity(0, 1, 1) == 0
    assert round_order_quantity(12.1, 1, 5) == 15
    assert round_order_quantity(0.5, 10, 5) == 10
