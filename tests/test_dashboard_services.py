import pandas as pd

from dashboard.services.forecast_service import add_state_id, filter_frame, forecast_by_day, forecast_by_dimension
from dashboard.services.inventory_service import inventory_kpis, scenario_frame
from dashboard.services.metrics_service import dataframe_to_csv_bytes, overview_kpis, safe_divide


def test_forecast_service_adds_business_dimensions():
    df = pd.DataFrame(
        {
            "unique_id": ["CA_1_FOODS_1", "TX_2_HOBBIES_2"],
            "store_id": ["CA_1", "TX_2"],
            "dept_id": ["FOODS_1", "HOBBIES_2"],
            "forecast_date": pd.to_datetime(["2016-03-28", "2016-03-28"]),
            "forecast_units": [10.0, 5.0],
        }
    )
    out = add_state_id(df)
    assert out["state_id"].tolist() == ["CA", "TX"]
    assert out["cat_id"].tolist() == ["FOODS", "HOBBIES"]


def test_forecast_filters_and_aggregations():
    df = add_state_id(
        pd.DataFrame(
            {
                "unique_id": ["CA_1_FOODS_1", "CA_2_FOODS_1", "TX_1_HOUSEHOLD_1"],
                "store_id": ["CA_1", "CA_2", "TX_1"],
                "dept_id": ["FOODS_1", "FOODS_1", "HOUSEHOLD_1"],
                "forecast_date": pd.to_datetime(["2016-03-28", "2016-03-28", "2016-03-29"]),
                "forecast_units": [10.0, 20.0, 30.0],
            }
        )
    )
    filtered = filter_frame(df, {"state_id": "CA", "cat_id": "FOODS"})
    assert len(filtered) == 2
    assert forecast_by_day(filtered)["forecast_units"].sum() == 30.0
    by_store = forecast_by_dimension(filtered, "store_id")
    assert set(by_store["store_id"]) == {"CA_1", "CA_2"}


def test_inventory_kpis_by_scenario():
    df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "scenario": ["base", "lean"],
            "initial_inventory": [100.0, 50.0],
            "expected_demand_during_lead_time": [40.0, 40.0],
            "safety_stock": [10.0, 8.0],
            "reorder_point": [50.0, 48.0],
            "order_up_to_level": [120.0, 90.0],
            "recommended_order_quantity": [20.0, 5.0],
            "projected_stockout_units": [0.0, 3.0],
            "overstock_units": [2.0, 0.0],
            "estimated_holding_cost": [1.0, 1.0],
            "estimated_stockout_cost": [0.0, 9.0],
            "estimated_ordering_cost": [75.0, 75.0],
            "estimated_total_inventory_cost": [76.0, 85.0],
        }
    )
    base = inventory_kpis(df, "base")
    assert scenario_frame(df, "base").shape[0] == 1
    assert base["initial_inventory"] == 100.0
    assert base["recommended_order"] == 20.0
    assert base["total_cost"] == 76.0


def test_overview_kpis_and_csv_conversion():
    forecast = pd.DataFrame(
        {
            "unique_id": ["A", "A", "B"],
            "forecast_units": [10.0, 15.0, 20.0],
            "fallback_used": [False, False, True],
        }
    )
    inventory = pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "scenario": ["base", "base"],
            "recommended_order_quantity": [1.0, 2.0],
            "estimated_total_inventory_cost": [100.0, 200.0],
        }
    )
    holdout = pd.DataFrame({"model": ["hybrid_holdout_w3"], "weighted_wape": [0.1], "weighted_rmsse": [0.8]})
    policy = pd.DataFrame(
        {
            "policy": ["hybrid_official", "baseline_seasonal_naive_28"],
            "scenario": ["base", "base"],
            "total_cost": [100.0, 130.0],
            "simulated_savings_vs_baseline": [30.0, 0.0],
        }
    )
    kpis = overview_kpis(forecast, inventory, holdout, policy, "base")
    assert safe_divide(3, 2) == 1.5
    assert safe_divide(3, 0) == 0
    assert kpis["forecast_28d"] == 45.0
    assert kpis["fallback_series"] == 1
    assert kpis["simulated_savings_vs_baseline"] == 30.0
    assert b"unique_id" in dataframe_to_csv_bytes(forecast)
