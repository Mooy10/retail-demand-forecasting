import pandas as pd

from dashboard.services.data_loader import load_forecast_uncertainty, load_inventory_recommendations, load_policy_comparison
from dashboard.services.inventory_service import executive_summary, generate_alerts


def test_generate_alerts_is_deterministic_on_synthetic_data():
    inventory = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C"],
            "scenario": ["base", "base", "base"],
            "recommended_order_quantity": [0.0, 100.0, 20.0],
            "estimated_total_inventory_cost": [10.0, 500.0, 20.0],
            "projected_stockout_units": [0.0, 50.0, 0.0],
            "stockout_risk_level": ["Low", "High", "Medium"],
            "overstock_units": [5.0, 0.0, 0.0],
        }
    )
    forecast = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C"],
            "fallback_used": [True, False, False],
            "selector_confidence": ["Low", "High", "Medium"],
        }
    )
    alerts = generate_alerts(inventory, forecast)
    assert not alerts.empty
    assert {"severity", "unique_id", "message", "metric"}.issubset(alerts.columns)
    assert alerts["message"].str.contains("fallback", case=False).any()
    assert alerts["message"].str.contains("Confianza baja", case=False).any()
    assert alerts["message"].str.contains("faltante", case=False).any()


def test_real_dashboard_business_rules_preserve_simulation_labels():
    inventory = load_inventory_recommendations()
    policy = load_policy_comparison()
    assert inventory["simulation_label"].eq("simulated_inventory_optimization").all()
    assert policy["simulation_label"].eq("simulated_policy_economic_comparison").all()
    assert set(inventory["scenario"].unique()) == {"lean", "base", "conservative"}


def test_real_dashboard_alerts_and_summary_are_available():
    forecast = load_forecast_uncertainty()
    inventory = load_inventory_recommendations()
    policy = load_policy_comparison()
    alerts = generate_alerts(inventory, forecast)
    summary = executive_summary(forecast, inventory, policy, "base")

    assert not alerts.empty
    assert alerts["unique_id"].notna().all()
    assert "tienda" in summary.lower()
    assert "fallback" in summary.lower()


def test_policy_comparison_contains_official_and_baseline():
    policy = load_policy_comparison()
    assert {"hybrid_official", "baseline_seasonal_naive_28"}.issubset(set(policy["policy"]))
    by_policy = policy.loc[policy["scenario"].eq("base")].groupby("policy")["total_cost"].sum()
    assert by_policy["hybrid_official"] >= 0
    assert by_policy["baseline_seasonal_naive_28"] >= 0
