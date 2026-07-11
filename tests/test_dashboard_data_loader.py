from pathlib import Path

import pandas as pd

from dashboard.services import data_loader


def test_dashboard_core_artifacts_are_available():
    required_paths = [
        data_loader.PROCESSED / "official_forecast_store_department.parquet",
        data_loader.PROCESSED / "official_forecast_28d_summary.parquet",
        data_loader.PROCESSED / "official_forecast_with_uncertainty.parquet",
        data_loader.PROCESSED / "inventory_recommendations.parquet",
        data_loader.PROCESSED / "inventory_daily_projection.parquet",
        data_loader.REPORTS / "holdout_w3_metrics_summary.csv",
        data_loader.REPORTS / "rolling_selector_metrics_summary.csv",
        data_loader.REPORTS / "inventory_policy_comparison.csv",
        data_loader.CONFIG / "inventory_assumptions.yaml",
    ]
    missing = [str(path) for path in required_paths if not Path(path).exists()]
    assert missing == []


def test_dashboard_loaders_return_non_empty_expected_contracts():
    forecast = data_loader.load_forecast_uncertainty()
    inventory = data_loader.load_inventory_recommendations()
    daily = data_loader.load_inventory_daily_projection()
    holdout = data_loader.load_holdout_metrics_summary()

    assert not forecast.empty
    assert not inventory.empty
    assert not daily.empty
    assert not holdout.empty
    assert {"unique_id", "forecast_date", "forecast_units", "baseline_forecast", "fallback_used"}.issubset(forecast.columns)
    assert {"unique_id", "scenario", "recommended_order_quantity", "estimated_total_inventory_cost"}.issubset(inventory.columns)
    assert {"unique_id", "scenario", "date", "projected_inventory_after_order"}.issubset(daily.columns)
    assert {"hybrid_holdout_w3", "seasonal_naive_28"}.issubset(set(holdout["model"]))


def test_dashboard_note_and_assumptions_are_visible():
    note = data_loader.PORTFOLIO_NOTE
    assumptions = data_loader.load_assumptions_text()
    assert "portafolio" in note.lower()
    assert "simulaciones" in note.lower()
    assert "service_level" in assumptions


def test_safe_read_missing_file_returns_empty_frame():
    df = data_loader.safe_read_parquet(data_loader.ROOT / "missing_dashboard_file.parquet", required=["x"])
    assert isinstance(df, pd.DataFrame)
    assert df.empty
