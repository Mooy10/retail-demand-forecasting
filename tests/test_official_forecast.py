import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.build_official_forecast import should_fallback


def test_official_forecast_contract_if_available():
    path = PROCESSED_DATA_DIR / "official_forecast_store_department.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert df["unique_id"].nunique() == 70
    assert set(df["horizon"].unique()) == set(range(1, 29))
    assert df.groupby("unique_id", observed=True).size().eq(28).all()
    assert (df["forecast_units"] >= 0).all()
    assert df["baseline_forecast"].notna().all()
    assert {"fallback_used", "fallback_reason", "selector_confidence"}.issubset(df.columns)


def test_official_forecast_summary_if_available():
    path = PROCESSED_DATA_DIR / "official_forecast_28d_summary.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert len(df) == 70
    assert (df["forecast_demand_28d"] >= 0).all()
    assert df["simulation_label"].eq("simulated_inventory_planning_input").all()


def test_fallback_rule_on_synthetic_low_confidence():
    row = pd.Series({"selected_model": "xgboost_phase7", "confidence": "Low", "score_difference": 0.01, "window_stability": 0.1})
    selected = pd.DataFrame({"prediction": [1.0, 2.0]})
    fallback, reason = should_fallback(row, selected)
    assert fallback
    assert "low_confidence" in reason
