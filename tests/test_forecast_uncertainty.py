import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.forecast_uncertainty import add_intervals, compute_uncertainty_stats


def test_uncertainty_intervals_contract_if_available():
    path = PROCESSED_DATA_DIR / "official_forecast_with_uncertainty.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert (df["forecast_lower_80"] >= 0).all()
    assert (df["forecast_lower_95"] >= 0).all()
    assert (df["forecast_upper_80"] >= df["forecast_units"]).all()
    assert (df["forecast_upper_95"] >= df["forecast_units"]).all()
    assert (df["forecast_units"] >= df["forecast_lower_80"]).all()
    assert (df["forecast_units"] >= df["forecast_lower_95"]).all()


def test_uncertainty_zero_error_synthetic():
    forecast = pd.DataFrame({
        "unique_id": ["A", "A"],
        "store_id": ["S", "S"],
        "dept_id": ["D", "D"],
        "forecast_units": [10.0, 20.0],
        "actual": [10.0, 20.0],
    })
    stats = compute_uncertainty_stats(forecast)
    out = add_intervals(forecast, stats)
    assert stats.loc[0, "mae_historical"] == 0
    assert (out["forecast_lower_95"] == out["forecast_units"]).all()
    assert (out["forecast_upper_95"] == out["forecast_units"]).all()
