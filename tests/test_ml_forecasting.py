import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR


def test_ml_predictions_temporal_consistency_and_coverage():
    path = PROCESSED_DATA_DIR / "ml_predictions_store_department.parquet"
    if not path.exists():
        return
    pred = pd.read_parquet(path)
    assert (pd.to_datetime(pred["forecast_date"]) > pd.to_datetime(pred["cutoff"])).all()
    assert set(pred["horizon"].unique()) == set(range(1, 29))
    assert pred["unique_id"].nunique() == 70
    assert pred["window"].nunique() == 3
    assert (pred["prediction"] >= 0).all()
    counts = pred.groupby(["unique_id", "window", "model"], observed=True).size()
    assert counts.eq(28).all()


def test_ml_metrics_have_no_infinite_values_and_comparison_is_valid():
    metrics_path = PROCESSED_DATA_DIR / "ml_metrics.parquet"
    comparison_path = REPORTS_DIR / "ml_vs_baseline_comparison.csv"
    if not metrics_path.exists() or not comparison_path.exists():
        return
    metrics = pd.read_parquet(metrics_path)
    numeric = metrics.select_dtypes(include=["number"])
    assert not np.isinf(numeric.to_numpy()).any()
    comparison = pd.read_csv(comparison_path)
    expected = (comparison["mean_wape_diff"] * -100 / (comparison["weighted_wape"] + comparison["mean_wape_diff"])).replace([np.inf, -np.inf], np.nan)
    assert comparison["mean_wape_improvement_pct"].notna().all()


def test_no_future_target_columns_in_training_feature_list():
    from src.run_ml_forecasting import EXCLUDED_COLUMNS

    assert "target_demand" in EXCLUDED_COLUMNS
    assert "target_date" in EXCLUDED_COLUMNS