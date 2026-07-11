import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.run_out_of_sample_model_selection import select_models


def test_holdout_window_3_not_used_for_selection_if_available():
    path = PROCESSED_DATA_DIR / "model_selection_registry_train_w1_w2.parquet"
    if not path.exists():
        return
    registry = pd.read_parquet(path)
    assert registry["unique_id"].nunique() == 70
    assert registry["evaluation_window"].eq("window_3").all()
    assert not registry["selector_training_windows"].str.contains("window_3").any()
    assert not registry["normalization_windows"].str.contains("window_3").any()


def test_target_window_never_appears_in_rolling_training_windows_if_available():
    path = PROCESSED_DATA_DIR / "rolling_selector_registry.parquet"
    if not path.exists():
        return
    registry = pd.read_parquet(path)
    assert registry["unique_id"].nunique() == 70
    for _, row in registry.iterrows():
        assert row["evaluation_window"] not in row["selector_training_windows"].split(",")
    window_2 = registry[registry["evaluation_window"].eq("window_2")]
    window_3 = registry[registry["evaluation_window"].eq("window_3")]
    assert window_2["selector_training_windows"].eq("window_1").all()
    assert window_3["selector_training_windows"].eq("window_1,window_2").all()
    assert not window_2["confidence"].eq("High").any()


def test_holdout_and_rolling_predictions_are_complete_and_nonnegative_if_available():
    for filename, expected_windows in [
        ("hybrid_predictions_holdout_w3.parquet", {"window_3"}),
        ("rolling_hybrid_predictions.parquet", {"window_2", "window_3"}),
    ]:
        path = PROCESSED_DATA_DIR / filename
        if not path.exists():
            continue
        pred = pd.read_parquet(path)
        assert set(pred["window"].unique()) == expected_windows
        assert pred["unique_id"].nunique() == 70
        assert (pred["prediction"] >= 0).all()
        counts = pred.groupby(["unique_id", "window"], observed=True).size()
        assert counts.eq(28).all()


def test_metrics_have_no_infinite_values_if_available():
    for filename in ["hybrid_metrics_holdout_w3.parquet", "rolling_hybrid_metrics.parquet"]:
        path = PROCESSED_DATA_DIR / filename
        if not path.exists():
            continue
        metrics = pd.read_parquet(path)
        numeric = metrics.select_dtypes(include=["number"])
        assert not np.isinf(numeric.to_numpy()).any()


def test_hybrid_uses_historical_registry_model_if_available():
    pred_path = PROCESSED_DATA_DIR / "hybrid_predictions_holdout_w3.parquet"
    reg_path = PROCESSED_DATA_DIR / "model_selection_registry_train_w1_w2.parquet"
    if not pred_path.exists() or not reg_path.exists():
        return
    pred = pd.read_parquet(pred_path)
    registry = pd.read_parquet(reg_path)
    check = pred[["unique_id", "source_model"]].drop_duplicates().merge(
        registry[["unique_id", "selected_model"]], on="unique_id", how="left", validate="one_to_one"
    )
    assert (check["source_model"] == check["selected_model"]).all()


def test_fallback_and_future_loss_on_synthetic_data():
    rows = []
    for window, naive_wape, ml_wape in [("window_1", 0.100, 0.099), ("window_2", 0.101, 0.100)]:
        rows.append({"dataset": "synthetic", "unique_id": "A", "window": window, "cutoff": pd.Timestamp("2020-01-01"), "model": "seasonal_naive_28", "wape": naive_wape, "rmsse": 1.00, "store_id": "S1", "dept_id": "D1", "actual_volume": 100, "prediction_volume": 100, "mae": 1, "rmse": 1, "smape": 1, "mape": 1})
        rows.append({"dataset": "synthetic", "unique_id": "A", "window": window, "cutoff": pd.Timestamp("2020-01-01"), "model": "xgboost_phase7", "wape": ml_wape, "rmsse": 0.99, "store_id": "S1", "dept_id": "D1", "actual_volume": 100, "prediction_volume": 100, "mae": 1, "rmse": 1, "smape": 1, "mape": 1})
    selected = select_models(pd.DataFrame(rows), ["window_1", "window_2"], "window_3")
    assert selected.loc[0, "selected_model"] == "seasonal_naive_28"
    future = pd.DataFrame({"model": ["seasonal_naive_28", "xgboost_phase7"], "window_3_wape": [0.12, 0.30]})
    assert future.loc[future["model"].eq(selected.loc[0, "selected_model"]), "window_3_wape"].iloc[0] < 0.30
