import numpy as np
import pandas as pd

from src.forecast_metrics import metrics_by_series


def _prediction_frame(actual, prediction):
    return pd.DataFrame(
        {
            "dataset": "unit",
            "unique_id": "series_1",
            "cutoff": pd.Timestamp("2020-01-01"),
            "model": "test_model",
            "actual": actual,
            "prediction": prediction,
            "rmsse_scale": 1.0,
            "demand_pattern": "Smooth",
            "abc_class": "A",
            "store_id": "S1",
            "dept_id": "D1",
            "item_id": "I1",
        }
    )


def test_metrics_are_zero_for_perfect_predictions():
    frame = _prediction_frame([1, 2, 3], [1, 2, 3])
    metrics = metrics_by_series(frame).iloc[0]
    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["wape"] == 0
    assert metrics["rmsse"] == 0


def test_rmse_is_greater_or_equal_mae():
    frame = _prediction_frame([1, 2, 3], [2, 2, 7])
    metrics = metrics_by_series(frame).iloc[0]
    assert metrics["rmse"] >= metrics["mae"]


def test_metrics_do_not_contain_infinite_values():
    frame = _prediction_frame([0, 0, 0], [1, 0, 2])
    metrics = metrics_by_series(frame)
    numeric = metrics.select_dtypes(include=["number"])
    assert not np.isinf(numeric.to_numpy()).any()