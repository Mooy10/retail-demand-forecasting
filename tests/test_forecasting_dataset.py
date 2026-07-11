import pandas as pd

from src.config import PROCESSED_DATA_DIR


STORE_DEPT = PROCESSED_DATA_DIR / "forecast_store_department.parquet"
SELECTED = PROCESSED_DATA_DIR / "forecast_selected_series.parquet"
REGISTRY = PROCESSED_DATA_DIR / "selected_series_registry.parquet"
STORE_PRED = PROCESSED_DATA_DIR / "baseline_predictions_store_department.parquet"
SELECTED_PRED = PROCESSED_DATA_DIR / "baseline_predictions_selected_series.parquet"


def test_store_department_dataset_has_70_series():
    df = pd.read_parquet(STORE_DEPT)
    assert df["unique_id"].nunique() == 70
    assert df.groupby("unique_id")["date"].nunique().eq(1913).all()


def test_selected_series_registry_is_class_a_and_not_over_100():
    registry = pd.read_parquet(REGISTRY)
    assert len(registry) <= 100
    assert (registry["abc_class"] == "A").all()


def test_selected_series_dataset_is_compact_and_complete():
    selected = pd.read_parquet(SELECTED)
    registry = pd.read_parquet(REGISTRY)
    assert selected["unique_id"].nunique() == len(registry)
    assert selected.groupby("unique_id")["date"].nunique().eq(1913).all()


def test_predictions_cover_three_windows_and_28_horizons():
    predictions = pd.concat([pd.read_parquet(STORE_PRED), pd.read_parquet(SELECTED_PRED)], ignore_index=True)
    assert predictions["window"].nunique() == 3
    assert set(predictions["horizon"].unique()) == set(range(1, 29))
    counts = predictions.groupby(["dataset", "unique_id", "cutoff", "model"], observed=True).size()
    assert counts.eq(28).all()


def test_no_future_leakage_in_predictions():
    predictions = pd.concat([pd.read_parquet(STORE_PRED), pd.read_parquet(SELECTED_PRED)], ignore_index=True)
    assert (pd.to_datetime(predictions["forecast_date"]) > pd.to_datetime(predictions["cutoff"])).all()


def test_prediction_observations_by_horizon_are_consistent():
    predictions = pd.concat([pd.read_parquet(STORE_PRED), pd.read_parquet(SELECTED_PRED)], ignore_index=True)
    counts = predictions.groupby(["dataset", "horizon"], observed=True).size()
    for dataset, group in counts.groupby(level=0):
        assert group.nunique() == 1