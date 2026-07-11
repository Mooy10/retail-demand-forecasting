import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


def test_hybrid_predictions_contract_if_available():
    path = PROCESSED_DATA_DIR / "hybrid_predictions_store_department.parquet"
    registry_path = PROCESSED_DATA_DIR / "model_selection_registry.parquet"
    if not path.exists() or not registry_path.exists():
        return
    pred = pd.read_parquet(path)
    registry = pd.read_parquet(registry_path)
    assert (pred["prediction"] >= 0).all()
    assert set(pred["horizon"].unique()) == set(range(1, 29))
    assert pred["window"].nunique() == 3
    assert pred["unique_id"].nunique() == 70
    counts = pred.groupby(["unique_id", "window"], observed=True).size()
    assert counts.eq(28).all()
    check = pred[["unique_id", "selected_model"]].drop_duplicates().merge(registry[["unique_id", "best_model"]], on="unique_id", how="left")
    assert (check["selected_model"] == check["best_model"]).all()


def test_hybrid_metrics_are_finite_if_available():
    path = PROCESSED_DATA_DIR / "hybrid_metrics.parquet"
    if not path.exists():
        return
    metrics = pd.read_parquet(path)
    numeric = metrics.select_dtypes(include=["number"])
    assert not np.isinf(numeric.to_numpy()).any()
