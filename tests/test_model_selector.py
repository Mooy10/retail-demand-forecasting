import pandas as pd

from src.model_selector import score_models, select_from_scores
from src.config import PROCESSED_DATA_DIR


def _rows(unique_id="A"):
    rows = []
    for cutoff in pd.date_range("2020-01-01", periods=3):
        rows.append({"unique_id": unique_id, "model": "seasonal_naive_28", "cutoff": cutoff, "wape": 0.10, "rmsse": 1.0, "store_id": "S1", "dept_id": "D1", "actual_volume": 100, "prediction_volume": 100})
        rows.append({"unique_id": unique_id, "model": "xgboost_phase7", "cutoff": cutoff, "wape": 0.099, "rmsse": 0.99, "store_id": "S1", "dept_id": "D1", "actual_volume": 100, "prediction_volume": 100})
    return pd.DataFrame(rows)


def test_simple_model_preference_when_score_gap_is_small():
    selected = select_from_scores(score_models(_rows()), tie_threshold=0.10)
    assert selected.loc[0, "best_model"] == "seasonal_naive_28"
    assert "simple baseline" in selected.loc[0, "selection_reason"]


def test_model_selection_registry_contract_if_available():
    path = PROCESSED_DATA_DIR / "model_selection_registry.parquet"
    if not path.exists():
        return
    registry = pd.read_parquet(path)
    assert registry["unique_id"].nunique() == 70
    assert registry["best_model"].notna().all()
    assert registry["confidence"].isin(["High", "Medium", "Low"]).all()
    assert registry["best_score"].le(registry["second_best_score"] + 1e-9).all()
