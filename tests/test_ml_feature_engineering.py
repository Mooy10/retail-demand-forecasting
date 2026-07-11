import pandas as pd

from src.ml_feature_engineering import add_demand_features
from src.config import PROCESSED_DATA_DIR


def test_lags_are_shifted_correctly_on_synthetic_data():
    df = pd.DataFrame({"unique_id": ["A"] * 70, "date": pd.date_range("2020-01-01", periods=70), "demand": range(70)})
    out = add_demand_features(df)
    assert out.loc[10, "lag_1"] == 9
    assert out.loc[10, "lag_7"] == 3
    assert out.loc[28, "lag_28"] == 0


def test_rolling_features_do_not_use_current_value():
    df = pd.DataFrame({"unique_id": ["A"] * 10, "date": pd.date_range("2020-01-01", periods=10), "demand": [1, 2, 3, 4, 100, 6, 7, 8, 9, 10]})
    out = add_demand_features(df)
    assert out.loc[4, "rolling_mean_7"] == 2.5
    assert out.loc[5, "rolling_max_7"] == 100


def test_ml_feature_table_has_no_future_target_columns_in_saved_features():
    path = PROCESSED_DATA_DIR / "ml_store_department_features.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        forbidden = [col for col in df.columns if col.startswith("target_")]
        assert forbidden == []
        assert df["unique_id"].nunique() == 70