import numpy as np
import pandas as pd

from src.advanced_feature_engineering import add_hierarchical_features, add_trend_and_smoothing
from src.config import PROCESSED_DATA_DIR


def test_advanced_rolling_features_use_prior_values_only():
    df = pd.DataFrame({"unique_id": ["A"] * 8, "date": pd.date_range("2020-01-01", periods=8), "demand": [1, 2, 3, 4, 100, 6, 7, 8]})
    out = add_trend_and_smoothing(df.copy())
    assert out.loc[4, "rolling_median_7"] == 2.5
    assert out.loc[5, "rolling_median_7"] == 3.0
    assert np.isnan(out.loc[0, "ewm_mean_7"])


def test_hierarchical_features_are_shifted_on_synthetic_data():
    rows = []
    for day in range(35):
        for uid, store, dept, state, cat, demand in [("A", "S1", "D1", "ST", "C", 1), ("B", "S1", "D2", "ST", "C", 2)]:
            rows.append({"unique_id": uid, "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day), "demand": demand, "store_id": store, "dept_id": dept, "state_id": state, "cat_id": cat, "rolling_mean_28": demand})
    out = add_hierarchical_features(pd.DataFrame(rows))
    day7 = out[(out["unique_id"] == "A") & (out["date"] == pd.Timestamp("2020-01-08"))].iloc[0]
    assert day7["store_total_lag_7"] == 3
    assert pd.isna(out.loc[0, "store_total_lag_7"])


def test_advanced_feature_table_contract_if_available():
    path = PROCESSED_DATA_DIR / "ml_store_department_advanced_features.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    expected = {"lag_364", "rolling_median_28", "store_total_lag_28", "price_volatility_12_weeks"}
    assert expected.issubset(df.columns)
    assert df["unique_id"].nunique() == 70
    assert not any(col.startswith("target_") for col in df.columns)
