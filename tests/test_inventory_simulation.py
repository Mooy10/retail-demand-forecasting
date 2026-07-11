import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


def test_inventory_initial_scenarios_if_available():
    path = PROCESSED_DATA_DIR / "inventory_initial_scenarios.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert set(df["scenario"].unique()) == {"lean", "base", "conservative"}
    assert df["unique_id"].nunique() == 70
    assert (df["initial_inventory"] >= 0).all()
    pivot = df.pivot(index="unique_id", columns="scenario", values="initial_inventory")
    assert (pivot["conservative"] >= pivot["base"]).all()
    assert (pivot["base"] >= pivot["lean"]).all()
    assert df["simulation_label"].eq("simulated_initial_inventory").all()


def test_inventory_daily_projection_if_available():
    path = PROCESSED_DATA_DIR / "inventory_daily_projection.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert df["unique_id"].nunique() == 70
    assert set(df["scenario"].unique()) == {"lean", "base", "conservative"}
    assert df.groupby(["unique_id", "scenario"], observed=True).size().eq(28).all()
    numeric = df.select_dtypes(include=["number"])
    assert not np.isinf(numeric.to_numpy()).any()
    assert (df["forecast_demand"] >= 0).all()
    assert (df["stockout_units"] >= 0).all()
