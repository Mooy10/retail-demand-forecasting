from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


EXPECTED_PARQUETS = {
    "daily_total_demand": PROCESSED_DATA_DIR / "daily_total_demand.parquet",
    "daily_demand_by_state": PROCESSED_DATA_DIR / "daily_demand_by_state.parquet",
    "daily_demand_by_store": PROCESSED_DATA_DIR / "daily_demand_by_store.parquet",
    "daily_demand_by_category": PROCESSED_DATA_DIR / "daily_demand_by_category.parquet",
    "daily_demand_by_department": PROCESSED_DATA_DIR / "daily_demand_by_department.parquet",
    "product_demand_summary": PROCESSED_DATA_DIR / "product_demand_summary.parquet",
    "series_demand_summary": PROCESSED_DATA_DIR / "series_demand_summary.parquet",
}


def test_processed_parquet_files_exist_and_are_not_empty():
    for path in EXPECTED_PARQUETS.values():
        assert path.exists(), f"Missing processed table: {path}"
        assert path.stat().st_size > 0
        df = pd.read_parquet(path)
        assert not df.empty


def test_daily_tables_have_expected_columns():
    expected_daily_base = {"d", "date", "wm_yr_wk", "weekday", "wday", "month", "year", "demand"}
    grouped_expectations = {
        "daily_total_demand": expected_daily_base,
        "daily_demand_by_state": expected_daily_base | {"state_id"},
        "daily_demand_by_store": expected_daily_base | {"store_id"},
        "daily_demand_by_category": expected_daily_base | {"cat_id"},
        "daily_demand_by_department": expected_daily_base | {"dept_id"},
    }

    for table_name, expected_columns in grouped_expectations.items():
        df = pd.read_parquet(EXPECTED_PARQUETS[table_name])
        assert expected_columns.issubset(df.columns)


def test_dates_are_valid_and_sorted_for_total_demand():
    daily_total = pd.read_parquet(EXPECTED_PARQUETS["daily_total_demand"])
    assert daily_total["date"].notna().all()
    assert daily_total["date"].is_monotonic_increasing
    assert daily_total["date"].min() < daily_total["date"].max()


def test_processed_demand_is_never_negative():
    for table_name, path in EXPECTED_PARQUETS.items():
        df = pd.read_parquet(path)
        demand_columns = [column for column in df.columns if "demand" in column]
        for column in demand_columns:
            assert (df[column] >= 0).all(), f"Negative values found in {table_name}.{column}"


def test_total_processed_demand_matches_raw_sales_file():
    raw_header = pd.read_csv(RAW_DATA_DIR / "sales_train_validation.csv", nrows=0)
    demand_cols = [column for column in raw_header.columns if column.startswith("d_")]
    dtype_map = {column: "int16" for column in demand_cols}
    raw_sales = pd.read_csv(RAW_DATA_DIR / "sales_train_validation.csv", usecols=demand_cols, dtype=dtype_map)
    raw_total = int(raw_sales.sum(axis=0).sum())

    daily_total = pd.read_parquet(EXPECTED_PARQUETS["daily_total_demand"])
    processed_total = int(daily_total["demand"].sum())

    assert processed_total == raw_total


def test_expected_state_and_store_counts_exist():
    states = pd.read_parquet(EXPECTED_PARQUETS["daily_demand_by_state"])["state_id"].nunique()
    stores = pd.read_parquet(EXPECTED_PARQUETS["daily_demand_by_store"])["store_id"].nunique()

    assert states == 3
    assert stores == 10
