"""Prepare compact forecasting datasets for baseline experiments.

Phase 5 intentionally avoids a full long table for all 30,490 item-store series.
It creates:
- 70 store-department daily series.
- Up to 100 selected class-A item-store series across demand patterns.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


SALES_FILE = RAW_DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"
SEGMENTATION_FILE = PROCESSED_DATA_DIR / "demand_segmentation.parquet"

STORE_DEPT_OUTPUT = PROCESSED_DATA_DIR / "forecast_store_department.parquet"
SELECTED_SERIES_OUTPUT = PROCESSED_DATA_DIR / "forecast_selected_series.parquet"
SELECTED_REGISTRY_OUTPUT = PROCESSED_DATA_DIR / "selected_series_registry.parquet"
METRICS_OUTPUT = PROCESSED_DATA_DIR / "forecasting_dataset_metrics.json"

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
PATTERN_ORDER = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
MAX_SERIES_PER_PATTERN = 25


def print_step(message: str) -> None:
    print(f"[forecasting_dataset] {message}")


def demand_columns_from_header() -> list[str]:
    header = pd.read_csv(SALES_FILE, nrows=0)
    cols = [col for col in header.columns if col.startswith("d_")]
    return sorted(cols, key=lambda value: int(value.split("_")[1]))


def read_calendar(demand_cols: list[str]) -> pd.DataFrame:
    calendar = pd.read_csv(CALENDAR_FILE, usecols=["d", "date"], parse_dates=["date"])
    calendar["d_order"] = calendar["d"].str.replace("d_", "", regex=False).astype("int16")
    aligned = pd.DataFrame({"d": demand_cols}).merge(calendar, on="d", how="left", validate="one_to_one")
    if aligned["date"].isna().any():
        missing = aligned.loc[aligned["date"].isna(), "d"].head(10).tolist()
        raise ValueError(f"Missing calendar rows for: {missing}")
    return aligned


def read_sales(demand_cols: list[str]) -> pd.DataFrame:
    dtype_map = {col: "int16" for col in demand_cols}
    dtype_map.update({col: "category" for col in ID_COLUMNS})
    dtype_map["id"] = "string"
    return pd.read_csv(SALES_FILE, dtype=dtype_map)


def build_store_department_dataset(sales: pd.DataFrame, calendar: pd.DataFrame, demand_cols: list[str]) -> pd.DataFrame:
    print_step("Building 70 store-department series...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        grouped = sales.groupby(["store_id", "dept_id"], observed=True)[demand_cols].sum().reset_index()
    long_df = grouped.melt(
        id_vars=["store_id", "dept_id"],
        value_vars=demand_cols,
        var_name="d",
        value_name="demand",
    )
    long_df = long_df.merge(calendar, on="d", how="left", validate="many_to_one")
    long_df["unique_id"] = long_df["store_id"].astype(str) + "_" + long_df["dept_id"].astype(str)
    long_df["item_id"] = pd.NA
    long_df["demand_pattern"] = "store_department"
    long_df["abc_class"] = "aggregate"
    long_df["demand"] = long_df["demand"].astype("int32")
    columns = [
        "unique_id",
        "date",
        "d",
        "d_order",
        "demand",
        "store_id",
        "dept_id",
        "item_id",
        "demand_pattern",
        "abc_class",
    ]
    return long_df[columns].sort_values(["unique_id", "date"]).reset_index(drop=True)


def select_class_a_series() -> pd.DataFrame:
    print_step("Selecting up to 25 class-A item-store series per demand pattern...")
    segmentation = pd.read_parquet(SEGMENTATION_FILE)
    selected_parts = []
    class_a = segmentation.loc[segmentation["abc_class"] == "A"].copy()
    class_a = class_a.sort_values(
        ["demand_pattern", "total_demand", "id"],
        ascending=[True, False, True],
    )
    for pattern in PATTERN_ORDER:
        selected = class_a.loc[class_a["demand_pattern"] == pattern].head(MAX_SERIES_PER_PATTERN)
        selected_parts.append(selected)
    registry = pd.concat(selected_parts, ignore_index=True)
    registry = registry.sort_values(["demand_pattern", "total_demand"], ascending=[True, False]).reset_index(drop=True)
    registry["selection_rank_within_pattern"] = registry.groupby("demand_pattern").cumcount() + 1
    return registry


def build_selected_series_dataset(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    demand_cols: list[str],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    print_step("Building compact long dataset for selected item-store series...")
    selected_sales = sales.loc[sales["id"].astype(str).isin(registry["id"].astype(str)), ID_COLUMNS + demand_cols].copy()
    selected_sales["id"] = selected_sales["id"].astype(str)
    registry_cols = ["id", "demand_pattern", "abc_class"]
    selected_sales = selected_sales.merge(registry[registry_cols], on="id", how="inner", validate="one_to_one")
    long_df = selected_sales.melt(
        id_vars=ID_COLUMNS + ["demand_pattern", "abc_class"],
        value_vars=demand_cols,
        var_name="d",
        value_name="demand",
    )
    long_df = long_df.merge(calendar, on="d", how="left", validate="many_to_one")
    long_df["unique_id"] = long_df["id"].astype(str)
    long_df["demand"] = long_df["demand"].astype("int16")
    columns = [
        "unique_id",
        "date",
        "d",
        "d_order",
        "demand",
        "store_id",
        "dept_id",
        "item_id",
        "demand_pattern",
        "abc_class",
    ]
    return long_df[columns].sort_values(["unique_id", "date"]).reset_index(drop=True)


def main() -> int:
    start = time.perf_counter()
    for path in [SALES_FILE, CALENDAR_FILE, SEGMENTATION_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    demand_cols = demand_columns_from_header()
    calendar = read_calendar(demand_cols)
    sales = read_sales(demand_cols)

    store_dept = build_store_department_dataset(sales, calendar, demand_cols)
    registry = select_class_a_series()
    selected_series = build_selected_series_dataset(sales, calendar, demand_cols, registry)

    store_dept.to_parquet(STORE_DEPT_OUTPUT, index=False)
    registry.to_parquet(SELECTED_REGISTRY_OUTPUT, index=False)
    selected_series.to_parquet(SELECTED_SERIES_OUTPUT, index=False)

    elapsed = time.perf_counter() - start
    metrics = {
        "execution_seconds": elapsed,
        "historical_days": len(demand_cols),
        "store_department_rows": int(store_dept.shape[0]),
        "store_department_series": int(store_dept["unique_id"].nunique()),
        "selected_series_rows": int(selected_series.shape[0]),
        "selected_series_count": int(registry.shape[0]),
        "selected_by_pattern": registry["demand_pattern"].value_counts().to_dict(),
        "store_department_demand": int(store_dept["demand"].sum()),
        "selected_series_demand": int(selected_series["demand"].sum()),
    }
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print_step(f"Saved {STORE_DEPT_OUTPUT.name}: {store_dept.shape[0]:,} rows")
    print_step(f"Saved {SELECTED_SERIES_OUTPUT.name}: {selected_series.shape[0]:,} rows")
    print_step(f"Saved {SELECTED_REGISTRY_OUTPUT.name}: {registry.shape[0]:,} selected series")
    print_step(f"Completed in {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())