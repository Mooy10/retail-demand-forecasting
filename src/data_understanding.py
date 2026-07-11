"""Generate memory-conscious data understanding assets for the M5 dataset.

This phase intentionally avoids melting the full item-store daily sales table into
~58 million rows. Instead, it keeps the sales matrix in wide format and only
reshapes compact aggregated outputs.
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR


CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"
SALES_FILE = RAW_DATA_DIR / "sales_train_validation.csv"
SELL_PRICES_FILE = RAW_DATA_DIR / "sell_prices.csv"
SUMMARY_FILE = REPORTS_DIR / "data_understanding_summary.md"

OUTPUT_FILES = {
    "daily_total_demand": PROCESSED_DATA_DIR / "daily_total_demand.parquet",
    "daily_demand_by_state": PROCESSED_DATA_DIR / "daily_demand_by_state.parquet",
    "daily_demand_by_store": PROCESSED_DATA_DIR / "daily_demand_by_store.parquet",
    "daily_demand_by_category": PROCESSED_DATA_DIR / "daily_demand_by_category.parquet",
    "daily_demand_by_department": PROCESSED_DATA_DIR / "daily_demand_by_department.parquet",
    "product_demand_summary": PROCESSED_DATA_DIR / "product_demand_summary.parquet",
    "series_demand_summary": PROCESSED_DATA_DIR / "series_demand_summary.parquet",
}

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_COLUMNS = [
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "d",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]


def print_step(message: str) -> None:
    """Print a standard progress message."""
    print(f"[data_understanding] {message}")


def require_raw_files() -> None:
    """Validate required raw files are present."""
    missing = [path for path in [CALENDAR_FILE, SALES_FILE, SELL_PRICES_FILE] if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required raw files:\n{formatted}")


def memory_mb(df: pd.DataFrame) -> float:
    """Return DataFrame memory usage in MB."""
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def read_calendar() -> pd.DataFrame:
    """Read calendar columns needed for date joins and profiling."""
    calendar = pd.read_csv(CALENDAR_FILE, usecols=CALENDAR_COLUMNS, parse_dates=["date"])
    calendar["d_order"] = calendar["d"].str.replace("d_", "", regex=False).astype("int32")
    return calendar.sort_values("d_order").reset_index(drop=True)


def read_sales() -> pd.DataFrame:
    """Read sales table with compact numeric dtypes for daily demand columns."""
    header = pd.read_csv(SALES_FILE, nrows=0)
    demand_cols = [col for col in header.columns if col.startswith("d_")]
    dtype_map = {col: "int16" for col in demand_cols}
    dtype_map.update({col: "category" for col in ID_COLUMNS})
    dtype_map["id"] = "string"
    return pd.read_csv(SALES_FILE, dtype=dtype_map)


def read_sell_prices() -> pd.DataFrame:
    """Read sell prices with compact dtypes for profiling."""
    return pd.read_csv(
        SELL_PRICES_FILE,
        dtype={
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": "int32",
            "sell_price": "float32",
        },
    )


def get_demand_columns(sales: pd.DataFrame) -> list[str]:
    """Return ordered demand columns from d_1 to d_n."""
    demand_cols = [col for col in sales.columns if col.startswith("d_")]
    return sorted(demand_cols, key=lambda value: int(value.split("_")[1]))


def calendar_for_sales_days(calendar: pd.DataFrame, demand_cols: Iterable[str]) -> pd.DataFrame:
    """Return calendar rows aligned to the sales demand columns."""
    day_frame = pd.DataFrame({"d": list(demand_cols)})
    sales_calendar = day_frame.merge(calendar, on="d", how="left", validate="one_to_one")
    if sales_calendar["date"].isna().any():
        missing = sales_calendar.loc[sales_calendar["date"].isna(), "d"].head(10).tolist()
        raise ValueError(f"Calendar is missing dates for sales days: {missing}")
    return sales_calendar


def with_calendar(demand_by_day: pd.DataFrame, sales_calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach real dates and calendar attributes to an aggregated daily table."""
    calendar_cols = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year"]
    result = sales_calendar[calendar_cols].merge(demand_by_day, on="d", how="inner")
    return result.sort_values("date").reset_index(drop=True)


def build_daily_total(
    sales: pd.DataFrame,
    demand_cols: list[str],
    sales_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total demand by day without reshaping the full raw table."""
    daily = sales[demand_cols].sum(axis=0).rename("demand").reset_index()
    daily.columns = ["d", "demand"]
    daily["demand"] = daily["demand"].astype("int64")
    return with_calendar(daily, sales_calendar)


def build_daily_group(
    sales: pd.DataFrame,
    demand_cols: list[str],
    sales_calendar: pd.DataFrame,
    group_col: str,
    output_group_col: str,
) -> pd.DataFrame:
    """Aggregate daily demand by one hierarchy level.

    The wide groupby produces only group_count x day_count cells. We then stack
    this compact matrix, not the original 30,490 x 1,913 sales matrix.
    """
    grouped_wide = sales.groupby(group_col, observed=True)[demand_cols].sum().T
    grouped_wide.index.name = "d"
    compact_long = grouped_wide.stack().rename("demand").reset_index()
    compact_long = compact_long.rename(columns={group_col: output_group_col})
    compact_long["demand"] = compact_long["demand"].astype("int64")
    result = with_calendar(compact_long, sales_calendar)
    del grouped_wide, compact_long
    gc.collect()
    return result


def build_series_summary(sales: pd.DataFrame, demand_cols: list[str]) -> pd.DataFrame:
    """Summarize total and zero-demand days per item-store series."""
    demand_matrix = sales[demand_cols]
    summary = sales[ID_COLUMNS].copy()
    summary["total_demand"] = demand_matrix.sum(axis=1).astype("int64")
    summary["zero_demand_days"] = demand_matrix.eq(0).sum(axis=1).astype("int16")
    summary["nonzero_demand_days"] = (len(demand_cols) - summary["zero_demand_days"]).astype("int16")
    summary["avg_daily_demand"] = (summary["total_demand"] / len(demand_cols)).astype("float32")
    return summary


def build_product_summary(series_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate demand from item-store series to product level."""
    product = (
        series_summary.groupby(["item_id", "dept_id", "cat_id"], observed=True)
        .agg(
            total_demand=("total_demand", "sum"),
            series_count=("id", "count"),
            avg_series_demand=("total_demand", "mean"),
            zero_demand_days=("zero_demand_days", "sum"),
            nonzero_demand_days=("nonzero_demand_days", "sum"),
        )
        .reset_index()
    )
    product["total_demand"] = product["total_demand"].astype("int64")
    product["series_count"] = product["series_count"].astype("int16")
    product["avg_series_demand"] = product["avg_series_demand"].astype("float32")
    product["zero_demand_days"] = product["zero_demand_days"].astype("int64")
    product["nonzero_demand_days"] = product["nonzero_demand_days"].astype("int64")
    return product.sort_values("total_demand", ascending=False).reset_index(drop=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a processed table as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print_step(f"Saved {path.name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")


def dimensions_for_outputs() -> dict[str, dict[str, int]]:
    """Return dimensions of generated Parquet outputs."""
    dimensions: dict[str, dict[str, int]] = {}
    for name, path in OUTPUT_FILES.items():
        df = pd.read_parquet(path)
        dimensions[name] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        del df
    return dimensions


def table_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""
    if df.empty:
        return "No rows available."

    headers = [str(column) for column in df.columns]
    rows = df.astype(str).values.tolist()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(
    profile: dict[str, object],
    hierarchy: dict[str, int],
    memory_profile: dict[str, float],
    output_dimensions: dict[str, dict[str, int]],
    top_products: pd.DataFrame,
    top_stores: pd.DataFrame,
    top_categories: pd.DataFrame,
    top_states: pd.DataFrame,
) -> str:
    """Build the Markdown summary report."""
    lines = [
        "# Data Understanding Summary",
        "",
        "## Dataset Dimensions",
        "",
        f"- Sales table rows: `{profile['sales_rows']:,}`",
        f"- Sales table columns: `{profile['sales_columns']:,}`",
        f"- Calendar rows: `{profile['calendar_rows']:,}`",
        f"- Sell prices rows: `{profile['sell_prices_rows']:,}`",
        f"- Historical sales days: `{profile['historical_days']:,}`",
        f"- Sales date range: `{profile['min_sales_date']}` to `{profile['max_sales_date']}`",
        "",
        "## Hierarchies Found",
        "",
    ]

    for key, value in hierarchy.items():
        lines.append(f"- {key}: `{value:,}`")

    lines.extend(
        [
            "",
            "## General Statistics",
            "",
            f"- Total item-store series: `{profile['total_series']:,}`",
            f"- Total demand: `{profile['total_demand']:,}` units",
            f"- Zero-demand values: `{profile['zero_demand_pct']:.2f}%`",
            f"- Approximate peak Python allocation during run: `{profile['peak_traced_memory_mb']:.2f} MB`",
            f"- Execution time: `{profile['execution_seconds']:.2f} seconds`",
            "",
            "## Approximate Memory Used By Loaded Files",
            "",
        ]
    )

    for file_name, memory in memory_profile.items():
        lines.append(f"- {file_name}: `{memory:.2f} MB`")

    lines.extend(["", "## Processed Parquet Dimensions", ""])
    for table_name, dims in output_dimensions.items():
        lines.append(f"- {table_name}: `{dims['rows']:,}` rows x `{dims['columns']:,}` columns")

    lines.extend(
        [
            "",
            "## Top Products By Volume",
            "",
            table_to_markdown(top_products),
            "",
            "## Top Stores By Volume",
            "",
            table_to_markdown(top_stores),
            "",
            "## Top Categories By Volume",
            "",
            table_to_markdown(top_categories),
            "",
            "## Top States By Volume",
            "",
            table_to_markdown(top_states),
            "",
            "## Initial Business Observations",
            "",
            "- Demand is highly hierarchical: the same product can behave differently by store and state.",
            "- Aggregated views by state, store, category, and department are enough for early business understanding without creating the full long sales table.",
            "- The zero-demand percentage is an important signal: many item-store combinations have intermittent demand, which will affect baseline and ML model choice.",
            "- Store and category concentration should guide inventory prioritization before modeling every SKU at equal depth.",
            "",
            "## Data Quality Risks",
            "",
            "- Zero sales may represent true no-demand days, stockouts, unavailable products, or pre-launch periods; they should not all be interpreted the same way.",
            "- Sell prices are weekly and may be missing before an item is available in a store.",
            "- Event columns in calendar contain expected nulls on non-event days.",
            "- The validation file ends before the evaluation horizon, so future phases must handle temporal splits carefully.",
            "",
            "## Memory And Performance Considerations",
            "",
            "- The script avoids a full melt of `sales_train_validation.csv`, which would create roughly 58 million rows.",
            "- Daily tables are created through vectorized column sums and compact groupby matrices.",
            "- Only aggregated matrices are stacked into long format.",
            "- Intermediate grouped objects are explicitly deleted and garbage collection is requested after each aggregation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run Phase 3 data understanding."""
    start = time.perf_counter()
    tracemalloc.start()
    require_raw_files()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print_step("Loading calendar.csv...")
    calendar = read_calendar()
    print_step("Loading sales_train_validation.csv with compact dtypes...")
    sales = read_sales()
    print_step("Loading sell_prices.csv...")
    sell_prices = read_sell_prices()

    memory_profile = {
        "calendar.csv": memory_mb(calendar),
        "sales_train_validation.csv": memory_mb(sales),
        "sell_prices.csv": memory_mb(sell_prices),
    }

    demand_cols = get_demand_columns(sales)
    sales_calendar = calendar_for_sales_days(calendar, demand_cols)
    demand_matrix = sales[demand_cols]

    print_step("Computing global demand profile...")
    total_demand = int(demand_matrix.sum(axis=0).sum())
    total_cells = int(sales.shape[0] * len(demand_cols))
    zero_values = int(demand_matrix.eq(0).sum(axis=0).sum())
    zero_demand_pct = zero_values / total_cells * 100

    hierarchy = {
        "states": int(sales["state_id"].nunique()),
        "stores": int(sales["store_id"].nunique()),
        "categories": int(sales["cat_id"].nunique()),
        "departments": int(sales["dept_id"].nunique()),
        "products": int(sales["item_id"].nunique()),
        "item_store_combinations": int(sales[["item_id", "store_id"]].drop_duplicates().shape[0]),
    }

    print_step("Building daily total demand...")
    daily_total = build_daily_total(sales, demand_cols, sales_calendar)
    save_parquet(daily_total, OUTPUT_FILES["daily_total_demand"])

    print_step("Building daily demand by state...")
    daily_state = build_daily_group(sales, demand_cols, sales_calendar, "state_id", "state_id")
    save_parquet(daily_state, OUTPUT_FILES["daily_demand_by_state"])

    print_step("Building daily demand by store...")
    daily_store = build_daily_group(sales, demand_cols, sales_calendar, "store_id", "store_id")
    save_parquet(daily_store, OUTPUT_FILES["daily_demand_by_store"])

    print_step("Building daily demand by category...")
    daily_category = build_daily_group(sales, demand_cols, sales_calendar, "cat_id", "cat_id")
    save_parquet(daily_category, OUTPUT_FILES["daily_demand_by_category"])

    print_step("Building daily demand by department...")
    daily_department = build_daily_group(sales, demand_cols, sales_calendar, "dept_id", "dept_id")
    save_parquet(daily_department, OUTPUT_FILES["daily_demand_by_department"])

    print_step("Building item-store series demand summary...")
    series_summary = build_series_summary(sales, demand_cols)
    save_parquet(series_summary, OUTPUT_FILES["series_demand_summary"])

    print_step("Building product demand summary...")
    product_summary = build_product_summary(series_summary)
    save_parquet(product_summary, OUTPUT_FILES["product_demand_summary"])

    top_products = product_summary[["item_id", "dept_id", "cat_id", "total_demand", "series_count"]].head(10)
    top_stores = (
        daily_store.groupby("store_id", observed=True)["demand"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_categories = (
        daily_category.groupby("cat_id", observed=True)["demand"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_states = (
        daily_state.groupby("state_id", observed=True)["demand"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    elapsed = time.perf_counter() - start

    profile = {
        "sales_rows": int(sales.shape[0]),
        "sales_columns": int(sales.shape[1]),
        "calendar_rows": int(calendar.shape[0]),
        "sell_prices_rows": int(sell_prices.shape[0]),
        "historical_days": int(len(demand_cols)),
        "min_sales_date": str(sales_calendar["date"].min().date()),
        "max_sales_date": str(sales_calendar["date"].max().date()),
        "total_series": int(sales.shape[0]),
        "total_demand": total_demand,
        "zero_demand_pct": zero_demand_pct,
        "peak_traced_memory_mb": peak_memory / 1024**2,
        "execution_seconds": elapsed,
    }

    output_dimensions = dimensions_for_outputs()
    report = build_report(
        profile=profile,
        hierarchy=hierarchy,
        memory_profile=memory_profile,
        output_dimensions=output_dimensions,
        top_products=top_products,
        top_stores=top_stores,
        top_categories=top_categories,
        top_states=top_states,
    )
    SUMMARY_FILE.write_text(report, encoding="utf-8")
    print_step(f"Summary written to {SUMMARY_FILE}")

    metrics_path = PROCESSED_DATA_DIR / "data_understanding_metrics.json"
    metrics = {
        "profile": profile,
        "hierarchy": hierarchy,
        "memory_profile_mb": memory_profile,
        "output_dimensions": output_dimensions,
        "top_products": json.loads(top_products.to_json(orient="records")),
        "top_stores": json.loads(top_stores.to_json(orient="records")),
        "top_categories": json.loads(top_categories.to_json(orient="records")),
        "top_states": json.loads(top_states.to_json(orient="records")),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_step(f"Metrics written to {metrics_path}")
    print_step(f"Completed in {elapsed:.2f} seconds")
    tracemalloc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

