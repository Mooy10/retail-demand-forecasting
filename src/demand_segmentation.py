"""Segment M5 demand series by behavior, volume, and variability.

This phase avoids creating a full 58M-row long table. Metrics are computed from
the wide sales matrix with vectorized NumPy/Pandas operations at item-store
series level.
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR


SALES_FILE = RAW_DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"
PRODUCT_SUMMARY_FILE = PROCESSED_DATA_DIR / "product_demand_summary.parquet"
SERIES_SUMMARY_FILE = PROCESSED_DATA_DIR / "series_demand_summary.parquet"

DEMAND_SEGMENTATION_FILE = PROCESSED_DATA_DIR / "demand_segmentation.parquet"
SEGMENT_SUMMARY_FILE = PROCESSED_DATA_DIR / "segment_summary.parquet"
ABC_XYZ_SUMMARY_FILE = PROCESSED_DATA_DIR / "abc_xyz_summary.parquet"
PATTERN_BY_CATEGORY_FILE = PROCESSED_DATA_DIR / "demand_pattern_by_category.parquet"
PATTERN_BY_STORE_FILE = PROCESSED_DATA_DIR / "demand_pattern_by_store.parquet"
METRICS_FILE = PROCESSED_DATA_DIR / "demand_segmentation_metrics.json"
REPORT_FILE = REPORTS_DIR / "demand_segmentation_summary.md"

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
XYZ_X_THRESHOLD = 0.50
XYZ_Y_THRESHOLD = 1.00


def print_step(message: str) -> None:
    """Print a standard progress message."""
    print(f"[demand_segmentation] {message}")


def require_inputs() -> None:
    """Validate all required inputs for this phase exist."""
    required = [SALES_FILE, CALENDAR_FILE, PRODUCT_SUMMARY_FILE, SERIES_SUMMARY_FILE]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required inputs:\n{formatted}")


def read_calendar() -> pd.DataFrame:
    """Read calendar and create an ordered day-to-date map."""
    calendar = pd.read_csv(CALENDAR_FILE, usecols=["d", "date"], parse_dates=["date"])
    calendar["d_order"] = calendar["d"].str.replace("d_", "", regex=False).astype("int32")
    return calendar.sort_values("d_order").reset_index(drop=True)


def read_sales() -> pd.DataFrame:
    """Read sales table with compact demand dtypes."""
    header = pd.read_csv(SALES_FILE, nrows=0)
    demand_cols = [col for col in header.columns if col.startswith("d_")]
    dtype_map = {col: "int16" for col in demand_cols}
    dtype_map.update({col: "category" for col in ID_COLUMNS})
    dtype_map["id"] = "string"
    return pd.read_csv(SALES_FILE, dtype=dtype_map)


def get_demand_columns(sales: pd.DataFrame) -> list[str]:
    """Return demand columns ordered by day number."""
    demand_cols = [col for col in sales.columns if col.startswith("d_")]
    return sorted(demand_cols, key=lambda value: int(value.split("_")[1]))


def align_sales_calendar(calendar: pd.DataFrame, demand_cols: list[str]) -> pd.DataFrame:
    """Return calendar rows aligned to sales demand columns."""
    day_frame = pd.DataFrame({"d": demand_cols})
    aligned = day_frame.merge(calendar, on="d", how="left", validate="one_to_one")
    if aligned["date"].isna().any():
        missing = aligned.loc[aligned["date"].isna(), "d"].head(10).tolist()
        raise ValueError(f"Calendar is missing dates for sales days: {missing}")
    return aligned


def classify_demand_pattern(adi: np.ndarray, cv_squared: np.ndarray) -> np.ndarray:
    """Classify demand pattern with Syntetos-Boylan thresholds."""
    conditions = [
        (adi < ADI_THRESHOLD) & (cv_squared < CV2_THRESHOLD),
        (adi < ADI_THRESHOLD) & (cv_squared >= CV2_THRESHOLD),
        (adi >= ADI_THRESHOLD) & (cv_squared < CV2_THRESHOLD),
        (adi >= ADI_THRESHOLD) & (cv_squared >= CV2_THRESHOLD),
    ]
    choices = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    return np.select(conditions, choices, default="Unclassified")


def assign_abc_classes(total_demand: pd.Series) -> pd.Series:
    """Assign ABC classes by cumulative demand contribution."""
    ordered = total_demand.sort_values(ascending=False)
    total = ordered.sum()
    if total == 0:
        return pd.Series("C", index=total_demand.index, dtype="string")

    cumulative_share = ordered.cumsum() / total
    abc_ordered = pd.Series("C", index=ordered.index, dtype="string")
    abc_ordered.loc[cumulative_share <= 0.80] = "A"
    abc_ordered.loc[(cumulative_share > 0.80) & (cumulative_share <= 0.95)] = "B"
    return abc_ordered.reindex(total_demand.index).astype("string")


def assign_xyz_classes(coefficient_of_variation: pd.Series) -> pd.Series:
    """Assign XYZ classes from coefficient of variation.

    X: CV < 0.50, low variability.
    Y: 0.50 <= CV < 1.00, medium variability.
    Z: CV >= 1.00, high variability.
    """
    xyz = pd.Series("Z", index=coefficient_of_variation.index, dtype="string")
    xyz.loc[coefficient_of_variation < XYZ_X_THRESHOLD] = "X"
    xyz.loc[
        (coefficient_of_variation >= XYZ_X_THRESHOLD)
        & (coefficient_of_variation < XYZ_Y_THRESHOLD)
    ] = "Y"
    return xyz


def compute_segmentation(sales: pd.DataFrame, sales_calendar: pd.DataFrame) -> pd.DataFrame:
    """Compute item-store segmentation metrics from the wide sales matrix."""
    demand_cols = get_demand_columns(sales)
    dates = sales_calendar["date"].to_numpy(dtype="datetime64[ns]")
    n_days = len(demand_cols)

    print_step("Converting demand matrix to NumPy view...")
    demand_values = sales[demand_cols].to_numpy(copy=False)
    values_float = demand_values.astype("float32", copy=False)

    print_step("Computing core demand statistics...")
    total_demand = demand_values.sum(axis=1, dtype="int64")
    mean_demand = values_float.mean(axis=1, dtype="float64")
    std_demand = values_float.std(axis=1, dtype="float64")
    coefficient_of_variation = np.divide(
        std_demand,
        mean_demand,
        out=np.zeros_like(std_demand, dtype="float64"),
        where=mean_demand > 0,
    )
    max_demand = demand_values.max(axis=1)
    median_demand = np.median(demand_values, axis=1)
    p90_demand = np.percentile(demand_values, 90, axis=1)

    print_step("Computing zero-demand and active-window statistics...")
    positive_mask = demand_values > 0
    positive_days = positive_mask.sum(axis=1).astype("int32")
    zero_demand_pct = (1 - positive_days / n_days) * 100
    has_positive = positive_days > 0

    first_idx = np.full(len(sales), -1, dtype="int32")
    last_idx = np.full(len(sales), -1, dtype="int32")
    first_idx[has_positive] = positive_mask.argmax(axis=1)[has_positive]
    last_idx[has_positive] = n_days - 1 - positive_mask[:, ::-1].argmax(axis=1)[has_positive]

    active_days = np.zeros(len(sales), dtype="int32")
    active_days[has_positive] = last_idx[has_positive] - first_idx[has_positive] + 1

    first_sale_date = np.full(len(sales), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    last_sale_date = np.full(len(sales), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    first_sale_date[has_positive] = dates[first_idx[has_positive]]
    last_sale_date[has_positive] = dates[last_idx[has_positive]]

    average_demand_interval = np.full(len(sales), np.nan, dtype="float64")
    multiple_positive = positive_days > 1
    average_demand_interval[multiple_positive] = (
        last_idx[multiple_positive] - first_idx[multiple_positive]
    ) / (positive_days[multiple_positive] - 1)
    average_demand_interval[positive_days == 1] = float(n_days)

    adi = np.divide(
        n_days,
        positive_days,
        out=np.full(len(sales), float(n_days), dtype="float64"),
        where=positive_days > 0,
    )

    print_step("Computing CV squared on non-zero demand only...")
    positive_sum = total_demand.astype("float64")
    positive_sq_sum = np.einsum("ij,ij->i", values_float, values_float, optimize=True)
    mean_positive = np.divide(
        positive_sum,
        positive_days,
        out=np.zeros(len(sales), dtype="float64"),
        where=positive_days > 0,
    )
    var_positive = np.divide(
        positive_sq_sum,
        positive_days,
        out=np.zeros(len(sales), dtype="float64"),
        where=positive_days > 0,
    ) - np.square(mean_positive)
    var_positive = np.clip(var_positive, a_min=0, a_max=None)
    cv_squared = np.divide(
        var_positive,
        np.square(mean_positive),
        out=np.zeros(len(sales), dtype="float64"),
        where=mean_positive > 0,
    )

    print_step("Assembling segmentation table...")
    segmentation = sales[ID_COLUMNS].copy()
    segmentation["total_demand"] = total_demand.astype("int64")
    segmentation["mean_demand"] = mean_demand.astype("float32")
    segmentation["std_demand"] = std_demand.astype("float32")
    segmentation["coefficient_of_variation"] = coefficient_of_variation.astype("float32")
    segmentation["zero_demand_pct"] = zero_demand_pct.astype("float32")
    segmentation["positive_demand_days"] = positive_days.astype("int32")
    segmentation["max_demand"] = max_demand.astype("int16")
    segmentation["median_demand"] = median_demand.astype("float32")
    segmentation["p90_demand"] = p90_demand.astype("float32")
    segmentation["first_sale_date"] = first_sale_date
    segmentation["last_sale_date"] = last_sale_date
    segmentation["active_days"] = active_days.astype("int32")
    segmentation["average_demand_interval"] = average_demand_interval.astype("float32")
    segmentation["adi"] = adi.astype("float32")
    segmentation["cv_squared"] = cv_squared.astype("float32")
    segmentation["demand_pattern"] = classify_demand_pattern(adi, cv_squared)
    segmentation["abc_class"] = assign_abc_classes(segmentation["total_demand"])
    segmentation["xyz_class"] = assign_xyz_classes(segmentation["coefficient_of_variation"])
    segmentation["abc_xyz_segment"] = segmentation["abc_class"].astype(str) + segmentation[
        "xyz_class"
    ].astype(str)

    del demand_values, values_float, positive_mask
    gc.collect()
    return segmentation


def demand_share(series: pd.Series) -> pd.Series:
    """Return demand share percentage for a grouped total demand series."""
    total = series.sum()
    if total == 0:
        return pd.Series(0.0, index=series.index)
    return series / total * 100


def build_segment_summary(segmentation: pd.DataFrame) -> pd.DataFrame:
    """Summarize demand behavior segments."""
    summary = (
        segmentation.groupby("demand_pattern", observed=True)
        .agg(
            series_count=("id", "count"),
            total_demand=("total_demand", "sum"),
            avg_mean_demand=("mean_demand", "mean"),
            avg_zero_demand_pct=("zero_demand_pct", "mean"),
            avg_adi=("adi", "mean"),
            avg_cv_squared=("cv_squared", "mean"),
        )
        .reset_index()
    )
    summary["series_share_pct"] = summary["series_count"] / summary["series_count"].sum() * 100
    summary["demand_share_pct"] = summary["total_demand"] / summary["total_demand"].sum() * 100
    return summary.sort_values("series_count", ascending=False).reset_index(drop=True)


def build_abc_xyz_summary(segmentation: pd.DataFrame) -> pd.DataFrame:
    """Summarize combined ABC-XYZ segments."""
    summary = (
        segmentation.groupby(["abc_class", "xyz_class", "abc_xyz_segment"], observed=True)
        .agg(
            series_count=("id", "count"),
            total_demand=("total_demand", "sum"),
            avg_zero_demand_pct=("zero_demand_pct", "mean"),
            avg_coefficient_of_variation=("coefficient_of_variation", "mean"),
        )
        .reset_index()
    )
    summary["series_share_pct"] = summary["series_count"] / summary["series_count"].sum() * 100
    summary["demand_share_pct"] = summary["total_demand"] / summary["total_demand"].sum() * 100
    return summary.sort_values(["abc_class", "xyz_class"]).reset_index(drop=True)


def build_pattern_by_dimension(segmentation: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Summarize demand patterns by a hierarchy dimension."""
    summary = (
        segmentation.groupby([dimension, "demand_pattern"], observed=True)
        .agg(
            series_count=("id", "count"),
            total_demand=("total_demand", "sum"),
            avg_zero_demand_pct=("zero_demand_pct", "mean"),
            avg_adi=("adi", "mean"),
            avg_cv_squared=("cv_squared", "mean"),
        )
        .reset_index()
    )
    dimension_totals = summary.groupby(dimension, observed=True)["total_demand"].transform("sum")
    summary["dimension_demand_share_pct"] = np.divide(
        summary["total_demand"],
        dimension_totals,
        out=np.zeros(len(summary), dtype="float64"),
        where=dimension_totals > 0,
    ) * 100
    return summary.sort_values([dimension, "total_demand"], ascending=[True, False]).reset_index(drop=True)


def write_outputs(segmentation: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Write segmentation outputs and return dimensions."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "demand_segmentation": (segmentation, DEMAND_SEGMENTATION_FILE),
        "segment_summary": (build_segment_summary(segmentation), SEGMENT_SUMMARY_FILE),
        "abc_xyz_summary": (build_abc_xyz_summary(segmentation), ABC_XYZ_SUMMARY_FILE),
        "demand_pattern_by_category": (
            build_pattern_by_dimension(segmentation, "cat_id"),
            PATTERN_BY_CATEGORY_FILE,
        ),
        "demand_pattern_by_store": (
            build_pattern_by_dimension(segmentation, "store_id"),
            PATTERN_BY_STORE_FILE,
        ),
    }

    dimensions: dict[str, dict[str, int]] = {}
    for name, (df, path) in outputs.items():
        df.to_parquet(path, index=False)
        dimensions[name] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        print_step(f"Saved {path.name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    return dimensions


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional dependencies."""
    if df.empty:
        return "No rows available."
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(
    segmentation: pd.DataFrame,
    segment_summary: pd.DataFrame,
    abc_xyz_summary: pd.DataFrame,
    pattern_by_category: pd.DataFrame,
    pattern_by_store: pd.DataFrame,
    dimensions: dict[str, dict[str, int]],
    metrics: dict[str, object],
) -> str:
    """Build the demand segmentation report."""
    abc_distribution = (
        segmentation.groupby("abc_class", observed=True)
        .agg(series_count=("id", "count"), total_demand=("total_demand", "sum"))
        .reset_index()
    )
    abc_distribution["series_share_pct"] = abc_distribution["series_count"] / len(segmentation) * 100
    abc_distribution["demand_share_pct"] = demand_share(abc_distribution["total_demand"])

    xyz_distribution = (
        segmentation.groupby("xyz_class", observed=True)
        .agg(series_count=("id", "count"), total_demand=("total_demand", "sum"))
        .reset_index()
    )
    xyz_distribution["series_share_pct"] = xyz_distribution["series_count"] / len(segmentation) * 100
    xyz_distribution["demand_share_pct"] = demand_share(xyz_distribution["total_demand"])

    top_category_patterns = pattern_by_category.sort_values("total_demand", ascending=False).head(12)
    top_store_patterns = pattern_by_store.sort_values("total_demand", ascending=False).head(12)

    lines = [
        "# Demand Segmentation Summary",
        "",
        "## Scope",
        "",
        "This phase classifies item-store demand series without building the full long sales table. Metrics are computed from the wide sales matrix using vectorized operations.",
        "",
        "## Output Dimensions",
        "",
    ]

    for name, shape in dimensions.items():
        lines.append(f"- {name}: `{shape['rows']:,}` rows x `{shape['columns']:,}` columns")

    lines.extend(
        [
            "",
            "## Syntetos-Boylan Demand Pattern Distribution",
            "",
            markdown_table(segment_summary),
            "",
            "## ABC Distribution",
            "",
            markdown_table(abc_distribution),
            "",
            "## XYZ Distribution",
            "",
            markdown_table(xyz_distribution),
            "",
            "## ABC-XYZ Matrix",
            "",
            markdown_table(abc_xyz_summary),
            "",
            "## Principal Category-Pattern Combinations",
            "",
            markdown_table(top_category_patterns),
            "",
            "## Principal Store-Pattern Combinations",
            "",
            markdown_table(top_store_patterns),
            "",
            "## Forecasting Implications",
            "",
            "- Smooth and high-volume series are the best starting point for baseline and machine learning forecasting.",
            "- Erratic and Lumpy series require robust error metrics and careful treatment of spikes.",
            "- Intermittent series may need specialized intermittent-demand baselines before advanced ML.",
            "- ABC-XYZ segments help decide where model complexity is worth the operational cost.",
            "",
            "## Inventory Implications",
            "",
            "- AX and AY series should receive the highest replenishment planning attention because they combine high volume with more predictable demand.",
            "- AZ series are high-value but volatile; safety stock and exception monitoring matter more than pure point forecasts.",
            "- C segments contain many low-volume series and may be better managed through simpler inventory rules.",
            "",
            "## Limitations",
            "",
            "- Zero demand can mean no customer demand, stockout, product unavailability, or pre-launch periods.",
            "- ADI and CVÂ² are historical descriptors, not causal explanations.",
            "- The analysis uses `sales_train_validation.csv`; future phases should handle validation/evaluation horizons explicitly.",
            "- XYZ thresholds are business rules based on CV: X < 0.50, Y < 1.00, Z >= 1.00. They can be tuned with planner feedback.",
            "",
            "## Runtime And Memory",
            "",
            f"- Execution time: `{metrics['execution_seconds']:.2f} seconds`",
            f"- Peak traced Python allocation: `{metrics['peak_traced_memory_mb']:.2f} MB`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run Phase 4 demand segmentation."""
    start = time.perf_counter()
    tracemalloc.start()
    require_inputs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print_step("Loading calendar and sales data...")
    calendar = read_calendar()
    sales = read_sales()
    demand_cols = get_demand_columns(sales)
    sales_calendar = align_sales_calendar(calendar, demand_cols)

    print_step("Loading Phase 3 summary Parquets for consistency checks...")
    product_summary = pd.read_parquet(PRODUCT_SUMMARY_FILE)
    series_summary = pd.read_parquet(SERIES_SUMMARY_FILE)

    segmentation = compute_segmentation(sales, sales_calendar)

    print_step("Validating consistency with Phase 3 summaries...")
    if int(segmentation["total_demand"].sum()) != int(series_summary["total_demand"].sum()):
        raise ValueError("Demand total mismatch between segmentation and series summary.")
    if int(segmentation["item_id"].nunique()) != int(product_summary["item_id"].nunique()):
        raise ValueError("Product count mismatch between segmentation and product summary.")

    dimensions = write_outputs(segmentation)
    segment_summary = pd.read_parquet(SEGMENT_SUMMARY_FILE)
    abc_xyz_summary = pd.read_parquet(ABC_XYZ_SUMMARY_FILE)
    pattern_by_category = pd.read_parquet(PATTERN_BY_CATEGORY_FILE)
    pattern_by_store = pd.read_parquet(PATTERN_BY_STORE_FILE)

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    elapsed = time.perf_counter() - start
    metrics = {
        "execution_seconds": elapsed,
        "peak_traced_memory_mb": peak_memory / 1024**2,
        "total_demand": int(segmentation["total_demand"].sum()),
        "series_count": int(segmentation.shape[0]),
        "dimensions": dimensions,
        "pattern_distribution": json.loads(segment_summary.to_json(orient="records")),
        "abc_distribution": json.loads(
            segmentation.groupby("abc_class", observed=True)
            .agg(series_count=("id", "count"), total_demand=("total_demand", "sum"))
            .reset_index()
            .to_json(orient="records")
        ),
        "xyz_distribution": json.loads(
            segmentation.groupby("xyz_class", observed=True)
            .agg(series_count=("id", "count"), total_demand=("total_demand", "sum"))
            .reset_index()
            .to_json(orient="records")
        ),
        "abc_xyz_summary": json.loads(abc_xyz_summary.to_json(orient="records")),
    }

    report = build_report(
        segmentation=segmentation,
        segment_summary=segment_summary,
        abc_xyz_summary=abc_xyz_summary,
        pattern_by_category=pattern_by_category,
        pattern_by_store=pattern_by_store,
        dimensions=dimensions,
        metrics=metrics,
    )
    REPORT_FILE.write_text(report, encoding="utf-8")
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_step(f"Summary written to {REPORT_FILE}")
    print_step(f"Metrics written to {METRICS_FILE}")
    print_step(f"Completed in {elapsed:.2f} seconds")
    tracemalloc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())