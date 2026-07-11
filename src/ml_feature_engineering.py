"""Feature engineering for store-department ML forecasting.

The output keeps one row per unique_id + date at store-department level. Demand
features are shifted so each row only uses information available before or at the
forecast origin handled by the ML backtesting script.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR


FORECAST_DATASET = PROCESSED_DATA_DIR / "forecast_store_department.parquet"
CALENDAR_FILE = RAW_DATA_DIR / "calendar.csv"
SELL_PRICES_FILE = RAW_DATA_DIR / "sell_prices.csv"
FEATURE_OUTPUT = PROCESSED_DATA_DIR / "ml_store_department_features.parquet"
FEATURE_REPORT = REPORTS_DIR / "ml_feature_summary.md"
FEATURE_METRICS = PROCESSED_DATA_DIR / "ml_feature_metrics.json"

LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 35, 42, 56]
ROLL_WINDOWS = [7, 14, 28, 56]


def print_step(message: str) -> None:
    print(f"[ml_feature_engineering] {message}")


def derive_state_id(store_id: pd.Series) -> pd.Series:
    return store_id.astype(str).str.split("_").str[0]


def derive_cat_id(dept_id: pd.Series) -> pd.Series:
    return dept_id.astype(str).str.rsplit("_", n=1).str[0]


def read_calendar() -> pd.DataFrame:
    calendar = pd.read_csv(CALENDAR_FILE, parse_dates=["date"])
    calendar["d_order"] = calendar["d"].str.replace("d_", "", regex=False).astype("int16")
    calendar["quarter"] = calendar["date"].dt.quarter.astype("int8")
    calendar["week_of_year"] = calendar["date"].dt.isocalendar().week.astype("int16")
    calendar["day_of_month"] = calendar["date"].dt.day.astype("int8")
    calendar["day_of_week"] = calendar["date"].dt.dayofweek.astype("int8")
    calendar["is_weekend"] = calendar["day_of_week"].isin([5, 6]).astype("int8")
    calendar["day_index"] = calendar["d_order"].astype("int16")
    calendar["sin_day_of_week"] = np.sin(2 * np.pi * calendar["day_of_week"] / 7).astype("float32")
    calendar["cos_day_of_week"] = np.cos(2 * np.pi * calendar["day_of_week"] / 7).astype("float32")
    calendar["sin_month"] = np.sin(2 * np.pi * calendar["month"] / 12).astype("float32")
    calendar["cos_month"] = np.cos(2 * np.pi * calendar["month"] / 12).astype("float32")
    calendar["has_event"] = (
        calendar["event_name_1"].notna() | calendar["event_name_2"].notna()
    ).astype("int8")
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        calendar[col] = calendar[col].fillna("none").astype("string")
    return calendar


def build_price_features() -> pd.DataFrame:
    print_step("Aggregating sell_prices to store_id + dept_id + wm_yr_wk...")
    prices = pd.read_csv(
        SELL_PRICES_FILE,
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    parts = prices["item_id"].str.split("_", expand=True)
    prices["dept_id"] = parts[0] + "_" + parts[1]
    agg = (
        prices.groupby(["store_id", "dept_id", "wm_yr_wk"], observed=True)
        .agg(
            mean_sell_price=("sell_price", "mean"),
            median_sell_price=("sell_price", "median"),
            min_sell_price=("sell_price", "min"),
            max_sell_price=("sell_price", "max"),
            price_std=("sell_price", "std"),
            item_count_with_price=("item_id", "nunique"),
        )
        .reset_index()
        .sort_values(["store_id", "dept_id", "wm_yr_wk"])
    )
    agg["price_std"] = agg["price_std"].fillna(0)
    agg["price_change_vs_previous_week"] = agg.groupby(["store_id", "dept_id"], observed=True)[
        "mean_sell_price"
    ].pct_change()
    dept_mean = agg.groupby(["store_id", "dept_id"], observed=True)["mean_sell_price"].transform("mean")
    agg["price_index_vs_store_department_mean"] = np.divide(
        agg["mean_sell_price"],
        dept_mean,
        out=np.ones(len(agg), dtype="float64"),
        where=dept_mean > 0,
    )
    price_cols = [
        "mean_sell_price",
        "median_sell_price",
        "min_sell_price",
        "max_sell_price",
        "price_std",
        "price_change_vs_previous_week",
        "price_index_vs_store_department_mean",
    ]
    agg[price_cols] = agg[price_cols].replace([np.inf, -np.inf], np.nan).astype("float32")
    agg["item_count_with_price"] = agg["item_count_with_price"].astype("int16")
    return agg


def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    print_step("Creating shifted lag, rolling, expanding and trend features...")
    df = df.sort_values(["unique_id", "date"]).copy()
    grouped = df.groupby("unique_id", observed=True)["demand"]
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    for window in ROLL_WINDOWS:
        rolling = shifted.groupby(df["unique_id"], observed=True).rolling(window, min_periods=1)
        df[f"rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{window}"] = rolling.std().reset_index(level=0, drop=True).fillna(0)
    for window in [7, 28]:
        rolling = shifted.groupby(df["unique_id"], observed=True).rolling(window, min_periods=1)
        df[f"rolling_min_{window}"] = rolling.min().reset_index(level=0, drop=True)
        df[f"rolling_max_{window}"] = rolling.max().reset_index(level=0, drop=True)
    zero_rolling = (shifted == 0).astype("float32").groupby(df["unique_id"], observed=True).rolling(28, min_periods=1)
    df["rolling_zero_pct_28"] = zero_rolling.mean().reset_index(level=0, drop=True) * 100

    df["expanding_mean"] = shifted.groupby(df["unique_id"], observed=True).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    df["expanding_std"] = shifted.groupby(df["unique_id"], observed=True).expanding(min_periods=2).std().reset_index(level=0, drop=True).fillna(0)
    df["demand_change_1"] = df["lag_1"] - df["lag_2"]
    df["demand_change_7"] = df["lag_7"] - df["lag_14"]
    df["lag_7_vs_28_ratio"] = np.divide(
        df["lag_7"],
        df["lag_28"],
        out=np.ones(len(df), dtype="float64"),
        where=df["lag_28"].fillna(0).to_numpy() > 0,
    )
    df["rolling_mean_7_vs_28_ratio"] = np.divide(
        df["rolling_mean_7"],
        df["rolling_mean_28"],
        out=np.ones(len(df), dtype="float64"),
        where=df["rolling_mean_28"].fillna(0).to_numpy() > 0,
    )
    return df


def build_features() -> pd.DataFrame:
    if not FORECAST_DATASET.exists():
        raise FileNotFoundError(f"Missing {FORECAST_DATASET}. Run src/forecasting_dataset.py first.")
    base = pd.read_parquet(FORECAST_DATASET)
    base["date"] = pd.to_datetime(base["date"])
    base["state_id"] = derive_state_id(base["store_id"])
    base["cat_id"] = derive_cat_id(base["dept_id"])

    calendar = read_calendar()
    calendar_cols = [
        "date",
        "wm_yr_wk",
        "year",
        "quarter",
        "month",
        "week_of_year",
        "day_of_month",
        "day_of_week",
        "is_weekend",
        "day_index",
        "sin_day_of_week",
        "cos_day_of_week",
        "sin_month",
        "cos_month",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "has_event",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    features = base.merge(calendar[calendar_cols], on="date", how="left", validate="many_to_one")
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    features["snap_active"] = [row[snap_map[state]] for state, row in zip(features["state_id"], features[["snap_CA", "snap_TX", "snap_WI"]].to_dict("records"), strict=False)]
    features["snap_active"] = pd.Series(features["snap_active"], index=features.index).astype("int8")

    price_features = build_price_features()
    features = features.merge(
        price_features,
        on=["store_id", "dept_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    price_cols = [
        "mean_sell_price",
        "median_sell_price",
        "min_sell_price",
        "max_sell_price",
        "price_std",
        "item_count_with_price",
        "price_change_vs_previous_week",
        "price_index_vs_store_department_mean",
    ]
    features = features.sort_values(["unique_id", "date"])
    features[price_cols] = features.groupby("unique_id", observed=True)[price_cols].ffill().bfill()
    features[price_cols] = features[price_cols].fillna(0)
    features = add_demand_features(features)

    numeric_float_cols = features.select_dtypes(include=["float64"]).columns
    features[numeric_float_cols] = features[numeric_float_cols].astype("float32")
    return features.sort_values(["unique_id", "date"]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    lines = ["| " + " | ".join(map(str, df.columns)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for row in df.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(features: pd.DataFrame, elapsed: float) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    nulls = (
        features.isna().sum().rename("null_count").reset_index().rename(columns={"index": "column"})
    )
    nulls["null_pct"] = (nulls["null_count"] / len(features) * 100).round(2)
    dtypes = pd.DataFrame({"column": features.columns, "dtype": [str(features[col].dtype) for col in features.columns]})
    max_lag = max(LAG_DAYS)
    lost_rows = int(features.loc[features["lag_56"].isna()].shape[0])
    lines = [
        "# ML Feature Summary",
        "",
        f"Rows: `{features.shape[0]:,}`",
        f"Columns: `{features.shape[1]:,}`",
        f"Date range: `{features['date'].min().date()}` to `{features['date'].max().date()}`",
        f"Unique series: `{features['unique_id'].nunique():,}`",
        f"Execution seconds: `{elapsed:.2f}`",
        "",
        "## Anti-Leakage Rules",
        "",
        "- Lag features are shifted by series and never use the current target value.",
        "- Rolling, expanding and zero-percentage features are computed from `demand.shift(1)`.",
        "- Price features are aggregated by store-department-week without demand weighting.",
        "- Direct multi-horizon modeling uses features at the forecast origin plus known target calendar fields.",
        "",
        "## Price Aggregation Methodology",
        "",
        "Item-store prices are mapped to department from `item_id`, then aggregated by `store_id`, `dept_id`, and `wm_yr_wk`. The pipeline uses mean, median, min, max, standard deviation, item count, week-over-week mean price change, and a price index against the store-department historical mean. No future demand is used as a weight.",
        "",
        "## Lag Row Loss",
        "",
        f"Rows with missing lag_{max_lag}: `{lost_rows:,}`. Training drops rows only when the selected model features are missing.",
        "",
        "## Columns And Types",
        "",
        markdown_table(dtypes),
        "",
        "## Null Summary",
        "",
        markdown_table(nulls.loc[nulls["null_count"] > 0].head(80)),
    ]
    FEATURE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    features = build_features()
    features.to_parquet(FEATURE_OUTPUT, index=False)
    elapsed = time.perf_counter() - start
    write_report(features, elapsed)
    metrics = {
        "execution_seconds": elapsed,
        "rows": int(features.shape[0]),
        "columns": int(features.shape[1]),
        "unique_series": int(features["unique_id"].nunique()),
        "min_date": str(features["date"].min().date()),
        "max_date": str(features["date"].max().date()),
        "rows_missing_lag_56": int(features["lag_56"].isna().sum()),
    }
    FEATURE_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_step(f"Saved {FEATURE_OUTPUT.name}: {features.shape[0]:,} rows x {features.shape[1]:,} columns")
    print_step(f"Completed in {elapsed:.2f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())