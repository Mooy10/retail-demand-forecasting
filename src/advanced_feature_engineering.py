"""Advanced feature engineering for Phase 7 store-department forecasting."""

from __future__ import annotations

import json
import time
import tracemalloc

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_DATA_DIR, REPORTS_DIR
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR, REPORTS_DIR

BASE_FEATURES = PROCESSED_DATA_DIR / "ml_store_department_features.parquet"
ADVANCED_FEATURES = PROCESSED_DATA_DIR / "ml_store_department_advanced_features.parquet"
ADVANCED_METRICS = PROCESSED_DATA_DIR / "advanced_feature_metrics.json"
ADVANCED_REPORT = REPORTS_DIR / "advanced_feature_summary.md"

ADVANCED_FEATURE_NAMES = [
    "rolling_median_7", "rolling_median_14", "rolling_median_28", "rolling_quantile_25_28", "rolling_quantile_75_28",
    "ewm_mean_7", "ewm_mean_14", "ewm_mean_28", "ewm_std_28", "rolling_trend_7", "rolling_trend_28",
    "lag_364", "lag_365", "lag_371", "rolling_mean_364", "previous_year_same_week_mean", "previous_year_same_day_of_week",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end",
    "days_to_next_event", "days_since_previous_event", "event_week", "pre_event_7_days", "post_event_7_days",
    "store_dept", "store_state", "dept_category", "month_day_of_week", "event_store", "snap_department",
    "store_total_lag_7", "store_total_lag_28", "store_rolling_mean_28", "department_total_lag_7", "department_total_lag_28",
    "department_rolling_mean_28", "state_total_lag_7", "state_total_lag_28", "category_total_lag_7", "category_total_lag_28",
    "series_share_of_store_28", "series_share_of_department_28", "series_share_of_state_28", "price_change_1_week",
    "price_change_4_weeks", "price_pct_change_1_week", "price_pct_change_4_weeks", "price_rolling_mean_4_weeks",
    "price_rolling_std_4_weeks", "price_relative_to_department", "price_relative_to_store", "discount_proxy", "price_volatility_12_weeks",
]


def print_step(message: str) -> None:
    print(f"[advanced_feature_engineering] {message}")


def rolling_slope(values: np.ndarray) -> float:
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype="float64")[mask]
    y = values[mask].astype("float64")
    denom = np.square(x - x.mean()).sum()
    if denom == 0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).sum() / denom)


def add_trend_and_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["unique_id", "date"]).copy()
    grouped = df.groupby("unique_id", observed=True)["demand"]
    shifted = grouped.shift(1)
    for window in [7, 14, 28]:
        rolling = shifted.groupby(df["unique_id"], observed=True).rolling(window, min_periods=1)
        df[f"rolling_median_{window}"] = rolling.median().reset_index(level=0, drop=True)
    rolling_28 = shifted.groupby(df["unique_id"], observed=True).rolling(28, min_periods=1)
    df["rolling_quantile_25_28"] = rolling_28.quantile(0.25).reset_index(level=0, drop=True)
    df["rolling_quantile_75_28"] = rolling_28.quantile(0.75).reset_index(level=0, drop=True)
    for span in [7, 14, 28]:
        df[f"ewm_mean_{span}"] = shifted.groupby(df["unique_id"], observed=True).transform(lambda s, span=span: s.ewm(span=span, adjust=False, min_periods=1).mean())
    df["ewm_std_28"] = shifted.groupby(df["unique_id"], observed=True).transform(lambda s: s.ewm(span=28, adjust=False, min_periods=2).std())
    for window in [7, 28]:
        df[f"rolling_trend_{window}"] = shifted.groupby(df["unique_id"], observed=True).rolling(window, min_periods=2).apply(rolling_slope, raw=True).reset_index(level=0, drop=True)
    return df


def add_long_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("unique_id", observed=True)["demand"]
    for lag in [364, 365, 371]:
        df[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    df["rolling_mean_364"] = shifted.groupby(df["unique_id"], observed=True).rolling(364, min_periods=28).mean().reset_index(level=0, drop=True)
    df["previous_year_same_week_mean"] = df[["lag_364", "lag_365", "lag_371"]].mean(axis=1)
    df["previous_year_same_day_of_week"] = df["lag_364"]
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(df["date"])
    df["is_month_start"] = date.dt.is_month_start.astype("int8")
    df["is_month_end"] = date.dt.is_month_end.astype("int8")
    df["is_quarter_start"] = date.dt.is_quarter_start.astype("int8")
    df["is_quarter_end"] = date.dt.is_quarter_end.astype("int8")
    df["is_year_start"] = date.dt.is_year_start.astype("int8")
    df["is_year_end"] = date.dt.is_year_end.astype("int8")
    calendar = df[["d_order", "has_event"]].drop_duplicates("d_order").sort_values("d_order").copy()
    event_orders = calendar.loc[calendar["has_event"].astype(int).eq(1), "d_order"].to_numpy(dtype="int32")
    all_orders = calendar["d_order"].to_numpy(dtype="int32")
    if len(event_orders):
        next_idx = np.searchsorted(event_orders, all_orders, side="left")
        prev_idx = next_idx - 1
        calendar["days_to_next_event"] = np.where(next_idx < len(event_orders), event_orders[np.minimum(next_idx, len(event_orders) - 1)] - all_orders, np.nan)
        calendar["days_since_previous_event"] = np.where(prev_idx >= 0, all_orders - event_orders[np.maximum(prev_idx, 0)], np.nan)
    else:
        calendar["days_to_next_event"] = np.nan
        calendar["days_since_previous_event"] = np.nan
    df = df.merge(calendar[["d_order", "days_to_next_event", "days_since_previous_event"]], on="d_order", how="left", validate="many_to_one")
    df["event_week"] = (df["days_to_next_event"].le(3) | df["days_since_previous_event"].le(3) | df["has_event"].astype(int).eq(1)).astype("int8")
    df["pre_event_7_days"] = df["days_to_next_event"].between(1, 7).astype("int8")
    df["post_event_7_days"] = df["days_since_previous_event"].between(1, 7).astype("int8")
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    event = df.get("event_name_1", pd.Series("none", index=df.index)).astype("string").fillna("none")
    df["store_dept"] = df["store_id"].astype(str) + "_" + df["dept_id"].astype(str)
    df["store_state"] = df["store_id"].astype(str) + "_" + df["state_id"].astype(str)
    df["dept_category"] = df["dept_id"].astype(str) + "_" + df["cat_id"].astype(str)
    df["month_day_of_week"] = df["month"].astype(str) + "_" + df["day_of_week"].astype(str)
    df["event_store"] = event.astype(str) + "_" + df["store_id"].astype(str)
    df["snap_department"] = df["snap_active"].astype(str) + "_" + df["dept_id"].astype(str)
    return df


def _hierarchy_features(df: pd.DataFrame, keys: list[str], prefix: str, rolling: bool) -> pd.DataFrame:
    daily = df.groupby(keys + ["date"], observed=True, as_index=False)["demand"].sum().sort_values(keys + ["date"])
    grouped = daily.groupby(keys, observed=True)["demand"]
    daily[f"{prefix}_total_lag_7"] = grouped.shift(7)
    daily[f"{prefix}_total_lag_28"] = grouped.shift(28)
    if rolling:
        shifted = grouped.shift(1)
        daily[f"{prefix}_rolling_mean_28"] = shifted.groupby([daily[k] for k in keys], observed=True).rolling(28, min_periods=1).mean().reset_index(level=list(range(len(keys))), drop=True)
    keep = keys + ["date", f"{prefix}_total_lag_7", f"{prefix}_total_lag_28"]
    if rolling:
        keep.append(f"{prefix}_rolling_mean_28")
    return daily[keep]


def add_hierarchical_features(df: pd.DataFrame) -> pd.DataFrame:
    for keys, prefix, rolling in [(["store_id"], "store", True), (["dept_id"], "department", True), (["state_id"], "state", False), (["cat_id"], "category", False)]:
        df = df.merge(_hierarchy_features(df, keys, prefix, rolling), on=keys + ["date"], how="left", validate="many_to_one")
    df["series_share_of_store_28"] = np.divide(df["rolling_mean_28"], df["store_rolling_mean_28"], out=np.zeros(len(df), dtype="float64"), where=df["store_rolling_mean_28"].fillna(0).to_numpy() > 0)
    df["series_share_of_department_28"] = np.divide(df["rolling_mean_28"], df["department_rolling_mean_28"], out=np.zeros(len(df), dtype="float64"), where=df["department_rolling_mean_28"].fillna(0).to_numpy() > 0)
    state_roll = df.groupby(["state_id", "date"], observed=True)["rolling_mean_28"].transform("sum")
    df["series_share_of_state_28"] = np.divide(df["rolling_mean_28"], state_roll, out=np.zeros(len(df), dtype="float64"), where=state_roll.fillna(0).to_numpy() > 0)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("unique_id", observed=True)["mean_sell_price"]
    df["price_change_1_week"] = grouped.transform(lambda s: s - s.shift(7))
    df["price_change_4_weeks"] = grouped.transform(lambda s: s - s.shift(28))
    df["price_pct_change_1_week"] = grouped.pct_change(7)
    df["price_pct_change_4_weeks"] = grouped.pct_change(28)
    shifted = grouped.shift(1)
    df["price_rolling_mean_4_weeks"] = shifted.groupby(df["unique_id"], observed=True).rolling(28, min_periods=7).mean().reset_index(level=0, drop=True)
    df["price_rolling_std_4_weeks"] = shifted.groupby(df["unique_id"], observed=True).rolling(28, min_periods=7).std().reset_index(level=0, drop=True)
    df["price_volatility_12_weeks"] = shifted.groupby(df["unique_id"], observed=True).rolling(84, min_periods=14).std().reset_index(level=0, drop=True)
    dept_price = df.groupby(["dept_id", "date"], observed=True)["mean_sell_price"].transform("mean")
    store_price = df.groupby(["store_id", "date"], observed=True)["mean_sell_price"].transform("mean")
    df["price_relative_to_department"] = np.divide(df["mean_sell_price"], dept_price, out=np.ones(len(df), dtype="float64"), where=dept_price.fillna(0).to_numpy() > 0)
    df["price_relative_to_store"] = np.divide(df["mean_sell_price"], store_price, out=np.ones(len(df), dtype="float64"), where=store_price.fillna(0).to_numpy() > 0)
    df["discount_proxy"] = np.divide(df["mean_sell_price"] - df["price_rolling_mean_4_weeks"], df["price_rolling_mean_4_weeks"], out=np.zeros(len(df), dtype="float64"), where=df["price_rolling_mean_4_weeks"].fillna(0).to_numpy() > 0)
    return df


def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["store_id", "dept_id", "state_id", "cat_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "store_dept", "store_state", "dept_category", "month_day_of_week", "event_store", "snap_department"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).astype("float32")
    return df


def build_advanced_features() -> pd.DataFrame:
    if not BASE_FEATURES.exists():
        raise FileNotFoundError(f"Missing {BASE_FEATURES}. Run src/ml_feature_engineering.py first.")
    df = pd.read_parquet(BASE_FEATURES)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["unique_id", "date"]).reset_index(drop=True)
    for message, func in [("trend and smoothing", add_trend_and_smoothing), ("long seasonality", add_long_seasonality), ("calendar", add_calendar_features), ("interactions", add_interactions), ("hierarchies", add_hierarchical_features), ("price", add_price_features)]:
        print_step(f"Adding {message} features...")
        df = func(df)
    return optimize_types(df).sort_values(["unique_id", "date"]).reset_index(drop=True)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in df.astype(str).values.tolist())
    return "\n".join(lines)


def write_report(df: pd.DataFrame, elapsed: float, peak_mb: float) -> None:
    new_cols = [col for col in ADVANCED_FEATURE_NAMES if col in df.columns]
    nulls = df[new_cols].isna().sum().rename("null_count").reset_index().rename(columns={"index": "feature"})
    nulls["null_pct"] = (nulls["null_count"] / len(df) * 100).round(2)
    lines = ["# Advanced Feature Summary", "", f"Rows: `{len(df):,}`", f"Columns: `{df.shape[1]:,}`", f"New advanced feature columns: `{len(new_cols):,}`", f"Execution seconds: `{elapsed:.2f}`", f"Peak traced memory MB: `{peak_mb:.2f}`", "", "## Anti-Leakage Rules", "", "- Demand lags, medians, quantiles, EWM and rolling trend features are computed from `demand.shift(1)` by series.", "- Long seasonal features preserve nulls when history is unavailable.", "- Hierarchical demand features aggregate by hierarchy and date, then shift or roll shifted demand.", "- Event variables use the known calendar and no future observed demand.", "- Price variables are predictive proxies, not causal elasticity estimates.", "", "## New Feature Columns", "", markdown_table(pd.DataFrame({"feature": new_cols})), "", "## Nulls In Advanced Features", "", markdown_table(nulls.loc[nulls["null_count"] > 0].head(80)), "", "## Methodological Limitation", "", "The downstream Phase 7 training keeps only the latest 180 origin days per cutoff to control memory and runtime; this can underuse older seasonality."]
    ADVANCED_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    tracemalloc.start()
    df = build_advanced_features()
    df.to_parquet(ADVANCED_FEATURES, index=False)
    current, peak = tracemalloc.get_traced_memory()
    elapsed = time.perf_counter() - start
    metrics = {"execution_seconds": elapsed, "peak_traced_memory_mb": peak / 1024**2, "rows": int(df.shape[0]), "columns": int(df.shape[1]), "unique_series": int(df["unique_id"].nunique()), "new_advanced_features": [col for col in ADVANCED_FEATURE_NAMES if col in df.columns]}
    ADVANCED_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(df, elapsed, peak / 1024**2)
    tracemalloc.stop()
    print_step(f"Saved {ADVANCED_FEATURES.name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print_step(f"Completed in {elapsed:.2f}s, peak traced memory {metrics['peak_traced_memory_mb']:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
