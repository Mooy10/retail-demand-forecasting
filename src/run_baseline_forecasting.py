"""Run Phase 5 baseline forecasting backtests."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_models import BASELINE_MODELS
from config import PROCESSED_DATA_DIR, REPORTS_DIR
from forecast_metrics import metrics_by_series, summarize_metrics


STORE_DEPT_DATASET = PROCESSED_DATA_DIR / "forecast_store_department.parquet"
SELECTED_SERIES_DATASET = PROCESSED_DATA_DIR / "forecast_selected_series.parquet"

STORE_DEPT_PREDICTIONS = PROCESSED_DATA_DIR / "baseline_predictions_store_department.parquet"
SELECTED_SERIES_PREDICTIONS = PROCESSED_DATA_DIR / "baseline_predictions_selected_series.parquet"
BASELINE_METRICS_PARQUET = PROCESSED_DATA_DIR / "baseline_metrics.parquet"
BASELINE_RUN_METRICS_JSON = PROCESSED_DATA_DIR / "baseline_run_metrics.json"

METRICS_BY_SERIES_CSV = REPORTS_DIR / "baseline_metrics_by_series.csv"
METRICS_SUMMARY_CSV = REPORTS_DIR / "baseline_metrics_summary.csv"
METRICS_BY_PATTERN_CSV = REPORTS_DIR / "baseline_metrics_by_pattern.csv"
METRICS_BY_STORE_DEPT_CSV = REPORTS_DIR / "baseline_metrics_by_store_department.csv"
EXECUTION_TIMES_CSV = REPORTS_DIR / "baseline_execution_times.csv"
SUMMARY_REPORT = REPORTS_DIR / "baseline_forecasting_summary.md"

HORIZON = 28
BACKTEST_WINDOWS = [
    {"window": "window_1", "cutoff_d": "d_1829", "valid_start_d": "d_1830", "valid_end_d": "d_1857"},
    {"window": "window_2", "cutoff_d": "d_1857", "valid_start_d": "d_1858", "valid_end_d": "d_1885"},
    {"window": "window_3", "cutoff_d": "d_1885", "valid_start_d": "d_1886", "valid_end_d": "d_1913"},
]


def print_step(message: str) -> None:
    print(f"[run_baseline_forecasting] {message}")


def parse_d_order(d_value: str) -> int:
    return int(str(d_value).replace("d_", ""))


def add_window_orders() -> list[dict[str, object]]:
    windows = []
    for window in BACKTEST_WINDOWS:
        item = dict(window)
        item["cutoff_order"] = parse_d_order(window["cutoff_d"])
        item["valid_start_order"] = parse_d_order(window["valid_start_d"])
        item["valid_end_order"] = parse_d_order(window["valid_end_d"])
        windows.append(item)
    return windows


def rmsse_scale(history: np.ndarray) -> float:
    if len(history) < 2:
        return 0.0
    diffs = np.diff(history.astype("float64"))
    return float(np.mean(np.square(diffs)))


def run_model(model_name: str, y_train: np.ndarray, train_dates, future_dates) -> np.ndarray:
    model = BASELINE_MODELS[model_name]
    if model_name == "seasonal_average_weekday":
        return model(y_train, history_dates=train_dates, future_dates=future_dates, horizon=HORIZON)
    return model(y_train, horizon=HORIZON)


def metadata_for_group(group: pd.DataFrame) -> dict[str, object]:
    first = group.iloc[0]
    return {
        "store_id": first.get("store_id", pd.NA),
        "dept_id": first.get("dept_id", pd.NA),
        "item_id": first.get("item_id", pd.NA),
        "demand_pattern": first.get("demand_pattern", pd.NA),
        "abc_class": first.get("abc_class", pd.NA),
    }


def backtest_dataset(data: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    timing_rows = []
    windows = add_window_orders()
    data = data.sort_values(["unique_id", "d_order"]).reset_index(drop=True)

    for unique_id, group in data.groupby("unique_id", observed=True, sort=False):
        group = group.sort_values("d_order")
        meta = metadata_for_group(group)
        for window in windows:
            train = group.loc[group["d_order"] <= window["cutoff_order"]]
            valid = group.loc[
                (group["d_order"] >= window["valid_start_order"])
                & (group["d_order"] <= window["valid_end_order"])
            ]
            if len(valid) != HORIZON:
                raise ValueError(f"{dataset_name}:{unique_id}:{window['window']} does not have {HORIZON} validation rows")

            y_train = train["demand"].to_numpy(dtype="float64")
            train_dates = pd.to_datetime(train["date"])
            future_dates = pd.to_datetime(valid["date"])
            scale = rmsse_scale(y_train)
            cutoff_date = pd.to_datetime(train["date"].max())

            for model_name in BASELINE_MODELS:
                started = time.perf_counter()
                error_message = None
                try:
                    pred = run_model(model_name, y_train, train_dates, future_dates)
                    pred = np.maximum(np.asarray(pred, dtype="float64"), 0.0)
                    if len(pred) != HORIZON:
                        raise ValueError(f"Model returned {len(pred)} predictions instead of {HORIZON}")
                except Exception as exc:  # Keep full backtest running if one model fails.
                    error_message = f"{type(exc).__name__}: {exc}"
                    pred = np.full(HORIZON, np.nan, dtype="float64")
                elapsed = time.perf_counter() - started
                timing_rows.append(
                    {
                        "dataset": dataset_name,
                        "unique_id": unique_id,
                        "window": window["window"],
                        "cutoff": cutoff_date,
                        "model": model_name,
                        "execution_seconds": elapsed,
                        "error": error_message,
                    }
                )

                for horizon, (_, actual_row) in enumerate(valid.iterrows(), start=1):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "unique_id": unique_id,
                            "window": window["window"],
                            "cutoff": cutoff_date,
                            "forecast_date": pd.to_datetime(actual_row["date"]),
                            "horizon": horizon,
                            "model": model_name,
                            "actual": float(actual_row["demand"]),
                            "prediction": float(pred[horizon - 1]),
                            "rmsse_scale": scale,
                            **meta,
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(timing_rows)



def markdown_table(df: pd.DataFrame) -> str:
    """Render small DataFrames as Markdown without optional dependencies."""
    if df.empty:
        return "No rows available."
    frame = df.copy()
    headers = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in frame.astype(str).values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)

def write_report(
    summary: pd.DataFrame,
    by_pattern: pd.DataFrame,
    by_store_dept: pd.DataFrame,
    execution_times: pd.DataFrame,
    metrics: pd.DataFrame,
    run_info: dict[str, object],
) -> None:
    global_store = summary.loc[summary["dataset"] == "store_department"].sort_values("weighted_wape").head(10)
    global_selected = summary.loc[summary["dataset"] == "selected_series"].sort_values("weighted_wape").head(10)
    pattern_view = by_pattern.sort_values(["demand_pattern", "weighted_wape"]).head(20)
    store_view = by_store_dept.sort_values("weighted_wape").head(20)
    timing = execution_times.groupby("model", observed=True)["execution_seconds"].sum().sort_values().reset_index()
    stability = summarize_metrics(metrics, ["dataset", "model", "cutoff"]).sort_values(["dataset", "model", "cutoff"])

    lines = [
        "# Baseline Forecasting Summary",
        "",
        "## Scope",
        "",
        "This phase evaluates baseline forecasting models with temporal backtesting only. It does not train machine learning models and does not use `sales_train_evaluation.csv`.",
        "",
        "## Modeling Levels",
        "",
        "- Store-department level: 70 complete daily series from 10 stores x 7 departments.",
        "- Controlled SKU-store sample: up to 25 class-A series per demand pattern, prioritized by total demand.",
        "",
        "## Backtesting Windows",
        "",
        "- Window 1: train through d_1829, validate d_1830 to d_1857.",
        "- Window 2: train through d_1857, validate d_1858 to d_1885.",
        "- Window 3: train through d_1885, validate d_1886 to d_1913.",
        "",
        "## Baselines",
        "",
        "Naive, Seasonal Naive 7, Seasonal Naive 28, Historical Mean, Moving Average 7, Moving Average 28, Seasonal Average by Weekday, Croston Classic, Croston SBA, and TSB.",
        "",
        "Croston, SBA, and TSB use alpha=0.1 by default; TSB also uses beta=0.1. Fully-zero series return zero forecasts and predictions are clipped at zero.",
        "",
        "## Global Metrics - Store Department",
        "",
        markdown_table(global_store),
        "",
        "## Global Metrics - Selected SKU Store Series",
        "",
        markdown_table(global_selected),
        "",
        "## Results By Demand Pattern",
        "",
        markdown_table(pattern_view),
        "",
        "## Best Store-Department Combinations By WAPE",
        "",
        markdown_table(store_view),
        "",
        "## Execution Time By Model",
        "",
        markdown_table(timing),
        "",
        "## Stability Across Windows",
        "",
        markdown_table(stability[["dataset", "model", "cutoff", "weighted_wape", "weighted_rmsse", "weighted_mae"]].head(60)),
        "",
        "## Error Analysis",
        "",
        "- WAPE is prioritized over MAPE because many actual values are zero.",
        "- RMSSE uses the mean squared first difference of each training history as the scale.",
        "- When the RMSSE scale is zero, RMSSE is set to zero only for perfect forecasts; otherwise it is left missing to avoid infinite values.",
        "- Mean and median metrics are both reported because intermittent series can produce skewed error distributions.",
        "",
        "## Implications For ML",
        "",
        "- Store-department baselines define a stable aggregate benchmark before item-level ML.",
        "- The selected class-A SKU-store sample highlights where intermittent methods help or fail by demand pattern.",
        "- ML in the next phase should be evaluated against these baselines by pattern, not only globally.",
        "",
        "## Limitations",
        "",
        "- This is not the official hierarchical WRMSSE M5 evaluation.",
        "- Hyperparameters for intermittent baselines are fixed defaults, not optimized.",
        "- The SKU-store experiment is a controlled high-volume sample, not the full 30,490 series.",
        "",
        "## Run Information",
        "",
        f"- Total execution seconds: `{run_info['execution_seconds']:.2f}`",
        f"- Store-department prediction rows: `{run_info['store_department_prediction_rows']:,}`",
        f"- Selected-series prediction rows: `{run_info['selected_series_prediction_rows']:,}`",
    ]
    SUMMARY_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print_step("Loading compact forecasting datasets...")
    store_dept = pd.read_parquet(STORE_DEPT_DATASET)
    selected_series = pd.read_parquet(SELECTED_SERIES_DATASET)

    print_step("Backtesting store-department baselines...")
    store_predictions, store_times = backtest_dataset(store_dept, "store_department")
    print_step("Backtesting selected SKU-store baselines...")
    selected_predictions, selected_times = backtest_dataset(selected_series, "selected_series")

    store_predictions.to_parquet(STORE_DEPT_PREDICTIONS, index=False)
    selected_predictions.to_parquet(SELECTED_SERIES_PREDICTIONS, index=False)

    all_predictions = pd.concat([store_predictions, selected_predictions], ignore_index=True)
    execution_times = pd.concat([store_times, selected_times], ignore_index=True)
    execution_times.to_csv(EXECUTION_TIMES_CSV, index=False)

    print_step("Calculating metrics...")
    metrics = metrics_by_series(all_predictions)
    metrics.to_parquet(BASELINE_METRICS_PARQUET, index=False)
    metrics.to_csv(METRICS_BY_SERIES_CSV, index=False)

    summary = summarize_metrics(metrics, ["dataset", "model"])
    summary.to_csv(METRICS_SUMMARY_CSV, index=False)

    by_pattern = summarize_metrics(metrics.loc[metrics["dataset"] == "selected_series"], ["demand_pattern", "model"])
    by_pattern.to_csv(METRICS_BY_PATTERN_CSV, index=False)

    by_store_dept = summarize_metrics(metrics.loc[metrics["dataset"] == "store_department"], ["store_id", "dept_id", "model"])
    by_store_dept.to_csv(METRICS_BY_STORE_DEPT_CSV, index=False)

    elapsed = time.perf_counter() - start
    run_info = {
        "execution_seconds": elapsed,
        "store_department_prediction_rows": int(store_predictions.shape[0]),
        "selected_series_prediction_rows": int(selected_predictions.shape[0]),
        "metrics_rows": int(metrics.shape[0]),
        "model_count": len(BASELINE_MODELS),
        "window_count": len(BACKTEST_WINDOWS),
        "store_department_series": int(store_dept["unique_id"].nunique()),
        "selected_series": int(selected_series["unique_id"].nunique()),
        "execution_errors": int(execution_times["error"].notna().sum()),
    }
    BASELINE_RUN_METRICS_JSON.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    write_report(summary, by_pattern, by_store_dept, execution_times, metrics, run_info)
    print_step(f"Saved predictions and metrics. Completed in {elapsed:.2f} seconds")
    if run_info["execution_errors"]:
        print_step(f"Execution errors recorded: {run_info['execution_errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())