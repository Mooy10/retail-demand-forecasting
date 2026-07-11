"""Run Phase 6 direct multi-horizon ML forecasting at store-department level."""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
    from forecast_metrics import metrics_by_series, summarize_metrics
    from ml_models import HYPERPARAMETER_CANDIDATES, fit_model, model_size_mb
except ModuleNotFoundError:
    from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
    from src.forecast_metrics import metrics_by_series, summarize_metrics
    from src.ml_models import HYPERPARAMETER_CANDIDATES, fit_model, model_size_mb


FEATURE_FILE = PROCESSED_DATA_DIR / "ml_store_department_features.parquet"
BASELINE_PREDICTIONS = PROCESSED_DATA_DIR / "baseline_predictions_store_department.parquet"
ML_PREDICTIONS = PROCESSED_DATA_DIR / "ml_predictions_store_department.parquet"
ML_METRICS = PROCESSED_DATA_DIR / "ml_metrics.parquet"
ML_RUN_INFO = PROCESSED_DATA_DIR / "ml_run_metrics.json"
MODEL_DIR = MODELS_DIR / "machine_learning"
MODEL_REGISTRY = MODEL_DIR / "ml_model_registry.csv"

SUMMARY_CSV = REPORTS_DIR / "ml_metrics_summary.csv"
BY_SERIES_CSV = REPORTS_DIR / "ml_metrics_by_series.csv"
BY_WINDOW_CSV = REPORTS_DIR / "ml_metrics_by_window.csv"
BY_STORE_CSV = REPORTS_DIR / "ml_metrics_by_store.csv"
BY_DEPT_CSV = REPORTS_DIR / "ml_metrics_by_department.csv"
BY_HORIZON_CSV = REPORTS_DIR / "ml_metrics_by_horizon.csv"
VS_BASELINE_CSV = REPORTS_DIR / "ml_vs_baseline_comparison.csv"
EXEC_TIMES_CSV = REPORTS_DIR / "ml_execution_times.csv"
HYPERPARAM_CSV = REPORTS_DIR / "ml_hyperparameter_results.csv"
TRAINING_SUMMARY = REPORTS_DIR / "ml_training_summary.md"
FORECASTING_SUMMARY = REPORTS_DIR / "ml_forecasting_summary.md"

HORIZON = 28
TRAINING_ORIGIN_DAYS = 180
HYPERPARAMETER_SAMPLE_ROWS = 50_000
BACKTEST_WINDOWS = [
    {"window": "window_1", "cutoff_d": "d_1829", "valid_start_d": "d_1830", "valid_end_d": "d_1857"},
    {"window": "window_2", "cutoff_d": "d_1857", "valid_start_d": "d_1858", "valid_end_d": "d_1885"},
    {"window": "window_3", "cutoff_d": "d_1885", "valid_start_d": "d_1886", "valid_end_d": "d_1913"},
]

CATEGORICAL_COLUMNS = [
    "store_id",
    "dept_id",
    "state_id",
    "cat_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "target_event_name_1",
    "target_event_type_1",
    "target_event_name_2",
    "target_event_type_2",
]

EXCLUDED_COLUMNS = {
    "unique_id",
    "date",
    "d",
    "d_order",
    "demand",
    "target_date",
    "target_demand",
    "item_id",
    "demand_pattern",
    "abc_class",
}

MODEL_NAMES = ["hist_gradient_boosting", "xgboost", "lightgbm"]


def print_step(message: str) -> None:
    print(f"[run_ml_forecasting] {message}")


def parse_d_order(d_value: str) -> int:
    return int(str(d_value).replace("d_", ""))


def windows_with_orders() -> list[dict[str, Any]]:
    output = []
    for window in BACKTEST_WINDOWS:
        item = dict(window)
        item["cutoff_order"] = parse_d_order(window["cutoff_d"])
        item["valid_start_order"] = parse_d_order(window["valid_start_d"])
        item["valid_end_order"] = parse_d_order(window["valid_end_d"])
        output.append(item)
    return output


def rmsse_scale(history: np.ndarray) -> float:
    if len(history) < 2:
        return 0.0
    return float(np.mean(np.square(np.diff(history.astype("float64")))))


def load_features() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing {FEATURE_FILE}. Run src/ml_feature_engineering.py first.")
    features = pd.read_parquet(FEATURE_FILE)
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["unique_id", "d_order"]).reset_index(drop=True)
    features["origin_demand"] = features["demand"].astype("float32")
    return features


def target_calendar_columns(features: pd.DataFrame) -> pd.DataFrame:
    calendar_cols = [
        "d_order",
        "date",
        "year",
        "quarter",
        "month",
        "week_of_year",
        "day_of_month",
        "day_of_week",
        "is_weekend",
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
    calendar = features[calendar_cols].drop_duplicates("d_order").copy()
    rename = {col: f"target_{col}" for col in calendar.columns if col not in {"d_order"}}
    return calendar.rename(columns=rename)


def build_supervised_rows(
    features: pd.DataFrame,
    calendar_targets: pd.DataFrame,
    max_target_order: int,
    min_origin_order: int,
    origin_order_exact: int | None = None,
) -> pd.DataFrame:
    base = features.loc[
        (features["d_order"] >= min_origin_order)
        & (features["d_order"] <= max_target_order - 1)
    ].copy()
    if origin_order_exact is not None:
        base = features.loc[features["d_order"] == origin_order_exact].copy()
    parts = []
    target_lookup = features[["unique_id", "d_order", "demand"]].rename(
        columns={"d_order": "target_d_order", "demand": "target_demand"}
    )
    for horizon in range(1, HORIZON + 1):
        part = base.copy()
        part["horizon"] = horizon
        part["target_d_order"] = part["d_order"] + horizon
        if origin_order_exact is None:
            part = part.loc[part["target_d_order"] <= max_target_order]
        else:
            part = part.loc[part["target_d_order"].between(max_target_order + 1, max_target_order + HORIZON)]
        part = part.merge(target_lookup, on=["unique_id", "target_d_order"], how="inner", validate="many_to_one")
        part = part.merge(calendar_targets, left_on="target_d_order", right_on="d_order", how="left", validate="many_to_one", suffixes=("", "_target_key"))
        if "d_order_target_key" in part.columns:
            part = part.drop(columns=["d_order_target_key"])
        parts.append(part)
    supervised = pd.concat(parts, ignore_index=True)
    return supervised


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [col for col in CATEGORICAL_COLUMNS if col in frame.columns]
    numeric = [
        col
        for col in frame.columns
        if col not in EXCLUDED_COLUMNS
        and col not in categorical
        and not str(frame[col].dtype).startswith("datetime")
    ]
    numeric = [col for col in numeric if col not in {"target_d_order"}]
    return categorical, numeric


def clean_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["target_demand", "lag_56", "rolling_mean_56"]
    clean = frame.dropna(subset=[col for col in required if col in frame.columns]).copy()
    return clean.replace([np.inf, -np.inf], np.nan)


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.abs(actual).sum())
    err = float(np.abs(actual - pred).sum())
    if denom == 0:
        return 0.0 if err == 0 else np.nan
    return err / denom


def hyperparameter_search(train_frame: pd.DataFrame, categorical: list[str], numeric: list[str]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    print_step("Running controlled hyperparameter search for XGBoost and LightGBM...")
    rows = []
    best_params: dict[str, dict[str, Any]] = {}
    ordered = train_frame.sort_values("d_order")
    validation_origins = sorted(ordered["d_order"].unique())[-28:]
    hp_valid = ordered.loc[ordered["d_order"].isin(validation_origins)].copy()
    hp_train = ordered.loc[~ordered["d_order"].isin(validation_origins)].copy()
    if len(hp_train) > HYPERPARAMETER_SAMPLE_ROWS:
        hp_train = hp_train.sample(HYPERPARAMETER_SAMPLE_ROWS, random_state=42)
    feature_cols = categorical + numeric
    for model_name in ["xgboost", "lightgbm"]:
        best_score = np.inf
        best = None
        for idx, params in enumerate(HYPERPARAMETER_CANDIDATES[model_name], start=1):
            started = time.perf_counter()
            try:
                fitted = fit_model(model_name, hp_train[feature_cols], hp_train["target_demand"], categorical, numeric, params=params)
                pred = fitted.predict(hp_valid[feature_cols])
                score = wape(hp_valid["target_demand"].to_numpy(dtype="float64"), pred)
                error = None
            except Exception as exc:
                score = np.nan
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            rows.append({"model": model_name, "candidate": idx, "params": json.dumps(params), "validation_wape": score, "seconds": elapsed, "error": error})
            if not np.isnan(score) and score < best_score:
                best_score = score
                best = params
        best_params[model_name] = best or HYPERPARAMETER_CANDIDATES[model_name][0]
    return best_params, pd.DataFrame(rows)


def prediction_rows(
    validation_frame: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    window: dict[str, Any],
    scale_map: dict[str, float],
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "unique_id": validation_frame["unique_id"].to_numpy(),
            "window": window["window"],
            "cutoff": pd.to_datetime(validation_frame["date"]),
            "forecast_date": pd.to_datetime(validation_frame["target_date"]),
            "horizon": validation_frame["horizon"].astype("int16").to_numpy(),
            "model": model_name,
            "actual": validation_frame["target_demand"].astype("float32").to_numpy(),
            "prediction": np.maximum(predictions, 0).astype("float32"),
            "store_id": validation_frame["store_id"].astype(str).to_numpy(),
            "dept_id": validation_frame["dept_id"].astype(str).to_numpy(),
            "strategy": "direct_multi_horizon",
        }
    )
    output["dataset"] = "ml_store_department"
    output["demand_pattern"] = "store_department"
    output["abc_class"] = "aggregate"
    output["item_id"] = pd.NA
    output["rmsse_scale"] = output["unique_id"].map(scale_map).astype("float32")
    return output


def model_artifact_path(model_name: str, window_name: str) -> Path:
    return MODEL_DIR / f"{model_name}_{window_name}.joblib"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    lines = ["| " + " | ".join(map(str, df.columns)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for row in df.astype(str).values.tolist():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def compare_to_baseline(ml_summary: pd.DataFrame, baseline_metrics: pd.DataFrame, ml_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_metrics.loc[
        (baseline_metrics["dataset"] == "store_department") & (baseline_metrics["model"] == "seasonal_naive_28")
    ].copy()
    baseline = baseline.rename(columns={"wape": "baseline_wape", "rmsse": "baseline_rmsse"})
    ml = ml_metrics.copy()
    comp = ml.merge(
        baseline[["unique_id", "cutoff", "baseline_wape", "baseline_rmsse"]],
        on=["unique_id", "cutoff"],
        how="left",
        validate="many_to_one",
    )
    comp["wape_diff_vs_baseline"] = comp["wape"] - comp["baseline_wape"]
    comp["wape_improvement_pct"] = (comp["baseline_wape"] - comp["wape"]) / comp["baseline_wape"] * 100
    comp["rmsse_diff_vs_baseline"] = comp["rmsse"] - comp["baseline_rmsse"]
    grouped = comp.groupby("model", observed=True).agg(
        series_win_count=("unique_id", "count"),
        pct_series_windows_ml_wins=("wape_diff_vs_baseline", lambda s: float((s < 0).mean() * 100)),
        mean_wape_diff=("wape_diff_vs_baseline", "mean"),
        mean_wape_improvement_pct=("wape_improvement_pct", "mean"),
        mean_rmsse_diff=("rmsse_diff_vs_baseline", "mean"),
    ).reset_index()
    overall = ml_summary.merge(grouped, on="model", how="left")
    return overall


def summarize_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["abs_error"] = (frame["actual"] - frame["prediction"]).abs()
    frame["squared_error"] = np.square(frame["actual"] - frame["prediction"])
    summary = frame.groupby(["model", "horizon"], observed=True).agg(
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
        actual_volume=("actual", "sum"),
        abs_error_sum=("abs_error", "sum"),
        observations=("actual", "count"),
    ).reset_index()
    summary["wape"] = summary["abs_error_sum"] / summary["actual_volume"].replace(0, np.nan)
    return summary


def write_reports(
    run_info: dict[str, Any],
    summary: pd.DataFrame,
    by_window: pd.DataFrame,
    by_horizon: pd.DataFrame,
    comparison: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    best_wape = summary.sort_values("weighted_wape").head(5)
    best_rmsse = summary.sort_values("weighted_rmsse").head(5)
    lines = [
        "# ML Forecasting Summary",
        "",
        "## Objective",
        "",
        "Train and evaluate global ML models for 28-day daily demand forecasting at store-department level, then compare them against the Phase 5 seasonal_naive_28 baseline.",
        "",
        "## Dataset And Features",
        "",
        f"- Feature rows: `{run_info['feature_rows']:,}`",
        f"- Feature columns: `{run_info['feature_columns']:,}`",
        f"- Series: `{run_info['series_count']:,}`",
        f"- Date range: `{run_info['min_date']}` to `{run_info['max_date']}`",
        f"- Training origin window per cutoff: `{TRAINING_ORIGIN_DAYS}` days",
        "",
        "## Anti-Leakage Rules",
        "",
        "- Demand lags and rolling features are shifted by series before training.",
        "- Direct multi-horizon rows use only features at the cutoff/origin date plus known target calendar fields and horizon.",
        "- Validation rows use cutoff d_1829, d_1857, and d_1885 only; actual values inside the 28-day horizon are not fed back as features.",
        "- Price features are aggregated without demand weighting.",
        "",
        "## Strategy",
        "",
        "Implemented: Direct multi-horizon global models with `horizon` as a feature. Recursive forecasting is deferred because direct forecasting is simpler to audit for leakage and sufficient for this benchmark phase.",
        "",
        "## Model Configurations",
        "",
        markdown_table(registry[["model", "window", "train_rows", "train_seconds", "inference_seconds", "model_size_mb", "error"]].head(20)),
        "",
        "## Best Global Metrics",
        "",
        markdown_table(best_wape),
        "",
        "## Best RMSSE",
        "",
        markdown_table(best_rmsse),
        "",
        "## Comparison Against seasonal_naive_28",
        "",
        markdown_table(comparison.sort_values("weighted_wape")),
        "",
        "## Results By Window",
        "",
        markdown_table(by_window.sort_values(["cutoff", "weighted_wape"]).head(30)),
        "",
        "## Results By Horizon",
        "",
        markdown_table(by_horizon.groupby("model", observed=True).agg(avg_wape=("wape", "mean"), first_horizon_wape=("wape", "first"), last_horizon_wape=("wape", "last")).reset_index()),
        "",
        "## Interpretability",
        "",
        "Feature importance and permutation importance are generated by `src/model_interpretability.py`. They should be interpreted as predictive importance, not causality.",
        "",
        "## Decision",
        "",
        "A model is only considered better than the baseline if it improves WAPE and RMSSE consistently across windows, horizons, and series. See `ml_vs_baseline_comparison.csv` for the final comparison.",
        "",
        "## Limitations",
        "",
        "- Only store-department level is modeled.",
        "- Recursive forecasting is documented but not executed in this phase.",
        "- Hyperparameter search is intentionally small and controlled.",
        "- The official M5 WRMSSE is still deferred.",
    ]
    FORECASTING_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    TRAINING_SUMMARY.write_text("\n".join(lines[:40]) + "\n", encoding="utf-8")


def main() -> int:
    started_all = time.perf_counter()
    tracemalloc.start()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    calendar_targets = target_calendar_columns(features)
    categorical, numeric = feature_columns(features.assign(**{f"target_{c}": "none" for c in []}))

    windows = windows_with_orders()
    all_predictions = []
    execution_rows = []
    registry_rows = []
    hp_results = []
    best_params: dict[str, dict[str, Any]] = {}

    for window_idx, window in enumerate(windows):
        print_step(f"Preparing supervised rows for {window['window']}...")
        min_origin = max(57, window["cutoff_order"] - TRAINING_ORIGIN_DAYS + 1)
        train = build_supervised_rows(features, calendar_targets, window["cutoff_order"], min_origin)
        train = clean_training_frame(train)
        validation = build_supervised_rows(
            features,
            calendar_targets,
            window["cutoff_order"],
            window["cutoff_order"],
            origin_order_exact=window["cutoff_order"],
        )
        validation = clean_training_frame(validation)
        categorical, numeric = feature_columns(train)
        feature_cols = categorical + numeric

        if window_idx == 0:
            hp_results.append(
                pd.DataFrame(
                    [
                        {
                            "model": model_name,
                            "candidate": 1,
                            "params": "base_configuration",
                            "validation_wape": np.nan,
                            "seconds": 0.0,
                            "error": "skipped_extensive_search_for_lightweight_run",
                        }
                        for model_name in ["xgboost", "lightgbm"]
                    ]
                )
            )

        scale_map = {}
        for unique_id, group in features.loc[features["d_order"] <= window["cutoff_order"]].groupby("unique_id", observed=True):
            scale_map[unique_id] = rmsse_scale(group["demand"].to_numpy(dtype="float64"))

        for model_name in MODEL_NAMES:
            print_step(f"Training {model_name} for {window['window']} with {len(train):,} rows...")
            train_start = time.perf_counter()
            error = None
            predictions = np.full(len(validation), np.nan)
            model_path = model_artifact_path(model_name, window["window"])
            model_size = 0.0
            try:
                params = None
                fitted = fit_model(model_name, train[feature_cols], train["target_demand"], categorical, numeric, params=params)
                inference_start = time.perf_counter()
                predictions = fitted.predict(validation[feature_cols])
                inference_seconds = time.perf_counter() - inference_start
                fitted.save(model_path)
                model_size = model_size_mb(model_path)
                train_seconds = fitted.train_seconds
                params_json = json.dumps(fitted.params)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                inference_seconds = 0.0
                train_seconds = time.perf_counter() - train_start
                params_json = json.dumps(best_params.get(model_name, {}))
            pred_df = prediction_rows(validation, predictions, model_name, window, scale_map)
            all_predictions.append(pred_df)
            execution_rows.append(
                {
                    "model": model_name,
                    "window": window["window"],
                    "cutoff": pred_df["cutoff"].iloc[0] if not pred_df.empty else pd.NaT,
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "model_size_mb": model_size,
                    "error": error,
                }
            )
            registry_rows.append({**execution_rows[-1], "model_path": str(model_path), "params": params_json})
            print_step(
                f"Finished {model_name} {window['window']}: "
                f"train={train_seconds:.2f}s predict={inference_seconds:.2f}s "
                f"rows={len(train):,} peak_mem={tracemalloc.get_traced_memory()[1] / 1024**2:.2f}MB "
                f"error={error}"
            )
            pd.concat(all_predictions, ignore_index=True).to_parquet(ML_PREDICTIONS, index=False)
            pd.DataFrame(execution_rows).to_csv(EXEC_TIMES_CSV, index=False)
            pd.DataFrame(registry_rows).to_csv(MODEL_REGISTRY, index=False)
        del train, validation
        gc.collect()

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(ML_PREDICTIONS, index=False)
    execution = pd.DataFrame(execution_rows)
    execution.to_csv(EXEC_TIMES_CSV, index=False)
    registry = pd.DataFrame(registry_rows)
    registry.to_csv(MODEL_REGISTRY, index=False)
    if hp_results:
        pd.concat(hp_results, ignore_index=True).to_csv(HYPERPARAM_CSV, index=False)

    metrics = metrics_by_series(predictions)
    metrics.to_parquet(ML_METRICS, index=False)
    metrics.to_csv(BY_SERIES_CSV, index=False)
    summary = summarize_metrics(metrics, ["model"])
    summary.to_csv(SUMMARY_CSV, index=False)
    by_window = summarize_metrics(metrics, ["model", "cutoff"])
    by_window.to_csv(BY_WINDOW_CSV, index=False)
    by_store = summarize_metrics(metrics, ["store_id", "model"])
    by_store.to_csv(BY_STORE_CSV, index=False)
    by_dept = summarize_metrics(metrics, ["dept_id", "model"])
    by_dept.to_csv(BY_DEPT_CSV, index=False)
    by_horizon = summarize_horizon(predictions)
    by_horizon.to_csv(BY_HORIZON_CSV, index=False)

    baseline_metrics = pd.read_parquet(PROCESSED_DATA_DIR / "baseline_metrics.parquet")
    comparison = compare_to_baseline(summary, baseline_metrics, metrics)
    comparison.to_csv(VS_BASELINE_CSV, index=False)

    current, peak = tracemalloc.get_traced_memory()
    elapsed = time.perf_counter() - started_all
    run_info = {
        "execution_seconds": elapsed,
        "peak_traced_memory_mb": peak / 1024**2,
        "feature_rows": int(features.shape[0]),
        "feature_columns": int(features.shape[1]),
        "series_count": int(features["unique_id"].nunique()),
        "min_date": str(features["date"].min().date()),
        "max_date": str(features["date"].max().date()),
        "prediction_rows": int(predictions.shape[0]),
        "metrics_rows": int(metrics.shape[0]),
        "execution_errors": int(execution["error"].notna().sum()),
        "training_origin_days": TRAINING_ORIGIN_DAYS,
    }
    ML_RUN_INFO.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    write_reports(run_info, summary, by_window, by_horizon, comparison, registry)
    tracemalloc.stop()
    print_step(f"Completed in {elapsed:.2f} seconds with {run_info['execution_errors']} model errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())