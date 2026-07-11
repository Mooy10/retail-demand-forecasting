"""Run Phase 7 advanced ML forecasting with XGBoost and LightGBM only."""

from __future__ import annotations

import gc
import json
import os
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

try:
    from config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
    from forecast_metrics import metrics_by_series, summarize_metrics
    from ml_models import fit_model, model_size_mb
except ModuleNotFoundError:
    from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
    from src.forecast_metrics import metrics_by_series, summarize_metrics
    from src.ml_models import fit_model, model_size_mb

FEATURE_FILE = PROCESSED_DATA_DIR / "ml_store_department_advanced_features.parquet"
PREDICTIONS_FILE = PROCESSED_DATA_DIR / "advanced_ml_predictions.parquet"
METRICS_FILE = PROCESSED_DATA_DIR / "advanced_ml_metrics.parquet"
RUN_INFO_FILE = PROCESSED_DATA_DIR / "advanced_ml_run_metrics.json"
MODEL_DIR = MODELS_DIR / "machine_learning" / "phase7"
MODEL_REGISTRY = MODEL_DIR / "advanced_ml_model_registry.csv"
SUMMARY_CSV = REPORTS_DIR / "advanced_ml_metrics_summary.csv"
BY_SERIES_CSV = REPORTS_DIR / "advanced_ml_by_series.csv"
BY_WINDOW_CSV = REPORTS_DIR / "advanced_ml_by_window.csv"
BY_STORE_CSV = REPORTS_DIR / "advanced_ml_by_store.csv"
BY_DEPT_CSV = REPORTS_DIR / "advanced_ml_by_department.csv"
BY_HORIZON_CSV = REPORTS_DIR / "advanced_ml_by_horizon.csv"
VS_PREVIOUS_CSV = REPORTS_DIR / "advanced_ml_vs_previous.csv"
EXEC_TIMES_CSV = REPORTS_DIR / "advanced_ml_execution_times.csv"
TRAINING_SUMMARY = REPORTS_DIR / "advanced_ml_training_summary.md"

HORIZON = 28
TRAINING_ORIGIN_DAYS = 180
BACKTEST_WINDOWS = [
    {"window": "window_1", "cutoff_d": "d_1829", "valid_start_d": "d_1830", "valid_end_d": "d_1857"},
    {"window": "window_2", "cutoff_d": "d_1857", "valid_start_d": "d_1858", "valid_end_d": "d_1885"},
    {"window": "window_3", "cutoff_d": "d_1885", "valid_start_d": "d_1886", "valid_end_d": "d_1913"},
]
MODEL_SPECS = {
    "xgboost_phase7": ("xgboost", {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.055, "n_jobs": 2}),
    "lightgbm_phase7": ("lightgbm", {"n_estimators": 60, "num_leaves": 31, "learning_rate": 0.045, "n_jobs": 2, "verbosity": -1}),
}
CATEGORICAL_COLUMNS = [
    "store_id", "dept_id", "state_id", "cat_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "store_dept", "store_state", "dept_category", "month_day_of_week", "event_store", "snap_department",
    "target_event_name_1", "target_event_type_1", "target_event_name_2", "target_event_type_2", "horizon_day_of_week", "horizon_month",
]
EXCLUDED_COLUMNS = {"unique_id", "date", "d", "d_order", "demand", "target_date", "target_demand", "item_id", "demand_pattern", "abc_class"}


def print_step(message: str) -> None:
    print(f"[run_advanced_ml_forecasting] {message}")


def parse_d_order(value: str) -> int:
    return int(str(value).replace("d_", ""))


def windows_with_orders() -> list[dict[str, Any]]:
    out = []
    for window in BACKTEST_WINDOWS:
        item = dict(window)
        item["cutoff_order"] = parse_d_order(window["cutoff_d"])
        item["valid_start_order"] = parse_d_order(window["valid_start_d"])
        item["valid_end_order"] = parse_d_order(window["valid_end_d"])
        out.append(item)
    return out


def rmsse_scale(history: np.ndarray) -> float:
    if len(history) < 2:
        return 0.0
    return float(np.mean(np.square(np.diff(history.astype("float64")))))


def load_features() -> pd.DataFrame:
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(f"Missing {FEATURE_FILE}. Run src/advanced_feature_engineering.py first.")
    df = pd.read_parquet(FEATURE_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["unique_id", "d_order"]).reset_index(drop=True)
    df["origin_demand"] = df["demand"].astype("float32")
    return df


def target_calendar_columns(features: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["d_order", "date", "year", "quarter", "month", "week_of_year", "day_of_month", "day_of_week", "is_weekend", "sin_day_of_week", "cos_day_of_week", "sin_month", "cos_month", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "has_event", "snap_CA", "snap_TX", "snap_WI", "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end", "days_to_next_event", "days_since_previous_event", "event_week", "pre_event_7_days", "post_event_7_days"]
    cols = [c for c in base_cols if c in features.columns]
    cal = features[cols].drop_duplicates("d_order").copy()
    return cal.rename(columns={c: f"target_{c}" for c in cal.columns if c != "d_order"})


def build_supervised_rows(features: pd.DataFrame, calendar_targets: pd.DataFrame, max_target_order: int, min_origin_order: int, origin_order_exact: int | None = None) -> pd.DataFrame:
    if origin_order_exact is None:
        base = features.loc[(features["d_order"] >= min_origin_order) & (features["d_order"] <= max_target_order - 1)].copy()
    else:
        base = features.loc[features["d_order"] == origin_order_exact].copy()
    target_lookup = features[["unique_id", "d_order", "demand"]].rename(columns={"d_order": "target_d_order", "demand": "target_demand"})
    parts = []
    for horizon in range(1, HORIZON + 1):
        part = base.copy()
        part["horizon"] = np.int16(horizon)
        part["target_d_order"] = part["d_order"] + horizon
        if origin_order_exact is None:
            part = part.loc[part["target_d_order"] <= max_target_order]
        else:
            part = part.loc[part["target_d_order"].between(max_target_order + 1, max_target_order + HORIZON)]
        part = part.merge(target_lookup, on=["unique_id", "target_d_order"], how="inner", validate="many_to_one")
        part = part.merge(calendar_targets, left_on="target_d_order", right_on="d_order", how="left", validate="many_to_one", suffixes=("", "_target_key"))
        if "d_order_target_key" in part.columns:
            part = part.drop(columns=["d_order_target_key"])
        part["horizon_day_of_week"] = part["horizon"].astype(str) + "_" + part["target_day_of_week"].astype(str)
        part["horizon_month"] = part["horizon"].astype(str) + "_" + part["target_month"].astype(str)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [col for col in CATEGORICAL_COLUMNS if col in frame.columns]
    numeric = [col for col in frame.columns if col not in EXCLUDED_COLUMNS and col not in categorical and col != "target_d_order" and not str(frame[col].dtype).startswith("datetime")]
    return categorical, numeric


def clean_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["target_demand", "lag_56", "rolling_mean_56"]
    clean = frame.dropna(subset=[col for col in required if col in frame.columns]).copy()
    return clean.replace([np.inf, -np.inf], np.nan)


def prediction_rows(validation_frame: pd.DataFrame, pred: np.ndarray, public_model_name: str, window: dict[str, Any], scale_map: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame({
        "unique_id": validation_frame["unique_id"].astype(str).to_numpy(), "window": window["window"], "cutoff": pd.to_datetime(validation_frame["date"]), "forecast_date": pd.to_datetime(validation_frame["target_date"]), "horizon": validation_frame["horizon"].astype("int16").to_numpy(), "model": public_model_name, "actual": validation_frame["target_demand"].astype("float32").to_numpy(), "prediction": np.maximum(np.asarray(pred, dtype="float64"), 0).astype("float32"), "store_id": validation_frame["store_id"].astype(str).to_numpy(), "dept_id": validation_frame["dept_id"].astype(str).to_numpy(), "strategy": "advanced_direct_multi_horizon",
    })
    out["dataset"] = "advanced_ml_store_department"
    out["demand_pattern"] = "store_department"
    out["abc_class"] = "aggregate"
    out["item_id"] = pd.NA
    out["rmsse_scale"] = out["unique_id"].map(scale_map).astype("float32")
    return out


def summarize_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["abs_error"] = (frame["actual"] - frame["prediction"]).abs()
    frame["squared_error"] = np.square(frame["actual"] - frame["prediction"])
    summary = frame.groupby(["model", "horizon"], observed=True).agg(mae=("abs_error", "mean"), rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))), actual_volume=("actual", "sum"), abs_error_sum=("abs_error", "sum"), observations=("actual", "count")).reset_index()
    summary["wape"] = summary["abs_error_sum"] / summary["actual_volume"].replace(0, np.nan)
    return summary


def combine_previous_metrics(advanced_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = pd.read_parquet(PROCESSED_DATA_DIR / "baseline_metrics.parquet")
    rows.append(baseline.loc[(baseline["dataset"] == "store_department") & (baseline["model"].isin(["seasonal_naive_28", "seasonal_average_weekday"]))].copy())
    phase6 = pd.read_parquet(PROCESSED_DATA_DIR / "ml_metrics.parquet")
    phase6 = phase6.loc[phase6["model"] == "xgboost"].copy()
    phase6["model"] = "xgboost_phase6"
    rows.append(phase6)
    rows.append(advanced_metrics.copy())
    return summarize_metrics(pd.concat(rows, ignore_index=True, sort=False), ["model"])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in df.astype(str).values.tolist())
    return "\n".join(lines)


def write_reports(run_info: dict[str, Any], summary: pd.DataFrame, by_window: pd.DataFrame, vs_previous: pd.DataFrame, execution: pd.DataFrame) -> None:
    lines = ["# Advanced ML Training Summary", "", f"Execution seconds: `{run_info['execution_seconds']:.2f}`", f"Peak traced memory MB: `{run_info['peak_traced_memory_mb']:.2f}`", f"Feature rows: `{run_info['feature_rows']:,}`", f"Feature columns: `{run_info['feature_columns']:,}`", f"Prediction rows: `{run_info['prediction_rows']:,}`", f"Training origin days per cutoff: `{TRAINING_ORIGIN_DAYS}`", "", "## Controlled Setup", "", "- Models: XGBoost Phase 7 and LightGBM Phase 7 only.", "- No extensive hyperparameter search; one reasonable configuration per model.", "- Direct multi-horizon strategy, 3 windows, 28-day horizon, 70 store-department series.", "- Environment thread limits: n_jobs=2, OMP/OPENBLAS/MKL=1, LOKY_MAX_CPU_COUNT=2.", "", "## Execution Times", "", markdown_table(execution), "", "## Phase 7 Metrics", "", markdown_table(summary.sort_values("weighted_wape")), "", "## Comparison With Baseline And Phase 6", "", markdown_table(vs_previous.sort_values("weighted_wape")), "", "## Metrics By Window", "", markdown_table(by_window.sort_values(["cutoff", "weighted_wape"]))]
    TRAINING_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start_all = time.perf_counter()
    tracemalloc.start()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    calendar_targets = target_calendar_columns(features)
    all_predictions = []
    execution_rows = []
    registry_rows = []
    for window in windows_with_orders():
        print_step(f"Preparing {window['window']} supervised rows...")
        min_origin = max(57, window["cutoff_order"] - TRAINING_ORIGIN_DAYS + 1)
        train = clean_training_frame(build_supervised_rows(features, calendar_targets, window["cutoff_order"], min_origin))
        validation = clean_training_frame(build_supervised_rows(features, calendar_targets, window["cutoff_order"], window["cutoff_order"], origin_order_exact=window["cutoff_order"]))
        categorical, numeric = feature_columns(train)
        feature_cols = categorical + numeric
        scale_map = {uid: rmsse_scale(g["demand"].to_numpy(dtype="float64")) for uid, g in features.loc[features["d_order"] <= window["cutoff_order"]].groupby("unique_id", observed=True)}
        for public_name, (fit_name, params) in MODEL_SPECS.items():
            print_step(f"Training {public_name} {window['window']} with {len(train):,} rows and {len(feature_cols):,} features...")
            train_start = time.perf_counter()
            error = None
            model_size = 0.0
            inference_seconds = 0.0
            predictions = np.full(len(validation), np.nan)
            model_path = MODEL_DIR / f"{public_name}_{window['window']}.joblib"
            try:
                fitted = fit_model(fit_name, train[feature_cols], train["target_demand"], categorical, numeric, params=params)
                inference_start = time.perf_counter()
                predictions = fitted.predict(validation[feature_cols])
                inference_seconds = time.perf_counter() - inference_start
                fitted.save(model_path)
                model_size = model_size_mb(model_path)
                train_seconds = fitted.train_seconds
                params_json = json.dumps(fitted.params)
            except Exception as exc:
                train_seconds = time.perf_counter() - train_start
                error = f"{type(exc).__name__}: {exc}"
                params_json = json.dumps(params)
            if error is None and train_seconds > 480:
                error = "timeout_exceeded_8_minutes_after_fit"
            pred_df = prediction_rows(validation, predictions, public_name, window, scale_map)
            all_predictions.append(pred_df)
            row = {"model": public_name, "window": window["window"], "cutoff": pred_df["cutoff"].iloc[0] if not pred_df.empty else pd.NaT, "train_rows": int(len(train)), "validation_rows": int(len(validation)), "feature_count": int(len(feature_cols)), "train_seconds": train_seconds, "inference_seconds": inference_seconds, "model_size_mb": model_size, "error": error}
            execution_rows.append(row)
            registry_rows.append({**row, "model_path": str(model_path), "params": params_json})
            print_step(f"Finished {public_name} {window['window']}: train={train_seconds:.2f}s predict={inference_seconds:.2f}s peak_mem={tracemalloc.get_traced_memory()[1] / 1024**2:.2f}MB error={error}")
            pd.concat(all_predictions, ignore_index=True).to_parquet(PREDICTIONS_FILE, index=False)
            pd.DataFrame(execution_rows).to_csv(EXEC_TIMES_CSV, index=False)
            pd.DataFrame(registry_rows).to_csv(MODEL_REGISTRY, index=False)
        del train, validation
        gc.collect()
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(PREDICTIONS_FILE, index=False)
    execution = pd.DataFrame(execution_rows)
    execution.to_csv(EXEC_TIMES_CSV, index=False)
    pd.DataFrame(registry_rows).to_csv(MODEL_REGISTRY, index=False)
    metrics = metrics_by_series(predictions)
    metrics.to_parquet(METRICS_FILE, index=False)
    metrics.to_csv(BY_SERIES_CSV, index=False)
    summary = summarize_metrics(metrics, ["model"])
    summary.to_csv(SUMMARY_CSV, index=False)
    by_window = summarize_metrics(metrics, ["model", "cutoff"])
    by_window.to_csv(BY_WINDOW_CSV, index=False)
    summarize_metrics(metrics, ["store_id", "model"]).to_csv(BY_STORE_CSV, index=False)
    summarize_metrics(metrics, ["dept_id", "model"]).to_csv(BY_DEPT_CSV, index=False)
    summarize_horizon(predictions).to_csv(BY_HORIZON_CSV, index=False)
    vs_previous = combine_previous_metrics(metrics)
    vs_previous.to_csv(VS_PREVIOUS_CSV, index=False)
    current, peak = tracemalloc.get_traced_memory()
    run_info = {"execution_seconds": time.perf_counter() - start_all, "peak_traced_memory_mb": peak / 1024**2, "feature_rows": int(features.shape[0]), "feature_columns": int(features.shape[1]), "series_count": int(features["unique_id"].nunique()), "prediction_rows": int(predictions.shape[0]), "metrics_rows": int(metrics.shape[0]), "execution_errors": int(execution["error"].notna().sum()), "training_origin_days": TRAINING_ORIGIN_DAYS}
    RUN_INFO_FILE.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    write_reports(run_info, summary, by_window, vs_previous, execution)
    tracemalloc.stop()
    print_step(f"Completed in {run_info['execution_seconds']:.2f}s with {run_info['execution_errors']} errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
