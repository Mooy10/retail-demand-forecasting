"""Build the official validated store-department forecast for inventory planning."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_DATA_DIR, REPORTS_DIR
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR, REPORTS_DIR

REGISTRY_FILE = PROCESSED_DATA_DIR / "model_selection_registry_train_w1_w2.parquet"
OFFICIAL_FORECAST = PROCESSED_DATA_DIR / "official_forecast_store_department.parquet"
OFFICIAL_28D_SUMMARY = PROCESSED_DATA_DIR / "official_forecast_28d_summary.parquet"
RUN_INFO = PROCESSED_DATA_DIR / "official_forecast_run_metrics.json"
CANDIDATE_MODELS = ["seasonal_naive_28", "seasonal_average_weekday", "xgboost_phase6", "xgboost_phase7", "lightgbm_phase7"]
BASELINE_MODEL = "seasonal_naive_28"
EVALUATION_WINDOW = "window_3"


def print_step(message: str) -> None:
    print(f"[build_official_forecast] {message}")


def load_candidate_predictions() -> pd.DataFrame:
    frames = []
    baseline = pd.read_parquet(PROCESSED_DATA_DIR / "baseline_predictions_store_department.parquet")
    frames.append(baseline.loc[baseline["model"].isin(["seasonal_naive_28", "seasonal_average_weekday"])].copy())
    phase6 = pd.read_parquet(PROCESSED_DATA_DIR / "ml_predictions_store_department.parquet")
    phase6 = phase6.loc[phase6["model"].eq("xgboost")].copy()
    phase6["model"] = "xgboost_phase6"
    frames.append(phase6)
    advanced = pd.read_parquet(PROCESSED_DATA_DIR / "advanced_ml_predictions.parquet")
    frames.append(advanced.loc[advanced["model"].isin(["xgboost_phase7", "lightgbm_phase7"])].copy())
    pred = pd.concat(frames, ignore_index=True, sort=False)
    pred = pred.loc[pred["window"].astype(str).eq(EVALUATION_WINDOW) & pred["model"].isin(CANDIDATE_MODELS)].copy()
    pred["forecast_date"] = pd.to_datetime(pred["forecast_date"])
    pred["prediction"] = pred["prediction"].astype("float64")
    return pred


def should_fallback(registry_row: pd.Series, selected_rows: pd.DataFrame) -> tuple[bool, str]:
    selected_model = str(registry_row["selected_model"])
    if selected_rows.empty:
        return True, "selected_model_predictions_missing"
    if selected_rows["prediction"].isna().any() or np.isinf(selected_rows["prediction"].to_numpy(dtype="float64")).any():
        return True, "selected_model_predictions_non_finite"
    if (selected_rows["prediction"] < 0).any():
        return True, "selected_model_predictions_negative"
    if registry_row.get("confidence") == "Low" and selected_model != BASELINE_MODEL:
        if float(registry_row.get("score_difference", 0.0)) < 0.03 or float(registry_row.get("window_stability", 0.0)) > 0.60:
            return True, "low_confidence_inconsistent_selected_model"
    return False, "selected_model_used"


def build_official_forecast() -> pd.DataFrame:
    registry = pd.read_parquet(REGISTRY_FILE)
    candidates = load_candidate_predictions()
    rows = []
    for _, reg in registry.iterrows():
        unique_id = reg["unique_id"]
        selected_model = reg["selected_model"]
        selected = candidates.loc[(candidates["unique_id"].eq(unique_id)) & (candidates["model"].eq(selected_model))].copy()
        baseline = candidates.loc[(candidates["unique_id"].eq(unique_id)) & (candidates["model"].eq(BASELINE_MODEL))].copy()
        fallback, reason = should_fallback(reg, selected)
        if fallback:
            chosen = baseline.copy()
            if chosen.empty:
                chosen = selected.copy()
                reason = "fallback_baseline_missing_selected_used"
        else:
            chosen = selected.copy()
        if chosen.empty:
            continue
        base_lookup = baseline[["forecast_date", "prediction"]].rename(columns={"prediction": "baseline_forecast"})
        out = chosen.merge(base_lookup, on="forecast_date", how="left", validate="one_to_one")
        out["forecast_units"] = out["prediction"].clip(lower=0).astype("float32")
        out["selected_model"] = selected_model
        out["source_model_used"] = out["model"]
        out["fallback_used"] = bool(fallback)
        out["fallback_reason"] = reason
        out["selector_confidence"] = reg["confidence"]
        out = out[["unique_id", "store_id", "dept_id", "forecast_date", "horizon", "forecast_units", "selected_model", "source_model_used", "fallback_used", "fallback_reason", "selector_confidence", "baseline_forecast", "actual"]]
        rows.append(out)
    official = pd.concat(rows, ignore_index=True).sort_values(["unique_id", "horizon"]).reset_index(drop=True)
    return official


def build_28d_summary(official: pd.DataFrame) -> pd.DataFrame:
    summary = official.groupby(["unique_id", "store_id", "dept_id"], observed=True).agg(
        forecast_start_date=("forecast_date", "min"),
        forecast_end_date=("forecast_date", "max"),
        forecast_demand_28d=("forecast_units", "sum"),
        average_daily_forecast=("forecast_units", "mean"),
        peak_daily_forecast=("forecast_units", "max"),
        selected_model=("source_model_used", "first"),
        selector_confidence=("selector_confidence", "first"),
        fallback_used=("fallback_used", "max"),
        actual_28d=("actual", "sum"),
        baseline_forecast_28d=("baseline_forecast", "sum"),
    ).reset_index()
    summary["simulation_label"] = "simulated_inventory_planning_input"
    return summary


def main() -> int:
    start = time.perf_counter()
    official = build_official_forecast()
    official.to_parquet(OFFICIAL_FORECAST, index=False)
    summary = build_28d_summary(official)
    summary.to_parquet(OFFICIAL_28D_SUMMARY, index=False)
    info = {
        "execution_seconds": time.perf_counter() - start,
        "forecast_rows": int(len(official)),
        "summary_rows": int(len(summary)),
        "unique_series": int(official["unique_id"].nunique()),
        "horizons": sorted(int(x) for x in official["horizon"].unique()),
        "fallback_rows": int(official["fallback_used"].sum()),
        "fallback_series": int(official.loc[official["fallback_used"], "unique_id"].nunique()),
        "source_model_distribution": official[["unique_id", "source_model_used"]].drop_duplicates()["source_model_used"].value_counts().to_dict(),
    }
    RUN_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print_step(f"Saved official forecast: {len(official):,} rows, {official['unique_id'].nunique():,} series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
