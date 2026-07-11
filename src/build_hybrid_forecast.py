"""Build a hybrid backtesting forecast from the Phase 7 model selection registry."""

from __future__ import annotations

import json
import time

import pandas as pd

try:
    from config import PROCESSED_DATA_DIR, REPORTS_DIR
    from forecast_metrics import metrics_by_series, summarize_metrics
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
    from src.forecast_metrics import metrics_by_series, summarize_metrics

REGISTRY_FILE = PROCESSED_DATA_DIR / "model_selection_registry.parquet"
HYBRID_PREDICTIONS = PROCESSED_DATA_DIR / "hybrid_predictions_store_department.parquet"
HYBRID_METRICS = PROCESSED_DATA_DIR / "hybrid_metrics.parquet"
HYBRID_SUMMARY_JSON = PROCESSED_DATA_DIR / "hybrid_run_metrics.json"
HYBRID_COMPARISON_CSV = REPORTS_DIR / "hybrid_vs_candidates.csv"


def print_step(message: str) -> None:
    print(f"[build_hybrid_forecast] {message}")


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
    return pd.concat(frames, ignore_index=True, sort=False)


def build_hybrid_predictions() -> pd.DataFrame:
    registry = pd.read_parquet(REGISTRY_FILE)
    candidates = load_candidate_predictions()
    selected = candidates.merge(registry[["unique_id", "best_model", "confidence"]], left_on=["unique_id", "model"], right_on=["unique_id", "best_model"], how="inner", validate="many_to_one")
    selected["selected_model"] = selected["model"]
    selected["model"] = "hybrid_selected"
    selected["strategy"] = "model_selection_hybrid"
    return selected.drop(columns=["best_model"]).sort_values(["unique_id", "window", "horizon"]).reset_index(drop=True)


def compare_hybrid(hybrid_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = pd.read_parquet(PROCESSED_DATA_DIR / "baseline_metrics.parquet")
    rows.append(baseline.loc[(baseline["dataset"] == "store_department") & baseline["model"].isin(["seasonal_naive_28", "seasonal_average_weekday"])].copy())
    phase6 = pd.read_parquet(PROCESSED_DATA_DIR / "ml_metrics.parquet")
    phase6 = phase6.loc[phase6["model"].eq("xgboost")].copy()
    phase6["model"] = "xgboost_phase6"
    rows.append(phase6)
    advanced = pd.read_parquet(PROCESSED_DATA_DIR / "advanced_ml_metrics.parquet")
    rows.append(advanced.loc[advanced["model"].isin(["xgboost_phase7", "lightgbm_phase7"])].copy())
    rows.append(hybrid_metrics.copy())
    return summarize_metrics(pd.concat(rows, ignore_index=True, sort=False), ["model"])


def hybrid_improvement_stats(hybrid_metrics: pd.DataFrame) -> dict[str, float]:
    baseline = pd.read_parquet(PROCESSED_DATA_DIR / "baseline_metrics.parquet")
    baseline = baseline.loc[(baseline["dataset"] == "store_department") & baseline["model"].eq("seasonal_naive_28"), ["unique_id", "cutoff", "wape", "rmsse"]].rename(columns={"wape": "baseline_wape", "rmsse": "baseline_rmsse"})
    h = hybrid_metrics.merge(baseline, on=["unique_id", "cutoff"], how="left", validate="one_to_one")
    return {"pct_series_windows_hybrid_beats_baseline_wape": float((h["wape"] < h["baseline_wape"]).mean() * 100), "pct_series_windows_hybrid_beats_baseline_rmsse": float((h["rmsse"] < h["baseline_rmsse"]).mean() * 100), "mean_wape_stability": float(h.groupby("unique_id", observed=True)["wape"].std().fillna(0).mean())}


def main() -> int:
    start = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = build_hybrid_predictions()
    predictions.to_parquet(HYBRID_PREDICTIONS, index=False)
    metrics = metrics_by_series(predictions)
    metrics.to_parquet(HYBRID_METRICS, index=False)
    comparison = compare_hybrid(metrics)
    comparison.to_csv(HYBRID_COMPARISON_CSV, index=False)
    registry = pd.read_parquet(REGISTRY_FILE)
    stats = hybrid_improvement_stats(metrics)
    stats.update({"execution_seconds": time.perf_counter() - start, "prediction_rows": int(len(predictions)), "metrics_rows": int(len(metrics)), "selected_model_distribution": registry["best_model"].value_counts().to_dict(), "confidence_distribution": registry["confidence"].value_counts().to_dict()})
    HYBRID_SUMMARY_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print_step(f"Saved {HYBRID_PREDICTIONS.name}: {len(predictions):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

