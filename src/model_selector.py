"""Select the best forecasting model per store-department series using backtesting metrics only."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_DATA_DIR
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR

OUTPUT = PROCESSED_DATA_DIR / "model_selection_registry.parquet"
SUMMARY_JSON = PROCESSED_DATA_DIR / "model_selection_metrics.json"
CANDIDATE_MODELS = ["seasonal_naive_28", "seasonal_average_weekday", "xgboost_phase6", "xgboost_phase7", "lightgbm_phase7"]
SIMPLE_MODEL = "seasonal_naive_28"
SIMPLE_TIE_THRESHOLD = 0.02


def print_step(message: str) -> None:
    print(f"[model_selector] {message}")


def load_candidate_metrics() -> pd.DataFrame:
    rows = []
    baseline = pd.read_parquet(PROCESSED_DATA_DIR / "baseline_metrics.parquet")
    rows.append(baseline.loc[(baseline["dataset"] == "store_department") & (baseline["model"].isin(["seasonal_naive_28", "seasonal_average_weekday"]))].copy())
    phase6 = pd.read_parquet(PROCESSED_DATA_DIR / "ml_metrics.parquet")
    phase6 = phase6.loc[phase6["model"] == "xgboost"].copy()
    phase6["model"] = "xgboost_phase6"
    rows.append(phase6)
    advanced = pd.read_parquet(PROCESSED_DATA_DIR / "advanced_ml_metrics.parquet")
    rows.append(advanced.loc[advanced["model"].isin(["xgboost_phase7", "lightgbm_phase7"])].copy())
    metrics = pd.concat(rows, ignore_index=True, sort=False)
    return metrics.loc[metrics["model"].isin(CANDIDATE_MODELS)].copy()


def _normalize(values: pd.Series) -> pd.Series:
    values = values.astype("float64")
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")
    return (values - min_value) / (max_value - min_value)


def score_models(metrics: pd.DataFrame) -> pd.DataFrame:
    per_model = metrics.groupby(["unique_id", "model"], observed=True).agg(store_id=("store_id", "first"), dept_id=("dept_id", "first"), mean_wape=("wape", "mean"), mean_rmsse=("rmsse", "mean"), wape_std=("wape", "std"), rmsse_std=("rmsse", "std"), windows=("cutoff", "nunique"), actual_volume=("actual_volume", "sum"), prediction_volume=("prediction_volume", "sum"), max_wape=("wape", "max")).reset_index()
    per_model["wape_std"] = per_model["wape_std"].fillna(0)
    per_model["rmsse_std"] = per_model["rmsse_std"].fillna(0)
    per_model["window_stability"] = (per_model["wape_std"] / per_model["mean_wape"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    per_window = metrics.copy()
    per_window["rank_wape"] = per_window.groupby(["unique_id", "cutoff"], observed=True)["wape"].rank(method="min")
    wins = per_window.loc[per_window["rank_wape"].eq(1)].groupby(["unique_id", "model"], observed=True).size().rename("window_wins").reset_index()
    per_model = per_model.merge(wins, on=["unique_id", "model"], how="left")
    per_model["window_wins"] = per_model["window_wins"].fillna(0).astype("int16")
    scored = []
    for _, group in per_model.groupby("unique_id", observed=True):
        g = group.copy()
        g["norm_wape"] = _normalize(g["mean_wape"])
        g["norm_rmsse"] = _normalize(g["mean_rmsse"])
        g["norm_stability"] = _normalize(g["window_stability"])
        g["penalty"] = 0.0
        g.loc[g["windows"].lt(3), "penalty"] += 0.25
        g.loc[g["window_stability"].gt(0.75), "penalty"] += 0.10
        g.loc[g["window_wins"].eq(1), "penalty"] += 0.04
        g.loc[g["max_wape"].gt(g["mean_wape"] * 2.5), "penalty"] += 0.05
        g["best_score"] = 0.50 * g["norm_wape"] + 0.30 * g["norm_rmsse"] + 0.20 * g["norm_stability"] + g["penalty"]
        scored.append(g)
    return pd.concat(scored, ignore_index=True)


def select_from_scores(scored: pd.DataFrame, tie_threshold: float = SIMPLE_TIE_THRESHOLD) -> pd.DataFrame:
    rows = []
    for unique_id, group in scored.groupby("unique_id", observed=True):
        ordered = group.sort_values(["best_score", "mean_wape", "mean_rmsse", "model"]).reset_index(drop=True)
        best = ordered.iloc[0].copy()
        second = ordered.iloc[1].copy() if len(ordered) > 1 else ordered.iloc[0].copy()
        reason = "lowest combined backtesting score"
        selected_score = float(best["best_score"])
        simple = ordered.loc[ordered["model"].eq(SIMPLE_MODEL)]
        if not simple.empty:
            simple_row = simple.iloc[0].copy()
            score_gap = float(simple_row["best_score"] - best["best_score"])
            raw_gap_is_small = (
                float(simple_row["mean_wape"]) <= float(best["mean_wape"]) * 1.02
                and float(simple_row["mean_rmsse"]) <= float(best["mean_rmsse"]) * 1.05
            )
            if score_gap <= tie_threshold or raw_gap_is_small:
                if best["model"] != SIMPLE_MODEL:
                    second = best.copy()
                best = simple_row
                selected_score = min(float(best["best_score"]), float(second["best_score"]))
                reason = "simple baseline preferred because score gap is small"
        second_score = max(float(second["best_score"]), selected_score)
        score_diff = float(second_score - selected_score)
        if score_diff >= 0.08 and best["window_stability"] <= 0.35:
            confidence = "High"
        elif score_diff >= 0.03 and best["window_stability"] <= 0.60:
            confidence = "Medium"
        else:
            confidence = "Low"
        rows.append({"unique_id": unique_id, "store_id": best["store_id"], "dept_id": best["dept_id"], "best_model": best["model"], "second_best_model": second["model"], "best_score": selected_score, "second_best_score": second_score, "score_difference": score_diff, "confidence": confidence, "mean_wape": float(best["mean_wape"]), "mean_rmsse": float(best["mean_rmsse"]), "window_stability": float(best["window_stability"]), "selection_reason": reason})
    return pd.DataFrame(rows).sort_values("unique_id").reset_index(drop=True)


def build_model_selection() -> pd.DataFrame:
    return select_from_scores(score_models(load_candidate_metrics()))


def main() -> int:
    start = time.perf_counter()
    registry = build_model_selection()
    registry.to_parquet(OUTPUT, index=False)
    metrics = {"execution_seconds": time.perf_counter() - start, "rows": int(len(registry)), "unique_series": int(registry["unique_id"].nunique()), "model_distribution": registry["best_model"].value_counts().to_dict(), "confidence_distribution": registry["confidence"].value_counts().to_dict()}
    SUMMARY_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_step(f"Saved {OUTPUT.name}: {len(registry):,} selected series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

