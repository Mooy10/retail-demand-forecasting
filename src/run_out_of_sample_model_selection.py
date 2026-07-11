"""Out-of-sample validation for Phase 7 model selection.

This script audits the in-sample hybrid from Phase 7 and then validates model
selection with strict temporal separation:
- holdout: select using windows 1-2, evaluate only window 3
- rolling-origin: select using window 1 -> evaluate window 2; select using
  windows 1-2 -> evaluate window 3
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_DATA_DIR, REPORTS_DIR
    from forecast_metrics import calculate_metric_row, summarize_metrics
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
    from src.forecast_metrics import calculate_metric_row, summarize_metrics

CANDIDATE_MODELS = [
    "seasonal_naive_28",
    "seasonal_average_weekday",
    "xgboost_phase6",
    "xgboost_phase7",
    "lightgbm_phase7",
]
SIMPLE_MODEL = "seasonal_naive_28"
SIMPLE_TIE_THRESHOLD = 0.02
WINDOWS = ["window_1", "window_2", "window_3"]

HOLDOUT_REGISTRY = PROCESSED_DATA_DIR / "model_selection_registry_train_w1_w2.parquet"
HOLDOUT_PREDICTIONS = PROCESSED_DATA_DIR / "hybrid_predictions_holdout_w3.parquet"
HOLDOUT_METRICS = PROCESSED_DATA_DIR / "hybrid_metrics_holdout_w3.parquet"
ROLLING_REGISTRY = PROCESSED_DATA_DIR / "rolling_selector_registry.parquet"
ROLLING_PREDICTIONS = PROCESSED_DATA_DIR / "rolling_hybrid_predictions.parquet"
ROLLING_METRICS = PROCESSED_DATA_DIR / "rolling_hybrid_metrics.parquet"
RUN_INFO = PROCESSED_DATA_DIR / "out_of_sample_selector_run_metrics.json"

AUDIT_REPORT = REPORTS_DIR / "model_selector_leakage_audit.md"
HOLDOUT_SUMMARY = REPORTS_DIR / "holdout_w3_metrics_summary.csv"
HOLDOUT_BY_SERIES = REPORTS_DIR / "holdout_w3_metrics_by_series.csv"
HOLDOUT_BY_STORE = REPORTS_DIR / "holdout_w3_metrics_by_store.csv"
HOLDOUT_BY_DEPT = REPORTS_DIR / "holdout_w3_metrics_by_department.csv"
HOLDOUT_BY_HORIZON = REPORTS_DIR / "holdout_w3_metrics_by_horizon.csv"
ROLLING_SUMMARY = REPORTS_DIR / "rolling_selector_metrics_summary.csv"
STABILITY_CSV = REPORTS_DIR / "model_selection_stability.csv"
TRANSITION_CSV = REPORTS_DIR / "model_transition_matrix.csv"
CONFIDENCE_CSV = REPORTS_DIR / "confidence_validation.csv"
OOS_SUMMARY_REPORT = REPORTS_DIR / "out_of_sample_model_selection_summary.md"
OOS_DECISION_REPORT = REPORTS_DIR / "out_of_sample_forecasting_decision.md"


def print_step(message: str) -> None:
    print(f"[out_of_sample_selector] {message}")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def normalize(values: pd.Series) -> pd.Series:
    values = values.astype("float64")
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype="float64")
    return (values - min_value) / (max_value - min_value)


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

    predictions = pd.concat(frames, ignore_index=True, sort=False)
    predictions["window"] = predictions["window"].astype(str)
    predictions["forecast_date"] = pd.to_datetime(predictions["forecast_date"])
    predictions["cutoff"] = pd.to_datetime(predictions["cutoff"])
    predictions["prediction"] = predictions["prediction"].clip(lower=0)
    return predictions.loc[predictions["model"].isin(CANDIDATE_MODELS)].copy()


def metrics_by_series_window(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "unique_id", "window", "cutoff", "model"]
    metadata_cols = ["demand_pattern", "abc_class", "store_id", "dept_id", "item_id", "evaluation_window", "selector_training_windows", "selected_model", "source_model", "confidence"]
    rows = []
    for keys, group in predictions.groupby(group_cols, observed=True, sort=False):
        row = dict(zip(group_cols, keys, strict=False))
        for col in metadata_cols:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        row.update(calculate_metric_row(group))
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metric_cols = ["mae", "rmse", "wape", "rmsse", "smape", "mape"]
    metrics[metric_cols] = metrics[metric_cols].replace([np.inf, -np.inf], np.nan)
    return metrics


def select_models(metrics: pd.DataFrame, training_windows: list[str], evaluation_window: str) -> pd.DataFrame:
    training = metrics.loc[metrics["window"].isin(training_windows) & metrics["model"].isin(CANDIDATE_MODELS)].copy()
    single_window = len(training_windows) == 1
    weights = (0.60, 0.40, 0.00) if single_window else (0.50, 0.30, 0.20)
    expected_windows = len(training_windows)
    per_model = training.groupby(["unique_id", "model"], observed=True).agg(
        store_id=("store_id", "first"),
        dept_id=("dept_id", "first"),
        mean_wape=("wape", "mean"),
        mean_rmsse=("rmsse", "mean"),
        wape_std=("wape", "std"),
        rmsse_std=("rmsse", "std"),
        windows=("window", "nunique"),
        actual_volume=("actual_volume", "sum"),
        prediction_volume=("prediction_volume", "sum"),
        max_wape=("wape", "max"),
    ).reset_index()
    per_model["wape_std"] = per_model["wape_std"].fillna(0)
    per_model["rmsse_std"] = per_model["rmsse_std"].fillna(0)
    if single_window:
        per_model["window_stability"] = 0.0
    else:
        per_model["window_stability"] = (per_model["wape_std"] / per_model["mean_wape"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)

    per_window = training.copy()
    per_window["rank_wape"] = per_window.groupby(["unique_id", "window"], observed=True)["wape"].rank(method="min")
    wins = per_window.loc[per_window["rank_wape"].eq(1)].groupby(["unique_id", "model"], observed=True).size().rename("window_wins").reset_index()
    per_model = per_model.merge(wins, on=["unique_id", "model"], how="left")
    per_model["window_wins"] = per_model["window_wins"].fillna(0).astype("int16")

    selected_rows = []
    for unique_id, group in per_model.groupby("unique_id", observed=True):
        g = group.copy()
        g["norm_wape"] = normalize(g["mean_wape"])
        g["norm_rmsse"] = normalize(g["mean_rmsse"])
        g["norm_stability"] = 0.0 if single_window else normalize(g["window_stability"])
        g["penalty"] = 0.0
        g.loc[g["windows"].lt(expected_windows), "penalty"] += 0.25
        if not single_window:
            g.loc[g["window_stability"].gt(0.75), "penalty"] += 0.10
        if not single_window:
            g.loc[g["window_wins"].eq(1), "penalty"] += 0.04
        g.loc[g["max_wape"].gt(g["mean_wape"] * 2.5), "penalty"] += 0.05
        g["score"] = weights[0] * g["norm_wape"] + weights[1] * g["norm_rmsse"] + weights[2] * g["norm_stability"] + g["penalty"]
        ordered = g.sort_values(["score", "mean_wape", "mean_rmsse", "model"]).reset_index(drop=True)
        best = ordered.iloc[0].copy()
        second = ordered.iloc[1].copy() if len(ordered) > 1 else ordered.iloc[0].copy()
        reason = "lowest historical selector score"
        selected_score = float(best["score"])
        simple = ordered.loc[ordered["model"].eq(SIMPLE_MODEL)]
        if not simple.empty:
            simple_row = simple.iloc[0].copy()
            score_gap = float(simple_row["score"] - best["score"])
            raw_gap_is_small = (
                float(simple_row["mean_wape"]) <= float(best["mean_wape"]) * 1.02
                and float(simple_row["mean_rmsse"]) <= float(best["mean_rmsse"]) * 1.05
            )
            if score_gap <= SIMPLE_TIE_THRESHOLD or raw_gap_is_small:
                if best["model"] != SIMPLE_MODEL:
                    second = best.copy()
                best = simple_row
                selected_score = min(float(best["score"]), float(second["score"]))
                reason = "seasonal_naive_28 fallback preferred because historical score gap is small"
        second_score = max(float(second["score"]), selected_score)
        score_diff = second_score - selected_score
        if single_window:
            confidence = "Medium" if score_diff >= 0.05 else "Low"
        elif score_diff >= 0.08 and best["window_stability"] <= 0.35:
            confidence = "High"
        elif score_diff >= 0.03 and best["window_stability"] <= 0.60:
            confidence = "Medium"
        else:
            confidence = "Low"
        selected_rows.append({
            "evaluation_window": evaluation_window,
            "selector_training_windows": ",".join(training_windows),
            "unique_id": unique_id,
            "store_id": best["store_id"],
            "dept_id": best["dept_id"],
            "best_model": best["model"],
            "selected_model": best["model"],
            "second_best_model": second["model"],
            "best_score": selected_score,
            "second_best_score": second_score,
            "score_difference": score_diff,
            "confidence": confidence,
            "mean_wape": float(best["mean_wape"]),
            "mean_rmsse": float(best["mean_rmsse"]),
            "window_stability": float(best["window_stability"]),
            "selection_reason": reason,
            "normalization_windows": ",".join(training_windows),
            "single_window_selector": bool(single_window),
        })
    registry = pd.DataFrame(selected_rows).sort_values("unique_id").reset_index(drop=True)
    known_series = sorted(training["unique_id"].unique())
    missing = sorted(set(known_series) - set(registry["unique_id"]))
    if missing:
        fallback_meta = training.drop_duplicates("unique_id").set_index("unique_id")
        for unique_id in missing:
            meta = fallback_meta.loc[unique_id]
            registry.loc[len(registry)] = {
                "evaluation_window": evaluation_window,
                "selector_training_windows": ",".join(training_windows),
                "unique_id": unique_id,
                "store_id": meta["store_id"],
                "dept_id": meta["dept_id"],
                "best_model": SIMPLE_MODEL,
                "selected_model": SIMPLE_MODEL,
                "second_best_model": SIMPLE_MODEL,
                "best_score": 1.0,
                "second_best_score": 1.0,
                "score_difference": 0.0,
                "confidence": "Low",
                "mean_wape": np.nan,
                "mean_rmsse": np.nan,
                "window_stability": np.nan,
                "selection_reason": "fallback to seasonal_naive_28 because no historical candidate score was available",
                "normalization_windows": ",".join(training_windows),
                "single_window_selector": bool(single_window),
            }
    return registry.sort_values("unique_id").reset_index(drop=True)


def build_selected_predictions(candidate_predictions: pd.DataFrame, registry: pd.DataFrame, evaluation_window: str, output_model: str) -> pd.DataFrame:
    candidates = candidate_predictions.loc[candidate_predictions["window"].eq(evaluation_window)].copy()
    selected = candidates.merge(
        registry[["evaluation_window", "selector_training_windows", "unique_id", "selected_model", "confidence"]],
        left_on=["window", "unique_id", "model"],
        right_on=["evaluation_window", "unique_id", "selected_model"],
        how="inner",
        validate="many_to_one",
    )
    selected["source_model"] = selected["model"]
    selected["model"] = output_model
    selected["strategy"] = "out_of_sample_model_selection"
    return selected.sort_values(["evaluation_window", "unique_id", "horizon"]).reset_index(drop=True)


def summarize_horizon(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    frame = predictions.copy()
    frame["abs_error"] = (frame["actual"] - frame["prediction"]).abs()
    frame["squared_error"] = np.square(frame["actual"] - frame["prediction"])
    summary = frame.groupby(group_cols + ["horizon"], observed=True).agg(
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda s: float(np.sqrt(np.mean(s)))),
        actual_volume=("actual", "sum"),
        abs_error_sum=("abs_error", "sum"),
        observations=("actual", "count"),
    ).reset_index()
    summary["wape"] = summary["abs_error_sum"] / summary["actual_volume"].replace(0, np.nan)
    return summary.replace([np.inf, -np.inf], np.nan)


def candidate_eval_metrics(candidate_metrics: pd.DataFrame, evaluation_window: str) -> pd.DataFrame:
    return candidate_metrics.loc[candidate_metrics["window"].eq(evaluation_window) & candidate_metrics["model"].isin(CANDIDATE_MODELS)].copy()


def compare_against_baseline(eval_metrics: pd.DataFrame, model_name: str) -> dict[str, float]:
    target = eval_metrics.loc[eval_metrics["model"].eq(model_name), ["unique_id", "wape", "rmsse"]].rename(columns={"wape": "model_wape", "rmsse": "model_rmsse"})
    baseline = eval_metrics.loc[eval_metrics["model"].eq(SIMPLE_MODEL), ["unique_id", "wape", "rmsse"]].rename(columns={"wape": "baseline_wape", "rmsse": "baseline_rmsse"})
    comp = target.merge(baseline, on="unique_id", how="inner", validate="one_to_one")
    return {
        "pct_series_win_wape": float((comp["model_wape"] < comp["baseline_wape"]).mean() * 100) if len(comp) else np.nan,
        "pct_series_win_rmsse": float((comp["model_rmsse"] < comp["baseline_rmsse"]).mean() * 100) if len(comp) else np.nan,
        "pct_series_loses_wape": float((comp["model_wape"] > comp["baseline_wape"]).mean() * 100) if len(comp) else np.nan,
    }


def selection_stability(reg_w1: pd.DataFrame, reg_w12: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = reg_w1[["unique_id", "selected_model", "confidence"]].rename(columns={"selected_model": "selected_model_w1", "confidence": "confidence_w1"})
    right = reg_w12[["unique_id", "selected_model", "confidence"]].rename(columns={"selected_model": "selected_model_w1_w2", "confidence": "confidence_w1_w2"})
    stability = left.merge(right, on="unique_id", how="inner", validate="one_to_one")
    stability["changed_model"] = stability["selected_model_w1"] != stability["selected_model_w1_w2"]
    matrix = pd.crosstab(stability["selected_model_w1"], stability["selected_model_w1_w2"])
    return stability, matrix


def confidence_validation(rolling_metrics: pd.DataFrame, rolling_predictions: pd.DataFrame) -> pd.DataFrame:
    baseline_metrics = metrics_by_series_window(load_candidate_predictions().query("model == @SIMPLE_MODEL"))
    baseline_metrics = baseline_metrics[["unique_id", "window", "wape", "rmsse"]].rename(columns={"wape": "baseline_wape", "rmsse": "baseline_rmsse"})
    metrics = rolling_metrics.merge(baseline_metrics, left_on=["unique_id", "window"], right_on=["unique_id", "window"], how="left", validate="one_to_one")
    metrics["beats_baseline_wape"] = metrics["wape"] < metrics["baseline_wape"]
    metrics["beats_baseline_rmsse"] = metrics["rmsse"] < metrics["baseline_rmsse"]
    metrics["evaluation_window"] = metrics.get("evaluation_window", metrics["window"])
    return metrics.groupby(["evaluation_window", "confidence"], observed=True).agg(
        series_count=("unique_id", "nunique"),
        mean_wape=("wape", "mean"),
        mean_rmsse=("rmsse", "mean"),
        pct_beats_baseline_wape=("beats_baseline_wape", lambda s: float(s.mean() * 100)),
        pct_beats_baseline_rmsse=("beats_baseline_rmsse", lambda s: float(s.mean() * 100)),
    ).reset_index()


def write_audit_report() -> None:
    lines = [
        "# Model Selector Leakage Audit", "",
        "## Current Phase 7 Selector", "",
        "`src/model_selector.py` loads `baseline_metrics.parquet`, `ml_metrics.parquet`, and `advanced_ml_metrics.parquet`. Those metric files contain all three backtesting windows: `window_1`, `window_2`, and `window_3`.", "",
        "## Current Phase 7 Hybrid Evaluation", "",
        "`src/build_hybrid_forecast.py` loads the registry created from all three windows and evaluates `hybrid_selected` on predictions from the same three windows.", "",
        "## Overlap", "",
        "There is complete overlap: the same windows used to choose the best model per series are also used to report the hybrid result.", "",
        "## Conclusion", "",
        "The Phase 7 WAPE `0.117336` is not a true out-of-sample model-selection estimate. It should be labeled as `in-sample model selection`. It is useful as an upper-bound diagnostic, but it is exposed to data snooping and selection bias.", "",
        "## Corrective Validation", "",
        "This phase adds a strict holdout validation that selects models using windows 1-2 only and evaluates on window 3, plus a rolling-origin validation that evaluates window 2 after training on window 1 and window 3 after training on windows 1-2.",
    ]
    AUDIT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_improvement_columns(summary: pd.DataFrame, baseline_model: str = SIMPLE_MODEL) -> pd.DataFrame:
    out = summary.copy()
    baseline = out.loc[out["model"].eq(baseline_model)]
    if baseline.empty:
        return out
    base = baseline.iloc[0]
    for metric in ["weighted_wape", "weighted_rmsse", "weighted_mae", "weighted_rmse", "weighted_smape"]:
        if metric in out.columns:
            out[f"{metric}_diff_vs_baseline"] = out[metric] - float(base[metric])
            out[f"{metric}_improvement_pct"] = (float(base[metric]) - out[metric]) / float(base[metric]) * 100
    return out


def build_reports(holdout_summary: pd.DataFrame, rolling_summary: pd.DataFrame, holdout_registry: pd.DataFrame, rolling_registry: pd.DataFrame, stability: pd.DataFrame, confidence: pd.DataFrame) -> None:
    phase7_hybrid = pd.read_csv(REPORTS_DIR / "hybrid_vs_candidates.csv").sort_values("weighted_wape").head(1)
    holdout_best = holdout_summary.sort_values("weighted_wape").iloc[0]
    holdout_hybrid = holdout_summary.loc[holdout_summary["model"].eq("hybrid_holdout_w3")].iloc[0]
    holdout_baseline = holdout_summary.loc[holdout_summary["model"].eq(SIMPLE_MODEL)].iloc[0]
    wape_improvement = (holdout_baseline["weighted_wape"] - holdout_hybrid["weighted_wape"]) / holdout_baseline["weighted_wape"] * 100
    rmsse_improvement = (holdout_baseline["weighted_rmsse"] - holdout_hybrid["weighted_rmsse"]) / holdout_baseline["weighted_rmsse"] * 100

    summary_lines = [
        "# Out-Of-Sample Model Selection Summary", "",
        "## Methodological Finding", "",
        "The Phase 7 hybrid result was selected and evaluated on the same three windows, so it is classified as `in-sample model selection` and should not be used as the main production estimate.", "",
        "## In-Sample Hybrid Reference", "", markdown_table(phase7_hybrid[["model", "weighted_wape", "weighted_rmsse", "series_count"]]), "",
        "## Strict Holdout Window 3", "", markdown_table(holdout_summary[["model", "weighted_mae", "weighted_wape", "weighted_rmsse", "series_count"]].sort_values("weighted_wape")), "",
        "## Rolling-Origin Validation", "", markdown_table(rolling_summary[["model", "evaluation_window", "weighted_wape", "weighted_rmsse", "series_count"]].sort_values(["evaluation_window", "weighted_wape"])), "",
        "## Holdout Selector Distribution", "", markdown_table(holdout_registry["selected_model"].value_counts().rename_axis("selected_model").reset_index(name="series")), "",
        "## Rolling Selector Distribution", "", markdown_table(rolling_registry.groupby(["evaluation_window", "selected_model"], observed=True).size().rename("series").reset_index()), "",
        "## Selection Stability", "", f"Series that changed model between selector trained on window 1 and selector trained on windows 1-2: `{int(stability['changed_model'].sum())}` of `{len(stability)}`.", "",
        "## Confidence Validation", "", markdown_table(confidence), "",
        "## Interpretation", "",
        f"On strict holdout window 3, `hybrid_holdout_w3` changes WAPE by `{wape_improvement:.2f}%` and RMSSE by `{rmsse_improvement:.2f}%` versus `seasonal_naive_28`.",
    ]
    OOS_SUMMARY_REPORT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if holdout_hybrid["weighted_wape"] < holdout_baseline["weighted_wape"] and holdout_hybrid["weighted_rmsse"] < holdout_baseline["weighted_rmsse"]:
        if wape_improvement < 2:
            decision = "Keep seasonal_naive_28 as official because WAPE improvement is below 2%, unless RMSSE is prioritized."
            status = "experimental"
        else:
            decision = "The holdout hybrid can be declared the official candidate, with seasonal_naive_28 retained as fallback."
            status = "candidate_official"
    elif holdout_hybrid["weighted_wape"] < holdout_baseline["weighted_wape"] or holdout_hybrid["weighted_rmsse"] < holdout_baseline["weighted_rmsse"]:
        decision = "The hybrid wins only one primary dimension, so keep it experimental and decide based on business priority."
        status = "experimental"
    else:
        decision = "seasonal_naive_28 should remain the official forecast; keep hybrid selection as future research."
        status = "baseline_official"

    decision_lines = [
        "# Out-Of-Sample Forecasting Decision", "",
        "## Leakage Audit Result", "",
        "Selection bias exists in the Phase 7 `hybrid_selected` result because model selection and evaluation used the same windows. The WAPE `0.117336` is in-sample model selection, not a strict production estimate.", "",
        "## Strict Holdout Decision", "",
        f"Holdout hybrid WAPE: `{holdout_hybrid['weighted_wape']:.6f}`. Baseline WAPE: `{holdout_baseline['weighted_wape']:.6f}`. Improvement: `{wape_improvement:.2f}%`.",
        f"Holdout hybrid RMSSE: `{holdout_hybrid['weighted_rmsse']:.6f}`. Baseline RMSSE: `{holdout_baseline['weighted_rmsse']:.6f}`. Improvement: `{rmsse_improvement:.2f}%`.", "",
        f"Decision status: `{status}`.", "", decision, "",
        "## Dashboard Forecast Recommendation", "",
        "Use the strict holdout conclusion, not the in-sample Phase 7 result, to decide the dashboard forecast. If the hybrid remains official, expose selected model and confidence; if not, use `seasonal_naive_28` and present the hybrid as experimental.", "",
        "## Required User-Facing Limitations", "",
        "- The selector was validated on only three backtesting windows.",
        "- Window 3 holdout is the main out-of-sample model-selection check.",
        "- The Phase 7 in-sample hybrid is an optimistic diagnostic and should not be the primary production metric.",
        "- Current modeling remains at store-department level only.",
    ]
    OOS_DECISION_REPORT.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_audit_report()
    candidate_predictions = load_candidate_predictions()
    candidate_metrics = metrics_by_series_window(candidate_predictions)

    print_step("Building strict holdout selector using windows 1-2 and evaluating window 3...")
    holdout_registry = select_models(candidate_metrics, ["window_1", "window_2"], "window_3")
    holdout_registry.to_parquet(HOLDOUT_REGISTRY, index=False)
    holdout_predictions = build_selected_predictions(candidate_predictions, holdout_registry, "window_3", "hybrid_holdout_w3")
    holdout_predictions.to_parquet(HOLDOUT_PREDICTIONS, index=False)
    holdout_metrics = metrics_by_series_window(holdout_predictions)
    holdout_metrics.to_parquet(HOLDOUT_METRICS, index=False)

    holdout_candidate_metrics = candidate_eval_metrics(candidate_metrics, "window_3")
    holdout_compare_metrics = pd.concat([holdout_candidate_metrics, holdout_metrics], ignore_index=True, sort=False)
    holdout_summary = add_improvement_columns(summarize_metrics(holdout_compare_metrics, ["model"]))
    holdout_summary.to_csv(HOLDOUT_SUMMARY, index=False)
    holdout_compare_metrics.to_csv(HOLDOUT_BY_SERIES, index=False)
    summarize_metrics(holdout_compare_metrics, ["store_id", "model"]).to_csv(HOLDOUT_BY_STORE, index=False)
    summarize_metrics(holdout_compare_metrics, ["dept_id", "model"]).to_csv(HOLDOUT_BY_DEPT, index=False)
    summarize_horizon(pd.concat([candidate_predictions.loc[candidate_predictions["window"].eq("window_3") & candidate_predictions["model"].isin(CANDIDATE_MODELS)], holdout_predictions], ignore_index=True, sort=False), ["model"]).to_csv(HOLDOUT_BY_HORIZON, index=False)

    print_step("Building rolling-origin selector validation...")
    rolling_regs = []
    rolling_preds = []
    for training_windows, evaluation_window in [(["window_1"], "window_2"), (["window_1", "window_2"], "window_3")]:
        reg = select_models(candidate_metrics, training_windows, evaluation_window)
        rolling_regs.append(reg)
        rolling_preds.append(build_selected_predictions(candidate_predictions, reg, evaluation_window, "rolling_hybrid"))
    rolling_registry = pd.concat(rolling_regs, ignore_index=True)
    rolling_registry.to_parquet(ROLLING_REGISTRY, index=False)
    rolling_predictions = pd.concat(rolling_preds, ignore_index=True)
    rolling_predictions.to_parquet(ROLLING_PREDICTIONS, index=False)
    rolling_metrics = metrics_by_series_window(rolling_predictions)
    rolling_metrics.to_parquet(ROLLING_METRICS, index=False)

    rolling_candidate_metrics = candidate_metrics.loc[candidate_metrics["model"].eq(SIMPLE_MODEL) & candidate_metrics["window"].isin(["window_2", "window_3"])].copy()
    rolling_compare = pd.concat([rolling_candidate_metrics, rolling_metrics], ignore_index=True, sort=False)
    rolling_summary = summarize_metrics(rolling_compare, ["model", "window"]).rename(columns={"window": "evaluation_window"})
    rolling_summary.to_csv(ROLLING_SUMMARY, index=False)

    reg_w1 = rolling_registry.loc[rolling_registry["evaluation_window"].eq("window_2")].copy()
    reg_w12 = rolling_registry.loc[rolling_registry["evaluation_window"].eq("window_3")].copy()
    stability, matrix = selection_stability(reg_w1, reg_w12)
    stability.to_csv(STABILITY_CSV, index=False)
    matrix.to_csv(TRANSITION_CSV)
    confidence = confidence_validation(rolling_metrics, rolling_predictions)
    confidence.to_csv(CONFIDENCE_CSV, index=False)

    build_reports(holdout_summary, rolling_summary, holdout_registry, rolling_registry, stability, confidence)

    holdout_stats = compare_against_baseline(holdout_compare_metrics, "hybrid_holdout_w3")
    run_info = {
        "execution_seconds": time.perf_counter() - start,
        "candidate_prediction_rows": int(len(candidate_predictions)),
        "candidate_metric_rows": int(len(candidate_metrics)),
        "holdout_prediction_rows": int(len(holdout_predictions)),
        "holdout_metric_rows": int(len(holdout_metrics)),
        "rolling_prediction_rows": int(len(rolling_predictions)),
        "rolling_metric_rows": int(len(rolling_metrics)),
        "holdout_selector_training_windows": ["window_1", "window_2"],
        "holdout_evaluation_window": "window_3",
        "rolling_evaluations": {"window_2": ["window_1"], "window_3": ["window_1", "window_2"]},
        "holdout_selected_model_distribution": holdout_registry["selected_model"].value_counts().to_dict(),
        "rolling_selected_model_distribution": {f"{key[0]}|{key[1]}": int(value) for key, value in rolling_registry.groupby(["evaluation_window", "selected_model"], observed=True).size().to_dict().items()},
        "selection_changes_w1_to_w12": int(stability["changed_model"].sum()),
        "holdout_vs_baseline": holdout_stats,
    }
    RUN_INFO.write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    print_step(f"Completed in {run_info['execution_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



