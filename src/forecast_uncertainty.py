"""Estimate empirical forecast uncertainty from out-of-sample backtesting errors."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

try:
    from config import PROCESSED_DATA_DIR, REPORTS_DIR
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR, REPORTS_DIR

OFFICIAL_FORECAST = PROCESSED_DATA_DIR / "official_forecast_store_department.parquet"
FORECAST_WITH_UNCERTAINTY = PROCESSED_DATA_DIR / "official_forecast_with_uncertainty.parquet"
UNCERTAINTY_STATS = PROCESSED_DATA_DIR / "forecast_uncertainty_by_series.parquet"
RUN_INFO = PROCESSED_DATA_DIR / "forecast_uncertainty_run_metrics.json"
REPORT = REPORTS_DIR / "forecast_uncertainty_summary.md"


def print_step(message: str) -> None:
    print(f"[forecast_uncertainty] {message}")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for row in df.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def compute_uncertainty_stats(forecast: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = forecast.copy()
    frame["error"] = frame["actual"] - frame["forecast_units"]
    frame["abs_error"] = frame["error"].abs()
    for unique_id, group in frame.groupby("unique_id", observed=True):
        error = group["error"].to_numpy(dtype="float64")
        abs_error = np.abs(error)
        forecast_units = group["forecast_units"].to_numpy(dtype="float64")
        actual = group["actual"].to_numpy(dtype="float64")
        p80 = float(np.nanpercentile(abs_error, 80))
        p90 = float(np.nanpercentile(abs_error, 90))
        p95 = float(np.nanpercentile(abs_error, 95))
        within_80 = np.mean((actual >= np.maximum(forecast_units - p80, 0)) & (actual <= forecast_units + p80))
        within_95 = np.mean((actual >= np.maximum(forecast_units - p95, 0)) & (actual <= forecast_units + p95))
        rows.append({
            "unique_id": unique_id,
            "store_id": group["store_id"].iloc[0],
            "dept_id": group["dept_id"].iloc[0],
            "error_std": float(np.nanstd(error, ddof=1)) if len(error) > 1 else 0.0,
            "mae_historical": float(np.nanmean(abs_error)),
            "rmse_historical": float(np.sqrt(np.nanmean(np.square(error)))),
            "error_percentile_80": p80,
            "error_percentile_90": p90,
            "error_percentile_95": p95,
            "bias_mean": float(np.nanmean(error)),
            "bias_absolute": float(abs(np.nanmean(error))),
            "coverage_80_approx": float(within_80),
            "coverage_95_approx": float(within_95),
            "simulation_label": "empirical_backtesting_uncertainty",
        })
    return pd.DataFrame(rows)


def add_intervals(forecast: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    out = forecast.merge(stats, on=["unique_id", "store_id", "dept_id"], how="left", validate="many_to_one")
    out["forecast_lower_80"] = (out["forecast_units"] - out["error_percentile_80"]).clip(lower=0).astype("float32")
    out["forecast_upper_80"] = (out["forecast_units"] + out["error_percentile_80"]).clip(lower=0).astype("float32")
    out["forecast_lower_95"] = (out["forecast_units"] - out["error_percentile_95"]).clip(lower=0).astype("float32")
    out["forecast_upper_95"] = (out["forecast_units"] + out["error_percentile_95"]).clip(lower=0).astype("float32")
    out["uncertainty_method"] = "empirical_absolute_error_from_holdout_backtesting"
    return out


def write_report(stats: pd.DataFrame, forecast: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# Forecast Uncertainty Summary",
        "",
        "## Objective",
        "",
        "Estimate empirical uncertainty around the official forecast using out-of-sample backtesting errors from the validated holdout forecast.",
        "",
        "## Important Limitation",
        "",
        "Intervals are empirical error bands based on historical backtesting. They are not exact probabilistic prediction intervals and should not be interpreted as calibrated probabilities without further validation.",
        "",
        f"Forecast rows: `{len(forecast):,}`",
        f"Series: `{forecast['unique_id'].nunique():,}`",
        f"Execution seconds: `{elapsed:.2f}`",
        "",
        "## Aggregate Error Statistics",
        "",
        markdown_table(stats[["error_std", "mae_historical", "rmse_historical", "error_percentile_80", "error_percentile_95", "coverage_80_approx", "coverage_95_approx"]].mean().round(3).rename("mean_value").reset_index().rename(columns={"index": "metric"})),
        "",
        "## Highest Historical MAE Series",
        "",
        markdown_table(stats.sort_values("mae_historical", ascending=False).head(10)[["unique_id", "store_id", "dept_id", "mae_historical", "rmse_historical", "bias_mean", "coverage_95_approx"]].round(3)),
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    forecast = pd.read_parquet(OFFICIAL_FORECAST)
    stats = compute_uncertainty_stats(forecast)
    stats.to_parquet(UNCERTAINTY_STATS, index=False)
    out = add_intervals(forecast, stats)
    out.to_parquet(FORECAST_WITH_UNCERTAINTY, index=False)
    elapsed = time.perf_counter() - start
    RUN_INFO.write_text(json.dumps({"execution_seconds": elapsed, "rows": int(len(out)), "series": int(out["unique_id"].nunique())}, indent=2), encoding="utf-8")
    write_report(stats, out, elapsed)
    print_step(f"Saved forecast with uncertainty: {len(out):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
