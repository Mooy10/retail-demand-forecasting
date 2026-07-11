"""Model interpretability artifacts for Phase 6."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    from config import PROCESSED_DATA_DIR, REPORTS_DIR
    from run_ml_forecasting import (
        FEATURE_FILE,
        MODEL_REGISTRY,
        build_supervised_rows,
        clean_training_frame,
        feature_columns,
        target_calendar_columns,
        windows_with_orders,
    )
    from ml_models import transformed_feature_names
except ModuleNotFoundError:
    from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
    from src.run_ml_forecasting import (
        FEATURE_FILE,
        MODEL_REGISTRY,
        build_supervised_rows,
        clean_training_frame,
        feature_columns,
        target_calendar_columns,
        windows_with_orders,
    )
    from src.ml_models import transformed_feature_names

FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PNG = REPORTS_DIR / "feature_importance.png"
PERMUTATION_CSV = REPORTS_DIR / "permutation_importance.csv"
SHAP_PNG = REPORTS_DIR / "shap_summary.png"
INTERPRETABILITY_JSON = PROCESSED_DATA_DIR / "model_interpretability_metrics.json"
ML_SUMMARY = REPORTS_DIR / "ml_metrics_summary.csv"


def print_step(message: str) -> None:
    print(f"[model_interpretability] {message}")


def get_best_model_name() -> str:
    summary = pd.read_csv(ML_SUMMARY)
    return str(summary.sort_values("weighted_wape").iloc[0]["model"])


def load_best_model() -> tuple[object, pd.Series]:
    best_model = get_best_model_name()
    registry = pd.read_csv(MODEL_REGISTRY)
    candidates = registry.loc[(registry["model"] == best_model) & registry["error"].isna()].copy()
    if candidates.empty:
        candidates = registry.loc[registry["model"] == best_model].copy()
    row = candidates.sort_values("window").iloc[-1]
    model = joblib.load(row["model_path"])
    return model, row


def validation_sample(window_name: str, max_rows: int = 3000) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_parquet(FEATURE_FILE)
    features["date"] = pd.to_datetime(features["date"])
    features["origin_demand"] = features["demand"].astype("float32")
    calendar_targets = target_calendar_columns(features)
    window = [w for w in windows_with_orders() if w["window"] == window_name][0]
    validation = build_supervised_rows(
        features,
        calendar_targets,
        window["cutoff_order"],
        window["cutoff_order"],
        origin_order_exact=window["cutoff_order"],
    )
    validation = clean_training_frame(validation)
    if len(validation) > max_rows:
        validation = validation.sample(max_rows, random_state=42)
    categorical, numeric = feature_columns(validation)
    X = validation[categorical + numeric]
    y = validation["target_demand"]
    return X, y


def save_bar_plot(df: pd.DataFrame, value_col: str, label_col: str, path: Path, title: str) -> None:
    top = df.sort_values(value_col, ascending=False).head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top[label_col], top[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    started = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(REPORTS_DIR.parent / ".matplotlib"))
    model, registry_row = load_best_model()
    X, y = validation_sample(str(registry_row["window"]))

    estimator = model.pipeline.named_steps["model"]
    feature_names = transformed_feature_names(model)
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        importances = np.zeros(len(feature_names), dtype="float64")
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    save_bar_plot(importance_df, "importance", "feature", FEATURE_IMPORTANCE_PNG, "Feature Importance")

    print_step("Computing permutation importance on validation sample...")
    perm = permutation_importance(model.pipeline, X, y, n_repeats=3, random_state=42, scoring="neg_mean_absolute_error")
    perm_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(PERMUTATION_CSV, index=False)

    shap_generated = False
    try:
        import shap  # type: ignore

        transformed = model.pipeline.named_steps["preprocess"].transform(X.head(1000))
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(SHAP_PNG, dpi=150)
        plt.close()
        shap_generated = True
    except Exception as exc:
        print_step(f"SHAP skipped: {type(exc).__name__}: {exc}")

    metrics = {
        "model": model.name,
        "window": str(registry_row["window"]),
        "sample_rows": int(len(X)),
        "execution_seconds": time.perf_counter() - started,
        "shap_generated": shap_generated,
    }
    INTERPRETABILITY_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_step(f"Saved interpretability artifacts for {model.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())