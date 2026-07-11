"""Cached data access for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"

PORTFOLIO_NOTE = (
    "Proyecto analítico de portafolio construido con el dataset M5. "
    "Las cifras de inventario, costos y ahorros son simulaciones basadas en supuestos configurables "
    "y no representan operaciones reales de Walmart."
)


def _cache_data(func):
    if st is not None:
        return st.cache_data(show_spinner=False)(func)
    return func


def safe_read_parquet(path: Path, columns: list[str] | None = None, required: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        if st is not None:
            st.warning(f"Archivo no disponible: {path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        if st is not None:
            st.error(f"No se pudo cargar {path.name}. Revisa que la fase previa haya generado el archivo.")
        print(f"[dashboard.data_loader] error reading {path}: {exc}")
        return pd.DataFrame()
    required = list(required or [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        if st is not None:
            st.warning(f"{path.name} no contiene columnas esperadas: {missing}")
        return pd.DataFrame()
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def safe_read_csv(path: Path, required: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        if st is not None:
            st.warning(f"Archivo no disponible: {path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        if st is not None:
            st.error(f"No se pudo cargar {path.name}.")
        print(f"[dashboard.data_loader] error reading {path}: {exc}")
        return pd.DataFrame()
    missing = [col for col in list(required or []) if col not in df.columns]
    if missing:
        if st is not None:
            st.warning(f"{path.name} no contiene columnas esperadas: {missing}")
        return pd.DataFrame()
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@_cache_data
def load_official_forecast() -> pd.DataFrame:
    return safe_read_parquet(PROCESSED / "official_forecast_store_department.parquet", required=["unique_id", "forecast_date", "horizon", "forecast_units"])


@_cache_data
def load_forecast_summary() -> pd.DataFrame:
    return safe_read_parquet(PROCESSED / "official_forecast_28d_summary.parquet", required=["unique_id", "forecast_demand_28d"])


@_cache_data
def load_forecast_uncertainty() -> pd.DataFrame:
    return safe_read_parquet(PROCESSED / "official_forecast_with_uncertainty.parquet", required=["unique_id", "forecast_units", "forecast_lower_95", "forecast_upper_95"])


@_cache_data
def load_model_registry() -> pd.DataFrame:
    return safe_read_parquet(PROCESSED / "model_selection_registry_train_w1_w2.parquet", required=["unique_id", "selected_model", "confidence"])


@_cache_data
def load_inventory_scenarios() -> pd.DataFrame:
    return safe_read_parquet(PROCESSED / "inventory_initial_scenarios.parquet", required=["unique_id", "scenario", "initial_inventory"])


@_cache_data
def load_inventory_recommendations() -> pd.DataFrame:
    return safe_read_parquet(PROCESSED / "inventory_recommendations.parquet", required=["unique_id", "scenario", "recommended_order_quantity", "simulation_label"])


@_cache_data
def load_inventory_daily_projection() -> pd.DataFrame:
    return safe_read_parquet(PROCESSED / "inventory_daily_projection.parquet", required=["unique_id", "scenario", "date", "forecast_demand"])


@_cache_data
def load_policy_comparison() -> pd.DataFrame:
    return safe_read_csv(REPORTS / "inventory_policy_comparison.csv", required=["policy", "scenario", "total_cost", "simulation_label"])


@_cache_data
def load_holdout_metrics_summary() -> pd.DataFrame:
    return safe_read_csv(REPORTS / "holdout_w3_metrics_summary.csv", required=["model", "weighted_wape", "weighted_rmsse"])


@_cache_data
def load_holdout_metrics_by_series() -> pd.DataFrame:
    return safe_read_csv(REPORTS / "holdout_w3_metrics_by_series.csv", required=["unique_id", "model", "wape", "rmsse"])


@_cache_data
def load_holdout_metrics_by_horizon() -> pd.DataFrame:
    return safe_read_csv(REPORTS / "holdout_w3_metrics_by_horizon.csv", required=["model", "horizon", "wape"])


@_cache_data
def load_rolling_metrics_summary() -> pd.DataFrame:
    return safe_read_csv(REPORTS / "rolling_selector_metrics_summary.csv", required=["model", "evaluation_window", "weighted_wape"])


@_cache_data
def load_baseline_metrics_summary() -> pd.DataFrame:
    return safe_read_csv(REPORTS / "baseline_metrics_summary.csv")


@_cache_data
def load_ml_vs_baseline() -> pd.DataFrame:
    return safe_read_csv(REPORTS / "ml_vs_baseline_comparison.csv")


@_cache_data
def load_confidence_validation() -> pd.DataFrame:
    return safe_read_csv(REPORTS / "confidence_validation.csv")


@_cache_data
def load_transition_matrix() -> pd.DataFrame:
    path = REPORTS / "model_transition_matrix.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, index_col=0)
    except Exception as exc:
        print(f"[dashboard.data_loader] transition matrix error: {exc}")
        return pd.DataFrame()


@_cache_data
def load_assumptions_text() -> str:
    path = CONFIG / "inventory_assumptions.yaml"
    if not path.exists():
        return "Archivo de supuestos no disponible."
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[dashboard.data_loader] assumptions error: {exc}")
        return "No se pudieron cargar los supuestos."


@_cache_data
def load_recent_history() -> pd.DataFrame:
    cols = ["unique_id", "store_id", "dept_id", "date", "demand"]
    df = safe_read_parquet(PROCESSED / "forecast_store_department.parquet", columns=cols, required=cols)
    if df.empty:
        return df
    max_date = df["date"].max()
    return df.loc[df["date"] >= max_date - pd.Timedelta(days=90)].copy()


def load_all_core() -> dict[str, pd.DataFrame]:
    return {
        "forecast": load_official_forecast(),
        "forecast_summary": load_forecast_summary(),
        "uncertainty": load_forecast_uncertainty(),
        "registry": load_model_registry(),
        "inventory": load_inventory_recommendations(),
        "daily_inventory": load_inventory_daily_projection(),
        "policy": load_policy_comparison(),
        "holdout": load_holdout_metrics_summary(),
    }
