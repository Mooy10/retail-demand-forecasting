from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note
from dashboard.components.charts import bar_chart, donut, model_metric_bars, PLOTLY_CONFIG
from dashboard.components.kpis import kpi_grid
from dashboard.components.tables import download_button, searchable_table
from dashboard.services.data_loader import (
    load_baseline_metrics_summary,
    load_confidence_validation,
    load_forecast_uncertainty,
    load_holdout_metrics_by_horizon,
    load_holdout_metrics_by_series,
    load_holdout_metrics_summary,
    load_ml_vs_baseline,
    load_model_registry,
    load_rolling_metrics_summary,
    load_transition_matrix,
)
from dashboard.services.forecast_service import add_state_id
from dashboard.services.metrics_service import model_label

st.set_page_config(page_title="Model Performance", layout="wide", initial_sidebar_state="auto")
styles = ROOT / "dashboard" / "assets" / "styles.css"
st.markdown(f"<style>{styles.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.title("Model Performance")
portfolio_note()

st.info(
    "La decisión oficial usa validación fuera de muestra W3. Los resultados de entrenamiento/rolling se muestran como contexto, "
    "pero no sustituyen el holdout estricto."
)

holdout = load_holdout_metrics_summary()
horizon = load_holdout_metrics_by_horizon()
series = add_state_id(load_holdout_metrics_by_series())
rolling = load_rolling_metrics_summary()
registry = load_model_registry()
forecast = load_forecast_uncertainty()
confidence = load_confidence_validation()
transition = load_transition_matrix()
baseline = load_baseline_metrics_summary()
ml_vs_baseline = load_ml_vs_baseline()

official = holdout.loc[holdout["model"].eq("hybrid_holdout_w3")] if not holdout.empty else pd.DataFrame()
baseline_row = holdout.loc[holdout["model"].eq("seasonal_naive_28")] if not holdout.empty else pd.DataFrame()

improvement = float(official["weighted_wape_improvement_pct"].iloc[0]) if not official.empty and "weighted_wape_improvement_pct" in official else 0
kpi_grid([
    {"label": "WAPE oficial", "value": float(official["weighted_wape"].iloc[0]) * 100 if not official.empty else 0, "help_text": "Holdout W3", "decimals": 2},
    {"label": "RMSSE oficial", "value": float(official["weighted_rmsse"].iloc[0]) if not official.empty else 0, "help_text": "Holdout W3", "decimals": 3},
    {"label": "Mejora vs baseline", "value": improvement, "help_text": "WAPE %", "decimals": 2},
    {"label": "WAPE baseline", "value": float(baseline_row["weighted_wape"].iloc[0]) * 100 if not baseline_row.empty else 0, "help_text": "seasonal_naive_28", "decimals": 2},
    {"label": "Series evaluadas", "value": int(official["series_count"].iloc[0]) if not official.empty else 0, "help_text": "store-dept", "decimals": 0},
])

cols = st.columns(2)
with cols[0]:
    st.plotly_chart(model_metric_bars(holdout, "weighted_wape", "WAPE ponderado por modelo - Holdout W3"), width="stretch", config=PLOTLY_CONFIG)
with cols[1]:
    st.plotly_chart(model_metric_bars(holdout, "weighted_rmsse", "RMSSE ponderado por modelo - Holdout W3"), width="stretch", config=PLOTLY_CONFIG)

cols = st.columns(2)
with cols[0]:
    if horizon.empty:
        st.plotly_chart(bar_chart(pd.DataFrame(), "horizon", "wape", "WAPE por horizonte"), width="stretch", config=PLOTLY_CONFIG)
    else:
        fig = px.line(horizon, x="horizon", y="wape", color="model", markers=True, title="WAPE por horizonte de pronóstico")
        fig.update_layout(template="plotly_white", autosize=True, margin=dict(l=20, r=20, t=60, b=80), yaxis_title="WAPE", xaxis_title="Horizonte", legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
with cols[1]:
    if rolling.empty:
        st.plotly_chart(bar_chart(pd.DataFrame(), "evaluation_window", "weighted_wape", "Rolling selector"), width="stretch", config=PLOTLY_CONFIG)
    else:
        fig = px.line(rolling, x="evaluation_window", y="weighted_wape", color="model", markers=True, title="Rolling selector - WAPE ponderado")
        fig.update_layout(template="plotly_white", autosize=True, margin=dict(l=20, r=20, t=60, b=80), yaxis_title="WAPE", xaxis_title="Ventana", legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

cols = st.columns(3)
with cols[0]:
    if forecast.empty:
        dist = pd.DataFrame()
    else:
        dist = forecast[["unique_id", "source_model_used"]].drop_duplicates().groupby("source_model_used")["unique_id"].nunique().reset_index(name="series")
    st.plotly_chart(donut(dist, "source_model_used", "series", "Modelo usado en forecast oficial"), width="stretch", config=PLOTLY_CONFIG)
with cols[1]:
    if registry.empty:
        conf = pd.DataFrame()
    else:
        conf = registry.groupby("confidence", observed=True)["unique_id"].nunique().reset_index(name="series")
    st.plotly_chart(donut(conf, "confidence", "series", "Confianza del selector"), width="stretch", config=PLOTLY_CONFIG)
with cols[2]:
    if confidence.empty:
        st.dataframe(pd.DataFrame(), width="stretch")
    else:
        st.markdown("#### Validación por confianza")
        st.dataframe(confidence, width="stretch", hide_index=True)

st.subheader("Matriz de transición del selector")
if transition.empty:
    st.warning("No hay matriz de transición disponible.")
else:
    st.dataframe(transition, width="stretch")

st.subheader("Detalle por serie")
series_view = series.copy()
if not series_view.empty:
    series_view["model_label"] = series_view["model"].map(model_label)
    series_view = series_view[["unique_id", "store_id", "dept_id", "model_label", "mae", "rmse", "wape", "rmsse", "actual_volume", "prediction_volume"]]
searchable_table(series_view, "performance", "performance_by_series_table", limit=100)
download_button(series_view, "Descargar métricas por serie", "model_performance_by_series.csv", "model_performance_by_series_download")

with st.expander("Tablas auxiliares"):
    st.markdown("**Baseline metrics**")
    st.dataframe(baseline.head(50), width="stretch")
    st.markdown("**ML vs baseline**")
    st.dataframe(ml_vs_baseline.head(50), width="stretch")
