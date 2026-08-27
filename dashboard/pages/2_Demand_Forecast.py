from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note
from dashboard.components.charts import forecast_with_uncertainty, forecast_line, PLOTLY_CONFIG
from dashboard.components.kpis import fallback_chip, kpi_grid
from dashboard.components.tables import searchable_table
from dashboard.services.data_loader import load_forecast_uncertainty, load_holdout_metrics_by_series, load_recent_history
from dashboard.services.forecast_service import add_state_id

st.set_page_config(page_title="Demand Forecast", layout="wide", initial_sidebar_state="auto")
styles = ROOT / "dashboard" / "assets" / "styles.css"
st.markdown(f"<style>{styles.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
st.title("Demand Forecast")
portfolio_note()

forecast = add_state_id(load_forecast_uncertainty())
history = add_state_id(load_recent_history())
metrics = load_holdout_metrics_by_series()
if forecast.empty:
    st.error("No hay forecast oficial disponible.")
    st.stop()

stores = sorted(forecast["store_id"].dropna().unique().tolist())
store = st.selectbox("Tienda", stores)
depts = sorted(forecast.loc[forecast["store_id"].eq(store), "dept_id"].dropna().unique().tolist())
dept = st.selectbox("Departamento", depts)
series = sorted(forecast.loc[forecast["store_id"].eq(store) & forecast["dept_id"].eq(dept), "unique_id"].unique().tolist())
unique_id = st.selectbox("Serie", series)
horizon = st.radio("Horizonte visible", [7, 14, 28], index=2, horizontal=True)

series_forecast = forecast.loc[forecast["unique_id"].eq(unique_id) & forecast["horizon"].le(horizon)].copy()
series_history = history.loc[history["unique_id"].eq(unique_id)].copy()
row = series_forecast.iloc[0]
series_metrics = metrics.loc[metrics["unique_id"].eq(unique_id) & metrics["model"].eq("hybrid_holdout_w3")]

st.markdown(f"### {unique_id}")
st.markdown(f"Estado `{row['state_id']}` · Tienda `{row['store_id']}` · Departamento `{row['dept_id']}` · Modelo `{row['source_model_used']}` · {fallback_chip(bool(row['fallback_used']))}", unsafe_allow_html=True)
kpi_grid([
    {"label": "Forecast total", "value": series_forecast["forecast_units"].sum(), "help_text": f"{horizon} días", "decimals": 1},
    {"label": "Promedio diario", "value": series_forecast["forecast_units"].mean(), "help_text": "Unidades", "decimals": 1},
    {"label": "Máximo diario", "value": series_forecast["forecast_units"].max(), "help_text": str(series_forecast.loc[series_forecast["forecast_units"].idxmax(), "forecast_date"].date()), "decimals": 1},
    {"label": "Incertidumbre prom.", "value": (series_forecast["forecast_upper_95"] - series_forecast["forecast_lower_95"]).mean(), "help_text": "Banda 95%", "decimals": 1},
    {"label": "WAPE serie", "value": float(series_metrics["wape"].iloc[0]) * 100 if not series_metrics.empty else 0, "help_text": "Holdout", "decimals": 2},
    {"label": "RMSSE serie", "value": float(series_metrics["rmsse"].iloc[0]) if not series_metrics.empty else 0, "help_text": "Holdout", "decimals": 3},
    {"label": "Confianza", "value": str(row["selector_confidence"]), "help_text": "Selector"},
])

st.plotly_chart(forecast_with_uncertainty(series_history, series_forecast), width="stretch", config=PLOTLY_CONFIG)

st.subheader("Tabla diaria")
table = series_forecast[["forecast_date", "horizon", "forecast_units", "forecast_lower_95", "forecast_upper_95", "source_model_used"]].rename(columns={"forecast_units": "forecast", "forecast_lower_95": "límite inferior", "forecast_upper_95": "límite superior", "source_model_used": "modelo"})
searchable_table(table, "forecast", "demand_forecast_table", limit=50)

st.subheader("Comparación rápida")
comp = series_forecast[["forecast_date", "forecast_units", "baseline_forecast"]].rename(columns={"forecast_units": "forecast oficial", "baseline_forecast": "seasonal_naive_28"})
comp = comp.melt("forecast_date", var_name="serie", value_name="unidades")
fig = px.line(comp, x="forecast_date", y="unidades", color="serie", markers=True, title="Forecast oficial vs seasonal_naive_28")
fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=50, b=20), yaxis_title="Unidades", xaxis_title="Fecha")
st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
st.caption("La comparación con baseline se muestra como referencia; el modelo complejo no se presenta como superior si no lo valida el holdout.")
