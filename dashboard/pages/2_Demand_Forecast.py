from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note
from dashboard.components.charts import forecast_with_uncertainty, forecast_line
from dashboard.components.kpis import fallback_chip, kpi_card
from dashboard.components.tables import searchable_table
from dashboard.services.data_loader import load_forecast_uncertainty, load_holdout_metrics_by_series, load_recent_history
from dashboard.services.forecast_service import add_state_id

st.set_page_config(page_title="Demand Forecast", layout="wide")
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
cols = st.columns(7)
with cols[0]: kpi_card("Forecast total", series_forecast["forecast_units"].sum(), f"{horizon} días", decimals=1)
with cols[1]: kpi_card("Promedio diario", series_forecast["forecast_units"].mean(), "Unidades", decimals=1)
with cols[2]: kpi_card("Máximo diario", series_forecast["forecast_units"].max(), str(series_forecast.loc[series_forecast["forecast_units"].idxmax(), "forecast_date"].date()), decimals=1)
with cols[3]: kpi_card("Incertidumbre prom.", (series_forecast["forecast_upper_95"] - series_forecast["forecast_lower_95"]).mean(), "Banda 95%", decimals=1)
with cols[4]: kpi_card("WAPE serie", float(series_metrics["wape"].iloc[0]) * 100 if not series_metrics.empty else 0, "Holdout", decimals=2)
with cols[5]: kpi_card("RMSSE serie", float(series_metrics["rmsse"].iloc[0]) if not series_metrics.empty else 0, "Holdout", decimals=3)
with cols[6]: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Confianza</div><div class='kpi-value'>{row['selector_confidence']}</div><div class='kpi-help'>Selector</div></div>", unsafe_allow_html=True)

st.plotly_chart(forecast_with_uncertainty(series_history, series_forecast), width="stretch")

st.subheader("Tabla diaria")
table = series_forecast[["forecast_date", "horizon", "forecast_units", "forecast_lower_95", "forecast_upper_95", "source_model_used"]].rename(columns={"forecast_units": "forecast", "forecast_lower_95": "límite inferior", "forecast_upper_95": "límite superior", "source_model_used": "modelo"})
searchable_table(table, "forecast", "demand_forecast_table", limit=50)

st.subheader("Comparación rápida")
comp = series_forecast[["forecast_date", "forecast_units", "baseline_forecast"]].rename(columns={"forecast_units": "forecast oficial", "baseline_forecast": "seasonal_naive_28"})
comp = comp.melt("forecast_date", var_name="serie", value_name="unidades")
fig = px.line(comp, x="forecast_date", y="unidades", color="serie", markers=True, title="Forecast oficial vs seasonal_naive_28")
fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), yaxis_title="Unidades", xaxis_title="Fecha")
st.plotly_chart(fig, width="stretch")
st.caption("La comparación con baseline se muestra como referencia; el modelo complejo no se presenta como superior si no lo valida el holdout.")

