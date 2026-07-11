from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note
from dashboard.components.charts import bar_chart, donut, forecast_line
from dashboard.components.filters import apply_dashboard_filters, global_filters
from dashboard.components.kpis import kpi_card
from dashboard.components.tables import download_button, searchable_table
from dashboard.services.data_loader import load_holdout_metrics_summary, load_inventory_recommendations, load_official_forecast, load_policy_comparison, load_recent_history
from dashboard.services.forecast_service import forecast_by_dimension
from dashboard.services.inventory_service import executive_summary, scenario_frame
from dashboard.services.metrics_service import historical_daily_average, overview_kpis

st.set_page_config(page_title="Executive Overview", layout="wide")
styles = ROOT / "dashboard" / "assets" / "styles.css"
st.markdown(f"<style>{styles.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
st.title("Executive Overview")
portfolio_note()

forecast = load_official_forecast()
inventory = load_inventory_recommendations()
holdout = load_holdout_metrics_summary()
policy = load_policy_comparison()
history = load_recent_history()
filters = global_filters(forecast, "exec")
scenario = filters.get("scenario", "base")
filtered_forecast = apply_dashboard_filters(forecast, filters)
filtered_inventory = apply_dashboard_filters(scenario_frame(inventory, scenario), filters)
filtered_history = apply_dashboard_filters(history, filters)

kpis = overview_kpis(filtered_forecast, filtered_inventory, holdout, policy, scenario)
cols = st.columns(5)
with cols[0]: kpi_card("Promedio histórico diario", historical_daily_average(filtered_history), "Últimos 90 días", decimals=1)
with cols[1]: kpi_card("Forecast 28 días", kpis["forecast_28d"], "Unidades", decimals=1)
with cols[2]: kpi_card("WAPE validado", kpis["wape"] * 100, "Holdout W3", decimals=2)
with cols[3]: kpi_card("Safety stock", filtered_inventory.get("safety_stock", []).sum() if not filtered_inventory.empty else 0, "Simulado", decimals=1)
with cols[4]: kpi_card("Costo total", filtered_inventory.get("estimated_total_inventory_cost", []).sum() if not filtered_inventory.empty else 0, "Simulado", currency=True)
cols = st.columns(5)
with cols[0]: kpi_card("RMSSE validado", kpis["rmsse"], "Holdout W3", decimals=3)
with cols[1]: kpi_card("Reorder point", filtered_inventory.get("reorder_point", []).sum() if not filtered_inventory.empty else 0, "Simulado", decimals=1)
with cols[2]: kpi_card("Órdenes recomendadas", (filtered_inventory.get("recommended_order_quantity", []) > 0).sum() if not filtered_inventory.empty else 0, scenario, decimals=0)
with cols[3]: kpi_card("Ahorro simulado", kpis["simulated_savings_vs_baseline"], "vs baseline", currency=True)
with cols[4]: kpi_card("Fallback series", filtered_forecast.loc[filtered_forecast.get("fallback_used", False).astype(bool), "unique_id"].nunique() if not filtered_forecast.empty else 0, "Series", decimals=0)

st.subheader("Visualizaciones ejecutivas")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(forecast_line(filtered_forecast, "Forecast diario total próximos 28 días"), width="stretch")
with col2:
    st.plotly_chart(bar_chart(forecast_by_dimension(filtered_forecast, "store_id"), "store_id", "forecast_units", "Forecast por tienda"), width="stretch")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(bar_chart(forecast_by_dimension(filtered_forecast, "dept_id"), "dept_id", "forecast_units", "Forecast por departamento"), width="stretch")
with col2:
    model_dist = filtered_forecast[["unique_id", "source_model_used"]].drop_duplicates()["source_model_used"].value_counts().reset_index()
    model_dist.columns = ["model", "series"] if not model_dist.empty else ["model", "series"]
    st.plotly_chart(donut(model_dist, "model", "series", "Distribución de modelos utilizados"), width="stretch")
col1, col2 = st.columns(2)
with col1:
    conf = filtered_forecast[["unique_id", "selector_confidence"]].drop_duplicates()["selector_confidence"].value_counts().reset_index()
    conf.columns = ["confidence", "series"] if not conf.empty else ["confidence", "series"]
    st.plotly_chart(donut(conf, "confidence", "series", "Distribución de confianza"), width="stretch")
with col2:
    risk = filtered_inventory["stockout_risk_level"].value_counts().reset_index() if not filtered_inventory.empty else filtered_inventory
    if not risk.empty: risk.columns = ["risk", "series"]
    st.plotly_chart(donut(risk, "risk", "series", "Distribución de riesgo de inventario"), width="stretch")

st.subheader("Resumen ejecutivo automático")
st.markdown(f"<div class='rpfa-card'>{executive_summary(filtered_forecast, filtered_inventory, policy, scenario)}</div>", unsafe_allow_html=True)

st.subheader("Comparación de costos por política")
policy_filtered = policy.loc[policy["scenario"].eq(scenario)] if not policy.empty else policy
st.plotly_chart(bar_chart(policy_filtered.groupby("policy", observed=True)["total_cost"].sum().reset_index(), "policy", "total_cost", "Costo total simulado por política"), width="stretch")
download_button(policy_filtered, "Descargar comparación de políticas", "policy_comparison_filtered.csv", "exec_policy")

st.subheader("Recomendaciones principales")
searchable_table(filtered_inventory.sort_values("recommended_order_quantity", ascending=False), "recomendaciones", "exec_recommendations", limit=100)

