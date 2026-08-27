from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note
from dashboard.components.charts import bar_chart, donut, forecast_line, PLOTLY_CONFIG
from dashboard.components.filters import apply_dashboard_filters, global_filters
from dashboard.components.kpis import kpi_grid
from dashboard.components.tables import download_button, searchable_table
from dashboard.services.data_loader import load_holdout_metrics_summary, load_inventory_recommendations, load_official_forecast, load_policy_comparison, load_recent_history
from dashboard.services.forecast_service import forecast_by_dimension
from dashboard.services.inventory_service import executive_summary, scenario_frame
from dashboard.services.metrics_service import historical_daily_average, overview_kpis

st.set_page_config(page_title="Executive Overview", layout="wide", initial_sidebar_state="auto")
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
kpi_grid([
    {"label": "Promedio histórico diario", "value": historical_daily_average(filtered_history), "help_text": "Últimos 90 días", "decimals": 1},
    {"label": "Forecast 28 días", "value": kpis["forecast_28d"], "help_text": "Unidades", "decimals": 1},
    {"label": "WAPE validado", "value": kpis["wape"] * 100, "help_text": "Holdout W3", "decimals": 2},
    {"label": "Safety stock", "value": filtered_inventory.get("safety_stock", []).sum() if not filtered_inventory.empty else 0, "help_text": "Simulado", "decimals": 1},
    {"label": "Costo total", "value": filtered_inventory.get("estimated_total_inventory_cost", []).sum() if not filtered_inventory.empty else 0, "help_text": "Simulado", "currency": True},
    {"label": "RMSSE validado", "value": kpis["rmsse"], "help_text": "Holdout W3", "decimals": 3},
    {"label": "Reorder point", "value": filtered_inventory.get("reorder_point", []).sum() if not filtered_inventory.empty else 0, "help_text": "Simulado", "decimals": 1},
    {"label": "Órdenes recomendadas", "value": (filtered_inventory.get("recommended_order_quantity", []) > 0).sum() if not filtered_inventory.empty else 0, "help_text": scenario, "decimals": 0},
    {"label": "Ahorro simulado", "value": kpis["simulated_savings_vs_baseline"], "help_text": "vs baseline", "currency": True},
    {"label": "Fallback series", "value": filtered_forecast.loc[filtered_forecast.get("fallback_used", False).astype(bool), "unique_id"].nunique() if not filtered_forecast.empty else 0, "help_text": "Series", "decimals": 0},
])

st.subheader("Visualizaciones ejecutivas")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(forecast_line(filtered_forecast, "Forecast diario total próximos 28 días"), width="stretch", config=PLOTLY_CONFIG)
with col2:
    st.plotly_chart(bar_chart(forecast_by_dimension(filtered_forecast, "store_id"), "store_id", "forecast_units", "Forecast por tienda"), width="stretch", config=PLOTLY_CONFIG)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(bar_chart(forecast_by_dimension(filtered_forecast, "dept_id"), "dept_id", "forecast_units", "Forecast por departamento"), width="stretch", config=PLOTLY_CONFIG)
with col2:
    model_dist = filtered_forecast[["unique_id", "source_model_used"]].drop_duplicates()["source_model_used"].value_counts().reset_index()
    model_dist.columns = ["model", "series"] if not model_dist.empty else ["model", "series"]
    st.plotly_chart(donut(model_dist, "model", "series", "Distribución de modelos utilizados"), width="stretch", config=PLOTLY_CONFIG)
col1, col2 = st.columns(2)
with col1:
    conf = filtered_forecast[["unique_id", "selector_confidence"]].drop_duplicates()["selector_confidence"].value_counts().reset_index()
    conf.columns = ["confidence", "series"] if not conf.empty else ["confidence", "series"]
    st.plotly_chart(donut(conf, "confidence", "series", "Distribución de confianza"), width="stretch", config=PLOTLY_CONFIG)
with col2:
    risk = filtered_inventory["stockout_risk_level"].value_counts().reset_index() if not filtered_inventory.empty else filtered_inventory
    if not risk.empty: risk.columns = ["risk", "series"]
    st.plotly_chart(donut(risk, "risk", "series", "Distribución de riesgo de inventario"), width="stretch", config=PLOTLY_CONFIG)

st.subheader("Resumen ejecutivo automático")
st.markdown(f"<div class='rpfa-card'>{executive_summary(filtered_forecast, filtered_inventory, policy, scenario)}</div>", unsafe_allow_html=True)

st.subheader("Comparación de costos por política")
policy_filtered = policy.loc[policy["scenario"].eq(scenario)] if not policy.empty else policy
st.plotly_chart(bar_chart(policy_filtered.groupby("policy", observed=True)["total_cost"].sum().reset_index(), "policy", "total_cost", "Costo total simulado por política"), width="stretch", config=PLOTLY_CONFIG)
download_button(policy_filtered, "Descargar comparación de políticas", "policy_comparison_filtered.csv", "exec_policy")

st.subheader("Recomendaciones principales")
searchable_table(filtered_inventory.sort_values("recommended_order_quantity", ascending=False), "recomendaciones", "exec_recommendations", limit=100)
