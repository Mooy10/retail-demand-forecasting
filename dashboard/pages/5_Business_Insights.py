from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note, render_alerts, simulation_warning
from dashboard.components.charts import bar_chart, donut, PLOTLY_CONFIG
from dashboard.components.filters import apply_dashboard_filters, global_filters
from dashboard.components.kpis import kpi_grid
from dashboard.components.tables import download_button, searchable_table
from dashboard.services.data_loader import load_forecast_uncertainty, load_inventory_recommendations, load_policy_comparison
from dashboard.services.forecast_service import add_state_id
from dashboard.services.inventory_service import executive_summary, generate_alerts, scenario_frame
from dashboard.services.metrics_service import format_currency

st.set_page_config(page_title="Business Insights", layout="wide", initial_sidebar_state="auto")
styles = ROOT / "dashboard" / "assets" / "styles.css"
st.markdown(f"<style>{styles.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.title("Business Insights")
portfolio_note()
simulation_warning()

forecast = add_state_id(load_forecast_uncertainty())
inventory = add_state_id(load_inventory_recommendations())
policy = load_policy_comparison()

filters = global_filters(forecast, key_prefix="insights")
scenario = filters.get("scenario") or "base"
filtered_forecast = apply_dashboard_filters(forecast, filters)
series_ids = filtered_forecast["unique_id"].drop_duplicates().tolist() if "unique_id" in filtered_forecast else []
filtered_inventory = inventory.loc[inventory["unique_id"].isin(series_ids)] if series_ids else inventory.iloc[0:0].copy()
scenario_inventory = scenario_frame(filtered_inventory, scenario)
scenario_policy = policy.loc[policy["scenario"].eq(scenario)].copy() if not policy.empty and "scenario" in policy else policy.iloc[0:0].copy()

st.markdown("### Lectura ejecutiva")
st.write(executive_summary(filtered_forecast, filtered_inventory, scenario_policy, scenario))

if filtered_forecast.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

by_store = filtered_forecast.groupby("store_id", observed=True)["forecast_units"].sum().sort_values(ascending=False)
by_dept = filtered_forecast.groupby("dept_id", observed=True)["forecast_units"].sum().sort_values(ascending=False)
fallback_series = filtered_forecast.loc[filtered_forecast["fallback_used"].astype(bool), "unique_id"].nunique() if "fallback_used" in filtered_forecast else 0
low_confidence = filtered_forecast.loc[filtered_forecast["selector_confidence"].eq("Low"), "unique_id"].nunique() if "selector_confidence" in filtered_forecast else 0
orders = scenario_inventory["recommended_order_quantity"].sum() if not scenario_inventory.empty else 0
cost = scenario_inventory["estimated_total_inventory_cost"].sum() if not scenario_inventory.empty else 0

kpi_grid([
    {"label": "Tienda prioritaria", "value": by_store.iloc[0], "help_text": by_store.index[0], "decimals": 0},
    {"label": "Depto. prioritario", "value": by_dept.iloc[0], "help_text": by_dept.index[0], "decimals": 0},
    {"label": "Series con fallback", "value": fallback_series, "help_text": "Revisión", "decimals": 0},
    {"label": "Confianza baja", "value": low_confidence, "help_text": "Series", "decimals": 0},
    {"label": "Orden sugerida", "value": orders, "help_text": scenario, "decimals": 0},
    {"label": "Costo simulado", "value": cost, "help_text": format_currency(cost), "decimals": 0},
])

cols = st.columns(2)
with cols[0]:
    store_plot = by_store.reset_index(name="forecast_units")
    st.plotly_chart(bar_chart(store_plot, "store_id", "forecast_units", "Demanda esperada por tienda"), width="stretch", config=PLOTLY_CONFIG)
with cols[1]:
    dept_plot = by_dept.reset_index(name="forecast_units")
    st.plotly_chart(bar_chart(dept_plot, "dept_id", "forecast_units", "Demanda esperada por departamento"), width="stretch", config=PLOTLY_CONFIG)

cols = st.columns(2)
with cols[0]:
    model = filtered_forecast[["unique_id", "source_model_used"]].drop_duplicates().groupby("source_model_used")["unique_id"].nunique().reset_index(name="series")
    st.plotly_chart(donut(model, "source_model_used", "series", "Dependencia de modelos por serie"), width="stretch", config=PLOTLY_CONFIG)
with cols[1]:
    if scenario_inventory.empty:
        risk = pd.DataFrame()
    else:
        risk = scenario_inventory.groupby("stockout_risk_level", observed=True)["unique_id"].nunique().reset_index(name="series")
    st.plotly_chart(donut(risk, "stockout_risk_level", "series", "Riesgo de inventario simulado"), width="stretch", config=PLOTLY_CONFIG)

st.subheader("Alertas accionables")
alerts = generate_alerts(scenario_inventory, filtered_forecast)
render_alerts(alerts)
download_button(alerts, "Descargar alertas", "business_alerts.csv", "business_alerts_download")

st.subheader("Prioridades de compra")
priority_cols = [
    "unique_id",
    "store_id",
    "dept_id",
    "scenario",
    "forecast_demand_28d",
    "recommended_order_quantity",
    "projected_stockout_units",
    "estimated_total_inventory_cost",
    "stockout_risk_level",
    "selector_confidence",
]
priority = scenario_inventory[[col for col in priority_cols if col in scenario_inventory.columns]].copy()
if not priority.empty:
    priority = priority.sort_values(["recommended_order_quantity", "projected_stockout_units"], ascending=False)
searchable_table(priority, "business_priorities", "business_priorities_table", limit=80)
download_button(priority, "Descargar prioridades", "business_priorities.csv", "business_priorities_download")

st.subheader("Comparación de escenarios")
scenario_compare = inventory.groupby("scenario", observed=True).agg(
    recommended_order_quantity=("recommended_order_quantity", "sum"),
    projected_stockout_units=("projected_stockout_units", "sum"),
    overstock_units=("overstock_units", "sum"),
    estimated_total_inventory_cost=("estimated_total_inventory_cost", "sum"),
).reset_index() if not inventory.empty else pd.DataFrame()
st.dataframe(scenario_compare, width="stretch", hide_index=True)
