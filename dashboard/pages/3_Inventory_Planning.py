from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note, render_alerts, simulation_warning
from dashboard.components.charts import bar_chart, donut, inventory_projection, PLOTLY_CONFIG
from dashboard.components.filters import apply_dashboard_filters, global_filters
from dashboard.components.kpis import kpi_grid
from dashboard.components.tables import download_button, searchable_table
from dashboard.services.data_loader import (
    load_assumptions_text,
    load_forecast_uncertainty,
    load_inventory_daily_projection,
    load_inventory_recommendations,
    load_policy_comparison,
)
from dashboard.services.forecast_service import add_state_id
from dashboard.services.inventory_service import generate_alerts, inventory_kpis, scenario_frame
from dashboard.services.metrics_service import format_currency

st.set_page_config(page_title="Inventory Planning", layout="wide", initial_sidebar_state="auto")
styles = ROOT / "dashboard" / "assets" / "styles.css"
st.markdown(f"<style>{styles.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.title("Inventory Planning")
portfolio_note()
simulation_warning()

forecast = add_state_id(load_forecast_uncertainty())
inventory = add_state_id(load_inventory_recommendations())
daily = load_inventory_daily_projection()
policy = load_policy_comparison()

filters = global_filters(forecast, key_prefix="inventory")
scenario = filters.get("scenario") or "base"
filtered_forecast = apply_dashboard_filters(forecast, filters)
series_ids = filtered_forecast["unique_id"].drop_duplicates().tolist() if "unique_id" in filtered_forecast else []
filtered_inventory = inventory.loc[inventory["unique_id"].isin(series_ids)] if series_ids else inventory.iloc[0:0].copy()
filtered_daily = daily.loc[daily["unique_id"].isin(series_ids)] if series_ids else daily.iloc[0:0].copy()
scenario_inventory = scenario_frame(filtered_inventory, scenario)
scenario_daily = scenario_frame(filtered_daily, scenario)
scenario_policy = policy.loc[policy["scenario"].eq(scenario)].copy() if not policy.empty and "scenario" in policy else policy.iloc[0:0].copy()

kpis = inventory_kpis(filtered_inventory, scenario)
kpi_grid([
    {"label": "Inventario inicial", "value": kpis["initial_inventory"], "help_text": scenario, "decimals": 0},
    {"label": "Demanda lead time", "value": kpis["lead_time_demand"], "help_text": "Simulada", "decimals": 0},
    {"label": "Safety stock", "value": kpis["safety_stock"], "help_text": "Total", "decimals": 0},
    {"label": "Punto de reorden", "value": kpis["reorder_point"], "help_text": "Total", "decimals": 0},
    {"label": "Orden sugerida", "value": kpis["recommended_order"], "help_text": "Unidades", "decimals": 0},
    {"label": "Costo simulado", "value": kpis["total_cost"], "help_text": format_currency(kpis["total_cost"]), "decimals": 0},
    {"label": "Faltante proyectado", "value": kpis["stockout_units"], "help_text": "Unidades", "decimals": 0},
    {"label": "Exceso proyectado", "value": kpis["overstock_units"], "help_text": "Unidades", "decimals": 0},
    {"label": "Costo faltante", "value": kpis["stockout_cost"], "help_text": format_currency(kpis["stockout_cost"]), "decimals": 0},
    {"label": "Costo holding", "value": kpis["holding_cost"], "help_text": format_currency(kpis["holding_cost"]), "decimals": 0},
])

st.plotly_chart(inventory_projection(scenario_daily), width="stretch", config=PLOTLY_CONFIG)

cols = st.columns(3)
with cols[0]:
    if scenario_inventory.empty:
        st.plotly_chart(donut(pd.DataFrame(), "stockout_risk_level", "series", "Riesgo por serie"), width="stretch", config=PLOTLY_CONFIG)
    else:
        risk = scenario_inventory.groupby("stockout_risk_level", observed=True)["unique_id"].nunique().reset_index(name="series")
        st.plotly_chart(donut(risk, "stockout_risk_level", "series", "Riesgo simulado de faltante"), width="stretch", config=PLOTLY_CONFIG)
with cols[1]:
    orders = scenario_inventory.groupby("store_id", observed=True)["recommended_order_quantity"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(bar_chart(orders, "store_id", "recommended_order_quantity", "Orden recomendada por tienda"), width="stretch", config=PLOTLY_CONFIG)
with cols[2]:
    dept_cost = scenario_inventory.groupby("dept_id", observed=True)["estimated_total_inventory_cost"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(bar_chart(dept_cost, "dept_id", "estimated_total_inventory_cost", "Costo simulado por departamento"), width="stretch", config=PLOTLY_CONFIG)

if not scenario_policy.empty:
    policy_plot = scenario_policy.groupby("policy", observed=True)["total_cost"].sum().sort_values().reset_index()
    st.plotly_chart(bar_chart(policy_plot, "policy", "total_cost", "Costo por política de inventario"), width="stretch", config=PLOTLY_CONFIG)

st.subheader("Alertas operativas simuladas")
render_alerts(generate_alerts(scenario_inventory, filtered_forecast))

st.subheader("Recomendaciones por serie")
cols_to_show = [
    "unique_id",
    "store_id",
    "dept_id",
    "scenario",
    "forecast_demand_28d",
    "initial_inventory",
    "safety_stock",
    "reorder_point",
    "recommended_order_quantity",
    "projected_stockout_units",
    "overstock_units",
    "estimated_total_inventory_cost",
    "stockout_risk_level",
    "selected_forecast_model",
    "selector_confidence",
]
table = scenario_inventory[[col for col in cols_to_show if col in scenario_inventory.columns]].sort_values(
    ["stockout_risk_level", "recommended_order_quantity"], ascending=[True, False]
) if not scenario_inventory.empty else scenario_inventory
searchable_table(table, "inventory_recommendations", "inventory_recommendations_table", limit=80)
download_button(table, "Descargar recomendaciones filtradas", "inventory_recommendations_filtered.csv", "inventory_recommendations_filtered_download")

with st.expander("Supuestos configurables de la simulación"):
    st.code(load_assumptions_text(), language="yaml")
