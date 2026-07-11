from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note
from dashboard.components.kpis import kpi_card
from dashboard.services.data_loader import load_all_core, load_assumptions_text
from dashboard.services.metrics_service import format_currency, format_number, overview_kpis

st.set_page_config(page_title="Retail Planning & Forecasting Analytics", page_icon="📊", layout="wide")

styles = Path(__file__).parent / "assets" / "styles.css"
if styles.exists():
    st.markdown(f"<style>{styles.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.title("Retail Planning & Forecasting Analytics")
st.markdown('<div class="rpfa-subtitle">Demand Forecasting, Inventory Simulation and Decision Support</div>', unsafe_allow_html=True)
portfolio_note()

data = load_all_core()
forecast = data["forecast"]
inventory = data["inventory"]
holdout = data["holdout"]
policy = data["policy"]

max_date = forecast["forecast_date"].max().date().isoformat() if not forecast.empty else "N/D"
series = forecast["unique_id"].nunique() if not forecast.empty else 0
st.markdown(f"**Fecha máxima disponible:** {max_date} · **Nivel analítico:** tienda + departamento · **Horizonte:** 28 días · **Series:** {series}")

kpis = overview_kpis(forecast, inventory, holdout, policy, "base")
cols = st.columns(7)
with cols[0]: kpi_card("Forecast 28 días", kpis["forecast_28d"], "Unidades", decimals=1)
with cols[1]: kpi_card("WAPE validado", kpis["wape"] * 100, "Holdout W3", decimals=2)
with cols[2]: kpi_card("RMSSE validado", kpis["rmsse"], "Holdout W3", decimals=3)
with cols[3]: kpi_card("Series fallback", kpis["fallback_series"], "Fallback visible", decimals=0)
with cols[4]: kpi_card("Órdenes base", kpis["recommended_orders"], "Escenario base", decimals=0)
with cols[5]: kpi_card("Costo simulado", kpis["simulated_total_cost"], "Escenario base", currency=True)
with cols[6]: kpi_card("Ahorro simulado", kpis["simulated_savings_vs_baseline"], "vs baseline", currency=True)

st.subheader("Navegación")
nav = [
    ("Executive Overview", "Visión ejecutiva de forecast, inventario y costos simulados."),
    ("Demand Forecast", "Exploración por tienda/departamento con incertidumbre empírica."),
    ("Inventory Planning", "Safety stock, ROP, EOQ y órdenes recomendadas por escenario."),
    ("Model Performance", "Comparación honesta entre baseline, ML e híbrido validado."),
    ("Business Insights", "Alertas y recomendaciones accionables para planeación."),
    ("Methodology", "Metodología, validación temporal y limitaciones."),
]
cols = st.columns(3)
for idx, (title, desc) in enumerate(nav):
    with cols[idx % 3]:
        st.markdown(f"<div class='rpfa-card'><h3>{title}</h3><p>{desc}</p></div>", unsafe_allow_html=True)

with st.expander("Supuestos de inventario simulados"):
    st.code(load_assumptions_text(), language="yaml")

st.markdown('<div class="rpfa-footer-note">No se usan logos ni identidad oficial de Walmart. No se reentrenan modelos desde la app.</div>', unsafe_allow_html=True)
