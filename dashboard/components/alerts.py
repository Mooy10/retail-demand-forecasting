"""Alert rendering components."""

from __future__ import annotations

import streamlit as st
import pandas as pd

SIMULATION_WARNING = "Estas recomendaciones no consideran inventario real, capacidad, caducidad, proveedores, mínimos de compra, entregas en tránsito ni restricciones logísticas. Son una simulación educativa basada en supuestos configurables."
PORTFOLIO_NOTE = "Proyecto analítico de portafolio construido con el dataset M5. Las cifras de inventario, costos y ahorros son simulaciones basadas en supuestos configurables y no representan operaciones reales de Walmart."


def portfolio_note() -> None:
    st.markdown(f'<div class="rpfa-note">{PORTFOLIO_NOTE}</div>', unsafe_allow_html=True)


def simulation_warning() -> None:
    st.warning(SIMULATION_WARNING)


def render_alerts(alerts: pd.DataFrame) -> None:
    if alerts.empty:
        st.success("No hay alertas con los filtros actuales.")
        return
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}
    out = alerts.copy()
    out["_order"] = out["severity"].map(severity_order).fillna(9)
    st.dataframe(out.sort_values(["_order", "metric"], ascending=[True, False]).drop(columns=["_order"]).head(200), width="stretch", hide_index=True)

