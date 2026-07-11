"""Reusable KPI components."""

from __future__ import annotations

import streamlit as st

from dashboard.services.metrics_service import format_currency, format_number


def kpi_card(label: str, value, help_text: str = "", currency: bool = False, decimals: int = 1) -> None:
    formatted = format_currency(value) if currency else format_number(value, decimals)
    st.markdown(
        f"""
        <div class="rpfa-kpi">
          <div class="rpfa-kpi-label">{label}</div>
          <div class="rpfa-kpi-value">{formatted}</div>
          <div class="rpfa-kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_chip(confidence: str) -> str:
    key = str(confidence).lower()
    css = "rpfa-chip-high" if key == "high" else "rpfa-chip-medium" if key == "medium" else "rpfa-chip-low"
    return f'<span class="rpfa-chip {css}">{confidence}</span>'


def fallback_chip(fallback: bool) -> str:
    css = "rpfa-chip-risk" if fallback else "rpfa-chip-high"
    text = "Fallback" if fallback else "Modelo seleccionado"
    return f'<span class="rpfa-chip {css}">{text}</span>'
