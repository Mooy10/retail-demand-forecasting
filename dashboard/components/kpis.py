"""Reusable KPI components."""

from __future__ import annotations

import streamlit as st

from dashboard.services.metrics_service import format_currency, format_number


def kpi_card(label: str, value, help_text: str = "", currency: bool = False, decimals: int = 1) -> None:
    formatted = format_currency(value) if currency else str(value) if isinstance(value, str) else format_number(value, decimals)
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


def kpi_grid(items: list[dict], columns: int = 4) -> None:
    """Render KPI cards in balanced rows to avoid cramped wide layouts."""
    if not items:
        return
    for start in range(0, len(items), columns):
        row_items = items[start : start + columns]
        cols = st.columns(len(row_items))
        for col, item in zip(cols, row_items):
            with col:
                kpi_card(**item)


def confidence_chip(confidence: str) -> str:
    key = str(confidence).lower()
    css = "rpfa-chip-high" if key == "high" else "rpfa-chip-medium" if key == "medium" else "rpfa-chip-low"
    return f'<span class="rpfa-chip {css}">{confidence}</span>'


def fallback_chip(fallback: bool) -> str:
    css = "rpfa-chip-risk" if fallback else "rpfa-chip-high"
    text = "Fallback" if fallback else "Modelo seleccionado"
    return f'<span class="rpfa-chip {css}">{text}</span>'
