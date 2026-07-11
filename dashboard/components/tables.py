"""Downloadable table components."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.services.metrics_service import dataframe_to_csv_bytes


def searchable_table(df: pd.DataFrame, label: str, key: str, limit: int = 500) -> pd.DataFrame:
    if df.empty:
        st.info("No hay registros para los filtros seleccionados.")
        return df
    query = st.text_input("Buscar", key=f"{key}_search", placeholder="Buscar en tabla...")
    out = df.copy()
    if query:
        mask = out.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
        out = out.loc[mask]
    st.dataframe(out.head(limit), width="stretch", hide_index=True)
    st.download_button(f"Descargar {label} CSV", dataframe_to_csv_bytes(out), file_name=f"{key}.csv", mime="text/csv", key=f"{key}_download")
    return out


def download_button(df: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    st.download_button(label, dataframe_to_csv_bytes(df), file_name=file_name, mime="text/csv", key=key, disabled=df.empty)

