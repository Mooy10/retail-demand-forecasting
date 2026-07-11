"""Reusable filters for dashboard pages."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.services.forecast_service import add_state_id


def option_list(values, all_label="Todos") -> list:
    clean = sorted(pd.Series(values).dropna().astype(str).unique().tolist())
    return [all_label] + clean


def global_filters(df: pd.DataFrame, key_prefix: str = "global", include_scenario: bool = True) -> dict:
    df = add_state_id(df) if not df.empty else df
    st.sidebar.markdown("### Filtros rápidos")
    states = option_list(df["state_id"] if "state_id" in df else [], "Todos")
    state = st.sidebar.selectbox("Estado", states, key=f"{key_prefix}_state")
    subset = df if state == "Todos" or "state_id" not in df else df.loc[df["state_id"].astype(str).eq(state)]
    stores = option_list(subset["store_id"] if "store_id" in subset else [], "Todos")
    store = st.sidebar.selectbox("Tienda", stores, key=f"{key_prefix}_store")
    subset = subset if store == "Todos" or "store_id" not in subset else subset.loc[subset["store_id"].astype(str).eq(store)]
    cats = option_list(subset["cat_id"] if "cat_id" in subset else [], "Todas")
    cat = st.sidebar.selectbox("Categoría", cats, key=f"{key_prefix}_cat")
    subset = subset if cat == "Todas" or "cat_id" not in subset else subset.loc[subset["cat_id"].astype(str).eq(cat)]
    depts = option_list(subset["dept_id"] if "dept_id" in subset else [], "Todos")
    dept = st.sidebar.selectbox("Departamento", depts, key=f"{key_prefix}_dept")
    model = st.sidebar.selectbox("Modelo usado", option_list(df.get("source_model_used", df.get("selected_model", [])), "Todos"), key=f"{key_prefix}_model")
    confidence = st.sidebar.selectbox("Confianza", option_list(df.get("selector_confidence", df.get("confidence", [])), "Todos"), key=f"{key_prefix}_confidence")
    fallback = st.sidebar.selectbox("Fallback", ["Todos", "Sí", "No"], key=f"{key_prefix}_fallback")
    scenario = st.sidebar.selectbox("Escenario de inventario", ["lean", "base", "conservative"], index=1, key=f"{key_prefix}_scenario") if include_scenario else None
    return {"state_id": state, "store_id": store, "cat_id": cat, "dept_id": dept, "source_model_used": model, "selector_confidence": confidence, "fallback_label": fallback, "scenario": scenario}


def apply_dashboard_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = add_state_id(df)
    mapping = {"state_id": filters.get("state_id"), "store_id": filters.get("store_id"), "cat_id": filters.get("cat_id"), "dept_id": filters.get("dept_id"), "source_model_used": filters.get("source_model_used"), "selector_confidence": filters.get("selector_confidence")}
    for col, val in mapping.items():
        if val in (None, "Todos", "Todas") or col not in out.columns:
            continue
        out = out.loc[out[col].astype(str).eq(str(val))]
    fb = filters.get("fallback_label")
    if fb == "Sí" and "fallback_used" in out.columns:
        out = out.loc[out["fallback_used"].astype(bool)]
    elif fb == "No" and "fallback_used" in out.columns:
        out = out.loc[~out["fallback_used"].astype(bool)]
    return out
