"""Inventory dashboard calculations and deterministic business rules."""

from __future__ import annotations

import pandas as pd


def scenario_frame(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if df.empty or "scenario" not in df.columns:
        return df.iloc[0:0].copy() if not df.empty else df
    return df.loc[df["scenario"].eq(scenario)].copy()


def inventory_kpis(inventory: pd.DataFrame, scenario: str = "base") -> dict[str, float]:
    df = scenario_frame(inventory, scenario)
    if df.empty:
        return {key: 0.0 for key in ["initial_inventory", "lead_time_demand", "safety_stock", "reorder_point", "order_up_to", "recommended_order", "stockout_units", "overstock_units", "holding_cost", "stockout_cost", "ordering_cost", "total_cost"]}
    return {
        "initial_inventory": float(df["initial_inventory"].sum()),
        "lead_time_demand": float(df.get("expected_demand_during_lead_time", pd.Series(dtype=float)).sum()),
        "safety_stock": float(df["safety_stock"].sum()),
        "reorder_point": float(df["reorder_point"].sum()),
        "order_up_to": float(df["order_up_to_level"].sum()),
        "recommended_order": float(df["recommended_order_quantity"].sum()),
        "stockout_units": float(df["projected_stockout_units"].sum()),
        "overstock_units": float(df["overstock_units"].sum()),
        "holding_cost": float(df["estimated_holding_cost"].sum()),
        "stockout_cost": float(df["estimated_stockout_cost"].sum()),
        "ordering_cost": float(df.get("estimated_ordering_cost", pd.Series(dtype=float)).sum()),
        "total_cost": float(df["estimated_total_inventory_cost"].sum()),
    }


def generate_alerts(inventory: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    alerts = []
    if not inventory.empty:
        base = scenario_frame(inventory, "base")
        if not base.empty:
            high_cost_threshold = base["estimated_total_inventory_cost"].quantile(0.90)
            high_order_threshold = base["recommended_order_quantity"].quantile(0.90)
            for _, row in base.iterrows():
                if row.get("stockout_risk_level") in {"High", "Critical"}:
                    alerts.append({"severity": row["stockout_risk_level"], "unique_id": row["unique_id"], "message": "Riesgo simulado de faltante elevado", "metric": row.get("projected_stockout_units", 0)})
                if row.get("recommended_order_quantity", 0) > high_order_threshold and row.get("recommended_order_quantity", 0) > 0:
                    alerts.append({"severity": "Medium", "unique_id": row["unique_id"], "message": "Orden recomendada por encima del percentil 90", "metric": row.get("recommended_order_quantity", 0)})
                if row.get("estimated_total_inventory_cost", 0) > high_cost_threshold:
                    alerts.append({"severity": "Medium", "unique_id": row["unique_id"], "message": "Costo total simulado por encima del percentil 90", "metric": row.get("estimated_total_inventory_cost", 0)})
                if row.get("overstock_units", 0) > 0:
                    alerts.append({"severity": "Low", "unique_id": row["unique_id"], "message": "Posible exceso de inventario simulado", "metric": row.get("overstock_units", 0)})
    if not forecast.empty:
        series = forecast[["unique_id", "fallback_used", "selector_confidence"]].drop_duplicates()
        for _, row in series.iterrows():
            if bool(row.get("fallback_used", False)):
                alerts.append({"severity": "Info", "unique_id": row["unique_id"], "message": "Forecast usa fallback visible", "metric": 1})
            if row.get("selector_confidence") == "Low":
                alerts.append({"severity": "Low", "unique_id": row["unique_id"], "message": "Confianza baja del selector", "metric": 1})
    return pd.DataFrame(alerts, columns=["severity", "unique_id", "message", "metric"])


def executive_summary(forecast: pd.DataFrame, inventory: pd.DataFrame, policy: pd.DataFrame, scenario: str = "base") -> str:
    if forecast.empty:
        return "No hay datos de forecast disponibles para generar el resumen ejecutivo."
    by_store = forecast.groupby("store_id", observed=True)["forecast_units"].sum().sort_values(ascending=False)
    by_dept = forecast.groupby("dept_id", observed=True)["forecast_units"].sum().sort_values(ascending=False)
    inv = scenario_frame(inventory, scenario)
    fallback_count = forecast.loc[forecast["fallback_used"].astype(bool), "unique_id"].nunique() if "fallback_used" in forecast else 0
    min_policy = "N/D"
    if not policy.empty:
        scen = policy.loc[policy["scenario"].eq(scenario)]
        if not scen.empty:
            min_policy = scen.groupby("policy", observed=True)["total_cost"].sum().sort_values().index[0]
    risk = inv["stockout_risk_level"].value_counts().idxmax() if not inv.empty and "stockout_risk_level" in inv else "N/D"
    return (
        f"La tienda con mayor demanda esperada es {by_store.index[0]} y el departamento prioritario es {by_dept.index[0]}. "
        f"El forecast usa fallback en {fallback_count} series. Bajo el escenario {scenario}, la política de menor costo simulado es {min_policy}. "
        f"El nivel de riesgo de inventario más frecuente es {risk}."
    )
