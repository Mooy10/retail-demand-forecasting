# BA-001 Dashboard Ejecutivo

Dashboard Streamlit para presentar el caso de negocio del proyecto **Retail Demand Forecasting con M5**.

> Proyecto analítico de portafolio construido con el dataset M5. Las cifras de inventario, costos y ahorros son simulaciones basadas en supuestos configurables y no representan operaciones reales de Walmart.

## Páginas

- `app.py`: portada ejecutiva, KPIs globales y navegación.
- `1_Executive_Overview.py`: resumen de forecast, performance e inventario.
- `2_Demand_Forecast.py`: detalle por serie store-department, incertidumbre y comparación contra `seasonal_naive_28`.
- `3_Inventory_Planning.py`: simulación de inventario, políticas, órdenes sugeridas y alertas.
- `4_Model_Performance.py`: métricas de holdout W3, rolling selector y distribución de modelos.
- `5_Business_Insights.py`: prioridades comerciales y recomendaciones accionables simuladas.
- `6_Methodology.py`: metodología, supuestos y limitaciones.

## Ejecución local

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```

## Dependencias de datos

El dashboard no carga archivos crudos ni entrena modelos. Consume únicamente artefactos procesados de fases previas:

- `data/processed/official_forecast_store_department.parquet`
- `data/processed/official_forecast_28d_summary.parquet`
- `data/processed/official_forecast_with_uncertainty.parquet`
- `data/processed/inventory_recommendations.parquet`
- `data/processed/inventory_daily_projection.parquet`
- `reports/holdout_w3_metrics_summary.csv`
- `reports/rolling_selector_metrics_summary.csv`
- `reports/inventory_policy_comparison.csv`
- `config/inventory_assumptions.yaml`
