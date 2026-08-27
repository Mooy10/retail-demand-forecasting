from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.alerts import portfolio_note, simulation_warning
from dashboard.services.data_loader import load_assumptions_text

st.set_page_config(page_title="Methodology", layout="wide", initial_sidebar_state="auto")
styles = ROOT / "dashboard" / "assets" / "styles.css"
st.markdown(f"<style>{styles.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.title("Methodology")
portfolio_note()
simulation_warning()

st.markdown(
    """
### Alcance
Este dashboard resume la capa ejecutiva del proyecto BA-001. Usa únicamente artefactos procesados de las fases anteriores:
forecast oficial store-department, métricas de validación, incertidumbre empírica y simulación de inventario.

### Pipeline analítico
"""
)

st.markdown(
    """
```text
M5 dataset
  -> Agregación store-department
  -> Feature engineering temporal
  -> Backtesting rolling
  -> Selector híbrido
  -> Holdout W3
  -> Forecast oficial 28 días
        -> Incertidumbre empírica
        -> Simulación de inventario
        -> Dashboard ejecutivo
```
"""
)

cols = st.columns(2)
with cols[0]:
    st.markdown(
        """
### Decisiones clave
- Nivel de forecast inicial: `store_id + dept_id`, 70 series.
- Horizonte oficial: 28 días.
- Benchmark obligatorio: `seasonal_naive_28`.
- Validación oficial: holdout W3 fuera de muestra.
- Modelos candidatos: baselines, modelos ML y selector híbrido.
- Inventario: simulación basada en supuestos configurables.
"""
    )
with cols[1]:
    st.markdown(
        """
### Métricas
- **MAE**: error absoluto promedio.
- **RMSE**: penaliza errores grandes.
- **WAPE**: error absoluto ponderado por volumen.
- **RMSSE**: escala el error contra un benchmark naive.
- **Stockout units**: faltante proyectado simulado.
- **Total cost**: costo simulado de holding, faltante y órdenes.
"""
    )

st.markdown(
    """
### Interpretación del modelo oficial
El selector híbrido se considera candidato oficial porque fue evaluado con un holdout temporal no usado para seleccionar modelos.
Cuando la confianza del selector es baja o una serie no tiene predicción confiable, el pipeline conserva un fallback visible para evitar
ocultar incertidumbre al negocio.

### Limitaciones
- El dataset M5 contiene ventas históricas, precios y calendario, pero no inventario real disponible, órdenes de compra ni costos operativos reales.
- Las recomendaciones de inventario, costos y ahorros son simulaciones. Sirven para mostrar razonamiento analítico y no para representar decisiones reales de Walmart.
- La granularidad store-department reduce ruido y costo computacional, pero oculta variación producto-tienda.
- La incertidumbre se aproxima con errores históricos de backtesting; no es un intervalo estadístico formal.
"""
)

with st.expander("Supuestos de inventario usados por la simulación"):
    st.code(load_assumptions_text(), language="yaml")
