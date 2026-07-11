# Demo Script

This is a 60-90 second Spanish narration script for a portfolio video.

Hola, este es mi proyecto Retail Planning & Forecasting Analytics, construido con el dataset publico M5.

El objetivo fue crear un caso empresarial completo de forecast de demanda para retail, no solo un notebook. El proyecto pronostica 28 dias de demanda para 70 series tienda-departamento y compara baselines, machine learning y un selector hibrido validado fuera de muestra.

En el dashboard se puede ver el forecast oficial, el WAPE de holdout, el RMSSE, las series donde se usa fallback y los escenarios simulados de inventario.

Una decision importante fue no forzar que machine learning ganara. El benchmark seasonal naive se mantiene visible y el forecast oficial se elige con validacion temporal.

La capa de inventario convierte el forecast en safety stock, punto de reorden, orden recomendada, riesgo de faltante y costo simulado. Estas cifras son simulaciones basadas en supuestos, no datos reales de Walmart.

El proyecto incluye pipeline reproducible, documentacion, pruebas automatizadas, dashboard ejecutivo y recomendaciones de negocio. Es una muestra de como conecto ciencia de datos, analytics engineering y comunicacion ejecutiva.
