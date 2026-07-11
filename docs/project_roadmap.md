# Project Roadmap

## Phase 1: Setup

Create the professional repository structure, configuration files, documentation skeleton, dependency list, and GitHub-ready base.

## Phase 2: Dataset Download

Download the M5 Forecasting Accuracy dataset from Kaggle into `data/raw` using the Kaggle API. Validate that all expected files are present.

## Phase 3: Data Understanding

Inspect the grain, dimensions, date ranges, table relationships, missing values, and business meaning of each dataset file. Generate memory-conscious Parquet aggregates for total demand, demand by hierarchy level, product summaries, and item-store series summaries without melting the full 58 million-row sales matrix.

## Phase 4: Demand Segmentation

Classify item-store series by demand behavior, volume, and variability using Syntetos-Boylan, ABC, and XYZ segmentation. Use these results to decide the best starting aggregation level for forecasting before building lags or models.

## Phase 5: Baseline Forecasting

Prepare compact forecasting datasets, run temporal backtesting windows, evaluate baseline forecasting models, and establish benchmarks before ML.

## Phase 6: SQL Analytics

Use DuckDB SQL to answer business questions about sales by product, store, category, department, state, date, event, and price behavior.

## Phase 7: Exploratory Data Analysis

Analyze demand trends, seasonality, intermittent demand, zero-sales behavior, category differences, store differences, state-level patterns, and event effects.

## Phase 8: Feature Engineering

Create calendar features, lag features, rolling statistics, price features, event indicators, SNAP indicators, hierarchy encodings, and target variables for forecasting.

## Phase 9: Baseline Forecasting

Build simple baseline models such as last-observation forecast, moving average forecast, and seasonal naive forecast. These baselines will define the minimum performance standard.

## Phase 10: Machine Learning Models

Train models using scikit-learn and advanced gradient boosting methods such as XGBoost and LightGBM.

## Phase 11: Evaluation

Evaluate models with MAE, RMSE, and MAPE using temporal validation. Compare model performance by product, store, category, and state.

## Phase 12: Dashboard

Build a Streamlit dashboard to explore actual vs. predicted demand, model errors, product rankings, store-level patterns, and business recommendations.

## Phase 13: Executive Report

Create an executive summary with business insights, model results, inventory recommendations, limitations, and next steps.

## Phase 14: GitHub Preparation

Finalize README, documentation, reproducibility instructions, project structure, sample figures, and clean repository hygiene for portfolio publication.


## Phase 6: Machine Learning Forecasting

Build store-department ML features, train lightweight global direct multi-horizon models, compare against `seasonal_naive_28`, generate interpretability artifacts, and document whether ML truly improves the baseline.

## Phase 7 - Advanced Features, Model Selection And Hybrid Forecast

Commands:

```powershell
python src\advanced_feature_engineering.py
python src\run_advanced_ml_forecasting.py
python src\model_selector.py
python src\build_hybrid_forecast.py
jupyter notebook notebooks\05_model_selection.ipynb
python -m pytest tests -p no:cacheprovider
```

Outputs include advanced features, Phase 7 XGBoost/LightGBM backtesting, per-series model selection, hybrid forecasts, and final forecasting decision reports.
## Phase 8 - Out-of-Sample Model Selection Validation

This phase audits the Phase 7 hybrid result and separates three concepts:

- `in-sample model selection`: model selection and evaluation use the same backtesting windows.
- `holdout model selection`: the selector is trained on windows 1-2 and evaluated only on window 3.
- `rolling-origin model selection`: the selector is trained only on windows available before each evaluation window.

Commands:

```powershell
python src\run_out_of_sample_model_selection.py
jupyter notebook notebooks\06_out_of_sample_model_validation.ipynb
python -m pytest tests -p no:cacheprovider -q
```

Main outputs:

- `reports/model_selector_leakage_audit.md`
- `reports/out_of_sample_model_selection_summary.md`
- `reports/out_of_sample_forecasting_decision.md`
- `data/processed/model_selection_registry_train_w1_w2.parquet`
- `data/processed/hybrid_predictions_holdout_w3.parquet`
- `data/processed/rolling_hybrid_predictions.parquet`
## Phase 9 - Simulated Inventory Optimization

This phase turns the validated out-of-sample forecast into a simulated inventory planning layer. M5 does not provide real inventory, open orders, logistics costs, supplier lead times, or stockout costs, so all inventory quantities and financial impact estimates are configurable simulation outputs.

Commands:

```powershell
python src\build_official_forecast.py
python src\forecast_uncertainty.py
python src\inventory_simulation.py
python src\inventory_optimization.py
python src\inventory_economic_analysis.py
jupyter notebook notebooks\07_inventory_optimization.ipynb
python -m pytest tests -p no:cacheprovider -q
```

Main outputs:

- `data/processed/official_forecast_store_department.parquet`
- `data/processed/official_forecast_with_uncertainty.parquet`
- `data/processed/inventory_initial_scenarios.parquet`
- `data/processed/inventory_recommendations.parquet`
- `data/processed/inventory_daily_projection.parquet`
- `reports/inventory_optimization_summary.md`
- `reports/executive_inventory_recommendations.md`
## Phase 10 - Executive Streamlit Dashboard

Build the final executive dashboard layer for portfolio presentation. The dashboard consumes processed outputs from forecasting, holdout validation, forecast uncertainty, and simulated inventory optimization.

Commands:

```powershell
.\.venv\Scripts\streamlit.exe run dashboard\app.py
.\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider -q
```

Main surfaces:

- Executive Overview
- Demand Forecast
- Inventory Planning
- Model Performance
- Business Insights
- Methodology

Validation expectations:

- Pages load without runtime errors.
- Filters update visible KPIs, charts, and tables.
- Lean, base, and conservative scenarios change inventory outputs.
- CSV downloads are available for executive tables.
- Simulated-data warnings are visible on inventory and business pages.
## Phase 11 - GitHub Portfolio Release Preparation

Prepare the completed project for public portfolio use. This phase adds the final README, case study, executive summary, architecture diagrams, dashboard screenshots, reproducibility guide, release checklist, license, contribution notes, security notes, and GitHub audit. It does not train new models or change analytical results.

