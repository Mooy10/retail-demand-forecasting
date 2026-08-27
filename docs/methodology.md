# Methodology

This project follows a professional analytics engineering and data science workflow for retail demand forecasting.

## Planned Approach

1. Establish a clean repository structure with reproducible configuration.
2. Download the real M5 Walmart dataset from Kaggle in a later phase.
3. Validate raw data files, keys, date ranges, and table relationships.
4. Create clean analytical datasets from sales, calendar, and price tables.
5. Use DuckDB SQL for repeatable analytical transformations and business queries.
6. Perform exploratory data analysis to understand demand patterns and data quality.
7. Build forecasting features such as lags, rolling statistics, calendar variables, price features, and hierarchy identifiers.
8. Establish baseline forecasts before training advanced models.
9. Train machine learning models using temporal validation.
10. Evaluate model performance with MAE, RMSE, and MAPE.
11. Create visualizations and a Streamlit dashboard for business users.
12. Document conclusions, limitations, and recommendations in an executive format.

## Modeling Scope

Models will not be trained during the setup phase. Later phases will include baseline methods, scikit-learn models, and advanced gradient boosting models such as XGBoost and LightGBM.

## Validation Principle

Because this is a forecasting problem, evaluation will use time-based validation rather than random train-test splits. This prevents future information from leaking into model training.

## Phase 3 Data Understanding Methodology

The M5 validation sales table contains 30,490 item-store series and 1,913 daily demand columns. A full melt would create roughly 58 million rows, so Phase 3 avoids that transformation.

Instead, the pipeline:

- Loads the sales matrix once using compact integer dtypes for demand columns.
- Uses vectorized column sums to create total daily demand.
- Uses groupby operations across the wide demand matrix to create compact daily aggregates by state, store, category, and department.
- Stacks only the small aggregated matrices, not the original raw sales matrix.
- Creates product and item-store summaries with row-wise vectorized sums.
- Explicitly deletes intermediate grouped objects and requests garbage collection after memory-sensitive steps.

The output of this phase is a set of Parquet analytical tables used by the notebook and future SQL/EDA phases.

## Phase 4 Demand Segmentation Methodology

Phase 4 classifies each item-store series without creating the full 58 million-row long table.

Demand pattern classification uses the Syntetos-Boylan framework:

- Smooth: ADI < 1.32 and CV² < 0.49.
- Erratic: ADI < 1.32 and CV² >= 0.49.
- Intermittent: ADI >= 1.32 and CV² < 0.49.
- Lumpy: ADI >= 1.32 and CV² >= 0.49.

ADI is calculated as the number of historical days divided by the number of days with positive demand. CV² is calculated on non-zero demand values only.

ABC classes are assigned by sorting item-store series by total demand and calculating cumulative demand contribution:

- A: series contributing approximately the first 80% of demand.
- B: next 15% of demand.
- C: remaining 5% of demand.

XYZ classes use the full-series coefficient of variation:

- X: CV < 0.50, lower variability.
- Y: 0.50 <= CV < 1.00, medium variability.
- Z: CV >= 1.00, high variability.

These thresholds are intentionally simple and transparent for a portfolio project. In a production setting, they should be reviewed with demand planners and inventory stakeholders.
## Phase 5 Baseline Forecasting Methodology

Baseline forecasting is evaluated at two levels:

- Store-department: 70 aggregate series from 10 stores and 7 departments.
- Selected SKU-store: up to 100 class-A item-store series selected by highest demand within Smooth, Erratic, Intermittent, and Lumpy patterns.

Backtesting uses three non-shuffled temporal windows with a 28-day horizon. Forecasts are produced only from training history available before each cutoff.

Metrics include MAE, RMSE, WAPE, RMSSE, and sMAPE. WAPE is emphasized over MAPE because many actual demand values are zero. RMSSE is scaled by the mean squared first difference of the training history for each series and cutoff. If the scale is zero, RMSSE is zero only for perfect forecasts and missing otherwise to avoid infinite values.

This phase does not implement the full official M5 WRMSSE hierarchy. That is reserved for a later evaluation phase.
## Phase 6 Machine Learning Methodology

Phase 6 trains global direct multi-horizon ML models at store-department level only. The run keeps 70 series, three backtesting windows, and the full 28-day horizon. To avoid blocking the workstation, training uses the latest 180 origin days per cutoff. This is a methodological limitation and should be revisited in a larger compute environment.

Models are HistGradientBoostingRegressor, XGBoost Regressor, and LightGBM Regressor with one lightweight base configuration each. XGBoost and LightGBM use `n_jobs=2`. No extensive hyperparameter search is performed in the lightweight run.

Anti-leakage rules:

- Demand lags and rolling statistics are shifted before use.
- Validation uses only features available at the cutoff/origin date.
- Target calendar fields are allowed because future calendar dates and events are known in advance.
- Price features are aggregated by store-department-week without demand weighting.

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
## Phase 10 - Dashboard Methodology

The dashboard is a presentation and decision-support layer. It does not train models, rebuild raw datasets, or generate new forecasts. It reads only validated processed artifacts from previous phases.

Design principles:

- Use Spanish executive language for the user interface.
- Separate validated model performance from simulated inventory impact.
- Keep `seasonal_naive_28` visible as the benchmark and fallback reference.
- Surface fallback usage and selector confidence instead of hiding model uncertainty.
- Show the portfolio and simulation warning on every executive surface where business impact is discussed.

The dashboard includes six pages: Executive Overview, Demand Forecast, Inventory Planning, Model Performance, Business Insights, and Methodology. Inventory cost, savings, reorder points, safety stock, and order quantities remain simulated outputs based on configurable assumptions, not real Walmart operations.
## Phase 11 - Release Methodology

The release phase audits the repository for large files, generated data, credentials, local paths, caches, and GitHub readiness. It separates publishable project assets from local reproducibility artifacts. The final README and case study are designed for recruiters, managers, and technical reviewers.

