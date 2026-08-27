# Final Portfolio Audit

## Executive Summary

This public audit summarizes the final portfolio-readiness review for the Retail Planning & Forecasting Analytics project. The project uses the public M5 Forecasting Accuracy dataset to build an end-to-end demand forecasting, model validation, inventory simulation, and executive dashboard workflow.

The review found a strong business case, a coherent analytical pipeline, clear distinction between measured forecast performance and simulated inventory impact, and a defensible final validation setup. The final public version should be presented as a portfolio project using public historical data, not as an official Walmart solution or a production replenishment system.

## Final Verdict

READY WITH MINOR FIXES

The project is ready for CV, LinkedIn, GitHub, and interviews after the final dashboard polish and documentation updates are committed and pushed. The main remaining caveat is reproducibility: large raw and processed data artifacts are intentionally excluded from GitHub, so a fresh clone must run the pipeline after configuring Kaggle access.

## Score

Portfolio Readiness: 88/100

| Area | Score |
|---|---:|
| Methodology | 18/20 |
| Technical implementation | 17/20 |
| Business relevance | 14/15 |
| Dashboard | 13/15 |
| Documentation | 9/10 |
| Reproducibility | 8/10 |
| GitHub presentation | 9/10 |

## Key Strengths

- Clear business framing around demand planning, inventory risk, purchasing, and commercial planning.
- Public dataset usage is transparent and properly separated from real company operations.
- Forecasting is validated with temporal backtesting instead of random train/test splits.
- The final claim uses strict holdout W3 validation rather than the optimistic in-sample selector result.
- `seasonal_naive_28` remains visible as a serious benchmark; machine learning is not forced to win.
- Inventory costs, stockouts, reorder points, and savings are labeled as simulated.
- Streamlit dashboard provides executive pages for forecast, uncertainty, inventory scenarios, model performance, business insights, and methodology.
- Automated tests cover metrics, feature engineering, model selection, official forecast, inventory logic, and dashboard services.

## Limitations

- The official forecasting layer uses 70 `store_id + dept_id` series, not the full item-store hierarchy.
- Advanced ML training uses the latest 180 origin days per cutoff to control local runtime and memory; this may underuse older seasonality.
- Only three backtesting windows are used, so results should be interpreted as a strong portfolio validation, not a production monitoring program.
- Inventory and economic results are simulations based on configurable assumptions; no real on-hand inventory, open orders, lead times, supplier constraints, unit costs, margins, or real stockout events are included.
- Generated raw data, processed Parquet files, model binaries, and report CSV outputs are intentionally excluded from GitHub.

## Methodology Audit

The methodology is appropriate for a professional portfolio project. The pipeline moves from raw M5 files to data understanding, demand segmentation, forecasting dataset creation, baseline forecasting, ML forecasting, advanced features, model selection, out-of-sample validation, official forecast generation, uncertainty estimation, inventory simulation, economic comparison, and dashboard communication.

The selected business grain, `store_id + dept_id`, is reasonable for an initial executive forecasting layer because it balances interpretability, memory use, runtime, and business relevance.

## Leakage Audit

Result: PASS with controlled warnings.

Demand lag and rolling features are shifted by series before being used for training. Calendar features are treated as known future information, which is appropriate for retail forecasting. The public, defensible result uses holdout W3 validation: models are selected using windows 1 and 2, then evaluated on window 3.

The earlier Phase 7 hybrid selector result is correctly treated as an in-sample diagnostic because model selection and evaluation used the same windows. It should not be presented as the main performance claim.

## Metrics Audit

The primary published holdout metrics are defensible:

| Metric | Value | Context |
|---|---:|---|
| Holdout WAPE | 0.115443 | `hybrid_holdout_w3`, weighted by actual volume |
| Holdout RMSSE | 0.762015 | `hybrid_holdout_w3`, scaled by historical series variation |
| Baseline WAPE | 0.125912 | `seasonal_naive_28` on the same holdout |
| Baseline RMSSE | 0.857261 | `seasonal_naive_28` on the same holdout |
| WAPE improvement | 8.31% | Hybrid holdout vs baseline |
| RMSSE improvement | 11.11% | Hybrid holdout vs baseline |

WAPE is aggregated with actual-volume weighting, not as a simple average of per-series WAPE. This is important for business interpretation.

## Hybrid Selector Audit

The hybrid selector chooses among baseline and ML candidates per series using previous validation evidence. It includes conservative fallback behavior when the simple seasonal benchmark is close enough or confidence is low. This is a strength because it reflects how demand planning teams often prefer robust, explainable choices over unnecessary model complexity.

The claim to make publicly is: the final hybrid forecast was validated on a strict holdout window and outperformed `seasonal_naive_28` on weighted WAPE and RMSSE in that setup.

## Inventory Audit

The inventory layer is valid as a simulation module. It translates the official forecast and empirical uncertainty into planning outputs such as safety stock, reorder point, order-up-to level, recommended order quantity, projected stockouts, and estimated cost.

All cost, savings, stockout, and order recommendations must remain described as simulated. The project should not claim real savings or real Walmart inventory decisions.

## Dashboard Audit

The dashboard was polished for public presentation. Public screenshots were refreshed with the corrected layout and should be used by the README. The dashboard should be described as a local Streamlit executive dashboard that reads generated artifacts and does not retrain models from the UI.

Validated dashboard pages:

- Home
- Executive Overview
- Demand Forecast
- Inventory Planning
- Model Performance
- Business Insights
- Methodology

Validated behaviors:

- Pages load without visible application errors.
- Filters update visible data.
- Scenario controls change simulated inventory outputs.
- Charts and tables contain data.
- CSV downloads are available.
- Public labels distinguish simulated inventory and economic outputs from measured forecast metrics.

## GitHub Audit

The public repository should exclude:

- raw M5 files
- processed Parquet files
- model binaries
- virtual environments
- local caches
- Kaggle credentials
- local logs
- temporary QA artifacts

The repository should include:

- README
- case study
- methodology and data documentation
- source pipeline scripts
- dashboard code
- tests
- public dashboard screenshots
- public final audit
- license and contribution/security docs

## Reproducibility Audit

A technical reviewer can reproduce the project by cloning the repository, creating a virtual environment, installing dependencies, configuring Kaggle access, downloading the M5 dataset, and running the scripts in the documented order.

Reproducibility is good for a portfolio project, but not instant, because generated data artifacts are intentionally not committed.

## Interview Claims

| Claim | Status | Notes |
|---|---|---|
| Built an end-to-end retail forecasting pipeline. | SAFE | Supported by scripts, docs, tests, and dashboard. |
| Compared statistical baselines and ML with temporal validation. | SAFE | Uses backtesting windows and 28-day horizon. |
| Built a per-series hybrid model selector. | SAFE | Explain confidence and fallback logic. |
| Achieved 11.54% WAPE on holdout validation. | SAFE | Specify holdout W3, not Kaggle leaderboard. |
| Built simulated inventory planning recommendations. | SAFE | Always say simulated. |
| ML always beat the baseline. | DO NOT CLAIM | The benchmark remained strong. |
| Estimated real Walmart savings. | DO NOT CLAIM | Savings are simulated under assumptions. |
| Forecasted the full item-store hierarchy as final output. | DO NOT CLAIM | Official layer is store-department. |

## Recommended Fixes

Before actively sharing the GitHub link:

1. Commit and push the final dashboard polish, refreshed screenshots, and this public audit.
2. Keep local-only audit details and QA artifacts out of GitHub.
3. Confirm that README images render correctly after the screenshot refresh.

Future improvements:

1. Add a small demo artifact package or hosted dashboard sample.
2. Add GitHub Actions for test automation.
3. Add a compact executive PDF.
4. Extend a future phase to prioritized item-store series.

## Final Publication Verdict

The project is ready to be shown publicly as a polished professional portfolio case, provided all claims remain framed around public M5 data, holdout validation, and simulated inventory impact.
