# Final Forecasting Decision

## 1. Best Global Model

The best global backtesting result is `hybrid_selected` with weighted WAPE `0.117336` and weighted RMSSE `0.788849`.

## 2. Single Model Or Hybrid

A hybrid strategy is preferable. No individual Phase 7 ML model beats `seasonal_naive_28` globally in WAPE, but the per-series selector improves the global result by combining baseline and ML strengths.

## 3. Priority Metric

Prioritize WAPE for inventory planning because it weights absolute error by actual demand volume and is easier to translate into business impact. RMSSE remains a secondary stability/scale-aware metric.

## 4. Series To Keep With Baseline

Keep baseline-family forecasts for `38` series selected as `seasonal_naive_28` or `seasonal_average_weekday`.

| unique_id | store_id | dept_id | best_model | confidence |
| --- | --- | --- | --- | --- |
| CA_1_FOODS_1 | CA_1 | FOODS_1 | seasonal_average_weekday | High |
| CA_1_FOODS_3 | CA_1 | FOODS_3 | seasonal_average_weekday | High |
| CA_1_HOBBIES_2 | CA_1 | HOBBIES_2 | seasonal_naive_28 | Low |
| CA_1_HOUSEHOLD_2 | CA_1 | HOUSEHOLD_2 | seasonal_naive_28 | High |
| CA_2_FOODS_1 | CA_2 | FOODS_1 | seasonal_naive_28 | High |
| CA_2_FOODS_3 | CA_2 | FOODS_3 | seasonal_naive_28 | Low |
| CA_2_HOBBIES_2 | CA_2 | HOBBIES_2 | seasonal_average_weekday | High |
| CA_2_HOUSEHOLD_2 | CA_2 | HOUSEHOLD_2 | seasonal_naive_28 | Medium |
| CA_3_FOODS_1 | CA_3 | FOODS_1 | seasonal_average_weekday | Medium |
| CA_3_FOODS_3 | CA_3 | FOODS_3 | seasonal_average_weekday | High |
| CA_3_HOBBIES_2 | CA_3 | HOBBIES_2 | seasonal_naive_28 | Medium |
| CA_3_HOUSEHOLD_2 | CA_3 | HOUSEHOLD_2 | seasonal_average_weekday | High |
| CA_4_FOODS_3 | CA_4 | FOODS_3 | seasonal_naive_28 | High |
| CA_4_HOBBIES_2 | CA_4 | HOBBIES_2 | seasonal_naive_28 | Low |
| CA_4_HOUSEHOLD_1 | CA_4 | HOUSEHOLD_1 | seasonal_naive_28 | Low |
| TX_1_FOODS_1 | TX_1 | FOODS_1 | seasonal_average_weekday | High |
| TX_1_FOODS_3 | TX_1 | FOODS_3 | seasonal_average_weekday | High |
| TX_1_HOBBIES_2 | TX_1 | HOBBIES_2 | seasonal_naive_28 | Medium |
| TX_1_HOUSEHOLD_2 | TX_1 | HOUSEHOLD_2 | seasonal_naive_28 | High |
| TX_2_FOODS_1 | TX_2 | FOODS_1 | seasonal_average_weekday | High |
| TX_2_FOODS_2 | TX_2 | FOODS_2 | seasonal_average_weekday | High |
| TX_2_FOODS_3 | TX_2 | FOODS_3 | seasonal_average_weekday | High |
| TX_2_HOBBIES_2 | TX_2 | HOBBIES_2 | seasonal_average_weekday | High |
| TX_3_HOBBIES_2 | TX_3 | HOBBIES_2 | seasonal_naive_28 | Medium |
| WI_1_FOODS_2 | WI_1 | FOODS_2 | seasonal_naive_28 | Low |
| WI_1_FOODS_3 | WI_1 | FOODS_3 | seasonal_naive_28 | Low |
| WI_1_HOBBIES_1 | WI_1 | HOBBIES_1 | seasonal_naive_28 | High |
| WI_1_HOBBIES_2 | WI_1 | HOBBIES_2 | seasonal_average_weekday | Medium |
| WI_1_HOUSEHOLD_1 | WI_1 | HOUSEHOLD_1 | seasonal_naive_28 | High |
| WI_1_HOUSEHOLD_2 | WI_1 | HOUSEHOLD_2 | seasonal_naive_28 | High |

## 5. Series That Can Use ML

Use ML for `32` series selected as `xgboost_phase6`, `xgboost_phase7`, or `lightgbm_phase7`.

| unique_id | store_id | dept_id | best_model | confidence |
| --- | --- | --- | --- | --- |
| CA_1_FOODS_2 | CA_1 | FOODS_2 | xgboost_phase6 | High |
| CA_1_HOBBIES_1 | CA_1 | HOBBIES_1 | xgboost_phase6 | Low |
| CA_1_HOUSEHOLD_1 | CA_1 | HOUSEHOLD_1 | lightgbm_phase7 | Medium |
| CA_2_FOODS_2 | CA_2 | FOODS_2 | xgboost_phase7 | Low |
| CA_2_HOBBIES_1 | CA_2 | HOBBIES_1 | xgboost_phase7 | High |
| CA_2_HOUSEHOLD_1 | CA_2 | HOUSEHOLD_1 | lightgbm_phase7 | High |
| CA_3_FOODS_2 | CA_3 | FOODS_2 | lightgbm_phase7 | High |
| CA_3_HOBBIES_1 | CA_3 | HOBBIES_1 | xgboost_phase7 | Medium |
| CA_3_HOUSEHOLD_1 | CA_3 | HOUSEHOLD_1 | xgboost_phase7 | Medium |
| CA_4_FOODS_1 | CA_4 | FOODS_1 | xgboost_phase7 | High |
| CA_4_FOODS_2 | CA_4 | FOODS_2 | xgboost_phase7 | High |
| CA_4_HOBBIES_1 | CA_4 | HOBBIES_1 | xgboost_phase7 | Medium |
| CA_4_HOUSEHOLD_2 | CA_4 | HOUSEHOLD_2 | xgboost_phase7 | Medium |
| TX_1_FOODS_2 | TX_1 | FOODS_2 | lightgbm_phase7 | High |
| TX_1_HOBBIES_1 | TX_1 | HOBBIES_1 | xgboost_phase6 | Medium |
| TX_1_HOUSEHOLD_1 | TX_1 | HOUSEHOLD_1 | xgboost_phase7 | Medium |
| TX_2_HOBBIES_1 | TX_2 | HOBBIES_1 | xgboost_phase7 | Medium |
| TX_2_HOUSEHOLD_1 | TX_2 | HOUSEHOLD_1 | xgboost_phase6 | Low |
| TX_2_HOUSEHOLD_2 | TX_2 | HOUSEHOLD_2 | xgboost_phase6 | Medium |
| TX_3_FOODS_1 | TX_3 | FOODS_1 | xgboost_phase7 | High |
| TX_3_FOODS_2 | TX_3 | FOODS_2 | lightgbm_phase7 | High |
| TX_3_FOODS_3 | TX_3 | FOODS_3 | xgboost_phase7 | Medium |
| TX_3_HOBBIES_1 | TX_3 | HOBBIES_1 | lightgbm_phase7 | Low |
| TX_3_HOUSEHOLD_1 | TX_3 | HOUSEHOLD_1 | lightgbm_phase7 | Medium |
| TX_3_HOUSEHOLD_2 | TX_3 | HOUSEHOLD_2 | xgboost_phase6 | Low |
| WI_1_FOODS_1 | WI_1 | FOODS_1 | lightgbm_phase7 | Low |
| WI_2_FOODS_1 | WI_2 | FOODS_1 | xgboost_phase6 | High |
| WI_2_HOBBIES_1 | WI_2 | HOBBIES_1 | xgboost_phase6 | High |
| WI_2_HOUSEHOLD_1 | WI_2 | HOUSEHOLD_1 | lightgbm_phase7 | Medium |
| WI_2_HOUSEHOLD_2 | WI_2 | HOUSEHOLD_2 | xgboost_phase6 | High |

## 6. Official Forecast For Dashboard

`hybrid_selected` should be the official backtesting forecast consumed by the future dashboard. `seasonal_naive_28` remains the global fallback and benchmark.

## 7. User-Facing Limitations

- Forecasts are validated at store-department level, not item-store level.
- Backtesting uses 3 windows and a 28-day horizon.
- ML training uses the latest 180 origin days per cutoff for runtime control.
- The selector is data-driven but still based on historical backtesting and should be monitored.
- Price variables are not causal elasticity estimates.
- Kaggle official WRMSSE is not implemented yet.

## Final Recommendation

Use the hybrid forecast for the portfolio dashboard, show the selected model and confidence per series, and keep the baseline visible so the business user can see when the simpler forecast is intentionally preferred.
