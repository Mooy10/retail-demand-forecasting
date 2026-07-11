# Model Selection Summary

## Advanced Features Created

Advanced feature table: `133,910` rows x `128` columns.
New advanced variables generated: `57`.

`rolling_median_7`, `rolling_median_14`, `rolling_median_28`, `rolling_quantile_25_28`, `rolling_quantile_75_28`, `ewm_mean_7`, `ewm_mean_14`, `ewm_mean_28`, `ewm_std_28`, `rolling_trend_7`, `rolling_trend_28`, `lag_364`, `lag_365`, `lag_371`, `rolling_mean_364`, `previous_year_same_week_mean`, `previous_year_same_day_of_week`, `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`, `is_year_start`, `is_year_end`, `days_to_next_event`, `days_since_previous_event`, `event_week`, `pre_event_7_days`, `post_event_7_days`, `store_dept`, `store_state`, `dept_category`, `month_day_of_week`, `event_store`, `snap_department`, `store_total_lag_7`, `store_total_lag_28`, `store_rolling_mean_28`, `department_total_lag_7`, `department_total_lag_28`, `department_rolling_mean_28`, `state_total_lag_7`, `state_total_lag_28`, `category_total_lag_7`, `category_total_lag_28`, `series_share_of_store_28`, `series_share_of_department_28`, `series_share_of_state_28`, `price_change_1_week`, `price_change_4_weeks`, `price_pct_change_1_week`, `price_pct_change_4_weeks`, `price_rolling_mean_4_weeks`, `price_rolling_std_4_weeks`, `price_relative_to_department`, `price_relative_to_store`, `discount_proxy`, `price_volatility_12_weeks`

## Anti-Leakage Rules

- Demand-derived advanced features use shifted history only.
- Hierarchical features aggregate demand by hierarchy and date, then shift or roll shifted series.
- Target calendar features are known calendar attributes, not future observed demand.
- Price variables are predictive proxies and must not be interpreted as causal elasticity.

## Phase 7 Model Results

| model | weighted_mae | weighted_wape | weighted_rmsse | series_count |
| --- | --- | --- | --- | --- |
| seasonal_naive_28 | 151.0752796835013 | 0.1367283284507579 | 0.9133816246977886 | 70 |
| xgboost_phase6 | 158.24820568778483 | 0.1405913191684483 | 0.8868082081907814 | 70 |
| lightgbm_phase7 | 160.89339294186513 | 0.1438792822748028 | 0.8969794060060086 | 70 |
| xgboost_phase7 | 167.24607555972253 | 0.1472261584572757 | 0.9279480577811527 | 70 |
| seasonal_average_weekday | 232.88432530215087 | 0.2117443348645168 | 1.3285391506724444 | 70 |

## Selector Distribution

| model | selected_series |
| --- | --- |
| seasonal_naive_28 | 22 |
| seasonal_average_weekday | 16 |
| xgboost_phase7 | 13 |
| lightgbm_phase7 | 10 |
| xgboost_phase6 | 9 |

## Selector Confidence

| confidence | series |
| --- | --- |
| High | 36 |
| Medium | 20 |
| Low | 14 |

## Model Chosen By Store And Department

| store_id | FOODS_1 | FOODS_2 | FOODS_3 | HOBBIES_1 | HOBBIES_2 | HOUSEHOLD_1 | HOUSEHOLD_2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CA_1 | seasonal_average_weekday | xgboost_phase6 | seasonal_average_weekday | xgboost_phase6 | seasonal_naive_28 | lightgbm_phase7 | seasonal_naive_28 |
| CA_2 | seasonal_naive_28 | xgboost_phase7 | seasonal_naive_28 | xgboost_phase7 | seasonal_average_weekday | lightgbm_phase7 | seasonal_naive_28 |
| CA_3 | seasonal_average_weekday | lightgbm_phase7 | seasonal_average_weekday | xgboost_phase7 | seasonal_naive_28 | xgboost_phase7 | seasonal_average_weekday |
| CA_4 | xgboost_phase7 | xgboost_phase7 | seasonal_naive_28 | xgboost_phase7 | seasonal_naive_28 | seasonal_naive_28 | xgboost_phase7 |
| TX_1 | seasonal_average_weekday | lightgbm_phase7 | seasonal_average_weekday | xgboost_phase6 | seasonal_naive_28 | xgboost_phase7 | seasonal_naive_28 |
| TX_2 | seasonal_average_weekday | seasonal_average_weekday | seasonal_average_weekday | xgboost_phase7 | seasonal_average_weekday | xgboost_phase6 | xgboost_phase6 |
| TX_3 | xgboost_phase7 | lightgbm_phase7 | xgboost_phase7 | lightgbm_phase7 | seasonal_naive_28 | lightgbm_phase7 | xgboost_phase6 |
| WI_1 | lightgbm_phase7 | seasonal_naive_28 | seasonal_naive_28 | seasonal_naive_28 | seasonal_average_weekday | seasonal_naive_28 | seasonal_naive_28 |
| WI_2 | xgboost_phase6 | seasonal_naive_28 | seasonal_naive_28 | xgboost_phase6 | seasonal_average_weekday | lightgbm_phase7 | xgboost_phase6 |
| WI_3 | seasonal_naive_28 | seasonal_naive_28 | seasonal_naive_28 | seasonal_average_weekday | seasonal_average_weekday | lightgbm_phase7 | xgboost_phase7 |

## Hybrid Performance

| model | weighted_mae | weighted_wape | weighted_rmsse | series_count |
| --- | --- | --- | --- | --- |
| hybrid_selected | 128.71790442298476 | 0.1173357104289727 | 0.7888491842071963 | 70 |
| seasonal_naive_28 | 151.0752796835013 | 0.1367283284507579 | 0.9133816246977886 | 70 |
| xgboost_phase6 | 158.24820568778483 | 0.1405913191684483 | 0.8868082081907814 | 70 |
| lightgbm_phase7 | 160.89339294186513 | 0.1438792822748028 | 0.8969794060060086 | 70 |
| xgboost_phase7 | 167.24607555972253 | 0.1472261584572757 | 0.9279480577811527 | 70 |
| seasonal_average_weekday | 232.88432530215087 | 0.2117443348645168 | 1.3285391506724444 | 70 |

Hybrid beats seasonal_naive_28 by average WAPE in `68.57%` of series and `56.19%` of series-window tests.
Hybrid beats seasonal_naive_28 by average RMSSE in `68.57%` of series.

## Low Confidence Series

| unique_id | store_id | dept_id | best_model | second_best_model | score_difference | mean_wape | mean_rmsse | window_stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA_1_HOBBIES_2 | CA_1 | HOBBIES_2 | seasonal_naive_28 | seasonal_average_weekday | 0.0 | 0.3282444605792775 | 1.099585253437838 | 0.12703002732296356 |
| CA_2_FOODS_3 | CA_2 | FOODS_3 | seasonal_naive_28 | lightgbm_phase7 | 0.0 | 0.12329288109060565 | 1.1060829805282248 | 0.2146487497788222 |
| CA_4_HOUSEHOLD_1 | CA_4 | HOUSEHOLD_1 | seasonal_naive_28 | xgboost_phase6 | 0.0 | 0.10193321905088681 | 0.9790066883698026 | 0.1203637542805411 |
| WI_1_FOODS_3 | WI_1 | FOODS_3 | seasonal_naive_28 | xgboost_phase6 | 0.0 | 0.11059928064759732 | 0.7035396321816018 | 0.06509575824852919 |
| WI_2_FOODS_2 | WI_2 | FOODS_2 | seasonal_naive_28 | xgboost_phase6 | 0.0 | 0.17732782158048566 | 1.5007500436544154 | 0.16400445494051338 |
| WI_3_FOODS_2 | WI_3 | FOODS_2 | seasonal_naive_28 | lightgbm_phase7 | 0.0 | 0.14481546042539714 | 0.892034474680329 | 0.1550984702653659 |
| WI_1_FOODS_2 | WI_1 | FOODS_2 | seasonal_naive_28 | xgboost_phase7 | 0.0 | 0.13565877536405965 | 1.1405616214738015 | 0.1140210398507287 |
| TX_3_HOUSEHOLD_2 | TX_3 | HOUSEHOLD_2 | xgboost_phase6 | xgboost_phase7 | 0.00047329442638469055 | 0.12697499508694562 | 0.9067220671087107 | 0.12932551961704597 |
| WI_1_FOODS_1 | WI_1 | FOODS_1 | lightgbm_phase7 | xgboost_phase6 | 0.005259099888095309 | 0.1670280140957947 | 1.0589647692692943 | 0.21051569779990967 |
| CA_4_HOBBIES_2 | CA_4 | HOBBIES_2 | seasonal_naive_28 | seasonal_average_weekday | 0.007166930855764864 | 0.44090182933685856 | 1.1256299739900262 | 0.1583678429992332 |
| CA_2_FOODS_2 | CA_2 | FOODS_2 | xgboost_phase7 | lightgbm_phase7 | 0.02121472151650955 | 0.15317133334413136 | 1.7375832985029749 | 0.27130886750700667 |
| TX_3_HOBBIES_1 | TX_3 | HOBBIES_1 | lightgbm_phase7 | xgboost_phase7 | 0.02355112320176604 | 0.1235100341784055 | 0.7532496728181767 | 0.08748613042580075 |
| TX_2_HOUSEHOLD_1 | TX_2 | HOUSEHOLD_1 | xgboost_phase6 | lightgbm_phase7 | 0.026932263334299045 | 0.09094584389414806 | 0.5735403735617727 | 0.2006699039099168 |
| CA_1_HOBBIES_1 | CA_1 | HOBBIES_1 | xgboost_phase6 | lightgbm_phase7 | 0.028455883110508062 | 0.14104230870648168 | 0.6784616523028557 | 0.15111608795381426 |

## Baseline Still Better Or Selected

The selector keeps a baseline-family model for `38` of 70 series.
| unique_id | store_id | dept_id | best_model | confidence | mean_wape | mean_rmsse |
| --- | --- | --- | --- | --- | --- | --- |
| CA_1_FOODS_1 | CA_1 | FOODS_1 | seasonal_average_weekday | High | 0.1407238009875585 | 0.7840692823721008 |
| CA_1_FOODS_3 | CA_1 | FOODS_3 | seasonal_average_weekday | High | 0.07039822214006002 | 0.3958828872400651 |
| CA_1_HOBBIES_2 | CA_1 | HOBBIES_2 | seasonal_naive_28 | Low | 0.3282444605792775 | 1.099585253437838 |
| CA_1_HOUSEHOLD_2 | CA_1 | HOUSEHOLD_2 | seasonal_naive_28 | High | 0.142001853936465 | 0.8605618530102 |
| CA_2_FOODS_1 | CA_2 | FOODS_1 | seasonal_naive_28 | High | 0.15216187110154827 | 0.9461428651010447 |
| CA_2_FOODS_3 | CA_2 | FOODS_3 | seasonal_naive_28 | Low | 0.12329288109060565 | 1.1060829805282248 |
| CA_2_HOBBIES_2 | CA_2 | HOBBIES_2 | seasonal_average_weekday | High | 0.2601080257890025 | 0.8546765583315682 |
| CA_2_HOUSEHOLD_2 | CA_2 | HOUSEHOLD_2 | seasonal_naive_28 | Medium | 0.17181103455622457 | 0.9463008868175112 |
| CA_3_FOODS_1 | CA_3 | FOODS_1 | seasonal_average_weekday | Medium | 0.1672676080271516 | 1.0447743483165954 |
| CA_3_FOODS_3 | CA_3 | FOODS_3 | seasonal_average_weekday | High | 0.056945849449033116 | 0.4484075705833404 |
| CA_3_HOBBIES_2 | CA_3 | HOBBIES_2 | seasonal_naive_28 | Medium | 0.23733972792528993 | 0.9166792045462383 |
| CA_3_HOUSEHOLD_2 | CA_3 | HOUSEHOLD_2 | seasonal_average_weekday | High | 0.12053569116804709 | 0.784351706693438 |
| CA_4_FOODS_3 | CA_4 | FOODS_3 | seasonal_naive_28 | High | 0.09655035537530549 | 0.777067991469916 |
| CA_4_HOBBIES_2 | CA_4 | HOBBIES_2 | seasonal_naive_28 | Low | 0.44090182933685856 | 1.1256299739900262 |
| CA_4_HOUSEHOLD_1 | CA_4 | HOUSEHOLD_1 | seasonal_naive_28 | Low | 0.10193321905088681 | 0.9790066883698026 |
| TX_1_FOODS_1 | TX_1 | FOODS_1 | seasonal_average_weekday | High | 0.14542748377134299 | 0.7860593976716146 |
| TX_1_FOODS_3 | TX_1 | FOODS_3 | seasonal_average_weekday | High | 0.07132952747481329 | 0.42598274228523875 |
| TX_1_HOBBIES_2 | TX_1 | HOBBIES_2 | seasonal_naive_28 | Medium | 0.4064103255451494 | 1.3561550947765733 |
| TX_1_HOUSEHOLD_2 | TX_1 | HOUSEHOLD_2 | seasonal_naive_28 | High | 0.14109775817977646 | 0.897621460355027 |
| TX_2_FOODS_1 | TX_2 | FOODS_1 | seasonal_average_weekday | High | 0.17885330714086936 | 0.9607851515440208 |

## ML Selected

The selector assigns ML to `32` of 70 series.
| unique_id | store_id | dept_id | best_model | confidence | mean_wape | mean_rmsse |
| --- | --- | --- | --- | --- | --- | --- |
| CA_1_FOODS_2 | CA_1 | FOODS_2 | xgboost_phase6 | High | 0.12774273321672233 | 0.7748458969196682 |
| CA_1_HOBBIES_1 | CA_1 | HOBBIES_1 | xgboost_phase6 | Low | 0.14104230870648168 | 0.6784616523028557 |
| CA_1_HOUSEHOLD_1 | CA_1 | HOUSEHOLD_1 | lightgbm_phase7 | Medium | 0.08170487144586501 | 0.5512470167873469 |
| CA_2_FOODS_2 | CA_2 | FOODS_2 | xgboost_phase7 | Low | 0.15317133334413136 | 1.7375832985029749 |
| CA_2_HOBBIES_1 | CA_2 | HOBBIES_1 | xgboost_phase7 | High | 0.17441861017265256 | 0.7647618417141286 |
| CA_2_HOUSEHOLD_1 | CA_2 | HOUSEHOLD_1 | lightgbm_phase7 | High | 0.13328949883725255 | 0.7558302214835141 |
| CA_3_FOODS_2 | CA_3 | FOODS_2 | lightgbm_phase7 | High | 0.11031053460249539 | 0.7622225974858051 |
| CA_3_HOBBIES_1 | CA_3 | HOBBIES_1 | xgboost_phase7 | Medium | 0.10910917071793295 | 0.6154578855063152 |
| CA_3_HOUSEHOLD_1 | CA_3 | HOUSEHOLD_1 | xgboost_phase7 | Medium | 0.08370874649108323 | 0.6942694205997234 |
| CA_4_FOODS_1 | CA_4 | FOODS_1 | xgboost_phase7 | High | 0.1884888837301849 | 1.1756105565354755 |
| CA_4_FOODS_2 | CA_4 | FOODS_2 | xgboost_phase7 | High | 0.11035767576984586 | 0.8278208701640053 |
| CA_4_HOBBIES_1 | CA_4 | HOBBIES_1 | xgboost_phase7 | Medium | 0.141315831943125 | 0.7839057909408589 |
| CA_4_HOUSEHOLD_2 | CA_4 | HOUSEHOLD_2 | xgboost_phase7 | Medium | 0.13511177008502892 | 0.8942737994056146 |
| TX_1_FOODS_2 | TX_1 | FOODS_2 | lightgbm_phase7 | High | 0.11991047733143272 | 0.604821623629865 |
| TX_1_HOBBIES_1 | TX_1 | HOBBIES_1 | xgboost_phase6 | Medium | 0.16733274512347204 | 0.773466382150692 |
| TX_1_HOUSEHOLD_1 | TX_1 | HOUSEHOLD_1 | xgboost_phase7 | Medium | 0.09267362973523524 | 0.6389571575926682 |
| TX_2_HOBBIES_1 | TX_2 | HOBBIES_1 | xgboost_phase7 | Medium | 0.11273635560497465 | 0.5122508338007435 |
| TX_2_HOUSEHOLD_1 | TX_2 | HOUSEHOLD_1 | xgboost_phase6 | Low | 0.09094584389414806 | 0.5735403735617727 |
| TX_2_HOUSEHOLD_2 | TX_2 | HOUSEHOLD_2 | xgboost_phase6 | Medium | 0.14279662802785179 | 0.9383897374904843 |
| TX_3_FOODS_1 | TX_3 | FOODS_1 | xgboost_phase7 | High | 0.18940387731757627 | 1.1602001430882434 |

## Limitations

- Phase 7 still models only 70 store-department aggregate series.
- The run keeps the latest 180 origin days per cutoff; older seasonal history is underused.
- Peak traced memory during advanced training was `1677.67` MB, slightly above the 1.5 GB guideline due supervised training-frame copies.
- Model selection is based on three historical windows only, so confidence should be reviewed before production use.
- Official M5 WRMSSE remains deferred.

## Recommendation For Simulated Production

Use the hybrid selected forecast as the dashboard forecast because it improves both WAPE and RMSSE globally while preserving simple baselines where they remain strongest. Keep `seasonal_naive_28` as the benchmark and fallback.
