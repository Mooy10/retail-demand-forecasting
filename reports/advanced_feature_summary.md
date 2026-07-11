# Advanced Feature Summary

Rows: `133,910`
Columns: `128`
New advanced feature columns: `57`
Execution seconds: `30.30`
Peak traced memory MB: `84.27`

## Anti-Leakage Rules

- Demand lags, medians, quantiles, EWM and rolling trend features are computed from `demand.shift(1)` by series.
- Long seasonal features preserve nulls when history is unavailable.
- Hierarchical demand features aggregate by hierarchy and date, then shift or roll shifted demand.
- Event variables use the known calendar and no future observed demand.
- Price variables are predictive proxies, not causal elasticity estimates.

## New Feature Columns

| feature |
| --- |
| rolling_median_7 |
| rolling_median_14 |
| rolling_median_28 |
| rolling_quantile_25_28 |
| rolling_quantile_75_28 |
| ewm_mean_7 |
| ewm_mean_14 |
| ewm_mean_28 |
| ewm_std_28 |
| rolling_trend_7 |
| rolling_trend_28 |
| lag_364 |
| lag_365 |
| lag_371 |
| rolling_mean_364 |
| previous_year_same_week_mean |
| previous_year_same_day_of_week |
| is_month_start |
| is_month_end |
| is_quarter_start |
| is_quarter_end |
| is_year_start |
| is_year_end |
| days_to_next_event |
| days_since_previous_event |
| event_week |
| pre_event_7_days |
| post_event_7_days |
| store_dept |
| store_state |
| dept_category |
| month_day_of_week |
| event_store |
| snap_department |
| store_total_lag_7 |
| store_total_lag_28 |
| store_rolling_mean_28 |
| department_total_lag_7 |
| department_total_lag_28 |
| department_rolling_mean_28 |
| state_total_lag_7 |
| state_total_lag_28 |
| category_total_lag_7 |
| category_total_lag_28 |
| series_share_of_store_28 |
| series_share_of_department_28 |
| series_share_of_state_28 |
| price_change_1_week |
| price_change_4_weeks |
| price_pct_change_1_week |
| price_pct_change_4_weeks |
| price_rolling_mean_4_weeks |
| price_rolling_std_4_weeks |
| price_relative_to_department |
| price_relative_to_store |
| discount_proxy |
| price_volatility_12_weeks |

## Nulls In Advanced Features

| feature | null_count | null_pct |
| --- | --- | --- |
| rolling_median_7 | 70 | 0.05 |
| rolling_median_14 | 70 | 0.05 |
| rolling_median_28 | 70 | 0.05 |
| rolling_quantile_25_28 | 70 | 0.05 |
| rolling_quantile_75_28 | 70 | 0.05 |
| ewm_mean_7 | 70 | 0.05 |
| ewm_mean_14 | 70 | 0.05 |
| ewm_mean_28 | 70 | 0.05 |
| ewm_std_28 | 140 | 0.1 |
| rolling_trend_7 | 140 | 0.1 |
| rolling_trend_28 | 140 | 0.1 |
| lag_364 | 25480 | 19.03 |
| lag_365 | 25550 | 19.08 |
| lag_371 | 25970 | 19.39 |
| rolling_mean_364 | 1960 | 1.46 |
| previous_year_same_week_mean | 25480 | 19.03 |
| previous_year_same_day_of_week | 25480 | 19.03 |
| days_to_next_event | 1960 | 1.46 |
| days_since_previous_event | 630 | 0.47 |
| store_total_lag_7 | 490 | 0.37 |
| store_total_lag_28 | 1960 | 1.46 |
| store_rolling_mean_28 | 70 | 0.05 |
| department_total_lag_7 | 490 | 0.37 |
| department_total_lag_28 | 1960 | 1.46 |
| department_rolling_mean_28 | 70 | 0.05 |
| state_total_lag_7 | 490 | 0.37 |
| state_total_lag_28 | 1960 | 1.46 |
| category_total_lag_7 | 490 | 0.37 |
| category_total_lag_28 | 1960 | 1.46 |
| price_change_1_week | 490 | 0.37 |
| price_change_4_weeks | 1960 | 1.46 |
| price_pct_change_1_week | 490 | 0.37 |
| price_pct_change_4_weeks | 1960 | 1.46 |
| price_rolling_mean_4_weeks | 490 | 0.37 |
| price_rolling_std_4_weeks | 490 | 0.37 |
| price_volatility_12_weeks | 980 | 0.73 |

## Methodological Limitation

The downstream Phase 7 training keeps only the latest 180 origin days per cutoff to control memory and runtime; this can underuse older seasonality.
