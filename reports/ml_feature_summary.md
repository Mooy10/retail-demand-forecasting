# ML Feature Summary

Rows: `133,910`
Columns: `71`
Date range: `2011-01-29` to `2016-04-24`
Unique series: `70`
Execution seconds: `15.56`

## Anti-Leakage Rules

- Lag features are shifted by series and never use the current target value.
- Rolling, expanding and zero-percentage features are computed from `demand.shift(1)`.
- Price features are aggregated by store-department-week without demand weighting.
- Direct multi-horizon modeling uses features at the forecast origin plus known target calendar fields.

## Price Aggregation Methodology

Item-store prices are mapped to department from `item_id`, then aggregated by `store_id`, `dept_id`, and `wm_yr_wk`. The pipeline uses mean, median, min, max, standard deviation, item count, week-over-week mean price change, and a price index against the store-department historical mean. No future demand is used as a weight.

## Lag Row Loss

Rows with missing lag_56: `3,920`. Training drops rows only when the selected model features are missing.

## Columns And Types

| column | dtype |
| --- | --- |
| unique_id | str |
| date | datetime64[us] |
| d | str |
| d_order | int16 |
| demand | int32 |
| store_id | str |
| dept_id | str |
| item_id | object |
| demand_pattern | str |
| abc_class | str |
| state_id | object |
| cat_id | object |
| wm_yr_wk | int64 |
| year | int64 |
| quarter | int8 |
| month | int64 |
| week_of_year | int16 |
| day_of_month | int8 |
| day_of_week | int8 |
| is_weekend | int8 |
| day_index | int16 |
| sin_day_of_week | float32 |
| cos_day_of_week | float32 |
| sin_month | float32 |
| cos_month | float32 |
| event_name_1 | string |
| event_type_1 | string |
| event_name_2 | string |
| event_type_2 | string |
| has_event | int8 |
| snap_CA | int64 |
| snap_TX | int64 |
| snap_WI | int64 |
| snap_active | int8 |
| mean_sell_price | float32 |
| median_sell_price | float32 |
| min_sell_price | float32 |
| max_sell_price | float32 |
| price_std | float32 |
| item_count_with_price | int16 |
| price_change_vs_previous_week | float32 |
| price_index_vs_store_department_mean | float32 |
| lag_1 | float32 |
| lag_2 | float32 |
| lag_3 | float32 |
| lag_7 | float32 |
| lag_14 | float32 |
| lag_21 | float32 |
| lag_28 | float32 |
| lag_35 | float32 |
| lag_42 | float32 |
| lag_56 | float32 |
| rolling_mean_7 | float32 |
| rolling_std_7 | float32 |
| rolling_mean_14 | float32 |
| rolling_std_14 | float32 |
| rolling_mean_28 | float32 |
| rolling_std_28 | float32 |
| rolling_mean_56 | float32 |
| rolling_std_56 | float32 |
| rolling_min_7 | float32 |
| rolling_max_7 | float32 |
| rolling_min_28 | float32 |
| rolling_max_28 | float32 |
| rolling_zero_pct_28 | float32 |
| expanding_mean | float32 |
| expanding_std | float32 |
| demand_change_1 | float32 |
| demand_change_7 | float32 |
| lag_7_vs_28_ratio | float32 |
| rolling_mean_7_vs_28_ratio | float32 |

## Null Summary

| column | null_count | null_pct |
| --- | --- | --- |
| item_id | 133910 | 100.0 |
| lag_1 | 70 | 0.05 |
| lag_2 | 140 | 0.1 |
| lag_3 | 210 | 0.16 |
| lag_7 | 490 | 0.37 |
| lag_14 | 980 | 0.73 |
| lag_21 | 1470 | 1.1 |
| lag_28 | 1960 | 1.46 |
| lag_35 | 2450 | 1.83 |
| lag_42 | 2940 | 2.2 |
| lag_56 | 3920 | 2.93 |
| rolling_mean_7 | 70 | 0.05 |
| rolling_mean_14 | 70 | 0.05 |
| rolling_mean_28 | 70 | 0.05 |
| rolling_mean_56 | 70 | 0.05 |
| rolling_min_7 | 70 | 0.05 |
| rolling_max_7 | 70 | 0.05 |
| rolling_min_28 | 70 | 0.05 |
| rolling_max_28 | 70 | 0.05 |
| expanding_mean | 70 | 0.05 |
| demand_change_1 | 140 | 0.1 |
| demand_change_7 | 980 | 0.73 |
