# Data Understanding Summary

## Dataset Dimensions

- Sales table rows: `30,490`
- Sales table columns: `1,919`
- Calendar rows: `1,969`
- Sell prices rows: `6,841,121`
- Historical sales days: `1,913`
- Sales date range: `2011-01-29` to `2016-04-24`

## Hierarchies Found

- states: `3`
- stores: `10`
- categories: `3`
- departments: `7`
- products: `3,049`
- item_store_combinations: `30,490`

## General Statistics

- Total item-store series: `30,490`
- Total demand: `65,695,409` units
- Zero-demand values: `68.20%`
- Approximate peak Python allocation during run: `307.47 MB`
- Execution time: `27.27 seconds`

## Approximate Memory Used By Loaded Files

- calendar.csv: `0.25 MB`
- sales_train_validation.csv: `112.62 MB`
- sell_prices.csv: `71.89 MB`

## Processed Parquet Dimensions

- daily_total_demand: `1,913` rows x `8` columns
- daily_demand_by_state: `5,739` rows x `9` columns
- daily_demand_by_store: `19,130` rows x `9` columns
- daily_demand_by_category: `5,739` rows x `9` columns
- daily_demand_by_department: `13,391` rows x `9` columns
- product_demand_summary: `3,049` rows x `8` columns
- series_demand_summary: `30,490` rows x `10` columns

## Top Products By Volume

| item_id | dept_id | cat_id | total_demand | series_count |
| --- | --- | --- | --- | --- |
| FOODS_3_090 | FOODS_3 | FOODS | 1002529 | 10 |
| FOODS_3_586 | FOODS_3 | FOODS | 920242 | 10 |
| FOODS_3_252 | FOODS_3 | FOODS | 565299 | 10 |
| FOODS_3_555 | FOODS_3 | FOODS | 491287 | 10 |
| FOODS_3_714 | FOODS_3 | FOODS | 396172 | 10 |
| FOODS_3_587 | FOODS_3 | FOODS | 396119 | 10 |
| FOODS_3_694 | FOODS_3 | FOODS | 390001 | 10 |
| FOODS_3_226 | FOODS_3 | FOODS | 363082 | 10 |
| FOODS_3_202 | FOODS_3 | FOODS | 295689 | 10 |
| FOODS_3_723 | FOODS_3 | FOODS | 284333 | 10 |

## Top Stores By Volume

| store_id | demand |
| --- | --- |
| CA_3 | 11188180 |
| CA_1 | 7698216 |
| TX_2 | 7214384 |
| WI_2 | 6544012 |
| WI_3 | 6427782 |
| TX_3 | 6089330 |
| CA_2 | 5685475 |
| TX_1 | 5595292 |
| WI_1 | 5149062 |
| CA_4 | 4103676 |

## Top Categories By Volume

| cat_id | demand |
| --- | --- |
| FOODS | 45089939 |
| HOUSEHOLD | 14480670 |
| HOBBIES | 6124800 |

## Top States By Volume

| state_id | demand |
| --- | --- |
| CA | 28675547 |
| TX | 18899006 |
| WI | 18120856 |

## Initial Business Observations

- Demand is highly hierarchical: the same product can behave differently by store and state.
- Aggregated views by state, store, category, and department are enough for early business understanding without creating the full long sales table.
- The zero-demand percentage is an important signal: many item-store combinations have intermittent demand, which will affect baseline and ML model choice.
- Store and category concentration should guide inventory prioritization before modeling every SKU at equal depth.

## Data Quality Risks

- Zero sales may represent true no-demand days, stockouts, unavailable products, or pre-launch periods; they should not all be interpreted the same way.
- Sell prices are weekly and may be missing before an item is available in a store.
- Event columns in calendar contain expected nulls on non-event days.
- The validation file ends before the evaluation horizon, so future phases must handle temporal splits carefully.

## Memory And Performance Considerations

- The script avoids a full melt of `sales_train_validation.csv`, which would create roughly 58 million rows.
- Daily tables are created through vectorized column sums and compact groupby matrices.
- Only aggregated matrices are stacked into long format.
- Intermediate grouped objects are explicitly deleted and garbage collection is requested after each aggregation.
