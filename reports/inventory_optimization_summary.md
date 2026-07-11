# Inventory Optimization Summary

## Objective

Convert the validated out-of-sample forecast into a simulated inventory planning module for 70 store-department series.

## Simulation Assumptions

- The M5 dataset does not include real on-hand inventory, open purchase orders, logistics costs, supplier lead times, or true stockout costs.
- Inventory, ordering cost, holding cost, lead time, service level, and stockout cost are configurable simulation assumptions.
- Default service level is 95%, lead time is 7 days, review period is 7 days, and base initial inventory is 14 days of recent demand.

## Forecast Used

Official forecast rows: `1,960` across `70` series and 28 horizons.
Fallback was used for `10` series.

## Methodology

- Safety stock uses `z * sigma_error * sqrt(lead_time_days)` from empirical out-of-sample forecast errors.
- Reorder point equals expected demand during lead time plus safety stock.
- EOQ uses simulated ordering cost and holding cost derived from a representative M5 sell price proxy.
- Recommended order quantity fills to order-up-to level and applies configurable minimum order quantity and rounding multiple.

## Key Inventory Metrics

Average safety stock: `371.70` units.
Average reorder point: `4535.60` units.
Scenario rows with recommended orders: `126` of `210`.
High/Critical risk rows: `0`.

## Departments With Highest Simulated Risk

| dept_id | scenario | projected_stockout_units | recommended_order_quantity | total_cost |
| --- | --- | --- | --- | --- |
| FOODS_3 | lean | 255475.57 | 138213.0 | 768651.31 |
| FOODS_3 | base | 252408.57 | 5830.0 | 760571.41 |
| FOODS_3 | conservative | 122788.57 | 0.0 | 375101.86 |
| HOUSEHOLD_1 | lean | 107435.81 | 60921.0 | 324169.32 |
| HOUSEHOLD_1 | base | 107435.31 | 4879.0 | 325136.43 |
| FOODS_2 | lean | 74660.36 | 44588.0 | 225384.87 |
| FOODS_2 | base | 74603.61 | 4663.0 | 225695.54 |
| HOUSEHOLD_1 | conservative | 56271.81 | 0.0 | 173577.93 |
| HOBBIES_1 | lean | 45828.81 | 28005.0 | 138886.57 |
| HOBBIES_1 | base | 45828.06 | 4047.0 | 139364.68 |

## Stores With Highest Simulated Risk

| store_id | scenario | projected_stockout_units | recommended_order_quantity | total_cost |
| --- | --- | --- | --- | --- |
| CA_3 | lean | 85813.44 | 46345.0 | 258653.2 |
| CA_3 | base | 85207.19 | 1705.0 | 257359.24 |
| WI_2 | lean | 69785.5 | 42090.0 | 210439.69 |
| WI_2 | base | 69719.5 | 5214.0 | 210608.72 |
| CA_1 | lean | 59405.12 | 32667.0 | 179232.79 |
| CA_1 | base | 58886.62 | 1874.0 | 177963.89 |
| TX_2 | lean | 54663.6 | 29007.0 | 164936.52 |
| CA_2 | lean | 53883.99 | 32997.0 | 162652.88 |
| CA_2 | base | 53882.74 | 4173.0 | 163013.86 |
| TX_2 | base | 53699.6 | 923.0 | 162253.32 |

## Priority Purchase Recommendations

| unique_id | store_id | dept_id | scenario | forecast_demand_28d | initial_inventory | safety_stock | reorder_point | recommended_order_quantity | stockout_risk_level | estimated_total_inventory_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA_3_FOODS_3 | CA_3 | FOODS_3 | lean | 81609.2 | 21104.5 | 968.55 | 21370.85 | 20669.0 | Medium | 119806.08 |
| WI_2_FOODS_3 | WI_2 | FOODS_3 | lean | 64948.0 | 16237.0 | 1634.53 | 17871.53 | 17872.0 | Medium | 92771.61 |
| WI_3_FOODS_3 | WI_3 | FOODS_3 | lean | 57311.0 | 14327.75 | 1199.18 | 15526.93 | 15527.0 | Medium | 82604.41 |
| CA_1_FOODS_3 | CA_1 | FOODS_3 | lean | 57411.37 | 14736.5 | 654.07 | 15006.91 | 14624.0 | Medium | 84391.12 |
| CA_2_FOODS_3 | CA_2 | FOODS_3 | lean | 50500.0 | 12625.0 | 1631.11 | 14256.11 | 14257.0 | Medium | 71079.65 |
| TX_2_FOODS_3 | TX_2 | FOODS_3 | lean | 54518.12 | 14332.25 | 602.55 | 14232.09 | 13530.0 | Medium | 80189.03 |
| WI_1_FOODS_3 | WI_1 | FOODS_3 | lean | 46830.79 | 11667.25 | 775.08 | 12482.78 | 12524.0 | Medium | 68125.15 |
| CA_3_HOUSEHOLD_1 | CA_3 | HOUSEHOLD_1 | lean | 39947.82 | 10081.5 | 552.81 | 10539.76 | 10446.0 | Medium | 58531.47 |
| TX_3_FOODS_3 | TX_3 | FOODS_3 | lean | 43815.01 | 12095.5 | 598.34 | 11552.09 | 10411.0 | Medium | 64119.28 |
| TX_1_FOODS_3 | TX_1 | FOODS_3 | lean | 40663.08 | 10441.25 | 518.71 | 10684.48 | 10409.0 | Medium | 59626.67 |
| WI_2_FOODS_2 | WI_2 | FOODS_2 | lean | 32245.67 | 7917.25 | 1287.68 | 9349.09 | 9494.0 | Medium | 44714.83 |
| CA_4_FOODS_3 | CA_4 | FOODS_3 | lean | 31532.0 | 7883.0 | 506.66 | 8389.66 | 8390.0 | Medium | 45938.32 |
| WI_2_HOUSEHOLD_1 | WI_2 | HOUSEHOLD_1 | lean | 28482.0 | 7120.5 | 1009.37 | 8129.87 | 8130.0 | Medium | 39903.81 |
| TX_2_HOUSEHOLD_1 | TX_2 | HOUSEHOLD_1 | lean | 23736.71 | 5871.0 | 326.47 | 6260.64 | 6324.0 | Medium | 34813.16 |
| CA_2_HOUSEHOLD_1 | CA_2 | HOUSEHOLD_1 | lean | 21300.24 | 5223.75 | 573.95 | 5899.01 | 6001.0 | Medium | 30411.01 |
| TX_1_HOUSEHOLD_1 | TX_1 | HOUSEHOLD_1 | lean | 21646.0 | 5411.5 | 481.11 | 5892.61 | 5893.0 | Medium | 31210.98 |
| CA_1_HOUSEHOLD_1 | CA_1 | HOUSEHOLD_1 | lean | 21790.0 | 5447.5 | 416.35 | 5863.85 | 5864.0 | Medium | 31621.75 |
| TX_3_HOUSEHOLD_1 | TX_3 | HOUSEHOLD_1 | lean | 22180.34 | 5573.5 | 319.03 | 5864.11 | 5836.0 | Medium | 32493.15 |
| WI_1_FOODS_2 | WI_1 | FOODS_2 | lean | 18680.0 | 4670.0 | 534.25 | 5204.25 | 5205.0 | Medium | 26569.75 |
| WI_3_HOUSEHOLD_1 | WI_3 | HOUSEHOLD_1 | lean | 18558.2 | 4623.75 | 402.99 | 5042.54 | 5059.0 | Medium | 26793.79 |

## Excess Inventory Opportunities

| unique_id | store_id | dept_id | scenario | overstock_units | excess_inventory_days | inventory_coverage_days | estimated_holding_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WI_3_HOBBIES_2 | WI_3 | HOBBIES_2 | conservative | 10.15 | 2.3 | 30.3 | 13.27 |

## Policy Comparison

| scenario | policy | total_cost | holding_cost | stockout_cost | ordering_cost | stockout_units | excess_inventory | service_level_approx | simulated_savings_vs_baseline | simulated_savings_vs_simple |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | baseline_seasonal_naive_28 | 1709180.96 | 17992.46 | 1685938.5 | 5250.0 | 561979.5 | 0.0 | 0.53 | 0.0 | -0.0 |
| base | hybrid_official | 1680537.51 | 17992.46 | 1658345.06 | 4200.0 | 552781.69 | 0.0 | 0.54 | 28643.44 | 28643.44 |
| base | no_forecast_historical_average | 1709180.96 | 17992.46 | 1685938.5 | 5250.0 | 561979.5 | 0.0 | 0.53 | 0.0 | 0.0 |
| conservative | baseline_seasonal_naive_28 | 909038.94 | 26988.69 | 882050.25 | 0.0 | 294016.75 | 0.0 | 0.75 | 0.0 | -0.0 |
| conservative | hybrid_official | 878662.41 | 26989.66 | 851672.75 | 0.0 | 283890.92 | 10.15 | 0.76 | 30376.53 | 30376.53 |
| conservative | no_forecast_historical_average | 909038.94 | 26988.69 | 882050.25 | 0.0 | 294016.75 | 0.0 | 0.75 | 0.0 | 0.0 |
| lean | baseline_seasonal_naive_28 | 1700192.98 | 8996.23 | 1685946.75 | 5250.0 | 561982.25 | 0.0 | 0.53 | 0.0 | -0.0 |
| lean | hybrid_official | 1684929.54 | 8996.23 | 1670683.31 | 5250.0 | 556894.44 | 0.0 | 0.53 | 15263.44 | 15263.44 |
| lean | no_forecast_historical_average | 1700192.98 | 8996.23 | 1685946.75 | 5250.0 | 561982.25 | 0.0 | 0.53 | 0.0 | 0.0 |

## Simulated Economic Impact

In the base scenario, hybrid_official simulated total cost is `1680537.51` versus baseline `1709180.96`.
Simulated savings versus baseline in base scenario: `28643.44`.
Simulated stockout reduction versus baseline in base scenario: `-9197.81` units change, where negative is better.

## Limitations

- Results are simulated and should not be interpreted as real Walmart inventory or real savings.
- Stockout risk is a categorical planning score, not an exact probability.
- Representative price is used as a proxy for unit value; true cost of goods is unavailable.
- Lead times, service levels, order constraints, holding cost, and stockout penalties must be supplied by a real company before production use.

## Data A Real Company Would Need

- On-hand inventory by SKU/location, open orders, supplier lead times, MOQ/case-pack rules, replenishment calendars, unit cost, margin, shelf-life, capacity, and actual stockout events.
