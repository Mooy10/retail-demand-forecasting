# Forecast Uncertainty Summary

## Objective

Estimate empirical uncertainty around the official forecast using out-of-sample backtesting errors from the validated holdout forecast.

## Important Limitation

Intervals are empirical error bands based on historical backtesting. They are not exact probabilistic prediction intervals and should not be interpreted as calibrated probabilities without further validation.

Forecast rows: `1,960`
Series: `70`
Execution seconds: `0.18`

## Aggregate Error Statistics

| metric | mean_value |
| --- | --- |
| error_std | 85.411 |
| mae_historical | 70.292 |
| rmse_historical | 91.288 |
| error_percentile_80 | 107.335 |
| error_percentile_95 | 172.882 |
| coverage_80_approx | 0.79 |
| coverage_95_approx | 0.929 |

## Highest Historical MAE Series

| unique_id | store_id | dept_id | mae_historical | rmse_historical | bias_mean | coverage_95_approx |
| --- | --- | --- | --- | --- | --- | --- |
| WI_2_FOODS_3 | WI_2 | FOODS_3 | 270.0 | 374.583 | 65.429 | 0.929 |
| CA_2_FOODS_3 | CA_2 | FOODS_3 | 254.321 | 403.771 | 166.036 | 0.929 |
| WI_2_FOODS_2 | WI_2 | FOODS_2 | 242.541 | 296.968 | 61.369 | 0.929 |
| WI_3_FOODS_3 | WI_3 | FOODS_3 | 212.571 | 270.693 | 7.5 | 0.929 |
| CA_1_FOODS_3 | CA_1 | FOODS_3 | 176.082 | 211.47 | 151.451 | 0.929 |
| CA_3_FOODS_3 | CA_3 | FOODS_3 | 172.23 | 218.788 | -10.221 | 0.929 |
| WI_2_HOUSEHOLD_1 | WI_2 | HOUSEHOLD_1 | 172.071 | 229.977 | -31.857 | 0.929 |
| TX_3_FOODS_3 | TX_3 | FOODS_3 | 159.686 | 180.443 | 119.714 | 0.929 |
| WI_1_FOODS_3 | WI_1 | FOODS_3 | 133.034 | 174.915 | -2.743 | 0.929 |
| TX_2_FOODS_3 | TX_2 | FOODS_3 | 109.971 | 137.045 | -17.183 | 0.929 |
