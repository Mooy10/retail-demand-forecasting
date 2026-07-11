# Out-Of-Sample Model Selection Summary

## Methodological Finding

The Phase 7 hybrid result was selected and evaluated on the same three windows, so it is classified as `in-sample model selection` and should not be used as the main production estimate.

## In-Sample Hybrid Reference

| model | weighted_wape | weighted_rmsse | series_count |
| --- | --- | --- | --- |
| hybrid_selected | 0.1173357104289727 | 0.7888491842071963 | 70 |

## Strict Holdout Window 3

| model | weighted_mae | weighted_wape | weighted_rmsse | series_count |
| --- | --- | --- | --- | --- |
| hybrid_holdout_w3 | 123.77594056084683 | 0.11544339493236289 | 0.7620148718796077 | 70 |
| xgboost_phase6 | 119.10769420488833 | 0.11829752060039579 | 0.7396796535216278 | 70 |
| lightgbm_phase7 | 122.19582982885477 | 0.12262952338706243 | 0.7581923846353527 | 70 |
| seasonal_naive_28 | 137.32857751760872 | 0.12591223917014327 | 0.8572611657733805 | 70 |
| xgboost_phase7 | 134.6738546934292 | 0.12830581266818683 | 0.8035107549788537 | 70 |
| seasonal_average_weekday | 237.31068970143033 | 0.2111325447684642 | 1.3229952823294355 | 70 |

## Rolling-Origin Validation

| model | evaluation_window | weighted_wape | weighted_rmsse | series_count |
| --- | --- | --- | --- | --- |
| rolling_hybrid | window_2 | 0.12308331784137677 | 0.8352088925123156 | 70 |
| seasonal_naive_28 | window_2 | 0.1405149536548513 | 0.9556363899061926 | 70 |
| rolling_hybrid | window_3 | 0.11544339493236289 | 0.7620148718796077 | 70 |
| seasonal_naive_28 | window_3 | 0.12591223917014327 | 0.8572611657733805 | 70 |

## Holdout Selector Distribution

| selected_model | series |
| --- | --- |
| seasonal_naive_28 | 23 |
| seasonal_average_weekday | 16 |
| lightgbm_phase7 | 13 |
| xgboost_phase7 | 10 |
| xgboost_phase6 | 8 |

## Rolling Selector Distribution

| evaluation_window | selected_model | series |
| --- | --- | --- |
| window_2 | lightgbm_phase7 | 14 |
| window_2 | seasonal_average_weekday | 19 |
| window_2 | seasonal_naive_28 | 24 |
| window_2 | xgboost_phase6 | 9 |
| window_2 | xgboost_phase7 | 4 |
| window_3 | lightgbm_phase7 | 13 |
| window_3 | seasonal_average_weekday | 16 |
| window_3 | seasonal_naive_28 | 23 |
| window_3 | xgboost_phase6 | 8 |
| window_3 | xgboost_phase7 | 10 |

## Selection Stability

Series that changed model between selector trained on window 1 and selector trained on windows 1-2: `25` of `70`.

## Confidence Validation

| evaluation_window | confidence | series_count | mean_wape | mean_rmsse | pct_beats_baseline_wape | pct_beats_baseline_rmsse |
| --- | --- | --- | --- | --- | --- | --- |
| window_2 | Low | 22 | 0.1808830819924523 | 1.066807207216793 | 50.0 | 50.0 |
| window_2 | Medium | 48 | 0.16369165857667178 | 0.8895457791641558 | 62.5 | 62.5 |
| window_3 | High | 28 | 0.16711545981400167 | 0.821938076058539 | 50.0 | 60.71428571428571 |
| window_3 | Low | 22 | 0.15156084962441269 | 0.9148833950050759 | 27.27272727272727 | 40.909090909090914 |
| window_3 | Medium | 20 | 0.17129184211444262 | 0.8966376793803452 | 45.0 | 50.0 |

## Interpretation

On strict holdout window 3, `hybrid_holdout_w3` changes WAPE by `8.31%` and RMSSE by `11.11%` versus `seasonal_naive_28`.
