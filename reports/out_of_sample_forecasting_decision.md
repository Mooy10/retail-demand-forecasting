# Out-Of-Sample Forecasting Decision

## Leakage Audit Result

Selection bias exists in the Phase 7 `hybrid_selected` result because model selection and evaluation used the same windows. The WAPE `0.117336` is in-sample model selection, not a strict production estimate.

## Strict Holdout Decision

Holdout hybrid WAPE: `0.115443`. Baseline WAPE: `0.125912`. Improvement: `8.31%`.
Holdout hybrid RMSSE: `0.762015`. Baseline RMSSE: `0.857261`. Improvement: `11.11%`.

Decision status: `candidate_official`.

The holdout hybrid can be declared the official candidate, with seasonal_naive_28 retained as fallback.

## Dashboard Forecast Recommendation

Use the strict holdout conclusion, not the in-sample Phase 7 result, to decide the dashboard forecast. If the hybrid remains official, expose selected model and confidence; if not, use `seasonal_naive_28` and present the hybrid as experimental.

## Required User-Facing Limitations

- The selector was validated on only three backtesting windows.
- Window 3 holdout is the main out-of-sample model-selection check.
- The Phase 7 in-sample hybrid is an optimistic diagnostic and should not be the primary production metric.
- Current modeling remains at store-department level only.
