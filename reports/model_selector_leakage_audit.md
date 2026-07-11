# Model Selector Leakage Audit

## Current Phase 7 Selector

`src/model_selector.py` loads `baseline_metrics.parquet`, `ml_metrics.parquet`, and `advanced_ml_metrics.parquet`. Those metric files contain all three backtesting windows: `window_1`, `window_2`, and `window_3`.

## Current Phase 7 Hybrid Evaluation

`src/build_hybrid_forecast.py` loads the registry created from all three windows and evaluates `hybrid_selected` on predictions from the same three windows.

## Overlap

There is complete overlap: the same windows used to choose the best model per series are also used to report the hybrid result.

## Conclusion

The Phase 7 WAPE `0.117336` is not a true out-of-sample model-selection estimate. It should be labeled as `in-sample model selection`. It is useful as an upper-bound diagnostic, but it is exposed to data snooping and selection bias.

## Corrective Validation

This phase adds a strict holdout validation that selects models using windows 1-2 only and evaluates on window 3, plus a rolling-origin validation that evaluates window 2 after training on window 1 and window 3 after training on windows 1-2.
