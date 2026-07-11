import numpy as np

from src.baseline_models import (
    BASELINE_MODELS,
    croston_classic_forecast,
    croston_sba_forecast,
    seasonal_naive_forecast,
    tsb_forecast,
)


def test_each_baseline_model_generates_28_nonnegative_predictions():
    y = np.array([0, 1, 0, 3, 0, 2, 4, 0, 1, 0, 5, 0, 2, 1], dtype=float)
    dates = np.datetime64("2020-01-01") + np.arange(len(y))
    future_dates = np.datetime64("2020-01-01") + np.arange(len(y), len(y) + 28)

    for model_name, model in BASELINE_MODELS.items():
        if model_name == "seasonal_average_weekday":
            pred = model(y, history_dates=dates, future_dates=future_dates, horizon=28)
        else:
            pred = model(y, horizon=28)
        assert len(pred) == 28
        assert np.isfinite(pred).all()
        assert (pred >= 0).all()


def test_seasonal_naive_7_uses_only_previous_7_days():
    y = np.arange(1, 21, dtype=float)
    pred = seasonal_naive_forecast(y, horizon=28, season_length=7)
    expected_first_week = y[-7:]
    assert np.array_equal(pred[:7], expected_first_week)
    assert np.array_equal(pred[7:14], expected_first_week)


def test_zero_series_does_not_break_intermittent_methods():
    y = np.zeros(50, dtype=float)
    for model in [croston_classic_forecast, croston_sba_forecast, tsb_forecast]:
        pred = model(y, horizon=28)
        assert len(pred) == 28
        assert np.array_equal(pred, np.zeros(28))