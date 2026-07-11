"""Reusable baseline forecasting models for Phase 5."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_array(y) -> np.ndarray:
    return np.asarray(y, dtype="float64")


def _nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype="float64"), 0.0)


def naive_forecast(y, horizon: int = 28) -> np.ndarray:
    y = _as_float_array(y)
    last_value = y[-1] if len(y) else 0.0
    return _nonnegative(np.repeat(last_value, horizon))


def seasonal_naive_forecast(y, horizon: int = 28, season_length: int = 7) -> np.ndarray:
    y = _as_float_array(y)
    if len(y) == 0:
        return np.zeros(horizon, dtype="float64")
    if len(y) < season_length:
        return naive_forecast(y, horizon)
    season = y[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return _nonnegative(np.tile(season, reps)[:horizon])


def historical_mean_forecast(y, horizon: int = 28) -> np.ndarray:
    y = _as_float_array(y)
    value = float(np.mean(y)) if len(y) else 0.0
    return _nonnegative(np.repeat(value, horizon))


def moving_average_forecast(y, horizon: int = 28, window: int = 7) -> np.ndarray:
    y = _as_float_array(y)
    if len(y) == 0:
        return np.zeros(horizon, dtype="float64")
    value = float(np.mean(y[-window:]))
    return _nonnegative(np.repeat(value, horizon))


def seasonal_average_by_weekday_forecast(
    y,
    history_dates,
    future_dates,
    horizon: int = 28,
) -> np.ndarray:
    y = _as_float_array(y)
    if len(y) == 0:
        return np.zeros(horizon, dtype="float64")
    history = pd.DataFrame({"date": pd.to_datetime(history_dates), "demand": y})
    history["weekday"] = history["date"].dt.dayofweek
    weekday_mean = history.groupby("weekday", observed=True)["demand"].mean()
    fallback = float(history["demand"].mean())
    future_weekdays = pd.Series(pd.to_datetime(future_dates)).dt.dayofweek.to_numpy()
    preds = [float(weekday_mean.get(day, fallback)) for day in future_weekdays[:horizon]]
    return _nonnegative(np.asarray(preds))


def croston_classic_forecast(y, horizon: int = 28, alpha: float = 0.1) -> np.ndarray:
    """Croston classic forecast for intermittent demand.

    alpha controls exponential smoothing for non-zero demand size and interval.
    Fully-zero series return zero forecasts.
    """
    y = _as_float_array(y)
    nonzero_idx = np.flatnonzero(y > 0)
    if len(nonzero_idx) == 0:
        return np.zeros(horizon, dtype="float64")

    z = y[nonzero_idx[0]]
    p = float(nonzero_idx[0] + 1)
    last_nonzero = nonzero_idx[0]
    for idx in nonzero_idx[1:]:
        interval = float(idx - last_nonzero)
        z = alpha * y[idx] + (1 - alpha) * z
        p = alpha * interval + (1 - alpha) * p
        last_nonzero = idx
    forecast = z / p if p > 0 else 0.0
    return _nonnegative(np.repeat(forecast, horizon))


def croston_sba_forecast(y, horizon: int = 28, alpha: float = 0.1) -> np.ndarray:
    """Syntetos-Boylan Approximation correction for Croston forecasts."""
    return _nonnegative((1 - alpha / 2) * croston_classic_forecast(y, horizon, alpha))


def tsb_forecast(y, horizon: int = 28, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    """Teunter-Syntetos-Babai intermittent-demand baseline.

    alpha smooths positive demand size and beta smooths demand occurrence
    probability. Fully-zero series return zero forecasts.
    """
    y = _as_float_array(y)
    if len(y) == 0 or np.all(y <= 0):
        return np.zeros(horizon, dtype="float64")

    first_positive = y[np.flatnonzero(y > 0)[0]]
    z = float(first_positive)
    p = 1.0 if y[0] > 0 else 0.0
    for value in y:
        occurrence = 1.0 if value > 0 else 0.0
        p = beta * occurrence + (1 - beta) * p
        if value > 0:
            z = alpha * value + (1 - alpha) * z
    return _nonnegative(np.repeat(p * z, horizon))


BASELINE_MODELS = {
    "naive": naive_forecast,
    "seasonal_naive_7": lambda y, horizon=28, **kwargs: seasonal_naive_forecast(y, horizon, 7),
    "seasonal_naive_28": lambda y, horizon=28, **kwargs: seasonal_naive_forecast(y, horizon, 28),
    "historical_mean": historical_mean_forecast,
    "moving_average_7": lambda y, horizon=28, **kwargs: moving_average_forecast(y, horizon, 7),
    "moving_average_28": lambda y, horizon=28, **kwargs: moving_average_forecast(y, horizon, 28),
    "seasonal_average_weekday": seasonal_average_by_weekday_forecast,
    "croston_classic": croston_classic_forecast,
    "croston_sba": croston_sba_forecast,
    "tsb": tsb_forecast,
}