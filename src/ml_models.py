"""Machine learning model wrappers for Phase 6 forecasting."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - dependency is validated by integration tests.
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


RANDOM_STATE = 42


@dataclass
class FittedMLModel:
    name: str
    pipeline: Pipeline
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    params: dict[str, Any]
    train_seconds: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.pipeline.predict(X[self.feature_columns])
        return np.maximum(np.asarray(predictions, dtype="float64"), 0.0)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


BASE_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "hist_gradient_boosting": {
        "max_iter": 40,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.05,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 2,
    },
    "lightgbm": {
        "n_estimators": 50,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "regression",
        "random_state": RANDOM_STATE,
        "n_jobs": 2,
        "verbosity": -1,
    },
}

HYPERPARAMETER_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "xgboost": [
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.95, "colsample_bytree": 0.9},
        {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.95},
    ],
    "lightgbm": [
        {"n_estimators": 50, "num_leaves": 31, "learning_rate": 0.07, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 50, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 50, "num_leaves": 45, "learning_rate": 0.035, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 50, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.9},
        {"n_estimators": 50, "num_leaves": 25, "learning_rate": 0.04, "subsample": 0.95, "colsample_bytree": 0.95},
    ],
}


def make_estimator(model_name: str, params: dict[str, Any] | None = None):
    model_params = dict(BASE_MODEL_CONFIGS[model_name])
    if params:
        model_params.update(params)
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**model_params)
    if model_name == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not available")
        return XGBRegressor(**model_params)
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not available")
        return LGBMRegressor(**model_params)
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(
    model_name: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
    params: dict[str, Any] | None = None,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", make_estimator(model_name, params))])


def fit_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    categorical_columns: list[str],
    numeric_columns: list[str],
    params: dict[str, Any] | None = None,
) -> FittedMLModel:
    feature_columns = categorical_columns + numeric_columns
    pipeline = build_pipeline(model_name, categorical_columns, numeric_columns, params)
    started = time.perf_counter()
    pipeline.fit(X_train[feature_columns], y_train)
    train_seconds = time.perf_counter() - started
    effective_params = dict(BASE_MODEL_CONFIGS[model_name])
    if params:
        effective_params.update(params)
    return FittedMLModel(
        name=model_name,
        pipeline=pipeline,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        params=effective_params,
        train_seconds=train_seconds,
    )


def transformed_feature_names(fitted: FittedMLModel) -> list[str]:
    return list(fitted.pipeline.named_steps["preprocess"].get_feature_names_out())


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024**2 if path.exists() else 0.0