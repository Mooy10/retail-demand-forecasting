"""Shared utilities for simulated inventory planning assumptions."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

try:
    from config import PROJECT_ROOT
except ModuleNotFoundError:
    from src.config import PROJECT_ROOT

ASSUMPTIONS_FILE = PROJECT_ROOT / "config" / "inventory_assumptions.yaml"

DEFAULT_ASSUMPTIONS = {
    "service_level_default": 0.95,
    "lead_time_days_default": 7,
    "review_period_days_default": 7,
    "ordering_cost_default": 75.0,
    "holding_cost_rate_annual_default": 0.20,
    "stockout_cost_per_unit_default": 3.0,
    "initial_inventory_days_default": 14,
    "minimum_order_quantity_default": 1,
    "order_rounding_multiple_default": 1,
    "simulated_unit_cost_default": 5.0,
    "days_per_year": 365,
}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "{}":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')


def load_inventory_assumptions(path: Path = ASSUMPTIONS_FILE) -> dict[str, Any]:
    """Load the simple YAML assumptions file without requiring PyYAML."""
    config: dict[str, Any] = {"defaults": dict(DEFAULT_ASSUMPTIONS), "overrides": {"state_id": {}, "store_id": {}, "dept_id": {}, "unique_id": {}}}
    if not path.exists():
        return config
    section = None
    override_type = None
    current_key = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            section = stripped[:-1]
            override_type = None
            current_key = None
            continue
        if section == "defaults" and indent >= 2 and ":" in stripped:
            key, value = stripped.split(":", 1)
            config["defaults"][key.strip()] = _parse_scalar(value)
        elif section == "overrides":
            if indent == 2 and stripped.endswith(":"):
                override_type = stripped[:-1]
                config["overrides"].setdefault(override_type, {})
                current_key = None
            elif indent == 4 and stripped.endswith(":") and override_type:
                current_key = stripped[:-1]
                config["overrides"].setdefault(override_type, {}).setdefault(current_key, {})
            elif indent >= 6 and ":" in stripped and override_type and current_key:
                key, value = stripped.split(":", 1)
                config["overrides"][override_type][current_key][key.strip()] = _parse_scalar(value)
    return config


def derive_state_id(store_id: str) -> str:
    return str(store_id).split("_")[0]


def assumptions_for_series(row: pd.Series | dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_inventory_assumptions()
    result = dict(config.get("defaults", DEFAULT_ASSUMPTIONS))
    unique_id = str(row.get("unique_id"))
    store_id = str(row.get("store_id"))
    dept_id = str(row.get("dept_id"))
    state_id = str(row.get("state_id", derive_state_id(store_id)))
    for level, key in [("state_id", state_id), ("store_id", store_id), ("dept_id", dept_id), ("unique_id", unique_id)]:
        result.update(config.get("overrides", {}).get(level, {}).get(key, {}))
    result["state_id"] = state_id
    result["assumption_source"] = "defaults_with_optional_overrides"
    return result


def z_score(service_level: float) -> float:
    service_level = min(max(float(service_level), 0.50), 0.999)
    return float(NormalDist().inv_cdf(service_level))


def round_order_quantity(quantity: float, minimum: float, multiple: float) -> float:
    quantity = max(0.0, float(quantity))
    if quantity <= 0:
        return 0.0
    quantity = max(quantity, float(minimum))
    multiple = max(float(multiple), 1.0)
    return float(math.ceil(quantity / multiple) * multiple)


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "Critical"
    if score >= 0.50:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or pd.isna(denominator):
        return default
    return float(numerator) / float(denominator)


def nonnegative(values):
    return np.maximum(np.asarray(values, dtype="float64"), 0.0)
