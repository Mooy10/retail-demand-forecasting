"""Project-wide path configuration for Retail Demand Forecasting."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SQL_DIR = PROJECT_ROOT / "sql"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
ML_MODELS_DIR = MODELS_DIR / "machine_learning"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

M5_COMPETITION_NAME = "m5-forecasting-accuracy"

EXPECTED_RAW_FILES = [
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
    "sample_submission.csv",
]
