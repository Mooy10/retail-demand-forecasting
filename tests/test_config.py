from pathlib import Path

from src import config


def test_project_root_is_repository_root():
    assert config.PROJECT_ROOT.name == "retail-demand-forecasting"
    assert (config.PROJECT_ROOT / "README.md").exists()


def test_core_directories_are_configured():
    expected_dirs = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.INTERIM_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.REPORTS_DIR,
        config.DOCS_DIR,
        config.TESTS_DIR,
    ]

    for directory in expected_dirs:
        assert isinstance(directory, Path)
        assert directory.exists()


def test_m5_expected_raw_files_are_declared():
    assert config.M5_COMPETITION_NAME == "m5-forecasting-accuracy"
    assert set(config.EXPECTED_RAW_FILES) == {
        "calendar.csv",
        "sell_prices.csv",
        "sales_train_validation.csv",
        "sample_submission.csv",
    }
