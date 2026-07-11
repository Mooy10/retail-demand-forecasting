import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


SEGMENTATION_PATH = PROCESSED_DATA_DIR / "demand_segmentation.parquet"
SEGMENT_SUMMARY_PATH = PROCESSED_DATA_DIR / "segment_summary.parquet"
ABC_XYZ_SUMMARY_PATH = PROCESSED_DATA_DIR / "abc_xyz_summary.parquet"
PATTERN_BY_CATEGORY_PATH = PROCESSED_DATA_DIR / "demand_pattern_by_category.parquet"
PATTERN_BY_STORE_PATH = PROCESSED_DATA_DIR / "demand_pattern_by_store.parquet"

EXPECTED_PATTERNS = {"Smooth", "Erratic", "Intermittent", "Lumpy"}


def test_segmentation_outputs_exist_and_are_not_empty():
    for path in [
        SEGMENTATION_PATH,
        SEGMENT_SUMMARY_PATH,
        ABC_XYZ_SUMMARY_PATH,
        PATTERN_BY_CATEGORY_PATH,
        PATTERN_BY_STORE_PATH,
    ]:
        assert path.exists(), f"Missing output: {path}"
        df = pd.read_parquet(path)
        assert not df.empty


def test_all_series_have_required_classes():
    segmentation = pd.read_parquet(SEGMENTATION_PATH)
    assert segmentation["demand_pattern"].notna().all()
    assert segmentation["abc_class"].notna().all()
    assert segmentation["xyz_class"].notna().all()
    assert (segmentation["demand_pattern"] != "Unclassified").all()


def test_adi_and_cv_squared_are_non_negative():
    segmentation = pd.read_parquet(SEGMENTATION_PATH)
    assert (segmentation["adi"] >= 0).all()
    assert (segmentation["cv_squared"] >= 0).all()


def test_segment_demand_matches_total_demand():
    segmentation = pd.read_parquet(SEGMENTATION_PATH)
    segment_summary = pd.read_parquet(SEGMENT_SUMMARY_PATH)
    abc_xyz_summary = pd.read_parquet(ABC_XYZ_SUMMARY_PATH)

    total_demand = int(segmentation["total_demand"].sum())
    assert int(segment_summary["total_demand"].sum()) == total_demand
    assert int(abc_xyz_summary["total_demand"].sum()) == total_demand


def test_processed_total_demand_matches_raw_sales_file():
    raw_header = pd.read_csv(RAW_DATA_DIR / "sales_train_validation.csv", nrows=0)
    demand_cols = [column for column in raw_header.columns if column.startswith("d_")]
    dtype_map = {column: "int16" for column in demand_cols}
    raw_sales = pd.read_csv(RAW_DATA_DIR / "sales_train_validation.csv", usecols=demand_cols, dtype=dtype_map)
    raw_total = int(raw_sales.sum(axis=0).sum())

    segmentation = pd.read_parquet(SEGMENTATION_PATH)
    assert int(segmentation["total_demand"].sum()) == raw_total


def test_expected_demand_patterns_when_present_in_real_data():
    segmentation = pd.read_parquet(SEGMENTATION_PATH)
    observed_patterns = set(segmentation["demand_pattern"].unique())
    assert observed_patterns.issubset(EXPECTED_PATTERNS)
    if len(observed_patterns) == 4:
        assert observed_patterns == EXPECTED_PATTERNS


def test_id_is_unique():
    segmentation = pd.read_parquet(SEGMENTATION_PATH)
    assert not segmentation["id"].duplicated().any()


def test_first_sale_date_is_not_after_last_sale_date():
    segmentation = pd.read_parquet(SEGMENTATION_PATH)
    dated = segmentation.dropna(subset=["first_sale_date", "last_sale_date"])
    assert (dated["first_sale_date"] <= dated["last_sale_date"]).all()


def test_expected_abc_xyz_values():
    segmentation = pd.read_parquet(SEGMENTATION_PATH)
    assert set(segmentation["abc_class"].unique()).issubset({"A", "B", "C"})
    assert set(segmentation["xyz_class"].unique()).issubset({"X", "Y", "Z"})
    assert segmentation["abc_xyz_segment"].str.match(r"^[ABC][XYZ]$").all()