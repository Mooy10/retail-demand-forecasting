"""Initial ingestion checks for the M5 Forecasting Accuracy dataset.

This script loads the main raw files, reports table shapes, columns, basic null
counts, and calendar date coverage. It writes a Markdown summary under reports/.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR, REPORTS_DIR


DATASET_FILES = {
    "calendar": "calendar.csv",
    "sales_train_validation": "sales_train_validation.csv",
    "sell_prices": "sell_prices.csv",
    "sample_submission": "sample_submission.csv",
}

SUMMARY_PATH = REPORTS_DIR / "data_ingestion_summary.md"


def print_step(message: str) -> None:
    """Print a standard progress message."""
    print(f"[data_ingestion] {message}")


def validate_required_files() -> list[Path]:
    """Ensure all required raw files are available before loading."""
    missing_files = [name for name in DATASET_FILES.values() if not (RAW_DATA_DIR / name).exists()]
    if missing_files:
        missing = "\n".join(f"  - {file_name}" for file_name in missing_files)
        raise FileNotFoundError(
            "Missing required raw dataset files in data/raw:\n"
            f"{missing}\n\n"
            "Run: python src/download_data.py"
        )
    return [RAW_DATA_DIR / name for name in DATASET_FILES.values()]


def load_tables() -> dict[str, pd.DataFrame]:
    """Load the required M5 raw files into pandas DataFrames."""
    validate_required_files()
    tables: dict[str, pd.DataFrame] = {}

    for table_name, file_name in DATASET_FILES.items():
        path = RAW_DATA_DIR / file_name
        print_step(f"Loading {file_name}...")
        tables[table_name] = pd.read_csv(path)

    return tables


def summarize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Return null counts and percentages for a DataFrame."""
    null_count = df.isna().sum()
    null_pct = (null_count / len(df) * 100).round(2) if len(df) else null_count
    return pd.DataFrame({"null_count": null_count, "null_pct": null_pct})


def calendar_date_summary(calendar: pd.DataFrame) -> dict[str, str]:
    """Validate and summarize the calendar date range."""
    if "date" not in calendar.columns:
        raise KeyError("calendar.csv must contain a 'date' column.")

    parsed_dates = pd.to_datetime(calendar["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())

    if invalid_dates == len(calendar):
        return {
            "min_date": "not available",
            "max_date": "not available",
            "invalid_dates": str(invalid_dates),
            "unique_dates": "0",
        }

    return {
        "min_date": parsed_dates.min().strftime("%Y-%m-%d"),
        "max_date": parsed_dates.max().strftime("%Y-%m-%d"),
        "invalid_dates": str(invalid_dates),
        "unique_dates": str(parsed_dates.nunique()),
    }


def build_summary_markdown(tables: dict[str, pd.DataFrame]) -> str:
    """Build a Markdown ingestion report."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# Data Ingestion Summary",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Source Directory",
        "",
        f"`{RAW_DATA_DIR}`",
        "",
        "## Loaded Tables",
        "",
    ]

    for table_name, df in tables.items():
        lines.extend(
            [
                f"### {table_name}",
                "",
                f"- Rows: `{df.shape[0]:,}`",
                f"- Columns: `{df.shape[1]:,}`",
                f"- Column names: `{', '.join(df.columns.astype(str))}`",
                "",
                "Top columns by null count:",
                "",
                "| column | null_count | null_pct |",
                "|---|---:|---:|",
            ]
        )

        null_summary = summarize_nulls(df).sort_values("null_count", ascending=False).head(15)
        for column_name, row in null_summary.iterrows():
            lines.append(f"| {column_name} | {int(row['null_count']):,} | {row['null_pct']:.2f}% |")
        lines.append("")

    calendar_summary = calendar_date_summary(tables["calendar"])
    lines.extend(
        [
            "## Calendar Date Validation",
            "",
            f"- Minimum date: `{calendar_summary['min_date']}`",
            f"- Maximum date: `{calendar_summary['max_date']}`",
            f"- Unique parsed dates: `{calendar_summary['unique_dates']}`",
            f"- Invalid date values: `{calendar_summary['invalid_dates']}`",
            "",
        ]
    )

    return "\n".join(lines)


def write_summary(summary_markdown: str) -> None:
    """Write the ingestion summary report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary_markdown, encoding="utf-8")
    print_step(f"Summary written to: {SUMMARY_PATH}")


def print_console_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Print key validation details to the console."""
    for table_name, df in tables.items():
        print_step(f"{table_name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
        print(f"  Columns: {', '.join(df.columns.astype(str))}")
        nulls = summarize_nulls(df)
        total_nulls = int(nulls["null_count"].sum())
        print(f"  Total null values: {total_nulls:,}")

    date_summary = calendar_date_summary(tables["calendar"])
    print_step(
        "calendar date range: "
        f"{date_summary['min_date']} to {date_summary['max_date']} "
        f"({date_summary['invalid_dates']} invalid date values)"
    )


def main() -> int:
    """Run initial raw data ingestion validation."""
    tables = load_tables()
    print_console_summary(tables)
    write_summary(build_summary_markdown(tables))
    print_step("Initial ingestion validation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
