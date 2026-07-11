"""Download the Kaggle M5 Forecasting Accuracy dataset into data/raw.

The script uses the official Kaggle CLI through subprocess. It does not perform
manual API-client authentication and is safe to run multiple times.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from config import EXPECTED_RAW_FILES, M5_COMPETITION_NAME, PROJECT_ROOT, RAW_DATA_DIR


ACCESS_TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
LEGACY_KAGGLE_JSON_PATH = Path.home() / ".kaggle" / "kaggle.json"

KAGGLE_AUTH_HELP = f"""
Kaggle authentication failed or is not configured.

Use one of these official authentication options:

1. Interactive login:
   kaggle auth login

2. API Token environment variable:
   $env:KAGGLE_API_TOKEN="your_kaggle_api_token"

3. API Token file:
   Save your token value at: {ACCESS_TOKEN_PATH}

4. Legacy kaggle.json file:
   Place kaggle.json at: {LEGACY_KAGGLE_JSON_PATH}

Also confirm that you accepted the competition rules at:
https://www.kaggle.com/competitions/m5-forecasting-accuracy
""".strip()


def print_step(message: str) -> None:
    """Print a standard progress message."""
    print(f"[download_data] {message}")


def ensure_raw_data_dir() -> None:
    """Create data/raw when it does not exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print_step(f"Raw data directory ready: {RAW_DATA_DIR}")


def expected_file_status() -> dict[str, bool]:
    """Return whether each expected raw file exists."""
    return {file_name: (RAW_DATA_DIR / file_name).exists() for file_name in EXPECTED_RAW_FILES}


def all_expected_files_exist() -> bool:
    """Check whether all required raw files are already available."""
    return all(expected_file_status().values())


def print_file_status() -> None:
    """Print availability for each expected dataset file."""
    print_step("Checking expected dataset files:")
    for file_name, exists in expected_file_status().items():
        status = "found" if exists else "missing"
        print(f"  - {file_name}: {status}")


def ensure_kaggle_cli_available() -> None:
    """Fail early when the Kaggle CLI is not available in PATH."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI was not found. Install dependencies with: "
            "pip install -r requirements.txt"
        )


def run_kaggle_download() -> None:
    """Download the M5 archive through the official Kaggle CLI."""
    ensure_kaggle_cli_available()
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        M5_COMPETITION_NAME,
        "-p",
        str(RAW_DATA_DIR),
    ]

    print_step("Running Kaggle CLI download command:")
    print(f"  {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        raise RuntimeError(KAGGLE_AUTH_HELP)

    if result.stderr.strip():
        print(result.stderr.strip())


def safe_extract_zip(zip_path: Path) -> None:
    """Extract a zip file while skipping files that already exist."""
    print_step(f"Extracting archive: {zip_path.name}")
    raw_root = RAW_DATA_DIR.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            target_path = (RAW_DATA_DIR / member_path).resolve()

            if not str(target_path).startswith(str(raw_root)):
                raise RuntimeError(f"Blocked unsafe zip member path: {member.filename}")

            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            if target_path.exists():
                print(f"  - skipped existing file: {member.filename}")
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                target.write(source.read())
            print(f"  - extracted: {member.filename}")


def extract_downloaded_archives() -> None:
    """Extract zip archives in data/raw without overwriting existing files."""
    zip_files = sorted(RAW_DATA_DIR.glob("*.zip"))
    if not zip_files:
        print_step("No zip archives found in data/raw.")
        return

    for zip_path in zip_files:
        safe_extract_zip(zip_path)


def main() -> int:
    """Run the dataset download workflow."""
    ensure_raw_data_dir()
    print_file_status()

    if all_expected_files_exist():
        print_step("All expected files already exist. Nothing to download.")
        return 0

    if list(RAW_DATA_DIR.glob("*.zip")):
        print_step("Existing zip archive found. Extracting before downloading again.")
        extract_downloaded_archives()
        print_file_status()
        if all_expected_files_exist():
            print_step("All expected files are now available after extraction.")
            return 0

    try:
        run_kaggle_download()
        extract_downloaded_archives()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_file_status()
    if not all_expected_files_exist():
        print_step("Download finished, but one or more expected files are still missing.")
        return 1

    print_step("Dataset is ready in data/raw.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())