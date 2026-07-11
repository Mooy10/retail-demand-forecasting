# GitHub Release Audit

Generated during Phase 11 repository preparation.

## Repository Size

- Approximate current local folder size, excluding .git: **1317.49 MB**.
- Approximate publishable size after excluding virtual environment, raw data, processed data, caches, and binary artifacts: **2.71 MB**.

## Files That Should Be Published

- README.md
- CASE_STUDY.md
- LICENSE
- CONTRIBUTING.md
- SECURITY.md
- requirements.txt
- pyproject.toml
- src/
- dashboard/
- tests/
- docs/
- config/inventory_assumptions.yaml
- Markdown reports in reports/
- Selected notebooks in notebooks/
- Placeholder .gitkeep files for empty data/model folders

## Files And Folders That Should Be Excluded

- .venv/
- .pytest_cache/
- .matplotlib/
- __pycache__/
- data/raw/*
- data/interim/*
- data/processed/*
- models/baseline/*
- models/machine_learning/*
- *.csv, *.parquet, *.zip, *.pkl, *.joblib, *.model, *.bin
- Local logs and temporary files
- Kaggle credentials and environment files

## Large Files Found Locally

- data\raw\sell_prices.csv - 193.97 MB
- data\raw\sales_train_evaluation.csv - 116.1 MB
- data\raw\sales_train_validation.csv - 114.45 MB
- .venv\Lib\site-packages\xgboost\lib\xgboost.dll - 95.27 MB
- data\raw\m5-forecasting-accuracy.zip - 45.79 MB
- .venv\Lib\site-packages\_duckdb.cp314-win_amd64.pyd - 35.59 MB
- data\processed\ml_store_department_advanced_features.parquet - 26.66 MB
- .venv\Lib\site-packages\pyarrow\arrow.dll - 21.1 MB
- .venv\Lib\site-packages\numpy.libs\libscipy_openblas64_-b788215d9d47792bcba3a2e2a7114320.dll - 19.46 MB
- .venv\Lib\site-packages\scipy.libs\libscipy_openblas-197ee2fc9b4d071f7e048078cac74115.dll - 19.32 MB
- .venv\share\jupyter\nbextensions\pydeck\index.js.map - 18.51 MB
- .venv\Lib\site-packages\pydeck\nbextension\static\index.js.map - 18.51 MB
- .venv\Lib\site-packages\pyarrow\arrow_flight.dll - 12.92 MB
- data\processed\ml_store_department_features.parquet - 11.71 MB
- .venv\Lib\site-packages\pyarrow\arrow_compute.dll - 9.06 MB

These are expected local artifacts. They are excluded by .gitignore and should not be committed.

## Possible Secrets Or Credentials

- .\.gitignore:84:kaggle.json
- .\.gitignore:86:access_token
- .\.gitignore:87:*access_token*
- .\SECURITY.md:20:If you find a secret committed by mistake or a security concern in the project, open a private communication channel with the repository owner before creating a public issue.
- .\src\download_data.py:18:ACCESS_TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
- .\src\download_data.py:19:LEGACY_KAGGLE_JSON_PATH = Path.home() / ".kaggle" / "kaggle.json"
- .\src\download_data.py:30:   $env:KAGGLE_API_TOKEN="your_kaggle_api_token"
- .\src\download_data.py:33:   Save your token value at: {ACCESS_TOKEN_PATH}
- .\src\download_data.py:35:4. Legacy kaggle.json file:
- .\src\download_data.py:36:   Place kaggle.json at: {LEGACY_KAGGLE_JSON_PATH}

The hits above were reviewed. The publishable project should contain only placeholder credential instructions, not real tokens.

## Local Paths Found

- .\docs\DEMO_RECORDING_GUIDE.md:11:2. Open `http://localhost:8501`.

The final README uses relative paths. Existing generated reports may still mention local execution paths; treat those as local artifacts or regenerate them with relative paths before publication if they are included.

## Recommendations

- Do not commit .venv/, raw M5 data, processed Parquet files, CSV metric outputs, or trained model binaries.
- Keep curated dashboard screenshots under docs/images/.
- Keep Mermaid source files and PNG diagrams under docs/.
- Review git status --short --ignored before the first public commit.
- If publishing notebooks, clear sensitive outputs and confirm they open correctly.
- Re-run tests before final commit.

## Release Readiness

The repository is structurally ready for a public portfolio release once ignored files are confirmed excluded and the final commit is created locally.



