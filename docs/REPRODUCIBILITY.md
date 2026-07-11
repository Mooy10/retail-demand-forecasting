# Reproducibility Guide

## Environment

- Python: 3.10 or higher.
- Validated local environment: Python 3.14 virtual environment on Windows PowerShell.
- Package versions are intentionally not pinned in `requirements.txt` because the project was developed as a portfolio workflow and exact production pinning has not been fully cross-platform tested.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

The project uses the public Kaggle M5 Forecasting Accuracy dataset. Before downloading, accept the competition rules on Kaggle and configure Kaggle authentication.

```powershell
python src\download_data.py
python src\data_ingestion.py
```

## Recommended Execution Order

```powershell
python src\data_understanding.py
python src\demand_segmentation.py
python src\forecasting_dataset.py
python src\run_baseline_forecasting.py
python src\ml_feature_engineering.py
python src\run_ml_forecasting.py
python src\advanced_feature_engineering.py
python src\run_advanced_ml_forecasting.py
python src\model_selector.py
python src\build_hybrid_forecast.py
python src\run_out_of_sample_model_selection.py
python src\build_official_forecast.py
python src\forecast_uncertainty.py
python src\inventory_simulation.py
python src\inventory_optimization.py
python src\inventory_economic_analysis.py
```

## Dashboard

```powershell
streamlit run dashboard\app.py
```

The dashboard expects processed artifacts under `data/processed/`, `reports/`, and `config/`.

## Thread Limits

For local machines, use conservative thread settings:

```powershell
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:OPENBLAS_NUM_THREADS="2"
```

XGBoost and LightGBM scripts use limited jobs where configured.

## Approximate Phase Timing

Timings depend on hardware and installed package versions.

- Data ingestion: seconds to a few minutes after download.
- Data understanding: minutes.
- Demand segmentation: minutes.
- Baselines: minutes.
- ML and advanced ML: longer-running local phases.
- Out-of-sample validation: minutes to longer depending on hardware.
- Inventory simulation: seconds to minutes.
- Dashboard startup: about 8 seconds in the validated local run.

## Expected Outputs

- Processed Parquet files in `data/processed/`.
- Markdown and CSV summaries in `reports/`.
- Streamlit dashboard under `dashboard/`.
- Diagrams and portfolio images under `docs/`.

## Troubleshooting

- If Kaggle download fails, confirm credentials and competition rule acceptance.
- If Parquet loading fails, install `pyarrow`.
- If LightGBM or XGBoost fails, reinstall dependencies inside the virtual environment.
- If Streamlit starts but pages are empty, confirm the processed artifacts exist.
- If tests fail because data files are missing, run the required pipeline phases first.
