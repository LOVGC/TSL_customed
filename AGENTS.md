# Workspace Context

## Project Overview
* the project is a deep learning project.
* the data is time series data, stores at ./dataset
* the tasks are prediction or classification

### Dataset and dataloader

* Dataset and dataloader are implemented at ./data_provider
* ./data_provider/data_factory.py is the dispatcher, def data_provider(args, flag) returns the dataset and dataloader
* ./data_provider/data_loader.py implements the specific Dataset 

### Model
* models are implemented at ./models using modules at ./layers

### Train, valid, test
* Train, valid, test are implemented at ./exp

### Define and Launch an experiment 
* the bash scripts used to run an experiment are defined at ./scripts 
* ./run.py is the top-level program that defined configuratio parameters for an experiment


## Codebase Summary
* Entry point: `run.py` defines CLI args, selects the experiment class, builds the run setting string, and handles train/test flow with checkpoints under `./checkpoints/` and outputs under `./results/` and `./test_results/`.
* Experiments: `exp/exp_long_term_forecasting.py` (forecasting, MSE + Adam, optional DTW metric) and `exp/exp_classification.py` (classification, CrossEntropy + RAdam, padding masks); `exp/exp_basic.py` handles device selection and model registry.
* Model: `models/TimesNet.py` implements TimesNet with FFT-based period discovery and Inception-style 2D conv blocks; supports forecast/imputation/anomaly/classification heads. Only TimesNet is present even though `run.py` lists other models/tasks.
* Data pipeline: `data_provider/data_factory.py` dispatches datasets and builds `DataLoader`s; `data_provider/data_loader.py` implements ETT hour/minute, custom CSV, Berkley sensor, simulated AR, C-MAPSS RUL, UEA .ts classification, and Real Doppler Kaggle (auto process/split). Variable-length classification uses `data_provider/uea.py` collate.
* Utilities: `utils/` provides time feature encoding, metrics, LR schedules + early stopping, and rich time-series augmentation (jitter/scaling/warps/DTW-based). Example runs live in `scripts/` (use `uv run python ...` per project rule).
* Misc: `GEMINI_code/` contains helper scripts for radar data processing/splitting/verification; `test_cmapss_loader.py` is a dataloader sanity check; `pyproject.toml` defines dependencies and torch index settings.

