# Gemini Workspace Context

## Project Overview
* the project is a deep learning project.
* the data is time series data, stores at ./data 
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


## Dependency and Virtual Env Management

This project uses `uv` for managing Python dependencies and running scripts.

* **Dependency Management:** The project uses `uv` for managing Python dependencies, as specified in `pyproject.toml` (or `requirements.txt`).
* **Python Environment:** When installing Python dependencies, prefer `uv pip install` and ensure you are working within a virtual environment created by `uv venv`.
* **Running Code:** When running Python tools or scripts, prefer to use `uv run` or `uvx` where possible.

## Example Commands for Gemini CLI
Here are some example commands for your reference:
*   To run a script (e.g., `main.py`): `uv run python main.py`


## 记住：当你需要写代码的时候，把代码存在 ./GEMINI_code 里面

