# Gemini Workspace Context

## Project Overview
* the project is a deep learning project.
* the data is time series data
* the tasks are prediction or classification


## Dependency and Virtual Env Management

This project uses `uv` for managing Python dependencies and running scripts.

* **Dependency Management:** The project uses `uv` for managing Python dependencies, as specified in `pyproject.toml` (or `requirements.txt`).
* **Python Environment:** When installing Python dependencies, prefer `uv pip install` and ensure you are working within a virtual environment created by `uv venv`.
* **Running Code:** When running Python tools or scripts, prefer to use `uv run` or `uvx` where possible.

## Example Commands for Gemini CLI
Here are some example commands for your reference:
*   To run a script (e.g., `main.py`): `uv run python main.py`
