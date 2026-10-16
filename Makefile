.PHONY: setup data hydro lab mlflow app test format

setup:
	uv sync --all-groups

# Kaub-only processed files + weather cache (re-downloads weather; add --offline to reuse the cache)
data:
	uv run python -m modeling.data_pipeline

# Daily table Kaub + Bingen + discharge + weather (data/processed/kaub_hydro_weather_daily.csv)
hydro:
	uv run python -m modeling.hydro_pipeline

lab:
	uv run python -m jupyterlab

app:
	uv run streamlit run modeling/App.py

mlflow:
	uv run mlflow ui

test:
	uv run python -m pytest

format:
	uv run black .
