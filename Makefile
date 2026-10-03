.PHONY: setup data train predict lab mlflow app test format

setup:
	uv sync --all-groups

data:
	uv run python -m modeling.data_pipeline

train:
	uv run python -m modeling.train

predict:
	uv run python -m modeling.predict models/linear data/X_test.csv data/y_test.csv

lab:
	uv run jupyter lab

app:
	uv run streamlit run modeling/App.py

mlflow:
	uv run mlflow ui

test:
	uv run pytest

format:
	uv run black .
