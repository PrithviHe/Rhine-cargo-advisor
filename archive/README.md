# Archive

Files that are **no longer used** by the pipelines, the Streamlit app, the tests or the notebooks.
They are kept for reference only. The folder layout mirrors where each file used to be.

| File | Was in | Why archived |
|---|---|---|
| `modeling/predict.py`, `modeling/feature_engineering.py` | `modeling/` | code from the neuefische coffee-quality template |
| `modeling/App.old.py` | `modeling/` | old version of the Streamlit app, replaced by `modeling/App.py` |
| `data/X_test.csv`, `data/y_test.csv` | `data/` | coffee-template test data, only used by `predict.py` |
| `data/processed/kaub_hydro_weather_hourly.csv` | `data/processed/` | hourly hydro-weather table from the original notebook 04; nothing reads it |
| `data/processed/kaub_live_daily.csv`, `kaub_realtime_daily.csv`, `live_status.json` | `data/processed/` | outputs of earlier live-data experiments; the live forecast now fetches data on demand |
| `images/0_general_tracking_mlflow.png` … `3_metadata.png` | `images/` | MLflow screenshots from the template README |
| `notebooks_models/*.joblib` | `notebooks/models/` | models saved by earlier notebook versions; the app uses `models/kaub_xgboost_04.joblib` |

Notebooks that were replaced by the pipeline notebooks are in [`notebooks/archive/`](../notebooks/archive).
See [`docs/pipelines.md`](../docs/pipelines.md) for the current project structure.
