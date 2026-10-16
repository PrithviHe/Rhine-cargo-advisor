# Archived notebooks

Earlier versions, kept for reference. They are **replaced** by the pipeline notebooks and are not
needed to train the model or run the Streamlit app.

| Notebook | Replaced by | Note |
|---|---|---|
| `04_eda_and_multivariate_main.ipynb` | `../04_eda_and_multivariate_pipeline.ipynb` | Original multivariate analysis, everything written in the cells. Data paths point to `../../data/`, so it still runs from this folder. |
| `05_Real_time_data.ipynb` | `../05_Real_time_pipeline.ipynb` | Original live forecast with its own copy of every function. It loads `models/water_level_model.joblib`, which no longer exists; the app now uses `models/kaub_xgboost_04.joblib`. |

See [`docs/pipelines.md`](../../docs/pipelines.md) for the current pipeline.
