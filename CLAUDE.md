# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state: template vs. actual project

This repo was forked from the neuefische `ds-modeling-pipeline` template. **Everything under
`modeling/` and `notebooks/EDA-and-modeling.ipynb` is still template code about the coffee quality
dataset** — altitude feature engineering, a linear regression on `quality_score`, MLflow logging.
None of it relates to HydroLoad-AI.

The actual project is water level / hydrological load forecasting for the Rhine gauge at **Kaub**:

- `data/raw/water_level_15min_kaub.csv` — 15-minute gauge readings from 2000 onward, ~935k rows.
  **Semicolon-separated**, columns `timestamp;value` (value in cm, `timestamp` format `YYYY-MM-DD HH:MM`).
  Read it with `pd.read_csv(..., sep=";")` — the default comma separator silently yields one column.
- `notebooks/Data_import.ipynb` — pulls hourly weather covariates for the Kaub area (lat 50.08,
  lon 7.76) from the Open-Meteo archive API (`temperature_2m`, `precipitation`, `rain`, `snowfall`,
  `pressure_msl`, `wind_*`, `et0_fao_evapotranspiration`), timezone `Europe/Berlin`.

When asked to build the real pipeline, treat the coffee code as a scaffold to replace, not as
something to keep consistent with. Confirm with the user before deleting template files.

## Commands

Everything runs through `uv`; there is no pip/conda/pyenv step. `uv run` re-syncs the env first,
so prefer it over activating `.venv`.

```bash
make setup      # uv sync --all-groups   (runtime + dev; downloads Python >=3.11 if needed)
make lab        # uv run jupyter lab
make mlflow     # uv run mlflow ui       -> http://127.0.0.1:5000
make train      # uv run python -m modeling.train
make predict    # uv run python -m modeling.predict models/linear data/X_test.csv data/y_test.csv
make test       # uv run pytest
make format     # uv run black .
```

Single test: `uv run pytest path/to/test_file.py::test_name`. No tests exist yet — pytest and
`testbook` (for testing notebooks) are installed but `tests/` has not been created.

Dependencies: `uv add <pkg>` / `uv add --group dev <pkg>` / `uv remove <pkg>` — never `pip install`,
it would desync `uv.lock`. CI should use `uv sync --frozen`. See `UV_SETUP.md` for the full guide.

## MLflow configuration

`modeling/config.py` resolves the tracking URI in this order: the contents of a local `.mlflow_uri`
file, then the `MLFLOW_URI` env var, then empty. `.mlflow_uri` and `mlflow.db` are gitignored and
must never be committed. `EXPERIMENT_NAME` is a module constant in the same file; the experiment must
already exist on the tracking server (`uv run mlflow experiments create --experiment-name <name>`).

Models are saved with `mlflow.sklearn.save_model` to a local path (`models/linear`), not logged as
MLflow artifacts — artifact logging needs an S3/AWS backend that is not set up here.

## Conventions that bite

- **Working directory matters for notebooks.** `Data_import.ipynb` uses repo-root-relative paths
  (`data/raw/...`), while `EDA-and-modeling.ipynb` uses `sys.path.append("..")` and `../.mlflow_uri`,
  i.e. it expects to run from `notebooks/`. Pick one and be explicit when adding cells.
- **`.gitignore` excludes `data/*.csv` and `models/*`**, so trained models and generated
  train/test splits stay local. `data/raw/` is not matched by that pattern, so large raw CSVs there
  *would* be committed — keep that in mind before `git add`.
- **Notebook merge conflicts**: nbdime is in the dev group. Run `nbdime config-git --enable` once,
  then `nbdime mergetool` when a `.ipynb` conflicts.
- Feature engineering functions in `modeling/feature_engineering.py` are pure `DataFrame -> DataFrame`
  transforms, and `train.py` applies the identical chain to train and test. Any new pipeline should
  keep that shape so `predict.py` can reuse it on raw input.

## GitHub automation

`.github/workflows/` contains four workflows that move issues and PRs across the neuefische org
project board (project 7) via the GraphQL API. They require org-level secrets and do no build,
test, or lint work — they will not validate a PR.
