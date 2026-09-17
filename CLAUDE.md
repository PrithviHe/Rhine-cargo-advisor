# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state: template vs. actual project

This repo was forked from the neuefische `ds-modeling-pipeline` template. **Everything under
`modeling/` and `notebooks/EDA-and-modeling.ipynb` is still template code about the coffee quality
dataset** — altitude feature engineering, a linear regression on `quality_score`, MLflow logging.
None of it relates to HydroLoad-AI.

The actual project is water level / hydrological load forecasting for the Rhine gauge at **Kaub**:

- `data/raw/water_level_15min_kaub.csv` — 15-minute gauge readings from 2000 onward, 935,128 rows.
  **Semicolon-separated**, columns `timestamp;value` (value in cm, `timestamp` format `YYYY-MM-DD HH:MM`).
  Read it with `pd.read_csv(..., sep=";")` — the default comma separator silently yields one column.
- `modeling/data_pipeline.py` — builds every file in `data/processed/` from that raw CSV plus the
  Open-Meteo archive API (lat 50.08, lon 7.76). Run it with `make data`, or
  `uv run python -m modeling.data_pipeline --offline` to reuse the cached weather CSV.
  This is the only thing that should write `data/processed/`.
- `notebooks/01. data_import.ipynb` — the exploration that produced the pipeline. Its `to_csv`
  calls are commented out on purpose so re-running it cannot clobber the corrected files.

## Timezones (the subtle bug in this dataset)

The two sources are on different clocks, and joining on their printed timestamps is wrong:

- **Gauge record**: Europe/Berlin wall-clock, DST-aware. The 02:00 hour is absent on all 27
  spring-forward dates and duplicated on all 26 fall-back dates (104 duplicated stamps).
- **Open-Meteo archive**: a **flat fixed-offset grid, not local time**, even though the request
  asks for `timezone=Europe/Berlin`. It applies one offset to the whole response and reports it
  as `utc_offset_seconds` — 7200 (UTC+2) for the 2000-01-01 → 2026-09-01 range — and happily
  returns hours like `2024-03-31T02:00` that do not exist in Berlin. **That offset depends on the
  requested date range, so never hard-code it when fetching; read it from the response.**

Joining on labels pairs water with the wrong hour of weather for the ~5 winter months on CET, and
averages the two passes of the repeated autumn hour into one. `data_pipeline.py` carries both sides
to UTC before joining. Daily aggregation is over **Europe/Berlin calendar days**, so two days a year
are 23 or 25 hours long — the daily file records `hours_expected` next to `hours_available`, and
completeness must be checked against that, never against a hard-coded 24.

`tests/test_data_pipeline.py` pins all of this; run it with `make test` before touching the joins.

When asked to build the real pipeline, treat the coffee code as a scaffold to replace, not as
something to keep consistent with. Confirm with the user before deleting template files.

## Commands

Everything runs through `uv`; there is no pip/conda/pyenv step. `uv run` re-syncs the env first,
so prefer it over activating `.venv`.

```bash
make setup      # uv sync --all-groups   (runtime + dev; downloads Python >=3.11 if needed)
make data       # uv run python -m modeling.data_pipeline   -> rebuilds data/processed/
make lab        # uv run jupyter lab
make mlflow     # uv run mlflow ui       -> http://127.0.0.1:5000
make train      # uv run python -m modeling.train
make predict    # uv run python -m modeling.predict models/linear data/X_test.csv data/y_test.csv
make test       # uv run pytest
make format     # uv run black .
```

Single test: `uv run pytest path/to/test_file.py::test_name`. `tests/test_data_pipeline.py` covers
the timezone handling; `testbook` is installed but no notebook tests exist yet. `pythonpath = ["."]`
in `pyproject.toml` is what lets pytest import `modeling` (the project is not installed as a package).

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
- **`.gitignore` does NOT exclude data.** The `data/*.csv` line is commented out, so every CSV
  under `data/` is tracked — roughly 53 MB is already committed (the 19 MB raw gauge file plus the
  processed hourly files). Re-running `make data` therefore shows up as a large diff.
  `models/*` *is* ignored.
- **Notebook merge conflicts**: nbdime is in the dev group. Run `nbdime config-git --enable` once,
  then `nbdime mergetool` when a `.ipynb` conflicts.
- Feature engineering functions in `modeling/feature_engineering.py` are pure `DataFrame -> DataFrame`
  transforms, and `train.py` applies the identical chain to train and test. Any new pipeline should
  keep that shape so `predict.py` can reuse it on raw input.

## GitHub automation

`.github/workflows/` contains four workflows that move issues and PRs across the neuefische org
project board (project 7) via the GraphQL API. They require org-level secrets and do no build,
test, or lint work — they will not validate a PR.
