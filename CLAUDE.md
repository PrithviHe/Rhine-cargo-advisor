# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state: template vs. actual project

This repo was forked from the neuefische `ds-modeling-pipeline` template. **`modeling/train.py`,
`modeling/predict.py`, `modeling/feature_engineering.py`, `notebooks/EDA-and-modeling.ipynb`,
`notebooks/Data_import.ipynb` and `README.md` are still template code about the coffee quality
dataset** — altitude feature engineering, a linear regression on `quality_score`. None of it
relates to HydroLoad-AI, and `make train` / `make predict` still run the coffee pipeline.

The actual project is water level forecasting for the Rhine gauge at **Kaub**, for barge load
planning: given today, will the level be below the loading threshold in N days?

The real work lives in `modeling/data_pipeline.py`, `modeling/config.py`, `tests/`, and
notebooks `01.`–`04.`. Treat the coffee code as a scaffold to replace, not something to stay
consistent with. Confirm with the user before deleting template files.

## Data

`data/raw/` — all three files are semicolon-separated, `timestamp;value`, `timestamp` format
`YYYY-MM-DD HH:MM`, 15-minute grid from 2000-01-01 01:00. **Read them with
`pd.read_csv(..., sep=";")`; the default comma separator silently yields one column.**

| file | rows | value | used by |
|---|---|---|---|
| `water_level_15min_kaub.csv` | 935,128 | level in cm, to 2026-09-01 | the pipeline, everything |
| `bingen_water_level15_min.csv` | 936,714 | level in cm, to 2026-09-18 | nothing yet |
| `kaub_water_discharge.csv` | 936,713 | discharge, to 2026-09-18 (unit not recorded anywhere; values ~3400 in Jan 2000) | nothing yet |

Bingen and discharge are committed but referenced nowhere — they are staged for the "upstream
gauge" next step in notebook 04's conclusions.

### `data/processed/` on disk is NOT what `data_pipeline.py` produces

The four committed processed files are the **legacy, timezone-naive output of the original
notebook 01**: the hourly files carry `date,time,value` and the daily file has no
`hours_available` / `hours_expected` columns. Every hourly local day in them is exactly 24 hours
long, which is the tell — the corrected pipeline yields 23- and 25-hour days at DST boundaries.

So notebooks 02, 03 and 04 all currently read data that still has the label-join defect described
below. Running `make data` fixes that but **changes the schema**: `value` becomes
`water_level_mean` / `_min` / `_max`, a `timestamp_utc` column appears, the daily file gains
`hours_available` and `hours_expected`, and the DST days start showing up as "incomplete" in any
check that compares against a hard-coded 24 (notebooks 02 and 04 both do this today and would
start dropping ~53 extra days). Expect to update those notebooks in the same change.

`modeling/data_pipeline.py` is the only thing that should write `data/processed/`. Run it with
`make data`, or `uv run python -m modeling.data_pipeline --offline` to reuse the cached weather
CSV instead of re-fetching 26 years of hourly weather from Open-Meteo (lat 50.08, lon 7.76).
Notebook 01's `to_csv` calls are commented out on purpose so re-running it cannot clobber files.

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
averages the two passes of the repeated autumn hour into one. `data_pipeline.py` carries both
sides to UTC before joining. Daily aggregation is over **Europe/Berlin calendar days**, so two
days a year are 23 or 25 hours long — the daily file records `hours_expected` next to
`hours_available`, and completeness must be checked against that, never against a hard-coded 24.

`tests/test_data_pipeline.py` pins all of this at the instant level (9 tests, all passing); run
`make test` before touching the joins. The daily aggregates are forgiving enough to hide a
one-hour misalignment, which is why the tests assert on UTC instants rather than daily means.

## The notebook chain

All of `01.`–`04.` use `../data/...` and expect to **run from `notebooks/`**. (The template
`Data_import.ipynb` uses repo-root-relative paths; `EDA-and-modeling.ipynb` uses
`sys.path.append("..")` and `../.mlflow_uri`.)

- **`01. data_import.ipynb`** — the exploration that produced `data_pipeline.py`. Writes nothing.
- **`02. eda.ipynb`** — daily EDA; found the 30-day cumulative precipitation window (r = 0.53)
  that notebook 04's features are built on. Its header still describes the *uncorrected* data
  ("flat UTC+1 grid", "seven months a year") — that text predates `data_pipeline.py` and is wrong
  in the details; trust the module and its tests over the notebook prose.
- **`03. baseline_model.ipynb`** — the only MLflow-logged notebook. Imports its constants from
  `modeling.config` (via `sys.path.append("..")`) and logs one run per model through a
  `run_baseline()` helper. 7-day horizon, linear AR on level lags vs. persistence.
- **`04. baseline_model.ipynb`** — baseline v2, self-contained: it does **not** import
  `modeling.config` and does **not** log to MLflow, redefining `HORIZON`, `TEST_YEARS`,
  `LOW_WATER_CM = 78`, `REGIME_BINS` and `VAL_YEARS` locally. Changing `modeling/config.py`
  therefore moves 03 and leaves 04 where it was. This is the notebook with the real result.

**The result that shapes any further modelling** (notebook 04, section 10): at a 7-day horizon
*no* model beats persistence at low water — persistence has the lowest MAE in every regime below
150 cm. OLS pulls low values toward the mean, over-predicting water by ~45 cm, which is the
dangerous direction for load planning. Adding weather features halves that damage; `log10` target
and gradient boosting do not fix it. Aggregate MAE hides all of this, so **every model must be
reported with a low-water column and a per-regime bias table**, and the alarm decision (recall of
"below threshold in N days") scored directly — a false negative is the expensive error.

## Commands

Everything runs through `uv`; there is no pip/conda/pyenv step. `uv run` re-syncs the env first,
so prefer it over activating `.venv`.

```bash
make setup      # uv sync --all-groups   (runtime + dev; downloads Python >=3.11 if needed)
make data       # uv run python -m modeling.data_pipeline   -> rebuilds data/processed/
make lab        # uv run jupyter lab
make mlflow     # uv run mlflow ui       -> http://127.0.0.1:5000
make test       # uv run pytest
make format     # uv run black .
make train      # COFFEE TEMPLATE, not this project
make predict    # COFFEE TEMPLATE, not this project
```

Single test: `uv run pytest tests/test_data_pipeline.py::test_name`. `testbook` is installed but
no notebook tests exist yet. `pythonpath = ["."]` in `pyproject.toml` is what lets pytest import
`modeling` (`tool.uv package = false`, so the project is not installed).

Dependencies: `uv add <pkg>` / `uv add --group dev <pkg>` / `uv remove <pkg>` — never `pip
install`, it would desync `uv.lock`. CI should use `uv sync --frozen`. See `UV_SETUP.md`.

## `modeling/config.py`

More than MLflow config — it is the shared modelling contract for notebook 03:

- `TRACKING_URI` resolves from the contents of a local `.mlflow_uri` file, then the `MLFLOW_URI`
  env var, then empty. Currently `sqlite:///mlflow.db`, i.e. a local file, not a server.
- A **relative** `sqlite:///` URI is rewritten to an absolute path under the repo root. This is
  deliberate: without it, code run from `notebooks/` logs to a second database. `notebooks/mlflow.db`
  is exactly that — a stray store with 5 orphaned runs from before the fix. The real store is
  `./mlflow.db` (11 runs). Both are gitignored, along with `.mlflow_uri`; never commit them.
- `EXPERIMENT_NAME = "hydroload-baseline"`. With the sqlite backend `mlflow.set_experiment`
  creates it on first use — no `mlflow experiments create` step needed.
- `DATE_COL`, `LEVEL_COL`, `DATA_PATH`, `HORIZON = 7`, `TEST_YEARS = 3`, `LAGS`, and
  `REGIME = {"lowwater_100": 100, "glw_78": 78, "critical_40": 40}`.

`make mlflow` picks up `./mlflow.db` only because MLflow 3.x defaults `--backend-store-uri` to
`sqlite:///mlflow.db` relative to the **working directory** — run it from the repo root.

Models are saved with `mlflow.sklearn.save_model` to a local path, not logged as MLflow artifacts;
artifact logging would need an S3/AWS backend that is not set up here.

## Conventions that bite

- **`.gitignore` does NOT exclude data.** The `data/*.csv` line is commented out, so every CSV
  under `data/` is tracked — ~100 MB is already committed (58 MB of raw gauge files plus 35 MB of
  processed ones). Re-running `make data` therefore shows up as a large diff. `models/*`, `mlruns`,
  `.mlflow_uri` and `mlflow.db` *are* ignored.
- **Notebook outputs are committed too** — the `01.`–`04.` files are 0.6–1.5 MB each, mostly
  rendered plots. nbdime is in the dev group: run `nbdime config-git --enable` once, then
  `nbdime mergetool` when a `.ipynb` conflicts.
- Any leakage-sensitive split must keep the embargo the notebooks establish: the last training
  target lands on or before the split date, the first test row is the day after, and validation is
  carved off the end of train the same way. Both 03 and 04 assert this.
- Feature engineering in `modeling/feature_engineering.py` is a chain of pure
  `DataFrame -> DataFrame` transforms applied identically to train and test (coffee code, but the
  shape is worth keeping so a future `predict.py` can reuse it on raw input).

## GitHub automation

`.github/workflows/` contains four workflows that move issues and PRs across the neuefische org
project board (project 7) via the GraphQL API. They require org-level secrets and do no build,
test, or lint work — they will not validate a PR.
