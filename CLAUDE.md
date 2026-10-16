# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**HydroLoad-AI**: forecast the Rhine water level at the **Kaub** gauge 7 days ahead and turn it
into a barge cargo estimate. The operational question: *will the level be below the low-water
threshold (77 cm) in 7 days?* The deliverable is a Streamlit app (`modeling/App.py`).

The repo started from the neuefische `ds-modeling-pipeline` coffee template. All template code,
data and screenshots have been moved to `archive/`; nothing outside
`archive/` refers to them. Don't revive them. Restrict UI work to **Streamlit** — the user removed
a FastAPI experiment on purpose.

The full, user-facing explanation of the pipelines is [`docs/pipelines.md`](docs/pipelines.md).
Keep it in sync when pipelines, notebooks or the app change.

## Layout

```
modeling/   data_pipeline.py  hydro_pipeline.py  kaub_pipeline.py
            realtime_pipeline.py  live_pipeline.py  cargo_calculation.py  App.py
notebooks/  01_data  02_eda  03_baseline_model
            04_eda_and_multivariate_pipeline  05_Real_time_pipeline
models/     kaub_xgboost_04.joblib        ← the model the app loads (committed)
data/raw/   Kaub, Bingen, discharge (15-min)   data/processed/   derived tables
docs/       pipelines.md, Final_Presentation_HydroLoadAI_final.pdf
tests/      test_data_pipeline.py  test_hydro_pipeline.py  test_live_pipeline.py
archive/    original notebooks 04/05 (archive/notebooks/), unused template code/data/images/old models
```

## Pipelines (how they depend on each other)

```
data_pipeline ─► hydro_pipeline ─┬─► kaub_pipeline ─► notebook 04 saves models/kaub_xgboost_04.joblib
                                  └─► live_pipeline ◄── realtime_pipeline (fetch + clean only)
```

- **`data_pipeline.py`** — raw Kaub gauge + Open-Meteo archive → UTC → joined. `build()` writes the
  Kaub-only processed files and the weather cache `data/processed/kaub_weather_hourly.csv`.
  Running it without `--offline` **re-downloads 26 years of weather**.
- **`hydro_pipeline.py`** — the historical daily table (Kaub + Bingen + discharge + weather) and
  the **single feature function `build_features()`**, plus `time_split()`, `metrics()`,
  `regime_table()`, `alarm_scores()` and the settings `HORIZON=7`, `TEST_YEARS=3`, `VAL_YEARS=2`,
  `LOW_WATER_CM=77`, `FEATURE_GROUPS`, `EXTENDED_FEATURES`.
- **`kaub_pipeline.py`** — the multivariate models (linear, LightGBM, random forest, XGBoost,
  polynomial ridge), `split_data()`, `train_model_groups()`, `refit_and_test()`, `save_model()`,
  `load_model()`, `predict()`. `MODEL_PATH = models/kaub_xgboost_04.joblib`.
- **`realtime_pipeline.py`** — fetches the last 30 days from Pegelonline and 35 days of daily
  weather from the Open-Meteo **forecast** API. Its own `create_live_features()` /
  `realtime_pipeline()` are legacy, kept only so tests and notebook 05 can prove the new path
  matches them. Don't call them from new code.
- **`live_pipeline.py`** — `live_pipeline(feature_names)` = realtime fetch → `build_features()` →
  `latest_inputs()` (stops with a clear error if any needed history is missing).

**The one rule: `hydro_pipeline.build_features()` is the only feature code.** Training
(`kp.make_features`), historical forecasts (`App.py` `load_features()`) and live forecasts
(`live_pipeline.create_live_features`) all call it. Never add a second copy of feature logic.
`build_features` creates Bingen/discharge/snow/evaporation features only when those columns exist,
so it also works on live data (which has only level, rain, temperature). It requires a complete
daily `DatetimeIndex` (`load_hydro_daily()` / `asfreq("D")`) so `shift(k)` means k calendar days.

## The saved model

`models/kaub_xgboost_04.joblib` = `{"model", "features", "horizon"}`: XGBoost (500 trees,
lr 0.03, depth 4) on the 13 features of group *"Kaub + precipitation + temperature"* (`lag_0/1/2/3/7/14/28`,
`precip_3d/7d/14d/30d`, `temp_3d/7d`), trained on **`train_fit` (2000-01-30 → 2021-08-18)**.
It is written by notebook 04's "Save Model for streamlit" cell. Retraining that way with the
current `build_features()` reproduces it exactly (deterministic).

**Trap:** `uv run python -m modeling.kaub_pipeline` (`run()`) saves a *different* model — refit on
`train` up to 2023 — to the **same path**. Don't run it unless the user wants to replace the app's
model; if it happens, `git restore models/kaub_xgboost_04.joblib`.

Split (7-day embargo on both cuts, asserted in `time_split`): train_fit → 2021-08-18,
validation 2021-08-26 → 2023-08-25, test 2023-09-02 → 2026-08-25.

## Streamlit app (`modeling/App.py`)

Loads the model above; **historical mode** → `load_hydro_daily()` + `build_features()`;
**live mode** → `live_pipeline()`; cargo via `cargo_calculation.estimate_cargo()` (simplified
linear draft–payload rule with example vessel defaults). The low-water threshold is a sidebar input
(default 77), not `LOW_WATER_CM`. Banner image: `images/4_kaub.jpg`. Streamlit warns that
`use_container_width` is deprecated in two calls; replacing it with `width="stretch"` is safe.

## Data

`data/raw/` — semicolon-separated `timestamp;value`, `YYYY-MM-DD HH:MM`, 15-min grid.
**Read with `pd.read_csv(..., sep=";")`**; the default comma silently yields one column.
`water_level_15min_kaub.csv` (to 2026-09-01), `bingen_water_level15_min.csv` and
`kaub_water_discharge.csv` (to 2026-09-18, discharge unit unrecorded).

`data/processed/`:
- `kaub_hydro_weather_daily.csv` — written by `hydro_pipeline` (timezone-correct, has
  `hours_available` and `hours_expected`). **This is what the app and pipelines use.**
- `kaub_weather_hourly.csv`, `kaub_water_level_hourly.csv`, `kaub_water_weather_hourly.csv`,
  `kaub_water_weather_daily.csv` — currently the **legacy notebook-01 output** (`date,time,value`,
  every day exactly 24 h, label-joined). Notebooks 02 and 03 read the daily one, so they still
  carry the timezone defect below. `load_weather_utc_from_cache()` handles this legacy cache via a
  fixed UTC+2 offset. Notebook 01's `to_csv` cells are **active** and overwrite these files.

`.gitignore` does **not** exclude data — all CSVs under `data/` are tracked (~100 MB). Regenerating
data shows up as large diffs.

## Timezones (the subtle bug in this dataset)

- **Gauge records**: Europe/Berlin wall-clock, DST-aware — the 02:00 hour is missing on spring-forward
  dates and repeated on fall-back dates.
- **Open-Meteo archive**: one fixed offset for the whole response (`utc_offset_seconds`, 7200 for the
  2000→2026 request), even with `timezone=Europe/Berlin`; it returns hours that don't exist locally.
  The offset depends on the requested range — read it from the response, never hard-code it when fetching.

Joining on printed labels misaligns water and weather by an hour for ~5 months a year and averages the
repeated autumn hour. The pipelines carry both sides to UTC before joining and aggregate to
**Berlin calendar days** (23/25-hour DST days; compare completeness to `hours_expected`, never to 24).
`tests/test_data_pipeline.py` pins this at the instant level — run the tests before touching joins.

Live data is aggregated differently (hourly in Berlin time, incomplete days → NaN) and its weather
comes from the forecast API, not the archive — feature *code* is identical, input values can differ slightly.

## Modelling results that shape further work

At a 7-day horizon persistence (`lag_0`) is hard to beat, especially at low water: the saved XGBoost
has test MAE 47.2–47.6 cm vs 50.8 for persistence overall, but ~36 cm vs ~24 cm at low water, with a
strong positive bias (promises water that isn't there — the dangerous direction). An experimental
variant predicting the 7-day *change* with Huber loss and extra features (`EXTENDED_FEATURES`)
cut low-water MAE to ~22.5 cm and raised alarm recall to 82 %, but it needs 60/90-day rain, snow and
evaporation that the live pipeline does not fetch, so it is not deployed.

**Always report a low-water row and a per-regime table (`regime_table`) and score the alarm decision
(`alarm_scores`, recall first)** — aggregate MAE hides the low-water behaviour.

Leakage rule: any split keeps the embargo — the last training target lands before the next period's
first origin. Choose models/settings on validation (or yearly validation folds), look at test once.

## Commands

Everything runs through `uv` (no pip/conda). There is **no Makefile** (removed; `make` isn't installed
on the user's Windows machine).

```bash
uv sync --all-groups                          # install runtime + dev
uv run python -m modeling.hydro_pipeline      # rebuild data/processed/kaub_hydro_weather_daily.csv
uv run python -m modeling.data_pipeline --offline   # rebuild Kaub-only files from the weather cache
uv run streamlit run modeling/App.py          # the app, http://localhost:8501
uv run python -m pytest                       # 22 tests
uv run python -m pytest tests/test_live_pipeline.py::test_live_features_equal_training_features
uv run python -m jupyterlab                   # notebooks — run them from notebooks/
uv run python -m nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb
uv run black .
```

**Windows: always use `python -m`.** The uv launchers in `.venv/Scripts/*.exe` fail with
*"uv trampoline failed to canonicalize script path"* (likely the space in `C:\Users\Prithvi Hegde`),
so `uv run pytest`, `uv run jupyter …`, `uv run uvicorn …` break; `uv run python -m <module>` works.
Shell output on this machine is cp1252 — set `PYTHONIOENCODING=utf-8` when printing notebook text
with emoji/arrows.

Dependencies: `uv add <pkg>` / `uv add --group dev <pkg>` / `uv remove <pkg>` — never `pip install`.
`mlflow`, `tabicl`, `parsenvy` and `statsmodels` are still declared but unused by current code
(MLflow config `modeling/config.py` was removed); confirm with the user before removing them.

## Notebooks

Run from `notebooks/` (they use `../data/...` and `sys.path.append("..")`). Outputs are committed.

| Notebook | Uses pipelines | Notes |
|---|---|---|
| `01_data` | no | data import; active `to_csv` overwrites `data/processed/` legacy files |
| `02_eda` | no | daily EDA (30-day rain window r≈0.53) |
| `03_baseline_model` | no | persistence vs linear AR; evaluates on **training data only** (test set built but unused) |
| `04_eda_and_multivariate_pipeline` | yes | all models via `kaub_pipeline`; **saves the app's model** |
| `05_Real_time_pipeline` | yes | live forecast via `live_pipeline`; needs internet |
| `../archive/notebooks/04_…_main`, `…/05_Real_time_data` | no | originals, hand-written; 05 loads a model file that no longer exists |

`notebooks/00. data_import.ipynb` is the user's personal copy and is git-ignored — leave it alone.

## Git workflow

Work on branch **`prithvi`**, push it, then bring it to `main` (fast-forward
`git push origin prithvi:main` when `main` has nothing new, or a PR). The user also edits on the
GitHub website (README, merges), so `git fetch` and pull before pushing; never force-push `main`.
Commit or push only when the user asks; the user often prefers to run git commands themselves —
give exact commands. `.github/workflows/` (neuefische project-board automation that needs org
secrets) is git-ignored and kept only locally; tracking moved to Airtable.
