"""Data pipeline for notebook 06: Kaub + Bingen + discharge + weather.

Run it with:

    uv run python -m modeling.hydro_pipeline

It writes data/processed/kaub_hydro_weather_daily.csv.

The gauge files are in German local time (with summer time), the weather is not.
So both are converted to UTC first, joined, and only then grouped into days.
"""

import numpy as np
import pandas as pd

from modeling.data_pipeline import (
    DAILY_AGG,
    LOCAL_TZ,
    PROCESSED_DIR,
    REPO_ROOT,
    load_water_utc,
    load_weather_utc_from_cache,
)

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
RAW_DIR = REPO_ROOT / "data" / "raw"

GAUGES = {
    "water_level": RAW_DIR / "water_level_15min_kaub.csv",
    "bingen_level": RAW_DIR / "bingen_water_level15_min.csv",
    "discharge": RAW_DIR / "kaub_water_discharge.csv",
}

WEATHER_FILE = PROCESSED_DIR / "kaub_weather_hourly.csv"
OUTPUT_FILE = PROCESSED_DIR / "kaub_hydro_weather_daily.csv"

HORIZON = 7  # forecast 7 days ahead
TEST_YEARS = 3  # last 3 years = test
VAL_YEARS = 2  # last 2 years of train = validation
LOW_WATER_CM = 78  # low-water mark (GlW) at Kaub

LAGS = [0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28]

KAUB = ["lag_0", "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28"]
PRECIP = ["precip_3d", "precip_7d", "precip_14d", "precip_30d"]
TEMP = ["temp_3d", "temp_7d"]
BINGEN = ["bingen_lag_1", "bingen_lag_2", "bingen_lag_3"]
SEASON = ["doy_sin", "doy_cos"]

FEATURE_GROUPS = {
    "Kaub only": KAUB,
    "Kaub + precipitation": KAUB + PRECIP,
    "Kaub + precipitation + temperature": KAUB + PRECIP + TEMP,
    "All selected features": KAUB + PRECIP + TEMP + BINGEN + SEASON,
}

# Extra features for the improved XGBoost (trend, longer rain memory, snow, evaporation)
TREND = [
    "level_change_1d", "level_change_3d", "level_change_7d", "level_change_14d",
    "level_min_7d", "level_max_7d", "level_mean_30d",
]
LONG_WEATHER = ["precip_60d", "precip_90d", "snow_30d", "et0_30d"]

EXTENDED_FEATURES = KAUB + PRECIP + TEMP + TREND + LONG_WEATHER + SEASON


# ---------------------------------------------------------------
# Step 1: build the daily table
# ---------------------------------------------------------------
def load_gauge_hourly(path, name):
    """Load one 15-minute gauge file and turn it into hourly mean/min/max (UTC)."""
    readings = load_water_utc(path)
    hourly = readings.resample("1h").agg(["mean", "min", "max"])
    hourly.columns = [name + "_mean", name + "_min", name + "_max"]
    return hourly


def day_length_hours(dates):
    """Hours in each local day: 24 normally, 23 or 25 on summer-time switch days."""
    start = dates.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="shift_forward")
    end = (dates + pd.Timedelta(days=1)).tz_localize(
        LOCAL_TZ, ambiguous=True, nonexistent="shift_forward"
    )
    return ((end - start) / pd.Timedelta(hours=1)).astype(int)


def make_daily(hourly):
    """Group the hourly table into German calendar days."""
    hourly = hourly.copy()
    hourly["date"] = hourly.index.tz_convert(LOCAL_TZ).date

    # How to combine the hours of one day
    rules = {}
    for name in GAUGES:
        rules[name + "_mean"] = "mean"
        rules[name + "_min"] = "min"
        rules[name + "_max"] = "max"
    for column, (new_name, how) in DAILY_AGG.items():
        if column in hourly.columns:
            rules[column] = how

    daily = hourly.groupby("date").agg(rules)

    # Rename weather columns, e.g. precipitation -> precipitation_sum
    for column, (new_name, how) in DAILY_AGG.items():
        daily = daily.rename(columns={column: new_name})

    daily["hours_available"] = hourly.groupby("date")["water_level_mean"].count()

    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"
    daily["hours_expected"] = day_length_hours(daily.index)
    return daily.reset_index()


def build_hydro_daily(save=True):
    """Load gauges + weather, join them on UTC time, group into days."""
    frames = []
    for name, path in GAUGES.items():
        frames.append(load_gauge_hourly(path, name))
    gauges = pd.concat(frames, axis=1)

    weather = load_weather_utc_from_cache(WEATHER_FILE)
    weather = weather.drop(columns="wind_direction_10m", errors="ignore")

    hourly = gauges.join(weather, how="inner")
    daily = make_daily(hourly)

    if save:
        daily.to_csv(OUTPUT_FILE, index=False)
    return daily


def load_hydro_daily(path=OUTPUT_FILE):
    """Read the daily table. Every calendar day gets a row (missing days = NaN)."""
    daily = pd.read_csv(path, parse_dates=["date"])
    daily = daily.set_index("date").sort_index()
    return daily.asfreq("D")


def complete_days(daily):
    """True for days that have all their hours."""
    return daily["hours_available"] >= daily["hours_expected"]


# ---------------------------------------------------------------
# Step 2: features and targets
# ---------------------------------------------------------------
def build_features(daily, horizon=HORIZON, round_inputs=True, drop_incomplete=False):
    """Create lag, rain, temperature, Bingen, season features and the targets.

    round_inputs=True rounds level, rain and temperature first, like notebook 04
    and realtime_pipeline.py do.

    Only water_level_mean, precipitation_sum and temperature_mean are required.
    Bingen, discharge, snow and evaporation features are only created when those
    columns exist, so the same function also works on live data.
    """
    if daily.index.inferred_freq != "D":
        raise ValueError("daily needs one row per day - use load_hydro_daily()")

    df = daily.copy()

    if drop_incomplete:
        df.loc[~complete_days(df), :] = np.nan

    if round_inputs:
        for column in ["water_level_mean", "precipitation_sum", "temperature_mean"]:
            df[column] = df[column].round()

    level = df["water_level_mean"]
    out = pd.DataFrame(index=df.index)

    # Kaub water level in the past
    for lag in LAGS:
        out[f"lag_{lag}"] = level.shift(lag)

    # Kaub trend: how much the level changed, and recent min / max / mean
    for days in [1, 3, 7, 14]:
        out[f"level_change_{days}d"] = level.diff(days)
    out["level_min_7d"] = level.rolling(7).min()
    out["level_max_7d"] = level.rolling(7).max()
    out["level_mean_30d"] = level.rolling(30).mean()

    # Upstream Bingen and discharge in the past (history only, not in live data)
    if "bingen_level_mean" in df.columns:
        for lag in [1, 2, 3, 7]:
            out[f"bingen_lag_{lag}"] = df["bingen_level_mean"].shift(lag)
    if "discharge_mean" in df.columns:
        for lag in [1, 2, 3, 7]:
            out[f"discharge_lag_{lag}"] = df["discharge_mean"].shift(lag)
        out["discharge_change_1d"] = df["discharge_mean"].diff(1)

    # Rain over the last N days
    for days in [3, 7, 14, 30, 60, 90]:
        out[f"precip_{days}d"] = df["precipitation_sum"].rolling(days).sum()

    # Average temperature over the last N days
    for days in [3, 7]:
        out[f"temp_{days}d"] = df["temperature_mean"].rolling(days).mean()

    # Snow and evaporation over the last 30 days (history only, not in live data)
    if "snowfall_sum" in df.columns:
        out["snow_30d"] = df["snowfall_sum"].rolling(30).sum()
    if "et0_sum" in df.columns:
        out["et0_30d"] = df["et0_sum"].rolling(30).sum()

    # Season (day of year as a circle)
    day_of_year = out.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Targets: water level 1..7 days ahead
    for h in range(1, horizon + 1):
        out[f"target_day_{h}"] = level.shift(-h)
    out["target"] = out[f"target_day_{horizon}"]

    return out


# ---------------------------------------------------------------
# Step 3: split by time (with a 7-day gap)
# ---------------------------------------------------------------
def time_split(data, horizon=HORIZON, test_years=TEST_YEARS, val_years=VAL_YEARS):
    """Return train, test, train_fit and validation.

    A gap of `horizon` days stops the last training target from overlapping
    the first test (or validation) day.
    """
    split_date = data.index.max() - pd.DateOffset(years=test_years)
    gap = pd.Timedelta(days=horizon)

    train = data.loc[: split_date - gap].dropna()
    test = data.loc[split_date + pd.Timedelta(days=1) :].dropna()

    val_start = train.index.max() - pd.DateOffset(years=val_years) + pd.Timedelta(days=1)
    train_fit = train.loc[: val_start - gap - pd.Timedelta(days=1)]
    validation = train.loc[val_start:]

    assert train.index.max() + gap < test.index.min()
    assert train_fit.index.max() + gap < validation.index.min()

    return {"train": train, "test": test, "train_fit": train_fit, "validation": validation}


# ---------------------------------------------------------------
# Step 4: evaluation
# ---------------------------------------------------------------
def metrics(actual, predicted, persistence=None):
    """MAE, RMSE, R², bias and (optional) skill vs persistence."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    error = predicted - actual  # positive = model predicts too much water

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    r2 = 1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)

    result = {
        "Samples": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Bias": np.mean(error),
        "Overprediction (%)": 100 * np.mean(error > 0),
    }

    if persistence is not None:
        persistence_mae = np.mean(np.abs(np.array(persistence, dtype=float) - actual))
        result["Persistence MAE"] = persistence_mae
        result["Skill vs persistence"] = 1 - mae / persistence_mae

    return result


def regime_table(actual, predicted, persistence, low_cm=LOW_WATER_CM, high_cm=None):
    """Error table per water condition (based on the actual level)."""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    persistence = np.array(persistence, dtype=float)

    groups = {"All levels": actual == actual, "Low water": actual < low_cm}
    if high_cm is None:
        groups["Other levels"] = actual >= low_cm
    else:
        groups["Normal water"] = (actual >= low_cm) & (actual < high_cm)
        groups["High water"] = actual >= high_cm

    rows = []
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        row = metrics(actual[mask], predicted[mask], persistence[mask])
        row["Condition"] = name
        rows.append(row)

    return pd.DataFrame(rows).set_index("Condition")


def alarm_scores(actual, predicted, low_cm=LOW_WATER_CM):
    """Score the yes/no question: will the level be below low_cm?"""
    is_low = np.array(actual, dtype=float) < low_cm
    alarm = np.array(predicted, dtype=float) < low_cm

    hits = int(np.sum(is_low & alarm))
    missed = int(np.sum(is_low & ~alarm))
    false_alarms = int(np.sum(~is_low & alarm))

    return {
        "Low-water days": int(is_low.sum()),
        "Hits": hits,
        "Missed alarms": missed,
        "False alarms": false_alarms,
        "Recall": hits / (hits + missed) if hits + missed > 0 else np.nan,
        "Precision": hits / (hits + false_alarms) if hits + false_alarms > 0 else np.nan,
    }


if __name__ == "__main__":
    daily = build_hydro_daily()
    print(f"Saved {len(daily)} days to {OUTPUT_FILE}")
