"""Fetch and prepare live inputs for the saved Kaub model."""

import numpy as np
import pandas as pd
import requests


# -------------------------------------------------
# Configuration
# -------------------------------------------------
TIMEZONE = "Europe/Berlin"
ROUND_BEFORE_FEATURES = True

LATITUDE = 50.08
LONGITUDE = 7.76

WATER_API = (
    "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
)
WEATHER_API = "https://api.open-meteo.com/v1/forecast"


def today_local():
    """Today's date as a timezone-naive midnight."""
    return (
        pd.Timestamp.now(tz=TIMEZONE)
        .normalize()
        .tz_localize(None)
    )


def get_json(url, params=None):
    response = requests.get(
        url,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# -------------------------------------------------
# Water data
# -------------------------------------------------
def fetch_water_data():
    stations = get_json(f"{WATER_API}/stations.json")

    station = next(
        (
            item for item in stations
            if item.get("shortname", "").strip().upper() == "KAUB"
        ),
        None,
    )

    if station is None:
        raise ValueError("Kaub station was not found.")

    measurements = get_json(
        f"{WATER_API}/stations/{station['uuid']}/W/measurements.json",
        params={"start": "P30D"},
    )

    water = pd.DataFrame(measurements)

    if water.empty:
        raise ValueError("No water measurements returned for Kaub.")

    return water


def clean_water_data(water):
    required = {"timestamp", "value"}

    if not required.issubset(water.columns):
        raise ValueError(
            "Water response must contain timestamp and value."
        )

    water = water.copy()

    water["timestamp"] = pd.to_datetime(
        water["timestamp"],
        utc=True,
        errors="coerce",
    )

    water["value"] = pd.to_numeric(
        water["value"],
        errors="coerce",
    )

    water = water.dropna(subset=["timestamp", "value"])

    if water.empty:
        raise ValueError("No valid water measurements remain.")

    # Average duplicate timestamps.
    water = (
        water.groupby("timestamp")[["value"]]
        .mean()
        .sort_index()
    )

    water.index = water.index.tz_convert(TIMEZONE)

    return water.rename(columns={"value": "water_level_mean"})


def create_daily_water(water):
    """Aggregate measurements to hourly, then daily means."""
    level = water["water_level_mean"]

    hourly = level.resample("h").mean()

    daily = hourly.resample("D").agg(["mean", "count"])
    daily.columns = ["water_level_mean", "hours_available"]

    # A local day can contain 23, 24, or 25 hours around DST.
    expected_hours = pd.Series(
        [
            len(
                pd.date_range(
                    start=day,
                    end=day + pd.DateOffset(days=1),
                    freq="h",
                    inclusive="left",
                )
            )
            for day in daily.index
        ],
        index=daily.index,
    )

    # Reject days that do not have a measurement in every hour.
    incomplete = daily["hours_available"] < expected_hours
    daily.loc[incomplete, "water_level_mean"] = np.nan

    daily.index = daily.index.tz_localize(None)
    daily.index.name = "date"

    # Never use the current, unfinished day.
    return daily.loc[daily.index < today_local()]


# -------------------------------------------------
# Weather data
# -------------------------------------------------
def fetch_weather_data():
    payload = get_json(
        WEATHER_API,
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": TIMEZONE,
            "past_days": 35,
            "forecast_days": 1,
        },
    )

    daily = payload.get("daily")

    if not daily or not daily.get("time"):
        raise ValueError("No daily weather data returned.")

    return pd.DataFrame(daily)


def clean_weather_data(weather):
    required = {
        "time",
        "temperature_2m_mean",
        "precipitation_sum",
    }

    if not required.issubset(weather.columns):
        raise ValueError("Weather response is missing required columns.")

    weather = weather.rename(
        columns={
            "time": "date",
            "temperature_2m_mean": "temperature_mean",
        }
    ).copy()

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce",
    )

    for column in ["temperature_mean", "precipitation_sum"]:
        weather[column] = pd.to_numeric(
            weather[column],
            errors="coerce",
        )

    weather = (
        weather.dropna(subset=["date"])
        .set_index("date")
        .sort_index()
    )

    if weather.index.has_duplicates:
        raise ValueError("Duplicate dates in weather data.")

    weather = weather.loc[
        weather.index < today_local(),
        ["temperature_mean", "precipitation_sum"],
    ]

    if weather.empty:
        raise ValueError("No completed weather days available.")

    return weather


# -------------------------------------------------
# Merge and validate
# -------------------------------------------------
def merge_live_data(water, weather):
    # Preserve the longer weather history needed by precip_30d.
    realtime = water.join(weather, how="outer").sort_index()
    realtime = realtime.asfreq("D")
    realtime.index.name = "date"

    return realtime.loc[realtime.index < today_local()]


def validate_live_data(realtime):
    if realtime.empty:
        raise ValueError("Merged live data is empty.")

    if not isinstance(realtime.index, pd.DatetimeIndex):
        raise ValueError("Live data must have a DatetimeIndex.")

    if realtime.index.has_duplicates:
        raise ValueError("Duplicate dates in live data.")

    required = [
        "water_level_mean",
        "temperature_mean",
        "precipitation_sum",
    ]

    missing = [
        column for column in required
        if column not in realtime.columns
    ]

    if missing:
        raise ValueError(f"Missing live columns: {missing}")

    yesterday = today_local() - pd.Timedelta(days=1)

    if realtime.index[-1] != yesterday:
        raise ValueError(
            f"Latest data date is {realtime.index[-1].date()}; "
            f"expected {yesterday.date()}."
        )

    if realtime.loc[yesterday, required].isna().any():
        raise ValueError(
            f"Water or weather data is incomplete for "
            f"{yesterday.date()}."
        )


# -------------------------------------------------
# Feature engineering
# -------------------------------------------------
def create_live_features(realtime):
    df = realtime.sort_index().copy()

    columns = [
        "water_level_mean",
        "precipitation_sum",
        "temperature_mean",
    ]

    if ROUND_BEFORE_FEATURES:
        df[columns] = df[columns].round()

    features = pd.DataFrame(index=df.index)

    for lag in [0, 1, 2, 3, 7, 14, 28]:
        features[f"lag_{lag}"] = (
            df["water_level_mean"].shift(lag)
        )

    for days in [3, 7, 14, 30]:
        features[f"precip_{days}d"] = (
            df["precipitation_sum"]
            .rolling(window=days, min_periods=days)
            .sum()
        )

    for days in [3, 7]:
        features[f"temp_{days}d"] = (
            df["temperature_mean"]
            .rolling(window=days, min_periods=days)
            .mean()
        )

    # Keep missing rows so an older date is never selected silently.
    return features


# -------------------------------------------------
# Complete pipeline called by Streamlit
# -------------------------------------------------
def realtime_pipeline(feature_names):
    water = fetch_water_data()
    water = clean_water_data(water)
    water = create_daily_water(water)

    weather = fetch_weather_data()
    weather = clean_weather_data(weather)

    realtime = merge_live_data(water, weather)
    validate_live_data(realtime)

    live_features = create_live_features(realtime)

    if live_features.empty:
        raise ValueError("No live features available.")

    missing = [
        name for name in feature_names
        if name not in live_features.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X_live = live_features.loc[:, feature_names].iloc[[-1]]

    invalid = ~np.isfinite(X_live.to_numpy(dtype=float))

    if invalid.any():
        bad_columns = X_live.columns[invalid[0]].tolist()
        raise ValueError(
            f"Latest input has incomplete history or invalid values "
            f"for: {bad_columns}. No forecast was generated."
        )

    return realtime, X_live