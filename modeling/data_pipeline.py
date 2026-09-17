"""Build the Kaub daily modelling table from the raw gauge record and Open-Meteo weather.

Run it with::

    uv run python -m modeling.data_pipeline          # re-fetch weather from the API
    uv run python -m modeling.data_pipeline --offline  # reuse the cached weather CSV

Outputs (all under ``data/processed/``):

``kaub_weather_hourly.csv``       weather on a true hourly grid
``kaub_water_level_hourly.csv``   gauge readings aggregated 15-minute -> hourly
``kaub_water_weather_hourly.csv`` the two joined on the real instant
``kaub_water_weather_daily.csv``  the daily table that notebooks 02 and 03 read

Why this module exists at all: the two sources are on *different clocks*, and joining
them on their printed timestamps silently pairs each water reading with weather from a
different hour for five months of every year. See ``TIMEZONE NOTES`` below.
"""

from __future__ import annotations

import argparse
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# TIMEZONE NOTES  (verified against the data and the API on 2026-09-16)
#
# Gauge record (data/raw/water_level_15min_kaub.csv)
#   Europe/Berlin *wall-clock* time, DST-aware. Proof: the 02:00 hour is absent on every
#   spring-forward date (27 of them, 2000-2026) and appears twice on every fall-back date
#   (26 of them, 104 duplicated 15-minute stamps).
#
# Weather (Open-Meteo archive API)
#   A flat, fixed-offset grid -- NOT DST-aware -- even though the request asks for
#   timezone=Europe/Berlin. The API applies one offset to the whole response and reports
#   it as `utc_offset_seconds`. For the 2000-01-01 -> 2026-09-01 range it returns 7200
#   (UTC+2), and the response contains hours such as 2024-03-31T02:00 that do not exist
#   in real Berlin local time.
#
#   That offset depends on the requested date range, so it must never be hard-coded when
#   fetching live -- read it from the response. WEATHER_UTC_OFFSET_HOURS below is only a
#   fallback for --offline runs against the cached CSV.
#
# Consequence of joining on the printed labels (what notebook 01 originally did):
#   summer (CEST, both +2) -> correctly aligned
#   winter (CET, water +1 vs weather +2) -> water paired with weather one hour early
#   i.e. roughly five months a year, not seven.
#
# The fix is to carry both sides to UTC, join on the instant, and only then aggregate.
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_WATER_PATH = REPO_ROOT / "data" / "raw" / "water_level_15min_kaub.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

LOCAL_TZ = "Europe/Berlin"
WEATHER_UTC_OFFSET_HOURS = 2  # --offline fallback only; see TIMEZONE NOTES

KAUB_LAT, KAUB_LON = 50.08, 7.76
WEATHER_START, WEATHER_END = "2000-01-01", "2026-09-01"
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "et0_fao_evapotranspiration",
]

# Hourly column -> (daily column, aggregation). wind_direction_10m is deliberately absent:
# a plain mean of compass degrees is wrong (350 deg and 10 deg average to 180, not 0), and
# nothing downstream uses it. Add it as a vector mean if a model ever needs it.
DAILY_AGG = {
    "temperature_2m": ("temperature_mean", "mean"),
    "relative_humidity_2m": ("humidity_mean", "mean"),
    "precipitation": ("precipitation_sum", "sum"),
    "rain": ("rain_sum", "sum"),
    "snowfall": ("snowfall_sum", "sum"),
    "pressure_msl": ("pressure_mean", "mean"),
    "wind_speed_10m": ("wind_speed_mean", "mean"),
    "et0_fao_evapotranspiration": ("et0_sum", "sum"),
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_water_utc(path: Path = RAW_WATER_PATH) -> pd.Series:
    """Read the 15-minute gauge record and return it indexed by UTC instant.

    The file is semicolon-separated; the default comma separator silently yields a single
    column. Values are centimetres on the Kaub gauge datum.
    """
    raw = pd.read_csv(
        path,
        sep=";",
        dtype={"value": "float64"},
        parse_dates=["timestamp"],
        date_format="%Y-%m-%d %H:%M",
    )
    idx = pd.DatetimeIndex(raw["timestamp"])

    # ambiguous="infer" resolves the repeated fall-back hour from the ordering of the
    # readings; nonexistent="shift_forward" is belt-and-braces, the skipped spring-forward
    # hour is simply absent from the file.
    local = idx.tz_localize(LOCAL_TZ, ambiguous="infer", nonexistent="shift_forward")

    series = pd.Series(
        raw["value"].to_numpy(), index=local.tz_convert("UTC"), name="value"
    )
    return series.sort_index()


def fetch_weather_utc() -> pd.DataFrame:
    """Fetch hourly weather from the Open-Meteo archive, indexed by UTC instant.

    The offset is taken from the response rather than assumed, because the API picks one
    fixed offset per request based on the date range.
    """
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": KAUB_LAT,
            "longitude": KAUB_LON,
            "start_date": WEATHER_START,
            "end_date": WEATHER_END,
            "hourly": ",".join(WEATHER_VARS),
            "timezone": LOCAL_TZ,
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()

    offset = timezone(timedelta(seconds=payload["utc_offset_seconds"]))
    frame = pd.DataFrame(payload["hourly"])
    idx = pd.DatetimeIndex(pd.to_datetime(frame.pop("time"))).tz_localize(offset)
    frame.index = idx.tz_convert("UTC")
    return frame.sort_index()


def load_weather_utc_from_cache(path: Path) -> pd.DataFrame:
    """Reinterpret the already-saved weather CSV onto a UTC index.

    Files written by this module carry an explicit ``timestamp_utc`` column, so no offset
    has to be guessed. Files written by the original notebook 01 do not; those are
    reinterpreted with WEATHER_UTC_OFFSET_HOURS, which is only correct for a cache
    produced by the WEATHER_START/WEATHER_END request above.
    """
    frame = pd.read_csv(path)

    if "timestamp_utc" in frame.columns:
        frame.index = pd.DatetimeIndex(
            pd.to_datetime(frame.pop("timestamp_utc"), utc=True)
        )
        frame = frame.drop(columns=["date", "time"], errors="ignore")
    else:
        idx = pd.DatetimeIndex(
            pd.to_datetime(frame.pop("date") + " " + frame.pop("time"))
        )
        offset = timezone(timedelta(hours=WEATHER_UTC_OFFSET_HOURS))
        frame.index = idx.tz_localize(offset).tz_convert("UTC")

    return frame.sort_index()


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def water_to_hourly(water_utc: pd.Series) -> pd.DataFrame:
    """Aggregate 15-minute readings to hourly mean/min/max on the UTC grid."""
    hourly = water_utc.resample("1h").agg(["mean", "min", "max", "count"])
    hourly.columns = [
        "water_level_mean",
        "water_level_min",
        "water_level_max",
        "n_readings",
    ]
    return hourly


def merge_hourly(water_hourly: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Inner-join gauge and weather on the UTC instant."""
    return water_hourly.join(weather, how="inner")


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the merged hourly table to Europe/Berlin calendar days.

    Local days are the operationally meaningful unit, which means two days a year are 23
    or 25 hours long. ``hours_expected`` records the true length so downstream
    completeness checks can compare against it instead of assuming 24.
    """
    local_index = hourly.index.tz_convert(LOCAL_TZ)
    grouper = pd.Series(local_index.date, index=hourly.index, name="date")

    daily = hourly.groupby(grouper).agg(
        water_level_mean=("water_level_mean", "mean"),
        water_level_min=("water_level_min", "min"),
        water_level_max=("water_level_max", "max"),
        hours_available=("water_level_mean", "count"),
        **{
            out: (src, how)
            for src, (out, how) in DAILY_AGG.items()
            if src in hourly.columns
        },
    )
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "date"

    # True length of each local day: 24 normally, 23 on spring-forward, 25 on fall-back.
    starts = pd.DatetimeIndex(daily.index).tz_localize(
        LOCAL_TZ, ambiguous=True, nonexistent="shift_forward"
    )
    ends = pd.DatetimeIndex(daily.index + pd.Timedelta(days=1)).tz_localize(
        LOCAL_TZ, ambiguous=True, nonexistent="shift_forward"
    )
    daily["hours_expected"] = ((ends - starts) / pd.Timedelta(hours=1)).astype(int)

    return daily.reset_index()


def to_local_date_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialise a UTC-indexed hourly frame for CSV.

    Keeps ``timestamp_utc`` as the unambiguous machine-readable instant and adds the
    local ``date``/``time`` columns the notebooks group on.
    """
    local = frame.index.tz_convert(LOCAL_TZ)
    out = frame.copy()
    out.insert(0, "time", local.time)
    out.insert(0, "date", local.date)
    out.insert(0, "timestamp_utc", frame.index.strftime("%Y-%m-%d %H:%M:%S%z"))
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
def build(offline: bool = False) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    weather_path = PROCESSED_DIR / "kaub_weather_hourly.csv"

    print("loading gauge record ...")
    water_utc = load_water_utc()
    print(
        f"  {len(water_utc):,} readings, {water_utc.index.min()} -> {water_utc.index.max()}"
    )

    if offline:
        print("loading cached weather ...")
        weather = load_weather_utc_from_cache(weather_path)
    else:
        print("fetching weather from Open-Meteo ...")
        weather = fetch_weather_utc()
    print(f"  {len(weather):,} hours, {weather.index.min()} -> {weather.index.max()}")

    water_hourly = water_to_hourly(water_utc)
    merged = merge_hourly(water_hourly, weather)
    print(f"merged hourly: {len(merged):,} rows")

    daily = hourly_to_daily(merged)
    print(
        f"daily: {len(daily):,} rows, {daily['date'].min().date()} -> {daily['date'].max().date()}"
    )

    to_local_date_time(weather).to_csv(weather_path, index=False)
    to_local_date_time(water_hourly.drop(columns="n_readings")).to_csv(
        PROCESSED_DIR / "kaub_water_level_hourly.csv", index=False
    )
    to_local_date_time(merged.drop(columns="n_readings")).to_csv(
        PROCESSED_DIR / "kaub_water_weather_hourly.csv", index=False
    )
    daily.to_csv(PROCESSED_DIR / "kaub_water_weather_daily.csv", index=False)
    print(f"wrote 4 files to {PROCESSED_DIR.relative_to(REPO_ROOT)}/")

    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="reuse data/processed/kaub_weather_hourly.csv instead of calling the API",
    )
    build(offline=parser.parse_args().offline)


if __name__ == "__main__":
    main()
