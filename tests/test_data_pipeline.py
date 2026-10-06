"""Regression tests for the timezone handling in modeling.data_pipeline.

The daily aggregates are forgiving - a one-hour misalignment barely moves a daily mean -
which is exactly why this bug survived unnoticed the first time. These tests pin the
behaviour at the instant level, where it is actually visible.
"""

from datetime import timedelta, timezone

import pandas as pd
import pytest

from modeling.data_pipeline import (
    LOCAL_TZ,
    hourly_to_daily,
    load_water_utc,
    load_weather_utc_from_cache,
)


def _write_gauge_csv(path, stamps, values):
    lines = ["timestamp;value"]
    lines += [f"{t};{v}" for t, v in zip(stamps, values)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_fall_back_hour_maps_to_two_distinct_utc_instants(tmp_path):
    """On 2024-10-27 Berlin repeats 02:00. Both copies must survive, one hour apart."""
    stamps = [
        "2024-10-27 01:30",  # CEST, 23:30 UTC on the 26th
        "2024-10-27 02:30",  # first pass, still CEST -> 00:30 UTC
        "2024-10-27 02:30",  # second pass, now CET   -> 01:30 UTC
        "2024-10-27 03:30",  # CET -> 02:30 UTC
    ]
    series = load_water_utc(
        _write_gauge_csv(tmp_path / "g.csv", stamps, [10, 20, 30, 40])
    )

    assert len(series) == 4, "a duplicated wall-clock hour must not collapse to one row"
    assert not series.index.has_duplicates
    assert list(series.index.strftime("%Y-%m-%d %H:%M")) == [
        "2024-10-26 23:30",
        "2024-10-27 00:30",
        "2024-10-27 01:30",
        "2024-10-27 02:30",
    ]
    # ordering decides which copy is which: 20 is the CEST pass, 30 the CET pass
    assert series.tolist() == [10, 20, 30, 40]


def test_spring_forward_gap_leaves_no_nan_hour(tmp_path):
    """02:00 does not exist on 2024-03-31. The UTC grid must simply be continuous."""
    stamps = ["2024-03-31 01:30", "2024-03-31 03:30", "2024-03-31 04:30"]
    series = load_water_utc(_write_gauge_csv(tmp_path / "g.csv", stamps, [10, 20, 30]))

    hourly = series.resample("1h").mean()
    assert hourly.isna().sum() == 0
    # 01:30 CET = 00:30 UTC, 03:30 CEST = 01:30 UTC -> consecutive, no hole
    assert list(series.index.strftime("%H:%M")) == ["00:30", "01:30", "02:30"]


def test_weather_cache_roundtrips_through_timestamp_utc(tmp_path):
    """A cache carrying timestamp_utc must be read back without guessing an offset."""
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"temperature_2m": [1.0, 2.0, 3.0]}, index=idx)
    out = frame.copy()
    out.insert(0, "time", idx.tz_convert(LOCAL_TZ).time)
    out.insert(0, "date", idx.tz_convert(LOCAL_TZ).date)
    out.insert(0, "timestamp_utc", idx.strftime("%Y-%m-%d %H:%M:%S%z"))
    out.to_csv(path := tmp_path / "w.csv", index=False)

    back = load_weather_utc_from_cache(path)
    assert list(back.columns) == ["temperature_2m"]
    assert back.index.equals(idx)


def test_legacy_weather_cache_uses_the_fixed_offset(tmp_path):
    """A cache from the original notebook has no timestamp_utc; it is a flat UTC+2 grid."""
    pd.DataFrame(
        {
            "date": ["2024-01-01"] * 2,
            "time": ["03:00:00", "04:00:00"],
            "temperature_2m": [1.0, 2.0],
        }
    ).to_csv(path := tmp_path / "w.csv", index=False)

    back = load_weather_utc_from_cache(path)
    assert list(back.index.strftime("%Y-%m-%d %H:%M")) == [
        "2024-01-01 01:00",
        "2024-01-01 02:00",
    ]


@pytest.mark.parametrize(
    "day, expected_hours",
    [("2024-03-31", 23), ("2024-10-27", 25), ("2024-06-01", 24)],
)
def test_hours_expected_tracks_dst(day, expected_hours):
    """Local calendar days are 23, 24 or 25 hours long; the daily table must say which."""
    start = pd.Timestamp(day, tz=LOCAL_TZ).tz_convert("UTC")
    idx = pd.date_range(start, periods=expected_hours, freq="h", tz="UTC")
    hourly = pd.DataFrame(
        {
            "water_level_mean": range(expected_hours),
            "water_level_min": range(expected_hours),
            "water_level_max": range(expected_hours),
            "precipitation": 1.0,
        },
        index=idx,
    )

    daily = hourly_to_daily(hourly)
    row = daily.loc[daily["date"] == pd.Timestamp(day)].iloc[0]
    assert row["hours_expected"] == expected_hours
    assert row["hours_available"] == expected_hours
    # a 23-hour day really does accumulate 23 hours of rain, not 24
    assert row["precipitation_sum"] == pytest.approx(float(expected_hours))


def test_daily_groups_on_local_days_not_utc_days():
    """23:30 local on 1 July is 21:30 UTC; it must land on 1 July, not 30 June."""
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-07-01 23:00", tz=LOCAL_TZ),
            pd.Timestamp("2024-07-02 00:00", tz=LOCAL_TZ),
        ]
    ).tz_convert("UTC")
    hourly = pd.DataFrame(
        {
            "water_level_mean": [100.0, 200.0],
            "water_level_min": [100.0, 200.0],
            "water_level_max": [100.0, 200.0],
        },
        index=idx,
    )

    daily = hourly_to_daily(hourly).set_index("date")["water_level_mean"]
    assert daily.loc[pd.Timestamp("2024-07-01")] == 100.0
    assert daily.loc[pd.Timestamp("2024-07-02")] == 200.0


def test_weather_offset_constant_matches_the_documented_api_behaviour():
    """Guard the --offline fallback: Open-Meteo returned 7200s for this date range."""
    from modeling.data_pipeline import WEATHER_UTC_OFFSET_HOURS

    assert timezone(timedelta(hours=WEATHER_UTC_OFFSET_HOURS)).utcoffset(
        None
    ) == timedelta(hours=2)
