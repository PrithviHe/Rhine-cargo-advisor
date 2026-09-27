"""
HydroLoad-AI | modeling/fetch_live.py

Pulls the latest observations from PEGELONLINE (gauge, discharge) and
Open-Meteo (weather), aggregates them to daily values in exactly the same way
as the historical pipeline, and appends them to the stored dataset.

No weather forecast is required: every model feature looks backwards only, so
the latest observations are sufficient to issue a seven-day forecast.

    python -m modeling.fetch_live            update the cache
    python -m modeling.fetch_live --status   show what is cached
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "processed" / "kaub_hydro_weather_daily.csv"
LIVE = ROOT / "data" / "processed" / "kaub_live_daily.csv"
STAMP = ROOT / "data" / "processed" / "live_status.json"

PEGEL = "https://www.pegelonline.wsv.de/webservices/rest-api/v2/stations"
METEO = "https://api.open-meteo.com/v1/forecast"

KAUB_LAT, KAUB_LON = 50.08, 7.76
# Kurznamen sind verifiziert; die UUIDs stehen als Rueckfallebene daneben.
STATION_LEVEL = "KAUB"          # Wasserstand W  [cm]   1d26e504-7f9e-480a-b52c-5932be6549ab
STATION_UPSTREAM = "BINGEN"     # Oberlieger W   [cm]   0309cd61-90c9-470e-99d4-2ee4fb2c5f84
STATION_DISCHARGE = "KAUB"      # Abfluss Q      [m3/s]

TZ = "Europe/Berlin"
DAYS = 40                       # Puffer: laengstes Feature-Fenster ist 30 Tage
TIMEOUT = 30

COLUMNS = ["date", "water_level_mean", "water_level_min", "water_level_max",
           "temperature_mean", "humidity_mean", "precipitation_sum", "rain_sum",
           "snowfall_sum", "pressure_mean", "wind_speed_mean",
           "bingen_level_mean", "discharge_mean", "hours_available"]


class LiveDataError(RuntimeError):
    """Raised when a source cannot be reached or returns unusable data."""


# ------------------------------------------------------------------ sources
def fetch_gauge(station: str, series: str = "W", days: int = DAYS,
                session: requests.Session | None = None) -> pd.Series:
    """
    One PEGELONLINE time series as a tz-aware Series.
    series: 'W' = water level in cm, 'Q' = discharge in m3/s.
    """
    url = f"{PEGEL}/{station}/{series}/measurements.json"
    get = (session or requests).get
    try:
        r = get(url, params={"start": f"P{days}D"}, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except requests.RequestException as e:
        raise LiveDataError(f"PEGELONLINE {station}/{series} unreachable: {e}") from e
    except ValueError as e:
        raise LiveDataError(f"PEGELONLINE {station}/{series} returned no JSON") from e

    if not isinstance(payload, list) or not payload:
        raise LiveDataError(f"PEGELONLINE {station}/{series} returned no measurements")

    df = pd.DataFrame(payload)
    if not {"timestamp", "value"} <= set(df.columns):
        raise LiveDataError(f"PEGELONLINE {station}/{series}: unexpected fields "
                            f"{list(df.columns)[:5]}")
    ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").dt.tz_convert(TZ)
    # PEGELONLINE liefert 15-Minuten-Werte mit Zeitzonenoffset
    return (pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=ts)
              .dropna().sort_index().rename(f"{station}_{series}"))


def fetch_weather(lat: float = KAUB_LAT, lon: float = KAUB_LON, days: int = DAYS,
                  session: requests.Session | None = None) -> pd.DataFrame:
    """Hourly observations for the last `days` days (no forecast horizon)."""
    params = {
        "latitude": lat, "longitude": lon, "timezone": TZ,
        "past_days": min(days, 92), "forecast_days": 1,
        "hourly": ("temperature_2m,relative_humidity_2m,precipitation,rain,"
                   "snowfall,pressure_msl,wind_speed_10m"),
    }
    get = (session or requests).get
    try:
        r = get(METEO, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except requests.RequestException as e:
        raise LiveDataError(f"Open-Meteo unreachable: {e}") from e
    except ValueError as e:
        raise LiveDataError("Open-Meteo returned no JSON") from e

    if "hourly" not in payload:
        raise LiveDataError(f"Open-Meteo: unexpected response {list(payload)[:5]}")

    w = pd.DataFrame(payload["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    return w.set_index("time").sort_index()


# ------------------------------------------------------------------ shaping
def to_daily(level: pd.Series, upstream: pd.Series, discharge: pd.Series,
             weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily values, matching the historical pipeline exactly."""
    def daily(series: pd.Series, how: str) -> pd.Series:
        """Resample to local calendar days and drop the timezone from the index."""
        out = series.resample("1D").agg(how)
        if getattr(out.index, "tz", None) is not None:
            out.index = out.index.tz_localize(None)
        out.index = out.index.normalize()
        return out

    d = pd.DataFrame({
        "water_level_mean": daily(level, "mean"),
        "water_level_min": daily(level, "min"),
        "water_level_max": daily(level, "max"),
        "n_readings": daily(level, "size"),
        "bingen_level_mean": daily(upstream, "mean"),
        "discharge_mean": daily(discharge, "mean"),
    })

    wx = weather.copy()
    if getattr(wx.index, "tz", None) is not None:
        wx.index = wx.index.tz_localize(None)
    wd = wx.resample("1D").agg(
        temperature_mean=("temperature_2m", "mean"),
        humidity_mean=("relative_humidity_2m", "mean"),
        precipitation_sum=("precipitation", "sum"),
        rain_sum=("rain", "sum"),
        snowfall_sum=("snowfall", "sum"),
        pressure_mean=("pressure_msl", "mean"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        hours_available=("temperature_2m", "size"))
    wd.index = wd.index.normalize()

    d = d.join(wd, how="inner")
    d.index.name = "date"
    d = d.reset_index()

    # Nur vollstaendige Tage: unvollstaendige verzerren Summen und Minima.
    # 96 Viertelstunden am Pegel, 24 Stunden beim Wetter.
    d = d[(d["hours_available"] >= 24) & (d["n_readings"] >= 86)]
    d = d.dropna(subset=["water_level_mean", "precipitation_sum"])
    return d[COLUMNS].reset_index(drop=True)


def merge_with_history(live: pd.DataFrame) -> pd.DataFrame:
    """Append live days to the stored history; live wins on overlapping dates."""
    if not HIST.exists():
        raise LiveDataError(f"Historical dataset missing: {HIST}")
    hist = pd.read_csv(HIST, parse_dates=["date"])
    hist = hist[[c for c in COLUMNS if c in hist.columns]]

    both = pd.concat([hist, live], ignore_index=True)
    both = (both.sort_values("date")
                .drop_duplicates("date", keep="last")
                .reset_index(drop=True))

    gaps = pd.date_range(both["date"].min(), both["date"].max(),
                         freq="D").difference(both["date"])
    if len(gaps):
        raise LiveDataError(f"{len(gaps)} calendar day(s) missing after the merge, "
                            f"first at {gaps[0].date()}")
    return both


# ------------------------------------------------------------------ entry
def update(days: int = DAYS, session: requests.Session | None = None) -> dict:
    """Fetch, aggregate, merge, cache. Returns a short status report."""
    level = fetch_gauge(STATION_LEVEL, "W", days, session)
    upstream = fetch_gauge(STATION_UPSTREAM, "W", days, session)
    discharge = fetch_gauge(STATION_DISCHARGE, "Q", days, session)
    weather = fetch_weather(days=days, session=session)

    live = to_daily(level, upstream, discharge, weather)
    if live.empty:
        raise LiveDataError("No complete day could be built from the live sources")

    merged = merge_with_history(live)
    LIVE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(LIVE, index=False)

    hist_end = pd.read_csv(HIST, usecols=["date"], parse_dates=["date"])["date"].max()
    status = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(merged)),
        "data_start": str(merged["date"].min().date()),
        "data_end": str(merged["date"].max().date()),
        "history_end": str(hist_end.date()),
        "live_days_added": int((merged["date"] > hist_end).sum()),
        "latest_level_cm": float(merged.iloc[-1]["water_level_mean"]),
        "sources": {"gauge": f"PEGELONLINE {STATION_LEVEL}/W",
                    "upstream": f"PEGELONLINE {STATION_UPSTREAM}/W",
                    "discharge": f"PEGELONLINE {STATION_DISCHARGE}/Q",
                    "weather": "Open-Meteo hourly observations"},
    }
    STAMP.write_text(json.dumps(status, indent=2))
    return status


def read_status() -> dict | None:
    return json.loads(STAMP.read_text()) if STAMP.exists() else None


def active_dataset() -> Path:
    """The live cache if it exists, otherwise the stored history."""
    return LIVE if LIVE.exists() else HIST


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Refresh the live dataset")
    ap.add_argument("--status", action="store_true", help="show cached status only")
    ap.add_argument("--days", type=int, default=DAYS)
    args = ap.parse_args()

    if args.status:
        st = read_status()
        print(json.dumps(st, indent=2) if st else "No live data cached yet.")
    else:
        try:
            print(json.dumps(update(args.days), indent=2))
        except LiveDataError as e:
            raise SystemExit(f"Live update failed: {e}")
