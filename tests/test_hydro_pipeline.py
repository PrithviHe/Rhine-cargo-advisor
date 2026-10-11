"""Alignment and leakage tests for modeling.hydro_pipeline (feature side, no raw files)."""

import numpy as np
import pandas as pd
import pytest

from modeling.hydro_pipeline import (
    FEATURE_GROUPS,
    alarm_scores,
    build_features,
    time_split,
)


def make_daily(n_days: int = 400, start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_days, freq="D", name="date")
    level = np.arange(n_days, dtype=float) + 100  # level == 100 + day number
    return pd.DataFrame(
        {
            "water_level_mean": level,
            "bingen_level_mean": level + 1000,
            "discharge_mean": level + 5000,
            "precipitation_sum": 1.0,
            "temperature_mean": 10.0,
            "snowfall_sum": 0.0,
            "et0_sum": 1.0,
            "hours_available": 24,
            "hours_expected": 24,
        },
        index=idx,
    )


def test_lags_and_targets_are_calendar_days():
    feats = build_features(make_daily())
    day = pd.Timestamp("2020-03-01")
    today = feats.loc[day, "lag_0"]
    assert feats.loc[day, "lag_7"] == today - 7
    assert feats.loc[day, "bingen_lag_1"] == today - 1 + 1000
    assert feats.loc[day, "target_day_1"] == today + 1
    assert feats.loc[day, "target"] == today + 7


def test_missing_day_yields_nan_not_a_wrong_lag():
    daily = make_daily()
    daily.loc["2020-03-01", :] = np.nan  # a gap, as asfreq("D") would produce
    feats = build_features(daily)
    assert np.isnan(feats.loc["2020-03-08", "lag_7"])
    assert np.isnan(feats.loc["2020-02-23", "target"])
    assert np.isnan(feats.loc["2020-03-05", "precip_7d"])
    assert feats.loc["2020-03-09", "lag_7"] == feats.loc["2020-03-02", "lag_0"]


def test_rejects_index_with_gaps():
    daily = make_daily().drop(pd.Timestamp("2020-03-01"))
    with pytest.raises(ValueError):
        build_features(daily)


def test_incomplete_days_masked_against_hours_expected():
    daily = make_daily()
    daily.loc["2020-03-29", "hours_expected"] = 23  # spring-forward day, complete
    daily.loc["2020-03-29", "hours_available"] = 23
    daily.loc["2020-04-10", "hours_available"] = 20  # genuinely incomplete
    feats = build_features(daily, drop_incomplete=True)
    assert not np.isnan(feats.loc["2020-03-29", "lag_0"])
    assert np.isnan(feats.loc["2020-04-10", "lag_0"])


def test_time_split_embargo():
    feats = build_features(make_daily(n_days=6 * 365))
    cols = FEATURE_GROUPS["Kaub only"] + ["target"]
    s = time_split(feats[cols], horizon=7, test_years=1, val_years=1)
    # last training target lands before the first test origin
    assert s["train"].index.max() + pd.Timedelta(days=7) < s["test"].index.min()
    assert s["train_fit"].index.max() + pd.Timedelta(days=7) < s["validation"].index.min()
    assert s["validation"].index.max() == s["train"].index.max()


def test_alarm_scores_counts_missed_alarms():
    scores = alarm_scores([70, 70, 90, 90], [70, 90, 70, 90], low_cm=78)
    assert (scores["Hits"], scores["Missed alarms"], scores["False alarms"]) == (1, 1, 1)
    assert scores["Recall"] == 0.5


def test_kaub_pipeline_predict_dates():
    from xgboost import XGBRegressor
    from modeling import kaub_pipeline as kp

    daily = make_daily()
    feats = kp.make_features(daily).dropna(subset=kp.FEATURES + ["target"])
    model = XGBRegressor(n_estimators=5).fit(feats[kp.FEATURES], feats["target"])
    saved = {"model": model, "features": kp.FEATURES, "horizon": 7}

    forecast = kp.predict(saved, daily)
    day = forecast.index[0]
    assert forecast.loc[day, "forecast_date"] == day + pd.Timedelta(days=7)
    assert forecast.loc[day, "today_level"] == daily.loc[day, "water_level_mean"]
    # the last day is forecast too, even though its target is unknown
    assert forecast.index[-1] == daily.index[-1]
