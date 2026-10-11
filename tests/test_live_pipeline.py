"""Live features must equal training features (no network needed)."""

import numpy as np
import pandas as pd
import pytest

from modeling import kaub_pipeline as kp
from modeling import live_pipeline as lp
from modeling import realtime_pipeline as rp

LIVE_COLUMNS = ["water_level_mean", "precipitation_sum", "temperature_mean"]


def make_live_frame(n_days=60):
    """Looks like realtime_pipeline output: 3 columns, one row per day, decimals."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-08-01", periods=n_days, freq="D", name="date")
    return pd.DataFrame(
        {
            "water_level_mean": 100 + rng.normal(0, 20, n_days).cumsum() / 5,
            "precipitation_sum": rng.gamma(0.5, 4, n_days),
            "temperature_mean": rng.normal(15, 4, n_days),
        },
        index=idx,
    )


def test_live_features_equal_training_features():
    realtime = make_live_frame()
    live = lp.create_live_features(realtime)[kp.FEATURES]
    training = kp.make_features(realtime)[kp.FEATURES]
    pd.testing.assert_frame_equal(live, training)


def test_live_features_equal_old_realtime_features():
    realtime = make_live_frame()
    new = lp.create_live_features(realtime)[kp.FEATURES]
    old = rp.create_live_features(realtime)[kp.FEATURES]
    pd.testing.assert_frame_equal(new, old)


def test_live_features_have_no_targets():
    live = lp.create_live_features(make_live_frame())
    assert not any(c.startswith("target") for c in live.columns)


def test_latest_inputs_is_last_day():
    realtime = make_live_frame()
    X = lp.latest_inputs(lp.create_live_features(realtime), kp.FEATURES)
    assert X.index[0] == realtime.index[-1]
    assert X["lag_0"].iloc[0] == round(realtime["water_level_mean"].iloc[-1])


def test_latest_inputs_stops_on_missing_history():
    realtime = make_live_frame()
    realtime.iloc[-8, 0] = np.nan  # water missing 7 days ago -> lag_7 is NaN
    with pytest.raises(ValueError, match="incomplete history"):
        lp.latest_inputs(lp.create_live_features(realtime), kp.FEATURES)


def test_latest_inputs_stops_on_feature_live_data_cannot_make():
    live = lp.create_live_features(make_live_frame())
    with pytest.raises(ValueError, match="cannot create"):
        lp.latest_inputs(live, kp.FEATURES + ["snow_30d"])
