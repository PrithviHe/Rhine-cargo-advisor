"""Live forecast inputs, built with the SAME feature code as training.

realtime_pipeline.py has its own copy of the feature code. This module keeps its
fetching and cleaning steps (realtime_pipeline.py itself is unchanged) but creates
the features with hydro_pipeline.build_features - the function the model was
trained with. So training and live features can never drift apart.

Use it like realtime_pipeline:

    from modeling.live_pipeline import live_pipeline
    realtime, X_live = live_pipeline(saved["features"])
"""

import numpy as np

from modeling import hydro_pipeline as hp
from modeling import realtime_pipeline as rp


# ---------------------------------------------------------------
# Step 1: fetch and clean live data (reused from realtime_pipeline)
# ---------------------------------------------------------------
def fetch_live_daily():
    """Last ~30 days of Kaub water level + weather, one row per day."""
    water = rp.fetch_water_data()
    water = rp.clean_water_data(water)
    water = rp.create_daily_water(water)

    weather = rp.fetch_weather_data()
    weather = rp.clean_weather_data(weather)

    realtime = rp.merge_live_data(water, weather)
    rp.validate_live_data(realtime)
    return realtime


# ---------------------------------------------------------------
# Step 2: features (the training function)
# ---------------------------------------------------------------
def create_live_features(realtime):
    """Same features as training. Target columns are removed (future is unknown)."""
    features = hp.build_features(realtime, round_inputs=rp.ROUND_BEFORE_FEATURES)

    target_columns = [c for c in features.columns if c.startswith("target")]
    return features.drop(columns=target_columns)


# ---------------------------------------------------------------
# Step 3: the latest row, checked
# ---------------------------------------------------------------
def latest_inputs(live_features, feature_names):
    """Model inputs for the latest day. Stops if anything is missing."""
    missing = [name for name in feature_names if name not in live_features.columns]
    if missing:
        raise ValueError(f"Live data cannot create these model features: {missing}")

    X_live = live_features[feature_names].iloc[[-1]]

    invalid = ~np.isfinite(X_live.to_numpy(dtype=float))
    if invalid.any():
        bad_columns = X_live.columns[invalid[0]].tolist()
        raise ValueError(
            f"Latest input has incomplete history for: {bad_columns}. "
            "No forecast was generated."
        )

    return X_live


# ---------------------------------------------------------------
# Everything together (same inputs/outputs as realtime_pipeline)
# ---------------------------------------------------------------
def live_pipeline(feature_names):
    realtime = fetch_live_daily()
    live_features = create_live_features(realtime)
    X_live = latest_inputs(live_features, feature_names)
    return realtime, X_live
