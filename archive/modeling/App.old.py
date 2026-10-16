"""Streamlit app for the saved XGBoost water-level model.

Run from the repository root:
    uv run streamlit run modeling/App.py
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from cargo_calculation import estimate_cargo


# -------------------------------------------------
# Project paths
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.realtime_pipeline import realtime_pipeline
from modeling.cargo_calculation import estimate_cargo


MODEL_PATH = ROOT / "models" / "water_level_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "kaub_hydro_weather_daily.csv"


# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="HydroLoad-AI",
    page_icon="🚢",
)

st.title("🚢 HydroLoad-AI")
st.caption("Water-level forecast and cargo advice")


# -------------------------------------------------
# Load the saved model
# -------------------------------------------------
@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)

    required = {"model", "features", "horizon"}

    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError(
            "The model file must contain model, features, and horizon."
        )

    return saved


# -------------------------------------------------
# Historical feature engineering
# -------------------------------------------------
@st.cache_data
def load_features(horizon):
    """Apply the historical preprocessing used in the notebook."""
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        index_col="date",
    ).sort_index()

    if df.index.has_duplicates:
        raise ValueError("Historical data contains duplicate dates.")

    # Preserve the preprocessing from your supplied notebook code.
    df = df.select_dtypes(include="number").round()

    # Keep missing calendar days visible so shifts represent days.
    df = df.asfreq("D")

    required = [
        "water_level_mean",
        "precipitation_sum",
        "temperature_mean",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing historical columns: {missing}")

    level = df["water_level_mean"]
    features = pd.DataFrame(index=df.index)

    for lag in [0, 1, 2, 3, 7, 14, 28]:
        features[f"lag_{lag}"] = level.shift(lag)

    for days in [3, 7, 14, 30]:
        features[f"precip_{days}d"] = (
            df["precipitation_sum"].rolling(days).sum()
        )

    for days in [3, 7]:
        features[f"temp_{days}d"] = (
            df["temperature_mean"].rolling(days).mean()
        )

    features["actual"] = level.shift(-horizon)

    return features


def predict_level(model, X, feature_names):
    """Validate inputs and return one prediction."""
    X = X.loc[:, feature_names]

    if len(X) != 1:
        raise ValueError("Prediction requires exactly one input row.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Model inputs contain missing or infinite values.")

    prediction = float(model.predict(X)[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    return prediction


try:
    saved = load_model()
    model = saved["model"]
    feature_names = list(saved["features"])
    horizon = int(saved["horizon"])

except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
mode = st.sidebar.radio(
    "Data source",
    ["Live forecast", "Historical forecast"],
)

threshold = st.sidebar.number_input(
    "Low-water threshold (cm)",
    value=77,
    step=1,
)

st.sidebar.caption(f"Forecast horizon: {horizon} days")


# -------------------------------------------------
# Live forecast
# -------------------------------------------------
if mode == "Live forecast":
    st.write(
        "Fetch the latest water and weather observations "
        "and generate an updated forecast."
    )

    if st.button("Update live forecast", type="primary"):
        try:
            with st.spinner("Fetching data and generating the forecast..."):
                realtime, X_live = realtime_pipeline(feature_names)

                forecast = predict_level(
                    model,
                    X_live,
                    feature_names,
                )

                origin = pd.Timestamp(X_live.index[0])

                if "water_level_mean" not in realtime.columns:
                    raise ValueError(
                        "Live data must contain water_level_mean."
                    )

                history = (
                    realtime["water_level_mean"]
                    .sort_index()
                    .loc[:origin]
                    .tail(90)
                )

                st.session_state["live_result"] = {
                    "row": X_live.copy(),
                    "origin": origin,
                    "forecast": forecast,
                    "history": history.copy(),
                    "updated_at": pd.Timestamp.now(tz="Europe/Berlin"),
                }

        except Exception as exc:
            st.session_state.pop("live_result", None)
            st.error(f"Could not update the live forecast: {exc}")

    result = st.session_state.get("live_result")

    if result is None:
        st.info("Click Update live forecast to begin.")
        st.stop()

    row = result["row"]
    origin = result["origin"]
    forecast = result["forecast"]
    history = result["history"]
    actual = None

    st.caption(
        f"Last successful fetch: "
        f"{result['updated_at']:%d %B %Y, %H:%M %Z}. "
        "Click Update live forecast to fetch again."
    )


# -------------------------------------------------
# Historical forecast
# -------------------------------------------------
else:
    try:
        data = load_features(horizon)

        missing = [
            feature for feature in feature_names
            if feature not in data.columns
        ]

        if missing:
            raise ValueError(f"Missing historical features: {missing}")

        rows = data.dropna(subset=feature_names)

        if rows.empty:
            raise ValueError("No complete historical inputs are available.")

    except Exception as exc:
        st.error(f"Could not prepare historical data: {exc}")
        st.stop()

    selected_date = st.date_input(
        "Forecast from",
        value=rows.index.max().date(),
        min_value=rows.index.min().date(),
        max_value=rows.index.max().date(),
    )

    origin = pd.Timestamp(selected_date)

    if origin not in rows.index:
        st.warning("Complete inputs are unavailable for this date.")
        st.stop()

    row = rows.loc[[origin]]

    try:
        forecast = predict_level(model, row, feature_names)
    except Exception as exc:
        st.error(f"Could not generate the forecast: {exc}")
        st.stop()

    actual = row["actual"].iloc[0]

    history = data.loc[
        origin - pd.Timedelta(days=90):origin,
        "lag_0",
    ]

    st.caption(
        "Historical predictions use the currently saved model. "
        "This view alone is not an out-of-sample evaluation."
    )


# -------------------------------------------------
# Shared forecast display
# -------------------------------------------------
observed_level = float(row["lag_0"].iloc[0])
target_date = (origin + pd.Timedelta(days=horizon)).date()

col1, col2 = st.columns(2)

col1.metric(
    f"Observed level on {origin.date()}",
    f"{observed_level:.1f} cm",
)

col2.metric(
    f"Forecast for {target_date}",
    f"{forecast:.1f} cm",
    delta=f"{forecast - observed_level:+.1f} cm",
    delta_color="off",
)

if actual is not None and pd.notna(actual):
    st.metric(
        f"Recorded level on {target_date}",
        f"{actual:.1f} cm",
    )


# -------------------------------------------------
# Low-water messages
# -------------------------------------------------
if observed_level < threshold:
    st.warning(
        f"Observed low water: the level on {origin.date()} "
        f"was below {threshold} cm."
    )

if forecast < threshold:
    st.warning(
        f"Forecast low water: the predicted level for "
        f"{target_date} is below {threshold} cm."
    )
else:
    st.info(
        f"The predicted level for {target_date} is at or above "
        f"{threshold} cm."
    )

st.caption(
    "This is a point forecast, not a guaranteed water level. "
    "Kaub gauge level is not navigation depth and does not "
    "by itself establish safe cargo capacity."
)


# -------------------------------------------------
# History and model inputs
# -------------------------------------------------
st.subheader("Observed water levels")
st.line_chart(history.rename("Water level (cm)"))

with st.expander("Model input"):
    st.dataframe(row[feature_names])


#Cargo calculation

requested_cargo = st.number_input(
    "Total cargo to transport (tonnes)",
    min_value=0.0,
    value=2000.0,
    step=100.0,
    key="cargo_requested_tonnes",
)

if st.button("Calculate cargo and trucks", key="calculate_cargo"):
    try:
        cargo = estimate_cargo(
            forecast_cm=float(forecast),       # Model prediction in cm
            requested_cargo_t=requested_cargo, # User-entered tonnes

            # Fixed example settings — replace with verified values.
            reference_gauge_cm=100.0,
            reference_depth_m=2.5,
            clearance_m=0.3,
            empty_draft_m=1.0,
            full_draft_m=3.0,
            max_payload_t=2500.0,
            truck_capacity_t=25.0,
        )

    except ValueError as error:
        st.error(str(error))

    else:
        st.metric(
            "Cargo by vessel",
            f"{cargo['vessel_cargo_t']:,.1f} tonnes",
        )

        st.metric(
            "Remaining cargo",
            f"{cargo['remaining_t']:,.1f} tonnes",
        )

        st.metric(
            "Truckloads needed",
            cargo["truckloads"],
        )

        if cargo["insufficient_empty_draft"]:
            st.warning(
                "Estimated depth is insufficient for the empty "
                "vessel with the configured clearance."
            )