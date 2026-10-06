"""Streamlit app for the saved XGBoost water-level model.
Run from the repository root:
    uv run streamlit run modeling/App.py
"""

from pathlib import Path
import base64
import io
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image


# Project paths
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modeling.realtime_pipeline import realtime_pipeline
from modeling.cargo_calculation import estimate_cargo
MODEL_PATH = ROOT / "models" / "water_level_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "kaub_hydro_weather_daily.csv"


# Page configuration
st.set_page_config(

    page_title="HydroLoad-AI",

    page_icon="🚢",
    layout="wide",

)

st.markdown("""
<style>
.stApp {background:#f3f7fb;color:#163650;}
[data-testid="stSidebar"] {background:#102f49;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stCaption,[data-testid="stSidebar"] summary {color:#e6f1fa!important;}
.block-container {padding-top:2rem;max-width:1600px;}
.metric-card {background:white;border:1px solid #dfe8f0;border-radius:10px;padding:18px 14px;min-height:142px;box-shadow:0 3px 12px #173e6510;}
.low-water-alert {animation:low-water-blink 1.4s ease-in-out infinite;}
@keyframes low-water-blink {0%,100% {background:white;} 50% {background:#fde2e2;}}
@media (prefers-reduced-motion:reduce){.low-water-alert{animation:none;background:#fde2e2;}}
.card-label {font-size:11px;font-weight:700;letter-spacing:.6px;color:#567086;}
.card-value {font-size:25px;font-weight:750;margin:12px 0 6px;color:#153b5a;}
.card-detail {font-size:12px;color:#60798b;}
.bar-label {display:flex;justify-content:space-between;font-size:13px;margin:18px 0 8px;gap:10px;}
.bar-track {background:#e7eff6;border-radius:8px;height:12px;margin-bottom:12px;}
.banner {position:relative;border-radius:12px;overflow:hidden;margin-bottom:1.5rem;box-shadow:0 3px 12px #173e6520;}
.banner img {display:block;width:100%;height:auto;}
.banner-text {position:absolute;inset:0;display:flex;flex-direction:column;justify-content:flex-start;padding:22px 28px;background:linear-gradient(to bottom,#102f49e6 0%,#102f4966 45%,transparent 100%);}
.banner-text h1 {color:white!important;font-size:30px;margin:0;padding:0;}
.banner-text p {color:#e6f1fa;font-size:14px;margin:6px 0 0;}
@media (max-width:700px){.banner-text{padding:12px 16px}.banner-text h1{font-size:18px}.banner-text p{font-size:12px}}
</style>
""", unsafe_allow_html=True)
st.sidebar.title("🚢 HydroLoad-AI")
st.sidebar.caption("RHINE CARGO CAPACITY ADVISOR")

BANNER_PATH = ROOT / "images" / "4_kaub.jpg"


@st.cache_data
def banner_base64(path, max_width=1800):
    """Downscale the banner photo once and return it as base64 JPEG."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_width, max_width))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=82)
    return base64.b64encode(buffer.getvalue()).decode()


title = "🚢 Rhine Water-Level Forecasting and Cargo Decision Support"
subtitle = "Rhine water-level forecast & cargo planning · Kaub, Germany"
if BANNER_PATH.exists():
    st.markdown(f"""
<div class="banner">
  <img src="data:image/jpeg;base64,{banner_base64(str(BANNER_PATH))}" alt="Rhine at Kaub">
  <div class="banner-text"><h1>{title}</h1><p>{subtitle}</p></div>
</div>
""", unsafe_allow_html=True)
else:
    st.title(title)
    st.caption(subtitle)

# Load the saved model
@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    required = {"model", "features", "horizon"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError(
            "The model file must contain model, features, and horizon."
        )
    return saved


# Historical feature engineering
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


# Sidebar
mode = st.sidebar.radio(
    "Data source",
    ["Live forecast", "Historical forecast"],
)
threshold = st.sidebar.number_input(
    "Low-water threshold (cm)",
    min_value=0,
    value=77,
    step=1,
)
st.sidebar.caption(f"Forecast horizon: {horizon} days")


# Live forecast
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

# Historical forecast

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

# Dashboard results
observed_level = float(history.loc[origin])
target_date = origin + pd.Timedelta(days=horizon)

with st.sidebar:
    st.markdown('---')
    st.subheader('Cargo planning')
    requested = st.number_input('Requested cargo (t)', min_value=0.0, value=2000.0, step=100.0)
    with st.expander('Vessel and depth assumptions', expanded=False):
        st.caption('Example defaults. Replace with verified values for your vessel and route.')
        reference_gauge = st.number_input('Reference gauge (cm)', value=100.0)
        reference_depth = st.number_input('Depth at reference gauge (m)', min_value=0.01, value=2.5)
        clearance = st.number_input('Under-keel clearance (m)', min_value=0.01, value=0.3)
        empty_draft = st.number_input('Empty draft (m)', min_value=0.01, value=1.0)
        full_draft = st.number_input('Full-load draft (m)', min_value=0.01, value=3.0)
        max_payload = st.number_input('Maximum vessel payload (t)', min_value=1.0, value=2500.0, step=100.0)
        truck_capacity = st.number_input('Truck payload (t)', min_value=0.1, value=25.0)
    st.caption('Cargo estimates update when you change the inputs.')

try:
    cargo = estimate_cargo(
        forecast_cm=forecast, reference_gauge_cm=reference_gauge,
        reference_depth_m=reference_depth, clearance_m=clearance,
        empty_draft_m=empty_draft, full_draft_m=full_draft,
        max_payload_t=max_payload, requested_cargo_t=requested,
        truck_capacity_t=truck_capacity,
    )
except ValueError as exc:
    st.error(f'Check vessel settings: {exc}')
    st.stop()

low_water = forecast < threshold
cards = st.columns(5)
for col, label, value, detail, colour in [
    (cards[0], 'OBSERVED GAUGE', f'{observed_level:.1f} cm', f'{origin:%d %b %Y} · daily mean', '#168ac0'),
    (cards[1], f'{horizon}-DAY FORECAST', f'{forecast:.1f} cm', f'{target_date:%d %b %Y} · {forecast-observed_level:+.1f} cm', '#f28b29'),
    (cards[2], 'AVAILABLE DRAFT', f"{cargo['available_draft_m']:.2f} m", 'Estimated depth minus clearance', '#4166b5'),
    (cards[3], 'CARGO BY VESSEL', f"{cargo['vessel_cargo_t']:,.0f} t", 'One vessel load · estimated', '#16a69a'),
    (cards[4], 'FORECAST STATUS', 'Low water' if low_water else 'Above threshold', f'User threshold: {threshold} cm', "#f00808" if low_water else '#16a69a'),
]:
    blink = ' low-water-alert' if label == 'FORECAST STATUS' and low_water else ''
    col.markdown(f'<div class="metric-card{blink}" style="border-top:4px solid {colour}"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-detail">{detail}</div></div>', unsafe_allow_html=True)

st.write('')
overview, details = st.tabs(['Overview', 'Model and assumptions'])
with overview:
    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.subheader("🌊 Water-level forecast · Kaub")
        st.caption("Recent observations and the predicted water level")

        days = st.radio(
            "Observed history",
            options=[14, 30, 90],
            index=1,
            horizontal=True,
            format_func=lambda n: f"{n} days",
            key="water_history_days",
        )

        origin_ts = pd.Timestamp(origin)
        target_ts = pd.Timestamp(target_date)

        shown = history.loc[
            origin_ts - pd.Timedelta(days=days - 1):origin_ts
        ]

        blue = "#168AC0"
        orange = "#F28B29"
        teal = "#16A69A"
        red = "#DC6373"
        navy = "#163650"

        fig = go.Figure()

        # Highlight the period beyond the latest observation.
        # This shading is not an uncertainty interval.
        fig.add_vrect(
            x0=origin_ts,
            x1=target_ts,
            fillcolor=orange,
            opacity=0.07,
            line_width=0,
            layer="below",
        )

        # Historical observations.
        fig.add_trace(
            go.Scatter(
                x=shown.index,
                y=shown.values,
                name="Observed daily mean",
                mode="lines",
                line=dict(color=blue, width=3),
                connectgaps=False,
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Observed: <b>%{y:.1f} cm</b>"
                    "<extra></extra>"
                ),
            )
        )

        # Highlight the observation at the forecast origin.
        if origin_ts in shown.index and pd.notna(shown.loc[origin_ts]):
            latest_level = float(shown.loc[origin_ts])

            fig.add_trace(
                go.Scatter(
                    x=[origin_ts],
                    y=[latest_level],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=blue,
                        line=dict(color="white", width=2),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        "<b>Forecast reference date</b><br>"
                        "%{x|%d %b %Y}<br>"
                        "Observed: %{y:.1f} cm"
                        "<extra></extra>"
                    ),
                )
            )

        # Single forecast point: no implied daily forecast trajectory.
        fig.add_trace(
            go.Scatter(
                x=[target_ts],
                y=[forecast],
                name=f"Day {horizon} forecast",
                mode="markers",
                marker=dict(
                    color=orange,
                    size=16,
                    symbol="diamond",
                    line=dict(color="white", width=2),
                ),
                hovertemplate=(
                    "<b>Forecast</b><br>"
                    "%{x|%d %b %Y}<br>"
                    "Gauge level: <b>%{y:.1f} cm</b>"
                    "<extra></extra>"
                ),
            )
        )

        # Label the forecast directly on the chart.
        fig.add_annotation(
            x=target_ts,
            y=forecast,
            text=(
                f"<b>{forecast:.1f} cm</b><br>"
                f"{target_ts:%d %b}"
            ),
            showarrow=True,
            arrowhead=0,
            arrowcolor=orange,
            ax=-45,
            ay=-55,
            bgcolor="white",
            bordercolor=orange,
            borderwidth=1,
            borderpad=7,
            font=dict(size=12, color=navy),
        )

        # Actual target value, when available in historical mode.
        if actual is not None and pd.notna(actual):
            fig.add_trace(
                go.Scatter(
                    x=[target_ts],
                    y=[actual],
                    name="Recorded target level",
                    mode="markers",
                    marker=dict(
                        color=teal,
                        size=15,
                        symbol="circle-open",
                        line=dict(width=3),
                    ),
                    hovertemplate=(
                        "<b>Recorded target level</b><br>"
                        "%{x|%d %b %Y}<br>"
                        "Observed: <b>%{y:.1f} cm</b>"
                        "<extra></extra>"
                    ),
                )
            )

        # Threshold label positioned on the left to avoid the forecast.
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=red,
            line_width=1.5,
            annotation_text=f"Low-water threshold · {threshold:g} cm",
            annotation_position="top left",
            annotation_font=dict(color=red, size=11),
        )

        # Boundary between observations and the forecast period.
        fig.add_shape(
            type="line",
            x0=origin_ts,
            x1=origin_ts,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#AAB9C7", width=1, dash="dot"),
        )

        fig.update_layout(
            height=430,
            margin=dict(l=15, r=25, t=35, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color=navy,
            ),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.20,
                xanchor="left",
                x=0,
                font=dict(size=11),
            ),
            yaxis_title="Kaub gauge level (cm)",
            xaxis_title=None,
        )

        fig.update_xaxes(
            range=[
                origin_ts - pd.Timedelta(days=days - 1),
                target_ts + pd.Timedelta(days=2),
            ],
            tickformat="%d %b",
            nticks=7,
            showgrid=False,
            showline=True,
            linecolor="#E2E8F0",
            ticks="outside",
            tickcolor="#CBD5E1",
            fixedrange=False,
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#EDF2F7",
            zeroline=False,
            ticksuffix=" cm",
            fixedrange=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "hydroload_kaub_forecast",
                    "scale": 2,
                },
            },
        )

        st.caption(
            f"Orange diamond: forecast for {target_ts:%d %B %Y} "
            f"({horizon} days ahead). Shading marks the forecast period, "
            "not uncertainty. Gauge level is not navigable depth."
        )

# with overview:
#     left, right = st.columns([1.65, 1])
#     with left:
#         st.subheader('Water-level forecast · Kaub')
#         days = st.radio('Observed history', [14, 30, 90], index=1, horizontal=True, format_func=lambda n: f'{n} days')
#         shown = history.loc[origin-pd.Timedelta(days=days-1):origin]
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=shown.index, y=shown.values, name='Observed daily mean', mode='lines', line=dict(color='#168ac0', width=3), connectgaps=False))
#         fig.add_trace(go.Scatter(x=[target_date], y=[forecast], name=f'Day {horizon} forecast', mode='markers', marker=dict(color='#f28b29', size=14, symbol='diamond')))
#         if actual is not None and pd.notna(actual):
#             fig.add_trace(go.Scatter(x=[target_date], y=[actual], name='Recorded target level', mode='markers', marker=dict(color='#16a69a', size=12, symbol='circle-open')))
#         fig.add_hline(y=threshold, line_dash='dash', line_color='#e05b68', annotation_text=f'Low-water threshold: {threshold} cm')
#         fig.update_layout(height=355, margin=dict(l=10,r=10,t=25,b=10), paper_bgcolor='white', plot_bgcolor='white', font=dict(color='#163650'), legend=dict(orientation='h', y=1.15), yaxis_title='Kaub gauge level (cm)', xaxis_title=None)
#         fig.update_xaxes(showgrid=False)
#         fig.update_yaxes(gridcolor="#728599")
#         st.plotly_chart(fig, use_container_width=True)
#         st.caption(f'The diamond is a single prediction for day {horizon}; no intermediate daily forecasts or uncertainty interval are implied.')
    with right:
        st.subheader('Cargo capacity')
        for label, amount, colour in [('Maximum vessel payload', max_payload, '#1677b8'), ('Estimated capacity at forecast level', cargo['capacity_t'], '#16a69a'), ('Planned vessel load', cargo['vessel_cargo_t'], '#4166b5')]:
            width = max(0, min(100, 100*amount/max_payload))
            st.markdown(f'<div class="bar-label">{label}<b>{amount:,.1f} t</b></div><div class="bar-track"><div style="width:{width}%;background:{colour};height:12px;border-radius:8px"></div></div>', unsafe_allow_html=True)
        st.write('')
        if cargo['insufficient_empty_draft']:
            st.error('Estimated available draft is below the empty-vessel draft. No cargo is assigned to this vessel.')
        elif low_water:
            st.warning(f'Forecast below the configured low-water threshold on {target_date:%d %b}. Review the loading plan and updated conditions.')
        else:
            st.info('Forecast above the configured threshold. Cargo capacity still depends on vessel and depth assumptions.')
        st.caption('Threshold status is a rule-based indicator, not a probability of risk or a navigation clearance.')
    st.subheader('Transport planning')
    a, b, c, d = st.columns(4)
    a.metric('Requested cargo', f'{requested:,.1f} t')
    b.metric('Remaining after one vessel load', f"{cargo['remaining_t']:,.1f} t")
    c.metric('Truckloads for remaining cargo', f"{cargo['truckloads']:,}")
    d.metric('Estimated water depth', f"{cargo['depth_m']:.2f} m")
    st.caption(f'Truckloads assume {truck_capacity:g} tonnes per truck. Cargo estimates use a simplified linear draft–payload relationship.')
    if actual is not None and pd.notna(actual):
        st.info(f'Recorded target level: {actual:.1f} cm. Forecast minus recorded level: {forecast-actual:+.1f} cm. This historical view is not an independent test evaluation.')
with details:
    st.subheader('Forecast inputs')
    st.dataframe(row[feature_names], use_container_width=True)
    st.write(f'The saved model predicts Kaub daily mean gauge level {horizon} days after the reference date. Live inputs use completed observation days.')
    st.subheader('Cargo assumptions')
    st.write('Estimated depth = reference depth + (forecast gauge − reference gauge) / 100. Available draft = estimated depth − clearance. Payload is scaled linearly between empty and full-load draft, and capped at maximum payload.')
    st.caption('Gauge level is not navigation depth. These prototype estimates require verified route depths and vessel assumptions and do not establish safe navigation.')
    st.download_button('Download observed levels (CSV)', history.rename('water_level_cm').to_csv(), file_name='kaub_observed_levels.csv', mime='text/csv')
