"""
HydroLoad-AI | app/streamlit_app.py

Decision-support dashboard for Rhine cargo planning at Kaub.

Information chain:
    current level -> forecast -> critical level -> risk -> cargo impact -> action

Start from the project root:
    streamlit run app/streamlit_app.py

The app trains nothing. It loads the artefacts produced by
modeling/train_forecast.py and applies modeling/cargo.py to them.
"""
from pathlib import Path
import json
import sys

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modeling.cargo import Assumptions, shortfall_range, payload_t   # noqa: E402
from modeling.train_forecast import build_features                   # noqa: E402

DATA = ROOT / "data" / "processed" / "kaub_hydro_weather_daily.csv"
MODELS = ROOT / "models"

# --- semantic colours, used consistently everywhere -----------------------
C_OBSERVED = "#9aa0a6"
C_FORECAST = "#4c8fd6"
C_CRITICAL = "#e05252"
C_WARN = "#e0913a"
C_OK = "#4fa87a"
HIST_DAYS = 14

st.set_page_config(page_title="HydroLoad-AI", page_icon="🚢", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      div[data-testid="stMetricLabel"] {font-size: .78rem; opacity: .75;}
      h1 {margin-bottom: .1rem;}
      .hl-sub {opacity:.65; font-size:.9rem; margin-top:-.4rem;}
      .hl-band {padding:.9rem 1.1rem; border-radius:.5rem; margin:.4rem 0 1.1rem 0;}
      .hl-band h4 {margin:0 0 .35rem 0; font-size:1rem; letter-spacing:.06em;}
      .hl-band p {margin:0; font-size:.92rem; opacity:.92;}
      .hl-kv {font-size:.86rem; opacity:.85; margin-top:.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================== loading
@st.cache_resource
def load_models():
    meta = json.loads((MODELS / "meta.json").read_text())
    mods = {h: {q: joblib.load(MODELS / f"q{int(q * 100):02d}_h{h}.pkl")
                for q in meta["quantiles"]} for h in meta["horizons"]}
    resid = {h: np.load(MODELS / f"residuals_h{h}.npy") for h in meta["horizons"]}
    return meta, mods, resid


@st.cache_data
def load_data():
    df = (pd.read_csv(DATA, parse_dates=["date"])
            .sort_values("date").reset_index(drop=True))
    return df, build_features(df)


if not (MODELS / "meta.json").exists():
    st.error("Model artefacts are missing. Run `python -m modeling.train_forecast` first.")
    st.stop()

META, MODS, RESID = load_models()
DF, XALL = load_data()
LEVEL = META["level_col"]
VALID = DF.index[XALL.notna().all(axis=1)]


# ======================================================== sidebar
st.sidebar.markdown("### Forecast")

if "ref_i" not in st.session_state:
    st.session_state.ref_i = int(VALID[-1])

sel = st.sidebar.date_input(
    "Reference date",
    value=DF.loc[st.session_state.ref_i, "date"].date(),
    min_value=DF.loc[VALID[0], "date"].date(),
    max_value=DF.loc[VALID[-1], "date"].date(),
    help="The day the forecast is issued. Only days with a complete feature "
         "row can be selected.",
)
hit = DF.index[DF["date"].dt.date == sel]
if len(hit) and hit[0] in VALID:
    st.session_state.ref_i = int(hit[0])

threshold = st.sidebar.number_input(
    "Critical gauge level (cm)", value=78.0, step=1.0,
    help="Below this level, loading restrictions apply. Default is the "
         "equivalent low-water level for Kaub (unverified).")

st.sidebar.markdown("### Scenarios")


def _jump(date_str: str) -> None:
    cand = DF.index[DF["date"] == pd.Timestamp(date_str)]
    if len(cand) and cand[0] in VALID:
        st.session_state.ref_i = int(cand[0])


if st.sidebar.button("Latest available", use_container_width=True):
    st.session_state.ref_i = int(VALID[-1])
if st.sidebar.button("Before record low", use_container_width=True):
    _jump("2026-08-10")
if st.sidebar.button("Normal conditions", use_container_width=True):
    _jump("2024-03-01")

st.sidebar.markdown("### Cargo assumptions")
st.sidebar.caption("Unverified - pending WSV / ELWIS sources.")
_d = Assumptions()
A = Assumptions(
    glw_cm=threshold,
    max_payload_t=st.sidebar.number_input("Max payload (t)", value=_d.max_payload_t, step=50.0),
    truck_payload_t=st.sidebar.number_input("Truck payload (t)", value=_d.truck_payload_t, step=1.0),
    train_payload_t=st.sidebar.number_input("Train payload (t)", value=_d.train_payload_t, step=50.0),
)


# ======================================================== forecast
i = st.session_state.ref_i
row = XALL.loc[[i]]
today = float(DF.loc[i, LEVEL])
ref_date = DF.loc[i, "date"]

rows = []
for h in META["horizons"]:
    qh = META[f"qhat_h{h}"]
    lo = max(0.0, today + MODS[h][0.1].predict(row)[0] - qh)
    mid = today + MODS[h][0.5].predict(row)[0]
    hi = today + MODS[h][0.9].predict(row)[0] + qh
    rows.append({
        "h": h,
        "date": ref_date + pd.Timedelta(days=h),
        "lo": lo, "mid": mid, "hi": hi,
        "p_below": float(np.mean(mid + RESID[h] < threshold)),
        "actual": float(DF.loc[i + h, LEVEL]) if i + h < len(DF) else np.nan,
    })
F = pd.DataFrame(rows)


# ======================================================== risk logic
def classify_day(lo: float, mid: float, thr: float) -> str:
    """
    Per-day status from the median forecast and the calibrated interval:
      Critical - median below the critical level
      At risk  - median above, but the critical level lies inside the interval
      Normal   - the whole interval sits above the critical level
    """
    if mid < thr:
        return "Critical"
    if lo < thr:
        return "At risk"
    return "Normal"


F["status"] = [classify_day(r.lo, r.mid, threshold) for r in F.itertuples()]
F["gap"] = F["mid"] - threshold

breach_days = int((F["status"] == "Critical").sum())
risk_days = int((F["status"] == "At risk").sum())
n_days = len(F)
max_p = float(F["p_below"].max())

if breach_days > 0:
    STATUS, C_STATUS = "LOW WATER RISK", C_CRITICAL
    HEADLINE = ("Median forecast falls below the critical level on "
                f"{breach_days} of {n_days} days.")
    ACTION = "Reduce loading and secure alternative capacity."
elif risk_days > 0:
    STATUS, C_STATUS = "ELEVATED UNCERTAINTY", C_WARN
    HEADLINE = ("Median forecast stays above the critical level, but the "
                f"80 % interval reaches below it on {risk_days} of {n_days} days.")
    ACTION = "Review loading conditions and monitor the forecast daily."
else:
    STATUS, C_STATUS = "NORMAL CONDITIONS", C_OK
    HEADLINE = "The 80 % interval stays above the critical level on all days."
    ACTION = "Normal loading conditions expected."

RISK_WORD = {"LOW WATER RISK": "HIGH", "ELEVATED UNCERTAINTY": "MODERATE",
             "NORMAL CONDITIONS": "LOW"}[STATUS]


# ======================================================== header
st.title("HydroLoad-AI")
st.markdown('<p class="hl-sub">Water level forecast &amp; cargo planning &middot; '
            f'Kaub gauge &middot; issued {ref_date:%d %b %Y}</p>',
            unsafe_allow_html=True)

k = st.columns(4)
k[0].metric("Current level", f"{today:.0f} cm")
k[1].metric("Forecast (day +7)", f"{F.iloc[-1]['mid']:.0f} cm",
            f"{F.iloc[-1]['lo']:.0f}-{F.iloc[-1]['hi']:.0f} cm", delta_color="off")
k[2].metric("Critical level", f"{threshold:.0f} cm")
k[3].metric("Risk", RISK_WORD)


# ======================================================== status band
gap_mean = float(F["gap"].mean())
st.markdown(
    f"""
    <div class="hl-band" style="background:{C_STATUS}1F;border-left:4px solid {C_STATUS};">
      <h4 style="color:{C_STATUS};">{STATUS}</h4>
      <p>{HEADLINE}</p>
      <p style="margin-top:.5rem;"><b>Recommended action:</b> {ACTION}</p>
      <div class="hl-kv">
        Forecast day +7: {F.iloc[-1]['lo']:.0f}-{F.iloc[-1]['hi']:.0f} cm
        &nbsp;&middot;&nbsp; Critical level: {threshold:.0f} cm
        &nbsp;&middot;&nbsp; Mean gap: {gap_mean:+.0f} cm
        &nbsp;&middot;&nbsp; Affected days: {breach_days} / {n_days}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ======================================================== chart
st.subheader("Seven-day water level forecast")
st.caption("Median forecast - 80 % prediction interval")

show_actual = st.checkbox("Show what actually happened (hindsight)", value=False)

hist = (DF.loc[max(0, i - HIST_DAYS):i, ["date", LEVEL]]
          .rename(columns={LEVEL: "level"}).copy())
hist["series"] = "Observed"
plot = F.copy()
plot["series"] = "Forecast"
plot["band"] = "80 % prediction interval"

pad = 0.08 * max(1.0, F["hi"].max() - min(F["lo"].min(), threshold))
ymin = max(0.0, min(hist["level"].min(), F["lo"].min(), threshold) - pad)
ymax = max(hist["level"].max(), F["hi"].max()) + pad
YS = alt.Scale(domain=[ymin, ymax], nice=False)
XA = alt.Axis(format="%d %b", labelAngle=-45, tickCount="day", grid=False)

COLORS = alt.Scale(
    domain=["Observed", "Forecast", "80 % prediction interval", "Critical level"],
    range=[C_OBSERVED, C_FORECAST, C_FORECAST, C_CRITICAL])
LEG = alt.Legend(title=None, orient="top", direction="horizontal", symbolSize=110)

band = alt.Chart(plot).mark_area(opacity=0.22).encode(
    x=alt.X("date:T", title=None, axis=XA),
    y=alt.Y("lo:Q", title="Gauge level (cm)", scale=YS), y2="hi:Q",
    color=alt.Color("band:N", scale=COLORS, legend=LEG))

obs = alt.Chart(hist).mark_line(strokeWidth=2.2).encode(
    x=alt.X("date:T", axis=XA), y=alt.Y("level:Q", scale=YS),
    color=alt.Color("series:N", scale=COLORS, legend=LEG),
    tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
             alt.Tooltip("level:Q", title="Observed (cm)", format=".0f")])

fc = alt.Chart(plot).mark_line(point=True, strokeDash=[6, 3], strokeWidth=2.6).encode(
    x=alt.X("date:T", axis=XA), y=alt.Y("mid:Q", scale=YS),
    color=alt.Color("series:N", scale=COLORS, legend=LEG),
    tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
             alt.Tooltip("mid:Q", title="Forecast (cm)", format=".0f"),
             alt.Tooltip("lo:Q", title="80 % low (cm)", format=".0f"),
             alt.Tooltip("hi:Q", title="80 % high (cm)", format=".0f"),
             alt.Tooltip("gap:Q", title="Gap to critical (cm)", format="+.0f"),
             alt.Tooltip("p_below:Q", title="P(below critical)", format=".0%"),
             alt.Tooltip("status:N", title="Status")])

crit_df = pd.DataFrame({"y": [threshold], "series": ["Critical level"]})
rule = alt.Chart(crit_df).mark_rule(strokeDash=[4, 4], strokeWidth=1.6).encode(
    y=alt.Y("y:Q", scale=YS),
    color=alt.Color("series:N", scale=COLORS, legend=LEG))
crit_lab = alt.Chart(pd.DataFrame(
    {"y": [threshold], "t": [f"Critical level - {threshold:.0f} cm"]})
).mark_text(align="left", baseline="bottom", dx=4, dy=-4,
            color=C_CRITICAL, fontSize=11).encode(
    x=alt.value(6), y=alt.Y("y:Q", scale=YS), text="t:N")

split = alt.Chart(pd.DataFrame({"x": [ref_date]})).mark_rule(
    color="#6b7280", strokeDash=[2, 3], strokeWidth=1).encode(x="x:T")
split_lab = alt.Chart(pd.DataFrame({"x": [ref_date], "y": [ymax],
                                    "t": ["Forecast starts"]})).mark_text(
    align="left", baseline="top", dx=5, dy=2, color="#6b7280", fontSize=10).encode(
    x="x:T", y=alt.Y("y:Q", scale=YS), text="t:N")

chart = band + obs + fc + rule + crit_lab + split + split_lab
if show_actual and F["actual"].notna().any():
    chart += alt.Chart(F.dropna(subset=["actual"])).mark_line(
        point=True, strokeWidth=2.4, color=C_CRITICAL).encode(
        x="date:T", y=alt.Y("actual:Q", scale=YS),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                 alt.Tooltip("actual:Q", title="Actual (cm)", format=".0f")])

st.altair_chart(chart.properties(height=370).configure_view(strokeOpacity=0),
                use_container_width=True)


# ======================================================== risk metrics
st.subheader("Risk")
r = st.columns(4)
r[0].metric("Days below critical", f"{breach_days} / {n_days}",
            help="Days on which the median forecast falls below the critical level.")
r[1].metric("Critical level inside interval", f"{risk_days} / {n_days}",
            help="Median above the critical level, but the 80 % interval reaches below it.")
r[2].metric("Highest probability below critical", f"{100 * max_p:.0f} %",
            help="From the empirical error distribution of the median forecast "
                 "on the calibration window.")
r[3].metric("Widest 80 % interval", f"{(F['hi'] - F['lo']).max():.0f} cm",
            help="Uncertainty grows with the forecast horizon.")


# ======================================================== cargo impact
st.subheader("Cargo impact")

last = F.iloc[-1]
rec_t = payload_t(last["lo"], A)                    # conservative: lower bound
mid_t = payload_t(last["mid"], A)
util = 100 * rec_t / A.max_payload_t
loss_t = A.max_payload_t - rec_t
trucks_needed = int(-(-loss_t // A.truck_payload_t))
trains_needed = int(-(-loss_t // A.train_payload_t))

c = st.columns(4)
c[0].metric("Recommended loading (day +7)", f"{rec_t:,.0f} t",
            f"{util:.0f} % of max", delta_color="off")
c[1].metric("Capacity loss vs. full load", f"-{loss_t:,.0f} t",
            f"-{100 - util:.0f} %", delta_color="off")
c[2].metric("Replacement capacity", f"{trucks_needed} trucks",
            f"or {trains_needed} train(s)", delta_color="off")
c[3].metric("Affected forecast days", f"{breach_days} / {n_days}")

st.caption(f"Recommended loading follows the lower bound of the interval, not the "
           f"median ({mid_t:,.0f} t). Overloading risks grounding; underloading only "
           f"costs revenue, so the two errors are not equivalent. Reference vessel: "
           f"{A.vessel_name}, {A.max_payload_t:,.0f} t at {A.max_draft_m:.2f} m draft.")


# ======================================================== detail table
with st.expander("Forecast detail by day", expanded=True):
    tbl = []
    for _, rr in F.iterrows():
        g = shortfall_range(rr["lo"], rr["mid"], rr["hi"], A)
        tbl.append({
            "Date": rr["date"].strftime("%d %b"),
            "Day": f"+{int(rr.h)}",
            "Forecast": f"{rr['mid']:.0f} cm",
            "80 % interval": f"{rr['lo']:.0f}-{rr['hi']:.0f} cm",
            "Gap": f"{rr['gap']:+.0f} cm",
            "P(below)": f"{100 * rr['p_below']:.0f} %",
            "Status": rr["status"],
            "Payload": f"{g['payload_t'][0]:,.0f}-{g['payload_t'][2]:,.0f} t",
            "Trucks": f"{g['trucks'][2]}-{g['trucks'][0]}",
            "Trains": f"{g['trains'][2]}-{g['trains'][0]}",
        })
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    st.caption("Payload and truck counts are ranges because the forecast is a range. "
               "A lower water level means a larger replacement requirement, so the "
               "order is reversed relative to the level.")


# ======================================================== model card
with st.expander("Model performance and limitations"):
    st.dataframe(pd.read_csv(MODELS / "forecast_report.csv").round(2),
                 use_container_width=True, hide_index=True)
    st.markdown(
        """
**Reading the table**

- `skill_pct` compares the model against the persistence baseline
  ("the level in *h* days equals today's level"). Positive means better.
- `coverage_calibrated` is the share of actual values that fell inside the
  80 % interval on the test period. The target is 0.80.
- `MAE_lowwater` is the error below 100 cm - the regime this product exists for.
  Where it exceeds `baseline_lowwater`, the naive forecast is still better.

**Limitations**

- Models are trained on *observed* precipitation. In live operation only weather
  *forecasts* are available, so real accuracy will be lower than shown here.
- Weather is measured at Kaub and acts as a proxy for the wider catchment.
- The upstream gauge and discharge peak at lag 0, so they add little forecast lead time.
- Every cargo constant is an assumption pending verification against WSV / ELWIS.
        """
    )
    st.json(A.as_dict())
