"""
HydroLoad-AI | app/core.py

All computation for the dashboard. No Streamlit calls in here, so the numbers
can be tested and reused. The UI layer only formats what this module returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from modeling.cargo import (Assumptions, payload_t, safe_draft_m,
                            available_depth_m, shortfall_range)
from modeling.train_forecast import build_features

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "kaub_hydro_weather_daily.csv"
MODELS = ROOT / "models"

KAUB_LAT, KAUB_LON = 50.085, 7.765


# ---------------------------------------------------------------- loading
def load_models() -> tuple[dict, dict, dict]:
    meta = json.loads((MODELS / "meta.json").read_text())
    mods = {h: {q: joblib.load(MODELS / f"q{int(q * 100):02d}_h{h}.pkl")
                for q in meta["quantiles"]} for h in meta["horizons"]}
    resid = {h: np.load(MODELS / f"residuals_h{h}.npy") for h in meta["horizons"]}
    return meta, mods, resid


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = (pd.read_csv(DATA, parse_dates=["date"])
            .sort_values("date").reset_index(drop=True))
    return df, build_features(df)


# ---------------------------------------------------------------- forecast
def make_forecast(df, X, meta, mods, resid, i: int, threshold: float) -> pd.DataFrame:
    """Seven-day forecast with calibrated interval and per-day status."""
    level_col = meta["level_col"]
    today = float(df.loc[i, level_col])
    ref = df.loc[i, "date"]
    row = X.loc[[i]]

    out = []
    for h in meta["horizons"]:
        qh = meta[f"qhat_h{h}"]
        lo = max(0.0, today + mods[h][0.1].predict(row)[0] - qh)
        mid = today + mods[h][0.5].predict(row)[0]
        hi = today + mods[h][0.9].predict(row)[0] + qh
        out.append({
            "h": h,
            "date": ref + pd.Timedelta(days=h),
            "lo": lo, "mid": mid, "hi": hi,
            "p_below": float(np.mean(mid + resid[h] < threshold)),
            "actual": float(df.loc[i + h, level_col]) if i + h < len(df) else np.nan,
        })
    F = pd.DataFrame(out)
    F["gap"] = F["mid"] - threshold
    F["status"] = np.where(F["mid"] < threshold, "Critical",
                           np.where(F["lo"] < threshold, "Watch", "Normal"))
    return F


# ---------------------------------------------------------------- risk
@dataclass(frozen=True)
class Risk:
    level: str            # NORMAL | WATCH | MODERATE | HIGH
    trend: str            # increasing | stable | easing
    reason: str           # full sentence, used in the recommendation panel
    short: str            # compact wording, used on the KPI card
    breach_days: int
    watch_days: int
    first_breach_h: int | None
    max_p: float


def classify_risk(F: pd.DataFrame, threshold: float) -> Risk:
    breach = int((F["status"] == "Critical").sum())
    watch = int((F["status"] == "Watch").sum())
    n = len(F)
    crossing = F.loc[F["status"] == "Critical", "h"]
    first = int(crossing.iloc[0]) if len(crossing) else None
    max_p = float(F["p_below"].max())

    if breach >= 4:
        lvl = "HIGH"
        reason = f"Forecast stays below the critical level on {breach} of {n} days."
        short = f"below critical on {breach}/{n} days"
    elif breach >= 1:
        lvl = "MODERATE"
        reason = f"Forecast crosses the critical level in {first} day(s)."
        short = f"crosses critical in {first} day(s)"
    elif watch >= 1:
        lvl = "WATCH"
        reason = ("Forecast stays above the critical level, but the 80 % interval "
                  f"reaches below it on {watch} of {n} days.")
        short = f"critical inside range on {watch}/{n} days"
    else:
        lvl = "NORMAL"
        reason = "Forecast and uncertainty range stay above the critical level."
        short = "forecast stays above critical"

    slope = F["p_below"].iloc[-1] - F["p_below"].iloc[0]
    trend = "increasing" if slope > 0.05 else ("easing" if slope < -0.05 else "stable")
    return Risk(lvl, trend, reason, short, breach, watch, first, max_p)


# ---------------------------------------------------------------- cargo
@dataclass(frozen=True)
class CargoView:
    normal_t: float
    forecast_t: float
    reduction_t: float
    reduction_pct: float
    draft_m: float
    normal_draft_m: float
    trucks: int
    trains: int
    extra_vessels: int
    stranded_7d_t: float


def cargo_view(F: pd.DataFrame, A: Assumptions, normal_level_cm: float) -> CargoView:
    """
    Conservative view: loading follows the LOWER bound of the day+7 interval,
    because overloading risks grounding while underloading only costs revenue.
    """
    lo7 = float(F.iloc[-1]["lo"])
    fc_t = payload_t(lo7, A)
    normal_t = A.max_payload_t
    red_t = max(0.0, normal_t - fc_t)

    # stranded over the whole horizon: daily deficit at the lower bound, summed
    stranded = float(sum(max(0.0, normal_t - payload_t(r.lo, A)) for r in F.itertuples()))

    return CargoView(
        normal_t=normal_t,
        forecast_t=fc_t,
        reduction_t=red_t,
        reduction_pct=100 * red_t / normal_t if normal_t else 0.0,
        draft_m=min(safe_draft_m(lo7, A), A.max_draft_m),
        normal_draft_m=min(safe_draft_m(normal_level_cm, A), A.max_draft_m),
        trucks=int(-(-red_t // A.truck_payload_t)),
        trains=int(-(-red_t // A.train_payload_t)),
        extra_vessels=int(-(-red_t // fc_t)) if fc_t > 0 else 0,
        stranded_7d_t=stranded,
    )


# ---------------------------------------------------------------- context
def normal_level(df: pd.DataFrame, level_col: str, ref_date, train_end: str) -> float:
    """Climatological median for this calendar day, from the training period."""
    tr = df[df["date"] <= train_end]
    doy = pd.Timestamp(ref_date).dayofyear
    win = tr[(tr["date"].dt.dayofyear - doy).abs() <= 7]
    return float(win[level_col].median()) if len(win) else float(tr[level_col].median())


def recommendations(risk: Risk, cv: CargoView, A: Assumptions) -> list[str]:
    """At most three concrete actions, derived from risk and cargo impact."""
    if risk.level in ("HIGH", "MODERATE"):
        return [
            f"Reduce vessel loading to {cv.forecast_t:,.0f} t "
            f"({cv.draft_m:.2f} m draft).",
            f"Reserve alternative capacity: about {cv.trucks} trucks "
            f"or {cv.trains} train(s).",
            "Monitor the gauge daily and re-run the forecast each morning.",
        ]
    if risk.level == "WATCH":
        return [
            f"Plan loading at {cv.forecast_t:,.0f} t until the outlook firms up.",
            "Check alternative capacity availability, do not book yet.",
            "Monitor the gauge daily.",
        ]
    return [
        "Normal loading conditions expected - no reduction required.",
        "No alternative capacity needed.",
        "Continue routine monitoring.",
    ]


def key_insights(F: pd.DataFrame, risk: Risk, cv: CargoView, threshold: float) -> list[str]:
    """Interpreted statements, not a repeat of the KPI values."""
    out = []
    slope = F["mid"].iloc[-1] - F["mid"].iloc[0]
    out.append("Water level trending downward over the horizon." if slope < -2
               else ("Water level trending upward over the horizon." if slope > 2
                     else "Water level roughly stable over the horizon."))
    if risk.first_breach_h:
        out.append(f"Critical level crossed on day +{risk.first_breach_h}.")
    elif risk.watch_days:
        out.append("Critical level lies inside the uncertainty range.")
    else:
        out.append("Critical level not reached within the forecast horizon.")
    if cv.reduction_t > 0:
        out.append(f"Recommended cargo down {cv.reduction_t:,.0f} t "
                   f"({cv.reduction_pct:.0f} %) versus full load.")
        out.append("Alternative transport capacity likely required.")
    else:
        out.append("No cargo reduction required.")
    return out[:4]


def executive_summary(F: pd.DataFrame, risk: Risk, cv: CargoView,
                      threshold: float) -> str:
    """One management-level paragraph, generated from the current forecast."""
    last = F.iloc[-1]
    if risk.level in ("HIGH", "MODERATE"):
        when = (f"in {risk.first_breach_h} day(s)" if risk.first_breach_h
                else "within the horizon")
        return (f"Water levels are expected to fall below the critical level of "
                f"{threshold:.0f} cm {when}, reaching {last['mid']:.0f} cm by day +7 "
                f"(80 % range {last['lo']:.0f}\u2013{last['hi']:.0f} cm). Vessel loading "
                f"should be reduced to {cv.forecast_t:,.0f} t, a cut of "
                f"{cv.reduction_t:,.0f} t ({cv.reduction_pct:.0f} %) against a full load. "
                f"Roughly {cv.trucks} trucks or {cv.trains} train(s) of alternative "
                f"capacity will be required to move the same volume.")
    if risk.level == "WATCH":
        return (f"The median forecast stays above the critical level of "
                f"{threshold:.0f} cm, but the 80 % range reaches below it on "
                f"{risk.watch_days} of {len(F)} days. Plan loading at "
                f"{cv.forecast_t:,.0f} t and check alternative capacity availability "
                f"without booking yet.")
    return (f"Water levels are expected to stay above the critical level of "
            f"{threshold:.0f} cm across the full seven-day horizon, with day +7 at "
            f"{last['mid']:.0f} cm. Full loading of {cv.normal_t:,.0f} t remains "
            f"possible and no alternative transport capacity is required.")


def sparkline_values(df: pd.DataFrame, level_col: str, i: int, n: int = 7) -> list[float]:
    """Last n observed levels, for the mini trend line on the KPI card."""
    return df.loc[max(0, i - n + 1):i, level_col].astype(float).tolist()
