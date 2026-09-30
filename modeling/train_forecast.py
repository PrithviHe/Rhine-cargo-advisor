"""
HydroLoad-AI | modeling/train_forecast.py

Trainiert je Prognosehorizont (1..7) drei Quantilsmodelle (p10/p50/p90),
kalibriert die Intervalle per Split-Conformal und legt alle Artefakte
in models/ ab.

Aufruf aus dem Projektwurzelordner:
    python -m modeling.train_forecast
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

# ----------------------------------------------------------------- Konfiguration
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "kaub_hydro_weather_daily.csv"
OUT = ROOT / "models"

DATE_COL = "date"
LEVEL_COL = "water_level_mean"
UPSTREAM_COL = "bingen_level_mean"
DISCHARGE_COL = "discharge_mean"
PRECIP_COL = "precipitation_sum"
TEMP_COL = "temperature_mean"

HORIZONS = range(1, 8)
QUANTILES = [0.1, 0.5, 0.9]
TARGET_COVERAGE = 0.80

TRAIN_END = "2021-08-31"     # model is fitted up to here
CALIB_END = "2023-08-31"     # calibration window, remainder is the test set

PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, max_depth=4,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=-1)


# ----------------------------------------------------------------- Features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature-Matrix. EINE Definitionsstelle - Training und App nutzen dieselbe.
    Lags werden relativ zum heutigen Stand kodiert, weil Baumverfahren nicht
    ueber ihren Trainingswertebereich hinaus extrapolieren koennen.
    """
    s = df[LEVEL_COL].astype(float)
    X = pd.DataFrame(index=df.index)

    # Gauge history
    for L in [0, 1, 2, 3, 7, 14, 28]:
        X[f"level_lag_{L}"] = s.shift(L)
    X["level_change_1d"] = s.diff(1)
    X["level_change_3d"] = s.diff(3)

    # Upstream gauge and discharge
    if UPSTREAM_COL in df.columns:
        u = df[UPSTREAM_COL].astype(float)
        for L in [1, 2, 3]:
            X[f"upstream_lag_{L}"] = u.shift(L)
        X["upstream_change_1d"] = u.diff(1)
    if DISCHARGE_COL in df.columns:
        q = df[DISCHARGE_COL].astype(float)
        for L in [1, 2, 3]:
            X[f"discharge_lag_{L}"] = q.shift(L)
        X["discharge_change_1d"] = q.diff(1)

    # Rainfall memory and temperature
    for w in [3, 7, 14, 30]:
        X[f"precip_{w}d"] = df[PRECIP_COL].astype(float).rolling(w).sum()
    for w in [3, 7]:
        X[f"temp_{w}d"] = df[TEMP_COL].astype(float).rolling(w).mean()

    # Season
    doy = df[DATE_COL].dt.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).values
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).values

    # relative encoding
    lvl = X["level_lag_0"]
    for c in X.columns:
        if c.startswith(("level_lag_", "upstream_lag_")) and c != "level_lag_0":
            X[c] = X[c] - lvl

    return X.astype(float)


# ----------------------------------------------------------------- Training
def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = (pd.read_csv(DATA, parse_dates=[DATE_COL])
            .sort_values(DATE_COL).reset_index(drop=True))
    s = df[LEVEL_COL].astype(float)
    X = build_features(df)
    d = df[DATE_COL]

    meta = {"features": list(X.columns), "horizons": list(HORIZONS),
            "quantiles": QUANTILES, "target_coverage": TARGET_COVERAGE,
            "train_end": TRAIN_END, "calib_end": CALIB_END,
            "level_col": LEVEL_COL, "date_col": DATE_COL,
            "data_end": str(d.max().date()), "n_rows": int(len(df))}
    report = []

    for h in HORIZONS:
        y = s.shift(-h) - s                          # target: the change, not the level
        m = X.notna().all(axis=1) & y.notna()
        Xh, yh, dh = X[m], y[m], d[m]
        lvl = X.loc[m, "level_lag_0"].values

        fit = (dh <= TRAIN_END).values
        cal = ((dh > TRAIN_END) & (dh <= CALIB_END)).values
        tst = (dh > CALIB_END).values

        models = {}
        for q in QUANTILES:
            mdl = lgb.LGBMRegressor(objective="quantile", alpha=q, **PARAMS)
            mdl.fit(Xh[fit], yh[fit])
            models[q] = mdl
            joblib.dump(mdl, OUT / f"q{int(q * 100):02d}_h{h}.pkl")

        # Split-conformal calibration on the calibration window
        lo_c = lvl[cal] + models[0.1].predict(Xh[cal])
        hi_c = lvl[cal] + models[0.9].predict(Xh[cal])
        mid_c = lvl[cal] + models[0.5].predict(Xh[cal])
        act_c = s.shift(-h)[m][cal].values
        scores = np.maximum(lo_c - act_c, act_c - hi_c)
        n = len(scores)
        k = min(int(np.ceil((n + 1) * TARGET_COVERAGE)) - 1, n - 1)
        qhat = float(np.sort(scores)[k])
        meta[f"qhat_h{h}"] = qhat

        # Residuals of the median forecast -> threshold probability in the app
        np.save(OUT / f"residuals_h{h}.npy", act_c - mid_c)

        # Evaluation on the test window
        lo_t = lvl[tst] + models[0.1].predict(Xh[tst])
        hi_t = lvl[tst] + models[0.9].predict(Xh[tst])
        mid_t = lvl[tst] + models[0.5].predict(Xh[tst])
        act_t = s.shift(-h)[m][tst].values
        low = act_t < 100

        mae = float(np.abs(mid_t - act_t).mean())
        base = float(np.abs(lvl[tst] - act_t).mean())
        report.append({
            "h": h,
            "MAE": mae,
            "baseline_MAE": base,
            "skill_pct": 100 * (1 - mae / base),
            "MAE_lowwater": float(np.abs(mid_t - act_t)[low].mean()),
            "baseline_lowwater": float(np.abs(lvl[tst] - act_t)[low].mean()),
            "coverage_raw": float(np.mean((act_t >= lo_t) & (act_t <= hi_t))),
            "coverage_calibrated": float(np.mean((act_t >= lo_t - qhat) &
                                                 (act_t <= hi_t + qhat))),
            "band_width_cm": float(np.mean((hi_t + qhat) - (lo_t - qhat))),
            "n_test": int(tst.sum()),
        })

    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    rep = pd.DataFrame(report)
    rep.to_csv(OUT / "forecast_report.csv", index=False)
    print(rep.round(2).to_string(index=False))
    print(f"\n{len(list(HORIZONS)) * len(QUANTILES)} models saved to {OUT}")


if __name__ == "__main__":
    main()
