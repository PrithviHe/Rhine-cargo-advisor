"""
HydroLoad-AI | app/streamlit_app.py
Rhine Cargo Capacity Advisor - operational decision support for Kaub.

    WATER -> FORECAST -> DRAFT -> CARGO -> RISK -> ACTION

Start from the project root:  streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modeling.cargo import Assumptions                    # noqa: E402
from app import core                                      # noqa: E402

st.set_page_config(page_title="HydroLoad-AI | Rhine Cargo Capacity Advisor",
                   page_icon="🚢", layout="wide")

# --- dark palette -------------------------------------------------------
BG       = "#0e1218"   # page background
PANEL    = "#161c24"   # cards and panels
PANEL_2  = "#1b232d"   # nested surfaces
BORDER   = "#27303c"
SIDEBAR  = "#111823"
TEXT     = "#e6eaf0"
MUTED    = "#8c97a5"

NAVY  = "#5aa9e6"      # headings / primary accent on dark
WATER = "#4c9fe0"      # forecast blue
OK    = "#35b37e"
WARN  = "#e0a13a"
CRIT  = "#ef5a5a"
HIST  = "#b7c0cb"      # historical line on dark
GRID  = "#242c37"
STATUS_COLOR = {"NORMAL": OK, "WATCH": WARN, "MODERATE": WARN, "HIGH": CRIT}

st.markdown(f"""
<style>
  .stApp {{background:{BG};}}
  .block-container {{padding-top:1.3rem; padding-bottom:2.5rem; max-width:1560px;}}

  /* ---------- Sidebar ---------- */
  section[data-testid="stSidebar"] {{background:{SIDEBAR};
      border-right:1px solid {BORDER};}}
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] small
      {{color:{TEXT} !important;}}
  section[data-testid="stSidebar"] hr {{border-color:{BORDER};}}

  section[data-testid="stSidebar"] button[kind="secondary"] {{
      background:{PANEL} !important; color:{TEXT} !important;
      border:1px solid {BORDER} !important; text-align:left !important;
      justify-content:flex-start !important; font-weight:500 !important;}}
  section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
      background:{PANEL_2} !important; border-color:{WATER} !important;
      color:#ffffff !important;}}
  section[data-testid="stSidebar"] button[kind="primary"] {{
      background:{PANEL_2} !important; color:#ffffff !important;
      border:1px solid {BORDER} !important; border-left:5px solid {WATER} !important;
      text-align:left !important; justify-content:flex-start !important;
      font-weight:700 !important;}}
  section[data-testid="stSidebar"] button p {{color:inherit !important;
      font-size:.92rem !important; margin:0 !important;}}
  section[data-testid="stSidebar"] input {{
      background:{PANEL} !important; color:{TEXT} !important;
      border:1px solid {BORDER} !important;}}

  /* ---------- Header ---------- */
  .hl-head {{background:linear-gradient(90deg,#14364e 0%,#1d5b84 100%);
             border:1px solid {BORDER}; border-radius:.6rem; padding:.9rem 1.3rem;
             color:#fff; display:flex; justify-content:space-between;
             align-items:center;}}
  .hl-head h2 {{margin:0; font-size:1.35rem; color:#fff; line-height:1.2;}}
  .hl-head p  {{margin:.12rem 0 0 0; font-size:.87rem; opacity:.9;}}
  .hl-meta    {{text-align:right; font-size:.79rem; opacity:.95; line-height:1.4;}}

  /* ---------- Executive summary ---------- */
  .exec {{background:{PANEL}; border:1px solid {BORDER}; border-left:4px solid {WATER};
          border-radius:.5rem; padding:.85rem 1.15rem; margin-top:.85rem;
          font-size:.9rem; line-height:1.5; color:{TEXT};}}
  .exec b {{color:{NAVY}; letter-spacing:.055em; font-size:.74rem;
            display:block; margin-bottom:.22rem; text-transform:uppercase;}}

  /* ---------- KPI cards ---------- */
  .kpi {{background:{PANEL}; border:1px solid {BORDER}; border-radius:.5rem;
         padding:.65rem .85rem; height:158px; display:flex; flex-direction:column;
         overflow:hidden;}}
  .kpi.pri {{background:{PANEL_2}; box-shadow:0 2px 10px rgba(0,0,0,.35);}}
  .kpi .top {{display:flex; align-items:center; gap:.4rem; margin-bottom:.3rem;}}
  .kpi .ico {{font-size:.98rem; line-height:1;}}
  .kpi .lab {{font-size:.70rem; color:#aab4c0; text-transform:uppercase;
              letter-spacing:.055em; font-weight:700;}}
  .kpi .val {{font-weight:700; line-height:1.08; margin-bottom:.25rem; color:{TEXT};}}
  .kpi.pri .val {{font-size:1.9rem; letter-spacing:-.01em;}}
  .kpi.sec .val {{font-size:1.52rem; letter-spacing:-.01em;}}
  .kpi .sub {{font-size:.765rem; color:#a7b1bd; line-height:1.4; flex:1;
              overflow:hidden;}}
  .kpi .spark {{margin-top:.15rem;}}

  /* ---------- Action banner ---------- */
  .act {{border-radius:.5rem; padding:1.05rem 1.25rem; margin:1.15rem 0 .4rem 0;
         border:1px solid {BORDER};}}
  .act .hdr {{font-weight:800; letter-spacing:.05em; font-size:.95rem;
              margin-bottom:.45rem;}}
  .act ul {{margin:.3rem 0 0 1.15rem; padding:0; columns:2; column-gap:2.6rem;}}
  .act li {{margin-bottom:.42rem; font-size:.9rem; break-inside:avoid;
            line-height:1.45; color:{TEXT};}}

  /* ---------- Panels ---------- */
  .ptitle {{font-size:.95rem; font-weight:700; color:{NAVY};
            margin:0 0 .55rem 0; display:flex; align-items:center; gap:.4rem;}}
  .bar-lab {{display:flex; justify-content:space-between; font-size:.8rem;
             color:{TEXT}; margin-bottom:.18rem;}}
  .bar-bg {{background:{PANEL_2}; border:1px solid {BORDER}; border-radius:3px;
            height:12px; margin-bottom:.7rem;}}
  .bar-fill {{height:10px; border-radius:3px;}}
  .loss {{background:{CRIT}1A; border:1px solid {CRIT}55; border-radius:.4rem;
          padding:.45rem .65rem; margin-top:.15rem; font-size:.85rem; color:{TEXT};
          display:flex; justify-content:space-between; align-items:center;}}
  .ins {{font-size:.855rem; margin-bottom:.42rem; line-height:1.4; color:{TEXT};}}
  .chain-row {{background:{PANEL_2}; border:1px solid {BORDER}; border-radius:.35rem;
               padding:.32rem .55rem;}}
  .chain-lab {{font-size:.70rem; color:#aab4c0; text-transform:uppercase;
               letter-spacing:.045em; font-weight:600;}}
  .chain-val {{font-size:1.02rem; font-weight:700; color:{NAVY}; line-height:1.15;}}
  .chain-note {{font-size:.73rem; color:#a7b1bd;}}
  .chain-arrow {{text-align:center; color:{WATER}; font-size:.85rem;
                 line-height:1; margin:.12rem 0;}}

  div[data-testid="stMetricValue"] {{font-size:1.45rem; color:{TEXT};}}
  div[data-testid="stMetricLabel"] {{font-size:.76rem; color:#aab4c0;}}
  div[data-testid="stCaptionContainer"], .stCaption, [data-testid="stCaptionContainer"] p
      {{color:#9aa5b1 !important;}}
  div[data-testid="stVerticalBlockBorderWrapper"] {{background:{PANEL};
      border-color:{BORDER} !important; border-radius:.5rem;}}
</style>""", unsafe_allow_html=True)


# ================================================================= data
@st.cache_resource
def _models():
    return core.load_models()


@st.cache_data
def _data():
    return core.load_data()


if not (core.MODELS / "meta.json").exists():
    st.error("Model artefacts missing. Run `python -m modeling.train_forecast` first.")
    st.stop()

META, MODS, RESID = _models()
DF, XALL = _data()
LEVEL = META["level_col"]
VALID = DF.index[XALL.notna().all(axis=1)]


# ================================================================= sidebar
st.sidebar.markdown("### 🚢 HydroLoad-AI")
st.sidebar.caption("Rhine Cargo Capacity Advisor")

NAV = [("Overview", "\U0001F3E0"), ("Forecast", "\U0001F4C8"),
       ("Cargo Planning", "\U0001F6A2"), ("Transport Options", "\U0001F69B"),
       ("Validation", "\U0001F50D"), ("Reports", "\U0001F4CA"),
       ("Settings", "\u2699\uFE0F")]
if "page" not in st.session_state:
    st.session_state.page = "Overview"

for name, icon in NAV:
    if st.sidebar.button(f"{icon}\u2002{name}", key=f"nav_{name}",
                         use_container_width=True,
                         type="primary" if st.session_state.page == name else "secondary"):
        st.session_state.page = name
        st.rerun()          # sofort neu zeichnen, sonst bleibt der Balken eine Runde zurueck
PAGE = st.session_state.page

st.sidebar.divider()
if "ref_i" not in st.session_state:
    st.session_state.ref_i = int(VALID[-1])

# Platzhalter: erscheint OBEN in der Anzeige, wird aber erst unten befuellt.
# Streamlit verbietet es, den Zustand eines bereits erzeugten Widgets zu aendern -
# deshalb muessen die Szenario-Buttons im Code VOR dem Datumsfeld stehen.
date_slot = st.sidebar.container()

st.sidebar.markdown("**Scenarios**")
SCEN = [("\U0001F4C9\u2002Latest available", None, "Current observed situation"),
        ("\u26A0\uFE0F\u2002Before record low", "2026-08-10", "Stress-test scenario"),
        ("\u2705\u2002Normal conditions", "2024-03-01", "Historical average")]
for label, date_str, note in SCEN:
    if st.sidebar.button(label, key=f"sc_{label}", use_container_width=True):
        if date_str is None:
            target = int(VALID[-1])
        else:
            c = DF.index[DF["date"] == pd.Timestamp(date_str)]
            target = int(c[0]) if len(c) and c[0] in VALID else st.session_state.ref_i
        st.session_state.ref_i = target
        st.session_state.ref_date = DF.loc[target, "date"].date()
        st.rerun()
    st.sidebar.caption(note)

if "ref_date" not in st.session_state:
    st.session_state.ref_date = DF.loc[st.session_state.ref_i, "date"].date()

with date_slot:
    st.markdown("**Reference date**")
    sel = st.date_input(
        "Reference date", key="ref_date",
        min_value=DF.loc[VALID[0], "date"].date(),
        max_value=DF.loc[VALID[-1], "date"].date(), label_visibility="collapsed")
hit = DF.index[DF["date"].dt.date == sel]
if len(hit) and hit[0] in VALID:
    st.session_state.ref_i = int(hit[0])

THRESHOLD = st.session_state.get("threshold", 77.0)   # GlW Kaub, ab 01.01.2023
A = st.session_state.get("assumptions", Assumptions(glw_cm=THRESHOLD))
st.sidebar.divider()
_age = (pd.Timestamp.today().normalize() - DF["date"].max()).days
st.sidebar.markdown("**Data**")
st.sidebar.caption(f"Through {DF['date'].max():%d %b %Y} \u00b7 "
                   + ("up to date" if _age <= 1 else f"{_age} days old"))
if st.sidebar.button("\U0001F504\u2002Refresh live data", use_container_width=True):
    from modeling import fetch_live
    with st.spinner("Fetching PEGELONLINE and Open-Meteo\u2026"):
        try:
            _new = fetch_live.update()
            st.cache_data.clear()
            st.success(f"Updated through {_new['data_end']}.")
            st.rerun()
        except fetch_live.LiveDataError as e:
            st.sidebar.error(f"Refresh failed: {e}")
st.sidebar.caption("The newest complete day is always yesterday \u2014 a partial "
                   "day would distort the daily mean and rainfall total.")

st.sidebar.divider()
st.sidebar.caption("Smarter forecasts.\nSmoother supply chains.")


# ================================================================= compute
i = st.session_state.ref_i
REF = DF.loc[i, "date"]
TODAY = float(DF.loc[i, LEVEL])
F = core.make_forecast(DF, XALL, META, MODS, RESID, i, THRESHOLD)
RISK = core.classify_risk(F, THRESHOLD)
NORMAL_LVL = core.normal_level(DF, LEVEL, REF, META["train_end"])
CV = core.cargo_view(F, A, NORMAL_LVL)
SC = STATUS_COLOR[RISK.level]


def data_age():
    """Alter des neuesten vollstaendigen Tages im Datensatz, in Tagen."""
    return (pd.Timestamp.today().normalize() - DF["date"].max()).days


def header():
    age = data_age()
    if age <= 1:
        fresh, col = "up to date", OK
    elif age == 2:
        fresh, col = f"{age} days old", WARN
    else:
        fresh, col = f"{age} days old", CRIT
    st.markdown(f"""
    <div class="hl-head">
      <div><h2>HydroLoad-AI</h2><p>Rhine Cargo Capacity Advisor</p></div>
      <div class="hl-meta">
        <b>Kaub, Middle Rhine</b><br>{core.KAUB_LAT:.3f}&deg; N, {core.KAUB_LON:.3f}&deg; E<br>
        Data through {DF["date"].max():%d %b %Y}
        &nbsp;<span style="background:{col};color:#0e1218;border-radius:.25rem;
              padding:.06rem .38rem;font-size:.71rem;font-weight:700;">{fresh}</span>
      </div>
    </div>""", unsafe_allow_html=True)
    if age > 2:
        st.warning(f"The dataset ends on {DF['date'].max():%d %b %Y}, {age} days ago. "
                   "Open **Settings** and refresh from the live sources before using "
                   "this forecast operationally.")


def sparkline(vals, w=118, h=26, colour=WATER):
    """Inline SVG trend line for the KPI card."""
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pts = " ".join(f"{k * w / (len(vals) - 1):.1f},{h - (v - lo) / rng * (h - 5) - 2.5:.1f}"
                   for k, v in enumerate(vals))
    return (f'<svg width="{w}" height="{h}" style="display:block;">'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="1.9" stroke-linejoin="round"/></svg>')


def kpi(col, icon, label, value, sub, color=None, accent=WATER,
        primary=False, spark=""):
    col.markdown(f"""
    <div class="kpi {'pri' if primary else 'sec'}"
         style="border-left:{'5px' if primary else '3px'} solid {accent};">
      <div class="top"><span class="ico">{icon}</span><span class="lab">{label}</span></div>
      <div class="val" style="color:{color or TEXT};">{value}</div>
      <div class="sub">{sub}</div>
      <div class="spark">{spark}</div>
    </div>""", unsafe_allow_html=True)


def bar(label, value, maxv, colour, unit="t"):
    pct = 0 if maxv <= 0 else max(0.0, min(100.0, 100 * value / maxv))
    st.markdown(f"""
      <div class="bar-lab"><span>{label}</span><b>{value:,.0f} {unit}</b></div>
      <div class="bar-bg"><div class="bar-fill"
           style="width:{pct:.1f}%;background:{colour};"></div></div>""",
                unsafe_allow_html=True)


def forecast_chart(height=330, days=8):
    hist = (DF.loc[max(0, i - days):i, ["date", LEVEL]]
              .rename(columns={LEVEL: "level"}).assign(series="Historical"))
    p = F.assign(series="Forecast", band="80 % uncertainty range")
    # Die kritische Linie muss IMMER sichtbar sein - sie ist der Bezugspunkt
    # der gesamten Entscheidung, auch wenn sie ueber allen Messwerten liegt.
    lo_all = min(hist.level.min(), F.lo.min(), THRESHOLD)
    hi_all = max(hist.level.max(), F.hi.max(), THRESHOLD)
    pad = 0.10 * max(1.0, hi_all - lo_all)
    ys = alt.Scale(domain=[lo_all - pad, hi_all + pad], nice=False)
    xa = alt.Axis(format="%d %b", labelAngle=-40, tickCount="day", grid=False,
                  labelColor=MUTED, titleColor=MUTED, domainColor=BORDER,
                  tickColor=BORDER)
    cs = alt.Scale(domain=["Historical", "Forecast", "80 % uncertainty range",
                           "Critical level"],
                   range=[HIST, WATER, WATER, CRIT])
    lg = alt.Legend(title=None, orient="top", direction="horizontal", symbolSize=90)

    band = alt.Chart(p).mark_area(opacity=0.28).encode(
        x=alt.X("date:T", title=None, axis=xa),
        y=alt.Y("lo:Q", title="Gauge level (cm)", scale=ys,
                axis=alt.Axis(labelColor=MUTED, titleColor=MUTED,
                              gridColor=GRID, domainColor=BORDER,
                              tickColor=BORDER, titlePadding=12,
                              labelPadding=4)), y2="hi:Q",
        color=alt.Color("band:N", scale=cs, legend=lg))
    obs = alt.Chart(hist).mark_line(strokeWidth=2.3).encode(
        x=alt.X("date:T", axis=xa), y=alt.Y("level:Q", scale=ys),
        color=alt.Color("series:N", scale=cs, legend=lg),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                 alt.Tooltip("level:Q", title="Observed (cm)", format=".0f")])
    fc = alt.Chart(p).mark_line(point=True, strokeDash=[6, 3], strokeWidth=2.6).encode(
        x=alt.X("date:T", axis=xa), y=alt.Y("mid:Q", scale=ys),
        color=alt.Color("series:N", scale=cs, legend=lg),
        tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                 alt.Tooltip("mid:Q", title="Forecast (cm)", format=".0f"),
                 alt.Tooltip("lo:Q", title="80 % low (cm)", format=".0f"),
                 alt.Tooltip("hi:Q", title="80 % high (cm)", format=".0f"),
                 alt.Tooltip("gap:Q", title="Difference to critical (cm)", format="+.0f"),
                 alt.Tooltip("p_below:Q", title="P(below critical)", format=".0%"),
                 alt.Tooltip("status:N", title="Status")])
    crit = alt.Chart(pd.DataFrame({"y": [THRESHOLD], "series": ["Critical level"]})
                     ).mark_rule(strokeDash=[6, 4], strokeWidth=2.6).encode(
        y=alt.Y("y:Q", scale=ys), color=alt.Color("series:N", scale=cs, legend=lg))
    clab = alt.Chart(pd.DataFrame(
        {"y": [THRESHOLD], "t": [f"Critical level · {THRESHOLD:.0f} cm"]})
    ).mark_text(align="left", baseline="bottom", dx=4, dy=-3, color=CRIT,
                fontSize=10.5).encode(x=alt.value(5), y=alt.Y("y:Q", scale=ys), text="t:N")
    now = alt.Chart(pd.DataFrame({"x": [REF]})).mark_rule(
        color=MUTED, strokeDash=[2, 3], opacity=0.8).encode(x="x:T")
    nlab = alt.Chart(pd.DataFrame({"x": [REF], "y": [ys.domain[1]],
                                   "t": [f"Today · {TODAY:.0f} cm"]})).mark_text(
        align="left", baseline="top", dx=5, dy=2, color=MUTED, fontSize=10).encode(
        x="x:T", y=alt.Y("y:Q", scale=ys), text="t:N")

    layers = [band, obs, fc, crit, clab, now, nlab]
    cross = F[F.status == "Critical"]
    if len(cross):
        # nur ein dezenter Marker - der Tag steht als Text in der KPI-Karte
        layers.append(alt.Chart(cross.iloc[[0]]).mark_point(
            size=130, shape="diamond", filled=True, color=CRIT,
            stroke=PANEL, strokeWidth=1.2).encode(
            x="date:T", y=alt.Y("mid:Q", scale=ys),
            tooltip=[alt.Tooltip("date:T", title="Crosses critical", format="%d %b"),
                     alt.Tooltip("mid:Q", title="Forecast (cm)", format=".0f")]))
    if RISK.breach_days:
        br = F[F.status == "Critical"]
        layers.insert(0, alt.Chart(pd.DataFrame(
            {"x": [br.date.min() - pd.Timedelta(hours=12)],
             "x2": [br.date.max() + pd.Timedelta(hours=12)]})).mark_rect(
            opacity=0.07, color=CRIT).encode(x="x:T", x2="x2:T"))
    st.altair_chart(alt.layer(*layers).properties(
                        height=height,
                        padding={"left": 14, "right": 10, "top": 8, "bottom": 6})
                    .configure_view(strokeOpacity=0, fill=PANEL)
                    .configure_legend(labelColor=MUTED, titleColor=MUTED)
                    .configure(background=PANEL), use_container_width=True)


# ================================================================= Overview
if PAGE == "Overview":
    header()

    # ---------- executive summary ----------
    st.markdown(f'<div class="exec"><b>Executive summary</b>'
                f'{core.executive_summary(F, RISK, CV, THRESHOLD)}</div>',
                unsafe_allow_html=True)
    st.write("")

    # ---------- KPI row: chain order, decision-critical cards emphasised ----------
    c = st.columns([1, 1, 1.12, 1.12, 1.12], gap="small")
    spark = core.sparkline_values(DF, LEVEL, i, 10)
    # Der Pegel hat einen willkuerlichen Nullpunkt - eine Prozentangabe waere
    # irrefuehrend. Deshalb die Differenz in Zentimetern.
    dev_cm = TODAY - NORMAL_LVL
    trend_word = ("falling" if spark[-1] < spark[0] - 1 else
                  "rising" if spark[-1] > spark[0] + 1 else "stable")
    trend_ico = "&#8600;" if trend_word == "falling" else (
        "&#8599;" if trend_word == "rising" else "&#8594;")

    kpi(c[0], "&#127754;", "Current water level", f"{TODAY:.0f} cm",
        f"<b style='color:{CRIT if dev_cm < 0 else OK}'>{dev_cm:+.0f} cm</b> "
        f"vs. {NORMAL_LVL:.0f} cm seasonal normal<br>"
        f"{trend_ico} {trend_word} over 10 days<br>"
        f"<span style='opacity:.8;'>reading against gauge datum, not water depth</span>",
        spark=sparkline(spark, colour=CRIT if trend_word == "falling" else WATER))

    # Wert und Spanne beziehen sich beide auf Tag +7. Der Zeitpunkt der
    # Unterschreitung steht als Zusatzzeile darunter, nicht als Hauptwert.
    d7 = F.iloc[-1]
    if RISK.first_breach_h:
        f_sub = (f"Critical level reached on day +{RISK.first_breach_h} "
                 f"({F.loc[F.h == RISK.first_breach_h, 'date'].iloc[0]:%d %b})")
        f_col = f_acc = CRIT
    elif RISK.watch_days:
        f_sub, f_col = "Critical level inside the range", WARN
        f_acc = WARN
    else:
        f_sub, f_col = "Stays above the critical level", OK
        f_acc = OK
    kpi(c[1], "&#128197;", "Forecast day +7", f"{d7['mid']:.0f} cm",
        f"80 % range {d7['lo']:.0f} to {d7['hi']:.0f} cm<br>{f_sub}",
        f_col, f_acc,
        spark=sparkline(F["mid"].tolist(), colour=f_col))

    d_pct = 100 * CV.draft_m / CV.normal_draft_m if CV.normal_draft_m else 100
    kpi(c[2], "&#128674;", "Recommended draft", f"{CV.draft_m:.2f} m",
        f"from {CV.fairway_m:.2f} m fairway depth<br>"
        f"<b style='color:{CRIT if d_pct < 95 else OK}'>{d_pct:.0f} % of normal</b> "
        f"({CV.normal_draft_m:.2f} m)",
        accent=CRIT if d_pct < 95 else OK, primary=True)

    kpi(c[3], "&#128230;", "Recommended cargo", f"{CV.forecast_t:,.0f} t",
        f"vs. {CV.normal_t:,.0f} t normal<br>"
        f"<b style='color:{CRIT if CV.reduction_t else OK}'>"
        f"&minus;{CV.reduction_t:,.0f} t ({CV.reduction_pct:.0f} %)</b>",
        color=CRIT if CV.reduction_pct > 25 else None,
        accent=CRIT if CV.reduction_t else OK, primary=True)

    light = {"HIGH": "&#128308;", "MODERATE": "&#128992;",
             "WATCH": "&#128993;", "NORMAL": "&#128994;"}[RISK.level]
    trend_txt = {"increasing": "risk rising over the horizon",
                 "easing": "risk easing towards day +7",
                 "stable": "risk steady across the horizon"}[RISK.trend]
    kpi(c[4], light, "Risk status", RISK.level.title(),
        f"{RISK.short}<br>{trend_txt}", SC, SC, primary=True)

    # ---------- recommended actions, directly under the KPIs ----------
    acts = "".join(f"<li>{a}</li>" for a in core.recommendations(RISK, CV, A))
    title = {"HIGH": "&#9888;&#65039; RECOMMENDED ACTIONS &middot; LOW-WATER WARNING",
             "MODERATE": "&#9888;&#65039; RECOMMENDED ACTIONS &middot; LOW-WATER WATCH",
             "WATCH": "&#9888;&#65039; RECOMMENDED ACTIONS &middot; ELEVATED UNCERTAINTY",
             "NORMAL": "&#9989; NORMAL CONDITIONS"}[RISK.level]
    st.markdown(f"""
    <div class="act" style="background:{SC}16;border-left:5px solid {SC};">
      <div class="hdr" style="color:{SC};">{title}</div>
      <ul>{acts}</ul>
    </div>""", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([2.05, 1], gap="small")

    with left:
        with st.container(border=True):
            st.markdown('<div class="ptitle">&#128200; Water level forecast '
                        '&middot; Kaub</div>', unsafe_allow_html=True)
            forecast_chart(height=352)

    with right:
        with st.container(border=True):
            st.markdown('<div class="ptitle">&#128230; Cargo capacity</div>',
                        unsafe_allow_html=True)
            bar("Normal capacity", CV.normal_t, CV.normal_t, NAVY)
            bar("Available capacity", CV.forecast_t, CV.normal_t,
                OK if not CV.reduction_t else WATER)
            st.markdown(
                f'<div class="loss"><span><b>Capacity loss</b></span>'
                f'<span style="color:{CRIT};font-weight:700;">'
                f'{CV.reduction_t:,.0f} t &nbsp;(&minus;{CV.reduction_pct:.0f} %)</span>'
                f'</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="ptitle">&#128666; Logistics impact</div>',
                        unsafe_allow_html=True)
            m = st.columns(2)
            m[0].metric("Shortfall per daily sailing",
                        f"{CV.stranded_7d_t / len(F):,.0f} t")
            m[0].caption(f"average over the 7 forecast days &middot; "
                         f"{CV.stranded_7d_t:,.0f} t in total if one vessel "
                         "sails every day")
            m[1].metric("Replacement capacity", f"{CV.trucks} trucks")
            vessel_note = (f"or {CV.trains} train(s) &middot; "
                           f"+{CV.extra_vessels} vessel trips"
                           if CV.extra_vessels <= 6 else
                           f"or {CV.trains} train(s) &middot; waterway no longer "
                           "a practical option at this level")
            m[1].caption(vessel_note)

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="ptitle">&#128161; Key insights &amp; situation'
                    '</div>', unsafe_allow_html=True)
        ic, ch, mc = st.columns([1.5, 1.25, 1.25], gap="medium")

        # -- interpretierte Aussagen --
        ins = "".join(f'<div class="ins">&#10003; {t}</div>'
                      for t in core.key_insights(F, RISK, CV, THRESHOLD))
        ic.markdown(f"<div style='font-size:.76rem;color:{MUTED};text-transform:uppercase;"
                    "letter-spacing:.05em;font-weight:600;margin-bottom:.45rem;'>"
                    "What the forecast means</div>" + ins, unsafe_allow_html=True)

        # -- die Kette Wasserstand -> Tiefgang -> Ladung, als Kette lesbar --
        chain = [("Gauge level", f"{F.iloc[-1]['lo']:.0f} cm",
                  "day +7, lower bound &middot; relative to gauge datum"),
                 ("Fairway depth", f"{CV.fairway_m:.2f} m",
                  f"derived from the {THRESHOLD:.0f} cm reference level"),
                 ("Safe draft", f"{CV.draft_m:.2f} m",
                  f"minus {A.under_keel_clearance_m:.2f} m under-keel clearance"),
                 ("Payload", f"{CV.forecast_t:,.0f} t",
                  f"&minus;{CV.reduction_pct:.0f} % vs. full load"),
                 ("Replacement", f"{CV.trucks} trucks",
                  f"or {CV.trains} train(s)")]
        rows = ""
        for k, (lab, val, note) in enumerate(chain):
            rows += (f"<div class='chain-row'>"
                     f"<div class='chain-lab'>{lab}</div>"
                     f"<div class='chain-val'>{val}</div>"
                     f"<div class='chain-note'>{note}</div></div>")
            if k < len(chain) - 1:
                rows += "<div class='chain-arrow'>&#8595;</div>"
        ch.markdown(f"<div style='font-size:.76rem;color:{MUTED};text-transform:uppercase;"
                    "letter-spacing:.05em;font-weight:600;margin-bottom:.45rem;'>"
                    "From level to action</div>" + rows, unsafe_allow_html=True)

        # -- farbige Karte --
        mc.markdown(f"<div style='font-size:.76rem;color:{MUTED};text-transform:uppercase;"
                    "letter-spacing:.05em;font-weight:600;margin-bottom:.45rem;'>"
                    "Location &middot; Kaub, Middle Rhine</div>", unsafe_allow_html=True)
        pts = pd.DataFrame({"lat": [core.KAUB_LAT], "lon": [core.KAUB_LON],
                            "name": ["Kaub gauge"]})
        try:
            with mc:
                st.pydeck_chart(pdk.Deck(
                    map_style="dark",
                    initial_view_state=pdk.ViewState(
                        latitude=core.KAUB_LAT, longitude=core.KAUB_LON,
                        zoom=7.4, pitch=0),
                    layers=[
                        pdk.Layer("ScatterplotLayer", data=pts,
                                  get_position="[lon, lat]",
                                  get_fill_color=[224, 82, 82, 210],
                                  get_radius=2600, pickable=True),
                        pdk.Layer("ScatterplotLayer", data=pts,
                                  get_position="[lon, lat]",
                                  get_fill_color=[224, 82, 82, 55],
                                  get_radius=9000),
                    ],
                    tooltip={"text": "{name}"}), height=215)
        except Exception:
            mc.map(pts, zoom=7, size=600, height=215)

        st.caption("All figures use the lower bound of the 80 % interval and the "
                   "cargo assumptions in Settings. Cargo at risk is the shortfall "
                   "against a full load on day +7; stranded tonnes sum the daily "
                   "shortfall across the horizon.")


# ================================================================= Forecast
elif PAGE == "Forecast":
    header()
    st.write("")
    with st.container(border=True):
        st.markdown('<div class="ptitle">&#128200; Water level forecast '
                    '&middot; Kaub</div>', unsafe_allow_html=True)
        forecast_chart(height=400, days=18)

    st.write("")
    st.markdown("#### Forecast detail")
    det = pd.DataFrame({
        "Date": F.date.dt.strftime("%d %b"),
        "Day": "+" + F.h.astype(str),
        "Forecast": F.mid.round(0).astype(int).astype(str) + " cm",
        "80 % range": (F.lo.round(0).astype(int).astype(str) + "–"
                       + F.hi.round(0).astype(int).astype(str) + " cm"),
        "Difference to critical": F.gap.round(0).astype(int).map("{:+d}".format) + " cm",
        "P(below critical)": (100 * F.p_below).round(0).astype(int).astype(str) + " %",
        "Status": F.status,
    })
    st.dataframe(det, use_container_width=True, hide_index=True)

    if F.actual.notna().any() and st.checkbox("Compare with what actually happened"):
        cmp = F.dropna(subset=["actual"]).copy()
        cmp["inside"] = (cmp.actual >= cmp.lo) & (cmp.actual <= cmp.hi)
        st.dataframe(pd.DataFrame({
            "Date": cmp.date.dt.strftime("%d %b"),
            "Forecast": cmp.mid.round(0).astype(int).astype(str) + " cm",
            "Actual": cmp.actual.round(0).astype(int).astype(str) + " cm",
            "Error": (cmp.mid - cmp.actual).round(0).astype(int).map("{:+d}".format) + " cm",
            "Inside 80 % range": cmp.inside.map({True: "yes", False: "no"}),
        }), use_container_width=True, hide_index=True)
        st.caption(f"{int(cmp.inside.sum())} of {len(cmp)} actual values fell inside "
                   "the calibrated interval.")


# ================================================================= Cargo Planning
elif PAGE == "Cargo Planning":
    header()
    st.write("")
    st.markdown("#### Loading plan by forecast day")
    from modeling.cargo import shortfall_range
    rows = []
    for r in F.itertuples():
        g = shortfall_range(r.lo, r.mid, r.hi, A)
        rows.append({"Date": r.date.strftime("%d %b"), "Day": f"+{r.h}",
                     "Level range": f"{r.lo:.0f}–{r.hi:.0f} cm",
                     "Safe draft": f"{g['safe_draft_m'][0]:.2f}–{g['safe_draft_m'][2]:.2f} m",
                     "Payload": f"{g['payload_t'][0]:,.0f}–{g['payload_t'][2]:,.0f} t",
                     "Shortfall": f"{g['deficit_t'][1]:,.0f} t (median)",
                     "Status": r.status})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info(f"**Loading recommendation for day +7: {CV.forecast_t:,.0f} t** at "
            f"{CV.draft_m:.2f} m draft. Derived from the lower bound of the interval, "
            "not the median: overloading risks grounding, underloading only costs "
            "revenue, so the two errors are not equivalent.")

    st.markdown("#### Sensitivity of the cargo assumptions")
    sens = []
    for f in (0.8, 0.9, 1.0, 1.1, 1.2):
        alt_A = Assumptions(glw_cm=THRESHOLD, max_payload_t=A.max_payload_t * f,
                            truck_payload_t=A.truck_payload_t,
                            train_payload_t=A.train_payload_t)
        v = core.cargo_view(F, alt_A, NORMAL_LVL)
        sens.append({"Max payload assumption": f"{alt_A.max_payload_t:,.0f} t "
                                               f"({f:+.0%} of current)".replace("+0%", "base"),
                     "Recommended cargo": f"{v.forecast_t:,.0f} t",
                     "Shortfall": f"{v.reduction_t:,.0f} t",
                     "Trucks": v.trucks, "Trains": v.trains})
    st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)
    st.caption("The cargo layer cannot be validated against ground truth, so its "
               "sensitivity to the assumptions is shown instead.")


# ================================================================= Transport Options
elif PAGE == "Transport Options":
    header()
    st.write("")
    if CV.reduction_t <= 0:
        st.success("No alternative capacity required - full loading is possible "
                   "across the forecast horizon.")
    else:
        st.markdown("#### Replacement capacity required")
        t = st.columns(3)
        t[0].metric("Road", f"{CV.trucks} trucks",
                    f"at {A.truck_payload_t:,.0f} t each", delta_color="off")
        t[1].metric("Rail", f"{CV.trains} train(s)",
                    f"at {A.train_payload_t:,.0f} t each", delta_color="off")
        if CV.extra_vessels <= 6:
            t[2].metric("Additional vessel trips", f"+{CV.extra_vessels}",
                        f"at {CV.forecast_t:,.0f} t each", delta_color="off")
        else:
            t[2].metric("Additional vessel trips", "not viable",
                        f"{CV.extra_vessels} trips at {CV.forecast_t:,.0f} t each",
                        delta_color="off")
            t[2].caption("At this loading limit the waterway cannot absorb the "
                         "shortfall; it has to move by road or rail.")

        st.markdown("#### Requirement by forecast day")
        from modeling.cargo import shortfall_range
        rows = []
        for r in F.itertuples():
            g = shortfall_range(r.lo, r.mid, r.hi, A)
            rows.append({"Date": r.date.strftime("%d %b"), "Day": f"+{r.h}",
                         "Shortfall range": f"{g['deficit_t'][2]:,.0f}–{g['deficit_t'][0]:,.0f} t",
                         "Trucks": f"{g['trucks'][2]}–{g['trucks'][0]}",
                         "Trains": f"{g['trains'][2]}–{g['trains'][0]}",
                         "Status": r.status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Ranges run from the best case to the worst case: a lower water "
                   "level means a larger replacement requirement.")
    st.warning("Cost and surcharge modelling is not implemented. Low-water surcharge "
               "brackets are not available in the project data, so no cost figures "
               "are shown rather than estimated ones.")


# ================================================================= Validation
elif PAGE == "Validation":
    header()
    st.write("")
    st.markdown("#### Retrospective validation")
    st.caption("The forecast is re-issued on every origin date in the window and "
               "compared with what actually happened. This is the only part of the "
               "product that can be checked against ground truth.")

    c = st.columns([1, 1, 1])
    v_start = c[0].date_input("From", value=pd.Timestamp("2023-09-01").date(),
                              min_value=DF.date.min().date(),
                              max_value=DF.date.max().date(), key="v_start")
    v_end = c[1].date_input("To", value=min(pd.Timestamp("2026-08-25"),
                                            DF.date.max()).date(),
                            min_value=DF.date.min().date(),
                            max_value=DF.date.max().date(), key="v_end")
    step = c[2].selectbox("Origin dates", [1, 3, 7], index=1,
                          format_func=lambda k: f"every {k} day(s)")

    if st.button("Run validation", type="primary"):
        st.session_state.bt_args = (str(v_start), str(v_end), int(step))

    if "bt_args" in st.session_state:
        a0, a1, a2 = st.session_state.bt_args
        with st.spinner("Re-issuing forecasts…"):
            BT = core.backtest(DF, XALL, META, MODS, RESID, THRESHOLD, a0, a1, a2)

        if BT.empty:
            st.warning("No complete origin dates in this window.")
        else:
            summ = core.backtest_summary(BT)
            k = st.columns(4)
            k[0].metric("Forecasts evaluated", f"{len(BT):,}")
            k[1].metric("Origin dates", f"{BT.origin.nunique():,}")
            k[2].metric("Mean coverage",
                        f"{100 * BT['inside'].mean():.0f} %",
                        f"target {100 * META['target_coverage']:.0f} %",
                        delta_color="off")
            k[3].metric("Skill at day +7",
                        f"{100 * summ.loc[summ.Horizon == 7, 'Skill vs baseline'].iloc[0]:.0f} %",
                        "vs. persistence", delta_color="off")

            st.markdown("##### Accuracy by horizon")
            show = summ.copy()
            show["Coverage"] = (100 * show["Coverage"]).round(0).astype(int).astype(str) + " %"
            show["Skill vs baseline"] = ((100 * show["Skill vs baseline"]).round(0)
                                         .astype(int).map("{:+d} %".format))
            show[["MAE (cm)", "Baseline MAE (cm)"]] = show[["MAE (cm)",
                                                            "Baseline MAE (cm)"]].round(1)
            st.dataframe(show, use_container_width=True, hide_index=True)

            long = summ.melt(id_vars="Horizon",
                             value_vars=["MAE (cm)", "Baseline MAE (cm)"],
                             var_name="series", value_name="mae")
            ch = alt.Chart(long).mark_line(point=True, strokeWidth=2.4).encode(
                x=alt.X("Horizon:O", title="Forecast horizon (days)",
                        axis=alt.Axis(labelColor=MUTED, titleColor=MUTED,
                                      domainColor=BORDER, tickColor=BORDER)),
                y=alt.Y("mae:Q", title="MAE (cm)",
                        axis=alt.Axis(labelColor=MUTED, titleColor=MUTED,
                                      gridColor=GRID, domainColor=BORDER,
                                      tickColor=BORDER)),
                color=alt.Color("series:N", scale=alt.Scale(
                    domain=["MAE (cm)", "Baseline MAE (cm)"], range=[WATER, MUTED]),
                    legend=alt.Legend(title=None, orient="top")),
                tooltip=["Horizon", "series", alt.Tooltip("mae:Q", format=".1f")])
            st.altair_chart(ch.properties(height=250)
                            .configure_view(strokeOpacity=0, fill=PANEL)
                            .configure_legend(labelColor=MUTED)
                            .configure(background=PANEL), use_container_width=True)

            st.markdown("##### Did the warning fire when it mattered?")
            st.caption(f"A warning is counted when the median forecast falls below "
                       f"{THRESHOLD:.0f} cm. Evaluated at day +7, the longest and "
                       "hardest horizon.")
            wq = core.warning_quality(BT[BT.h == 7])
            w = st.columns(5)
            w[0].metric("Correct warnings", wq["hits"])
            w[1].metric("False alarms", wq["false_alarms"])
            w[2].metric("Missed events", wq["missed"])
            w[3].metric("Recall", f"{100 * wq['recall']:.0f} %")
            w[3].caption("share of low-water days warned about")
            w[4].metric("Precision", f"{100 * wq['precision']:.0f} %")
            w[4].caption("share of warnings that were correct")

            st.markdown("##### Case study")
            cases = sorted(BT.origin.unique())
            pick = st.select_slider(
                "Origin date", options=cases,
                value=min(cases, key=lambda d: abs(
                    pd.Timestamp(d) - pd.Timestamp("2026-08-10"))),
                format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"))
            case = BT[BT.origin == pick].sort_values("h")
            cc = st.columns([2, 1])
            band = alt.Chart(case).mark_area(opacity=0.28, color=WATER).encode(
                x=alt.X("target_date:T", title=None,
                        axis=alt.Axis(format="%d %b", labelAngle=-40,
                                      labelColor=MUTED, domainColor=BORDER,
                                      tickColor=BORDER)),
                y=alt.Y("lo:Q", title="Gauge level (cm)",
                        axis=alt.Axis(labelColor=MUTED, titleColor=MUTED,
                                      gridColor=GRID, domainColor=BORDER,
                                      tickColor=BORDER)), y2="hi:Q")
            fline = alt.Chart(case).mark_line(point=True, strokeDash=[6, 3],
                                              strokeWidth=2.4, color=WATER).encode(
                x="target_date:T", y="mid:Q")
            aline = alt.Chart(case).mark_line(point=True, strokeWidth=2.4,
                                              color=CRIT).encode(
                x="target_date:T", y="actual:Q",
                tooltip=[alt.Tooltip("target_date:T", title="Date", format="%d %b"),
                         alt.Tooltip("actual:Q", title="Actual (cm)", format=".0f"),
                         alt.Tooltip("mid:Q", title="Forecast (cm)", format=".0f")])
            rule = alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(
                color=CRIT, strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
            with cc[0]:
                st.altair_chart((band + fline + aline + rule).properties(height=260)
                                .configure_view(strokeOpacity=0, fill=PANEL)
                                .configure(background=PANEL),
                                use_container_width=True)
                st.caption("Blue dashed: median forecast with 80 % band. "
                           "Red: what actually happened. Dashed red line: critical level.")
            with cc[1]:
                st.metric("Inside the 80 % band",
                          f"{int(case['inside'].sum())} of {len(case)} days")
                st.metric("MAE over the seven days",
                          f"{case['abs_error'].mean():.1f} cm")
                st.metric("Baseline MAE",
                          f"{case['baseline_abs'].mean():.1f} cm")
                st.dataframe(pd.DataFrame({
                    "Day": "+" + case.h.astype(str),
                    "Forecast": case.mid.round(0).astype(int).astype(str) + " cm",
                    "Actual": case.actual.round(0).astype(int).astype(str) + " cm",
                    "In band": case.inside.map({True: "yes", False: "no"}),
                }), use_container_width=True, hide_index=True, height=230)
    else:
        st.info("Choose a window and press **Run validation**.")


# ================================================================= Reports
elif PAGE == "Reports":
    header()
    st.write("")
    st.markdown("#### Low-water exposure, 2000 to today")
    hist = DF[["date", LEVEL]].copy()
    hist["year"] = hist.date.dt.year
    rep = (hist.groupby("year")[LEVEL]
           .agg(days_below_critical=lambda s: int((s < THRESHOLD).sum()),
                annual_minimum="min", annual_median="median")
           .round(0).astype(int).reset_index())
    ch = alt.Chart(rep).mark_bar(color=WATER).encode(
        x=alt.X("year:O", title=None,
                axis=alt.Axis(labelAngle=-45, labelColor=MUTED, domainColor=BORDER,
                              tickColor=BORDER)),
        y=alt.Y("days_below_critical:Q", title=f"Days below {THRESHOLD:.0f} cm",
                axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID,
                              domainColor=BORDER, tickColor=BORDER)),
        tooltip=["year", "days_below_critical", "annual_minimum", "annual_median"])
    st.altair_chart(ch.properties(height=250)
                    .configure_view(strokeOpacity=0, fill=PANEL)
                    .configure(background=PANEL), use_container_width=True)
    st.dataframe(rep.rename(columns={
        "year": "Year", "days_below_critical": f"Days below {THRESHOLD:.0f} cm",
        "annual_minimum": "Annual minimum (cm)", "annual_median": "Annual median (cm)"}),
        use_container_width=True, hide_index=True, height=240)

    st.markdown("#### Model performance")
    st.dataframe(pd.read_csv(core.MODELS / "forecast_report.csv").round(2),
                 use_container_width=True, hide_index=True)
    st.caption("skill_pct compares the model against the persistence baseline. "
               "coverage_calibrated is the measured share of actual values inside "
               "the 80 % interval; the target is 0.80. MAE_lowwater is the error "
               "below 100 cm, the regime this product exists for.")


# ================================================================= Settings
else:
    header()
    st.write("")
    st.markdown("#### Operating thresholds")
    thr = st.number_input(
        "Critical gauge level (cm)", value=float(THRESHOLD), step=1.0,
        help="Equivalent low-water level (GlW) for Kaub: 77 cm, valid from "
             "01.01.2023. Source: PEGELONLINE station master data.")

    st.markdown("#### Cargo assumptions")
    st.caption("Unverified — pending WSV / ELWIS and vessel loading tables. "
               "Each value is marked in modeling/cargo.py.")
    d = Assumptions()
    s = st.columns(3)
    mp = s[0].number_input("Max payload (t)", value=float(A.max_payload_t), step=50.0)
    tp = s[1].number_input("Truck payload (t)", value=float(A.truck_payload_t), step=1.0)
    rp = s[2].number_input("Train payload (t)", value=float(A.train_payload_t), step=50.0)

    if st.button("Apply", type="primary"):
        st.session_state.threshold = thr
        st.session_state.assumptions = Assumptions(
            glw_cm=thr, max_payload_t=mp, truck_payload_t=tp, train_payload_t=rp)
        st.rerun()

    st.markdown("#### Live data")
    from modeling import fetch_live
    stat = fetch_live.read_status()
    src = core.active_data()
    d1, d2 = st.columns([1, 2])
    with d1:
        if st.button("Refresh from live sources", type="primary",
                     use_container_width=True):
            with st.spinner("Fetching PEGELONLINE and Open-Meteo…"):
                try:
                    new_stat = fetch_live.update()
                    st.cache_data.clear()
                    st.success(f"Updated to {new_stat['data_end']} "
                               f"({new_stat['live_days_added']} new day(s)).")
                    st.rerun()
                except fetch_live.LiveDataError as e:
                    st.error(f"Live update failed: {e}")
    with d2:
        st.caption(f"Active dataset: `{src.name}` · "
                   f"{DF.date.min():%d %b %Y} – {DF.date.max():%d %b %Y} "
                   f"({len(DF):,} days)")
        if stat:
            st.caption(f"Last live refresh: {stat['updated_utc']} UTC · "
                       f"{stat['live_days_added']} day(s) beyond the stored history")
        else:
            st.caption("No live refresh yet — running on the stored history. "
                       "Sources: PEGELONLINE (gauge, discharge) and Open-Meteo "
                       "(weather). No API key required.")

    st.markdown("#### Model and data")
    st.json({"records": META["n_rows"], "data_end": META["data_end"],
             "train_end": META["train_end"], "calibration_end": META["calib_end"],
             "horizons": len(META["horizons"]), "features": len(META["features"]),
             "target_coverage": META["target_coverage"]})
    st.json(A.as_dict())
