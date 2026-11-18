"""Inpatient bed demand: a Streamlit dashboard over the bed model, on synthetic data.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # works without installing the package

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from bedmodel import ORIGINAL, RECOMMENDED, WARD_GROUPS, Method, fit, make_hospital, observe_year  # noqa: E402
from bedmodel import BASELINE, simulate, total, true_occupancy  # noqa: E402
from bedmodel.los import SpellPool  # noqa: E402
from bedmodel.metrics import beds_for, coverage  # noqa: E402
from bedmodel.model import DEFAULT_SCENARIOS, STANDARD_SCENARIOS, data_interval, discharge_targets  # noqa: E402

st.set_page_config(page_title="Bed demand model", page_icon="🛏️", layout="wide")

METHODOLOGY_FILE = ROOT / "docs" / "METHODOLOGY.md"
METHODOLOGY = (METHODOLOGY_FILE.read_text(encoding="utf-8") if METHODOLOGY_FILE.exists()
               else "The methodology is in docs/METHODOLOGY.md in the project repository.")
METHODS_FILE = ROOT / "results" / "methods.csv"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL, MUTED, GRID, BAND = "#52514e", "#898781", "#e1e0d9", "#f0efea"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
PRESETS = {"recommended": "Recommended", "original": "Original", "custom": "Custom"}
ALL = "All ward types"

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }}
[data-testid="stMetric"] {{ background: #ffffff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.75rem 1rem 0.8rem; }}
[data-testid="stMetricLabel"] p {{ color: {NEUTRAL}; font-size: 0.86rem; }}
.eyebrow {{ color: {NEUTRAL}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
            margin: 0.4rem 0 0.35rem; }}
.lede {{ color: #3a3936; font-size: 1.02rem; margin: -0.4rem 0 0.2rem; max-width: 62rem; }}
.badge {{ display: inline-block; font-size: 0.74rem; font-weight: 600; color: {NEUTRAL}; background: {BAND};
          border: 1px solid {GRID}; border-radius: 999px; padding: 0.08rem 0.55rem; margin-left: 0.45rem;
          vertical-align: middle; }}
.scenario-line {{ color: #3a3936; font-size: 0.95rem; margin: 0.2rem 0 0.9rem; }}
.kpi {{ background: #ffffff; border: 1px solid {GRID}; border-radius: 10px; padding: 0.8rem 1rem 0.85rem; height: 100%; }}
.kpi .label {{ color: {NEUTRAL}; font-size: 0.86rem; line-height: 1.3; min-height: 2.2rem; }}
.kpi .value {{ color: #1d1d1b; font-size: 2rem; font-weight: 600; line-height: 1.15; margin: 0.15rem 0 0.35rem;
               font-variant-numeric: tabular-nums; }}
.kpi .status {{ display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.82rem; color: #3a3936;
                background: {BAND}; border-radius: 999px; padding: 0.12rem 0.55rem 0.12rem 0.3rem; }}
.kpi .icon {{ display: inline-flex; width: 1.05rem; height: 1.05rem; border-radius: 50%; color: #fff;
              font-size: 0.72rem; font-weight: 700; align-items: center; justify-content: center; }}
.kpi .note {{ color: {MUTED}; font-size: 0.78rem; margin-top: 0.45rem; line-height: 1.35; }}
.callout {{ background: #ffffff; border: 1px solid {GRID}; border-left: 4px solid {BLUE}; border-radius: 8px;
            padding: 0.85rem 1.1rem; margin: 0.3rem 0 1rem; }}
.chart-note {{ color: {MUTED}; font-size: 0.8rem; margin: -0.3rem 0 0.4rem; }}
section[data-testid="stSidebar"][data-collapsed="false"] {{ min-width: 360px; }}
</style>
""", unsafe_allow_html=True)

DEFAULTS = {"hospital": 0, "preset": "recommended", "lag": 30, "core_pct": 85, "esc_pct": 95, "sims": 500,
            "keep_long": True, "tukey": False, "cap": False, "in_progress": True, "los_model": "pool",
            "seasonality": True, "surges": True, "long_warmup": True, "show_truth": True}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------- cached computation
@st.cache_resource(show_spinner=False, max_entries=16)
def get_extract(hospital: int, lag: int):
    return observe_year(make_hospital(hospital), lag_days=lag)


@st.cache_resource(show_spinner=False, max_entries=32)
def get_model(hospital: int, lag: int, method: tuple):
    return fit(get_extract(hospital, lag), Method(*method))


@st.cache_data(show_spinner=False, max_entries=48)
def get_occupancy(hospital: int, lag: int, method: tuple, sims: int, scenario: str = "Baseline",
                  years_ahead: int = 0, growth: float = 0.0, los_improvement: float = 0.0):
    sc = STANDARD_SCENARIOS.get(scenario, BASELINE)
    return simulate(get_model(hospital, lag, method), sc, n_sims=sims, seed=hospital,
                    years_ahead=years_ahead, growth=growth, los_improvement=los_improvement)


@st.cache_data(show_spinner=False, max_entries=16)
def get_truth(hospital: int, lag: int, sims: int):
    ex = get_extract(hospital, lag)
    return true_occupancy(make_hospital(hospital), ex.end + pd.Timedelta(days=1), n_sims=sims, seed=hospital + 500)


@st.cache_data(show_spinner=False, max_entries=16)
def get_interval(hospital: int, lag: int, method: tuple, core: float, esc: float):
    return data_interval(get_extract(hospital, lag), Method(*method), n_boot=30, n_sims=200, seed=hospital,
                         core=core, escalation=esc)


def method_tuple(m: Method) -> tuple:
    return tuple(asdict(m).values())


def current_method() -> Method:
    g = st.session_state
    if g["preset"] == "recommended":
        return RECOMMENDED
    if g["preset"] == "original":
        return ORIGINAL
    return Method("Custom", los=g["los_model"], max_los=None if g["keep_long"] else 60, tukey=g["tukey"],
                  cap=90 if g["cap"] else None, in_progress=g["in_progress"], seasonality=g["seasonality"],
                  surges=g["surges"], warmup=365 if g["long_warmup"] else 30)


def load_preset() -> None:
    m = {"recommended": RECOMMENDED, "original": ORIGINAL}.get(st.session_state["preset"])
    if m is None:
        return
    st.session_state.update(keep_long=m.max_los is None, tukey=m.tukey, cap=m.cap is not None,
                            in_progress=m.in_progress, los_model=m.los, seasonality=m.seasonality,
                            surges=m.surges, long_warmup=m.warmup >= 365)


def to_custom() -> None:
    st.session_state["preset"] = "custom"


# ---------------------------------------------------------------- method path (the long-stays tab)
STEPS = [
    ("Original method", ORIGINAL),
    ("+ keep stays over 60 days", replace(ORIGINAL, max_los=None)),
    ("+ no Tukey exclusion", replace(ORIGINAL, max_los=None, tukey=False)),
    ("+ no 90-day cap", replace(ORIGINAL, max_los=None, tukey=False, cap=None)),
    ("+ 365-day warm-up", replace(ORIGINAL, max_los=None, tukey=False, cap=None, warmup=365)),
    ("+ resample spells instead of fitting", replace(RECOMMENDED, in_progress=False, seasonality=False, surges=False)),
    ("+ use spells still in hospital (Kaplan-Meier)", replace(RECOMMENDED, seasonality=False, surges=False)),
    ("+ seasonality", replace(RECOMMENDED, surges=False)),
    ("+ surges (recommended)", RECOMMENDED),
]


@st.cache_data(show_spinner=False, max_entries=8)
def get_path(hospital: int, lag: int, sims: int, core: float) -> pd.DataFrame:
    truth = total(get_truth(hospital, lag, sims))
    rows = []
    for label, m in STEPS:
        tot = total(get_occupancy(hospital, lag, method_tuple(m), sims))
        b = beds_for(tot, core)
        rows.append({"Step": label, "Core beds": b, "Nights covered (truth)": coverage(truth, b)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- formatting
def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="chart-note">{text}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, text: str, status: tuple[bool, str] | None = None) -> str:
    badge = ""
    if status is not None:
        ok, words = status
        icon, colour = ("✓", GOOD) if ok else ("✕", CRITICAL)
        badge = (f'<span class="status"><span class="icon" style="background:{colour}">{icon}</span>{words}</span>')
    return (f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>{badge}'
            f'<div class="note">{text}</div></div>')


def short(name: str) -> str:
    return (name.replace("General surgery", "Gen surg").replace("Gastroenterology", "Gastro")
            .replace("Colorectal surgery", "Colorectal").replace("Trauma and orthopaedics", "T&O")
            .replace(" - emergency", " em").replace(" - elective", " el").replace(" - male", " M")
            .replace(" - female", " F").replace("Gynaecology", "Gynae"))


# ---------------------------------------------------------------- charts
def occupancy_chart(occ: np.ndarray, dates: pd.DatetimeIndex, core_beds: int, esc_beds: int,
                    truth: np.ndarray | None = None):
    d = pd.DataFrame({"date": dates, "median": np.median(occ, axis=0), "lo": np.percentile(occ, 5, axis=0),
                      "hi": np.percentile(occ, 95, axis=0)})
    lo = float(min(d["lo"].min(), core_beds) * 0.9)
    hi = float(max(d["hi"].max(), esc_beds) * 1.05)
    y = alt.Y("median:Q", title="Beds occupied at midnight", scale=alt.Scale(domain=[lo, hi], zero=False))
    band = alt.Chart(d).mark_area(color=BLUE, opacity=0.18).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b", labelAngle=0, values=[alt.DateTime(year=m.year, month=m.month, date=1)
                                                       for m in pd.date_range(dates[0], dates[-1], freq="MS")])),
        y=alt.Y("lo:Q", scale=alt.Scale(domain=[lo, hi], zero=False), title="Beds occupied at midnight"), y2="hi:Q")
    line = alt.Chart(d).mark_line(color=BLUE, strokeWidth=2).encode(
        x="date:T", y=y, tooltip=[alt.Tooltip("date:T", format="%a %d %b"), alt.Tooltip("median:Q", format=".0f"),
                                  alt.Tooltip("lo:Q", title="5th percentile", format=".0f"),
                                  alt.Tooltip("hi:Q", title="95th percentile", format=".0f")])
    rules = pd.DataFrame({"y": [core_beds, esc_beds], "label": [f"Core {core_beds}", f"Escalation {esc_beds}"],
                          "kind": ["core", "esc"]})
    rule = alt.Chart(rules).mark_rule(strokeDash=[5, 4], strokeWidth=1.5).encode(
        y=alt.Y("y:Q", scale=alt.Scale(domain=[lo, hi], zero=False)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["core", "esc"], range=[ORANGE, CRITICAL]), legend=None))
    text = alt.Chart(rules).mark_text(align="right", dy=-6, fontSize=11, color=NEUTRAL).encode(
        x=alt.datum(d["date"].max()), y=alt.Y("y:Q", scale=alt.Scale(domain=[lo, hi], zero=False)), text="label:N")
    layers = band + line + rule + text
    if truth is not None:
        t = pd.DataFrame({"date": dates, "truth": np.median(truth, axis=0)})
        layers += alt.Chart(t).mark_line(color=NEUTRAL, strokeDash=[2, 3], strokeWidth=1.3).encode(
            x="date:T", y=alt.Y("truth:Q", scale=alt.Scale(domain=[lo, hi], zero=False)))
    return layers.properties(height=300)


def los_chart(spells: pd.DataFrame, model_los, max_days: int = 30):
    fin = spells[spells["Finished"]]["Length of Stay"]
    obs = fin.clip(upper=max_days + 1).value_counts(normalize=True).reindex(range(1, max_days + 2), fill_value=0)
    d = pd.DataFrame({"days": obs.index, "share": 100 * obs.to_numpy(), "what": "Finished spells"})
    if isinstance(model_los, SpellPool):
        pm = pd.Series(model_los.weights, index=np.minimum(model_los.los, max_days + 1)).groupby(level=0).sum()
    else:
        draws = model_los.sample(200000, np.random.default_rng(0))
        pm = pd.Series(np.minimum(draws, max_days + 1)).value_counts(normalize=True)
    pm = pm.reindex(range(1, max_days + 2), fill_value=0)
    m = pd.DataFrame({"days": pm.index, "share": 100 * pm.to_numpy()})
    bars = alt.Chart(d).mark_bar(color=BLUE, opacity=0.55, size=11).encode(
        x=alt.X("days:Q", title=f"Length of stay (days; the last bar is over {max_days})",
                scale=alt.Scale(domain=[0.5, max_days + 1.5])),
        y=alt.Y("share:Q", title="Share of spells (%)"),
        tooltip=[alt.Tooltip("days:Q"), alt.Tooltip("share:Q", format=".1f")])
    pts = alt.Chart(m).mark_line(color=ORANGE, strokeWidth=2, point=alt.OverlayMarkDef(size=28, color=ORANGE)).encode(
        x="days:Q", y="share:Q", tooltip=[alt.Tooltip("days:Q"), alt.Tooltip("share:Q", title="model", format=".1f")])
    return (bars + pts).properties(height=260)


def path_chart(path: pd.DataFrame, true_core: int):
    d = path.copy()
    lo = min(d["Core beds"].min(), true_core) * 0.92
    hi = max(d["Core beds"].max(), true_core) * 1.04
    x = alt.X("Core beds:Q", scale=alt.Scale(domain=[lo, hi]), title=f"Core beds, all ward types (dashed: truth, {true_core})")
    y = alt.Y("Step:N", sort=list(d["Step"]), title=None, axis=alt.Axis(labelLimit=340))
    bars = alt.Chart(d).mark_bar(color=BLUE, size=22, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=y, x=x, x2=alt.datum(lo), tooltip=["Step", "Core beds", alt.Tooltip("Nights covered (truth):Q", format=".0%")])
    text = alt.Chart(d).mark_text(align="left", dx=5, color=NEUTRAL, fontSize=11).encode(y=y, x=x, text="Core beds:Q")
    rule = alt.Chart(pd.DataFrame({"Core beds": [true_core]})).mark_rule(
        color=NEUTRAL, strokeDash=[5, 4], strokeWidth=1.5).encode(x=x)
    return (bars + text + rule).properties(height=34 * len(d) + 20)


def bar_chart(d: pd.DataFrame, x: str, y: str, title: str, colour: str = BLUE):
    return alt.Chart(d).mark_bar(color=colour, size=18, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y(f"{y}:N", sort=list(d[y]), title=None, axis=alt.Axis(labelLimit=340)),
        x=alt.X(f"{x}:Q", title=title), tooltip=[y, alt.Tooltip(f"{x}:Q", format=".0f")]).properties(
        height=max(140, 30 * len(d)))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Data")
    st.slider("Synthetic hospital", 0, 19, key="hospital",
              help="Each number is a different invented hospital: the same fifteen ward types with different "
                   "admission rates, stays and shares of long stays.")
    st.slider("Extract taken this many days after the year ends", 0, 180, step=10, key="lag",
              help="Spells still in hospital on that date show their stay so far.")
    st.subheader("Method")
    st.radio("Method", list(PRESETS), format_func=PRESETS.get, key="preset", horizontal=True,
             on_change=load_preset, label_visibility="collapsed")
    with st.expander("Settings", expanded=st.session_state["preset"] == "custom"):
        st.toggle("Keep stays over 60 days", key="keep_long", on_change=to_custom,
                  help="The original extract kept spells of 1 to 60 days only, so longer stays dropped out of "
                       "the admissions as well as the stays.")
        st.toggle("Use spells still in hospital (Kaplan–Meier)", key="in_progress", on_change=to_custom)
        st.radio("Length of stay", ["pool", "fitted"], key="los_model", on_change=to_custom, horizontal=True,
                 format_func={"pool": "Resample spells", "fitted": "Fitted distribution"}.get)
        st.toggle("Exclude Tukey outliers from the stays", key="tukey", on_change=to_custom,
                  help="Stays above Q3 + 3 × IQR. Their admissions still count; they're given a typical stay instead.")
        st.toggle("Cap stays at 90 days", key="cap", on_change=to_custom)
        st.toggle("Seasonality", key="seasonality", on_change=to_custom)
        st.toggle("Emergency surges", key="surges", on_change=to_custom)
        st.toggle("365-day warm-up (otherwise 30)", key="long_warmup", on_change=to_custom)
    st.subheader("Planning")
    st.slider("Core: share of days covered (%)", 70, 95, key="core_pct")
    st.slider("Escalation: share of days covered (%)", 80, 99, key="esc_pct")
    st.select_slider("Simulated years", options=[200, 500, 1000, 2000], key="sims",
                     help="More: steadier numbers, slower.")
    st.toggle("Show the true answer", key="show_truth",
              help="Possible only because the data are synthetic: the following year simulated from the process "
                   "that generated the data.")

g = st.session_state
hosp, lag, sims = int(g["hospital"]), int(g["lag"]), int(g["sims"])
core_q, esc_q = g["core_pct"] / 100, max(g["esc_pct"], g["core_pct"] + 1) / 100
method = current_method()
mt = method_tuple(method)
extract = get_extract(hosp, lag)
model = get_model(hosp, lag, mt)
dates = model.plan_dates
with st.spinner("Simulating…"):
    occ = get_occupancy(hosp, lag, mt, sims)
truth = get_truth(hosp, lag, sims) if g["show_truth"] else None
fy = f"{dates[0]:%Y}/{dates[-1]:%y}"

st.title("Bed demand model")
st.markdown('<div class="lede">How many inpatient beds each surgical ward type needs: thousands of simulated years of '
            'admissions and stays, built from one year of spell data. <span class="badge">Synthetic data</span></div>',
            unsafe_allow_html=True)

tab_beds, tab_groups, tab_scen, tab_proj, tab_long, tab_about = st.tabs(
    ["Beds by ward type", "Ward groups", "Scenarios", "Projection", "Long stays", "Methodology"])

# ---------------------------------------------------------------- beds
with tab_beds:
    tot = total(occ)
    s = extract.spells
    st.markdown(f'<div class="scenario-line"><b>Hospital {hosp}</b> · data year {extract.start:%d %b %Y} to '
                f'{extract.end:%d %b %Y}, {len(s):,} spells ({int((~s["Finished"]).sum())} still in hospital when the '
                f'extract was taken) · <b>{method.name} method</b> · planning year {fy} · {sims:,} simulated years.</div>',
                unsafe_allow_html=True)
    b_core, b_esc = beds_for(tot, core_q), beds_for(tot, esc_q)
    cols = st.columns(4)
    if truth is not None:
        t_tot = total(truth)
        t_core, t_esc = beds_for(t_tot, core_q), beds_for(t_tot, esc_q)
        cov_c, cov_e = coverage(t_tot, b_core), coverage(t_tot, b_esc)
        cols[0].markdown(kpi(f"Core beds, all ward types ({g['core_pct']}% of days)", f"{b_core:,}",
                             f"The true answer is {t_core:,}. In the true year, {b_core:,} beds cover {cov_c:.0%} of days.",
                             (abs(cov_c - core_q) <= 0.05, f"Covers {cov_c:.0%} of true days")), unsafe_allow_html=True)
        cols[1].markdown(kpi(f"Escalation beds ({g['esc_pct']}% of days)", f"{b_esc:,}",
                             f"The true answer is {t_esc:,}. In the true year, {b_esc:,} beds cover {cov_e:.0%} of days.",
                             (abs(cov_e - esc_q) <= 0.03, f"Covers {cov_e:.0%} of true days")), unsafe_allow_html=True)
    else:
        cols[0].markdown(kpi(f"Core beds, all ward types ({g['core_pct']}% of days)", f"{b_core:,}",
                             f"Occupancy is at or below this on {coverage(tot, b_core):.0%} of simulated days."),
                         unsafe_allow_html=True)
        cols[1].markdown(kpi(f"Escalation beds ({g['esc_pct']}% of days)", f"{b_esc:,}",
                             f"Occupancy is at or below this on {coverage(tot, b_esc):.0%} of simulated days."),
                         unsafe_allow_html=True)
    cols[2].markdown(kpi("Average beds occupied", f"{tot.mean():,.0f}",
                         f"Busiest simulated night: {int(tot.max()):,}."), unsafe_allow_html=True)
    long_days = extract.spells.loc[extract.spells["Finished"], "Length of Stay"]
    share_long = long_days[long_days > 60].sum() / max(long_days.sum(), 1)
    cols[3].markdown(kpi("Bed-days from stays over 60 days", f"{share_long:.0%}",
                         f"{int((long_days > 60).sum())} finished spells ({(long_days > 60).mean():.1%}) in the data year. "
                         "The original extract left them out."), unsafe_allow_html=True)

    eyebrow("By ward type")
    rows = []
    for name, w in model.wards.items():
        o = occ[name]
        r = {"Ward type": name, "Admissions a day": w.arrivals.per_day, "Average stay": w.los.mean,
             "Average occupied": o.mean(), "Core beds": beds_for(o, core_q), "Escalation beds": beds_for(o, esc_q)}
        if truth is not None:
            r["True core"] = beds_for(truth[name], core_q)
            r["True nights covered"] = coverage(truth[name], r["Core beds"])
        rows.append(r)
    table = pd.DataFrame(rows)
    show_ci = st.toggle("Add uncertainty from having one year of data (about 20 seconds)", key="show_ci",
                        help="Refits the model to 30 bootstrap copies of the data year and reruns it each time.")
    if show_ci:
        with st.spinner("Refitting to 30 bootstrap copies of the year…"):
            di = get_interval(hosp, lag, mt, core_q, esc_q)
        ci = di.per_ward.set_index("ward_type")
        table["Core: 95% range"] = [f"{ci.at[n, 'core_low']:.0f}–{ci.at[n, 'core_high']:.0f}" for n in table["Ward type"]]
        note(f"All ward types: core {di.total['core_low']:.0f}–{di.total['core_high']:.0f}, escalation "
             f"{di.total['escalation_low']:.0f}–{di.total['escalation_high']:.0f}. These ranges are how far the bed "
             "numbers move when the model is refitted to bootstrap copies of the data year (weeks redrawn from nearby "
             "weeks, plus the uncertainty in the year's average surge level). They are much wider than the Monte Carlo "
             "noise, which more simulations would shrink.")
    st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3, column_config={
        "Ward type": st.column_config.TextColumn(width="medium"),
        "Admissions a day": st.column_config.NumberColumn(format="%.2f", width="small"),
        "Average stay": st.column_config.NumberColumn(format="%.1f", width="small", help="Days"),
        "Average occupied": st.column_config.NumberColumn(format="%.1f", width="small"),
        "Core beds": st.column_config.NumberColumn(width="small"),
        "Escalation beds": st.column_config.NumberColumn(width="small"),
        "True core": st.column_config.NumberColumn(width="small"),
        "True nights covered": st.column_config.NumberColumn(format="percent", width="small"),
    })
    if truth is not None:
        note("True core: the beds the true year needs. True nights covered: the share of the true year's nights on which "
             f"the model's core number is enough (aim: {g['core_pct']}%).")

    eyebrow("One ward type")
    pick = st.selectbox("Ward type", [ALL] + model.names, key="pick", label_visibility="collapsed")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Occupancy through {fy}**")
        o = tot if pick == ALL else occ[pick]
        t = (total(truth) if pick == ALL else truth[pick]) if truth is not None else None
        st.altair_chart(occupancy_chart(o, dates, beds_for(o, core_q), beds_for(o, esc_q), t), width="stretch")
        note("Line: median simulated occupancy each night; band: 5th to 95th percentile across simulated years"
             + ("; dotted: the true median." if t is not None else "."))
    with c2:
        if pick != ALL:
            st.markdown("**Length of stay**")
            ws = extract.spells[extract.spells["WardTypeRequired"] == pick]
            st.altair_chart(los_chart(ws, model.wards[pick].los), width="stretch")
            fin = ws[ws["Finished"]]["Length of Stay"]
            note(f"Bars: finished spells in the data year. Orange: the model's stays. {int((fin > 30).sum())} spells over "
                 f"30 days, {int((fin > 60).sum())} over 60; longest {int(fin.max())} days. "
                 f"{int((~ws['Finished']).sum())} still in hospital when the extract was taken.")
        else:
            st.markdown("**Where the beds go**")
            share = pd.DataFrame({"Ward type": [short(n) for n in model.names],
                                  "Beds": [occ[n].mean() for n in model.names]}).sort_values("Beds", ascending=False)
            st.altair_chart(bar_chart(share, "Beds", "Ward type", "Average beds occupied"), width="stretch")

# ---------------------------------------------------------------- ward groups
with tab_groups:
    st.markdown("Ward types that could share a ward. A shared ward needs fewer beds than its ward types would separately, "
                "because they aren't all busy on the same nights. The combined number comes from adding the simulated "
                "occupancies night by night, not from adding the ward types' bed numbers.")
    rows = []
    for gname, members in WARD_GROUPS.items():
        sep = sum(beds_for(occ[m], core_q) for m in members)
        comb = beds_for(total(occ, list(members)), core_q)
        sep_e = sum(beds_for(occ[m], esc_q) for m in members)
        comb_e = beds_for(total(occ, list(members)), esc_q)
        rows.append({"Ward group": gname, "Ward types": ", ".join(short(m) for m in members),
                     "Core, separate wards": sep, "Core, shared ward": comb, "Beds saved": sep - comb,
                     "Escalation, separate": sep_e, "Escalation, shared": comb_e, "Escalation beds saved": sep_e - comb_e})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    sep_all = sum(beds_for(occ[n], core_q) for n in model.names)
    note(f"The same effect at hospital scale: the {len(model.names)} ward types' core numbers add up to {sep_all} beds, "
         f"but all of them together need {beds_for(total(occ), core_q)}. With the recommended method, emergency ward "
         "types share surges, so they are busy together a little more often and the saving is slightly smaller than "
         "independent simulations would show.")

# ---------------------------------------------------------------- scenarios
with tab_scen:
    st.markdown("Ways of reducing demand, each run through the same simulation. With the recommended method, each "
                "scenario changes individual simulated patients (for example, an SDEC-eligible patient doesn't take a bed); "
                "with the original fitted method, it becomes an equivalent cut in admissions or in the average stay, as "
                "the original model did.")
    chosen = st.multiselect("Scenarios", list(STANDARD_SCENARIOS), default=DEFAULT_SCENARIOS, key="scenarios")
    run_scen = st.toggle(f"Run the scenarios (about {max(1, len(chosen)) * (1 if sims <= 500 else 3)} seconds the "
                         "first time)", key="run_scen")
    if run_scen and chosen:
        res = {"Baseline": occ}
        prog = st.progress(0.0, text="Simulating scenarios…")
        for i, name in enumerate(chosen):
            res[name] = get_occupancy(hosp, lag, mt, sims, name)
            prog.progress((i + 1) / len(chosen), text=f"Simulated: {name}")
        prog.empty()
        mat = pd.DataFrame({sc: {"All ward types": beds_for(total(o), core_q),
                                 **{short(n): beds_for(o[n], core_q) for n in model.names}} for sc, o in res.items()}).T
        saved = pd.DataFrame({"Scenario": chosen,
                              "Beds saved": [mat.at["Baseline", "All ward types"] - mat.at[c, "All ward types"]
                                             for c in chosen]}).sort_values("Beds saved", ascending=False)
        st.markdown(f"**Core beds saved, all ward types** (baseline {mat.at['Baseline', 'All ward types']})")
        st.altair_chart(bar_chart(saved, "Beds saved", "Scenario", "Core beds saved against the baseline", AQUA),
                        width="stretch")
        st.markdown("**Core beds by scenario and ward type**")
        st.dataframe(mat, width="stretch", height=35 * (len(mat) + 1) + 3)
        los_sc = [c for c in chosen if STANDARD_SCENARIOS[c].los_reduction > 0]
        if los_sc:
            r = STANDARD_SCENARIOS[los_sc[0]].los_reduction
            dt = discharge_targets(extract.spells, r)
            dt["ward_type"] = dt["ward_type"].map(short)
            st.markdown(f"**Discharge targets for '{los_sc[0]}'**")
            st.dataframe(dt.rename(columns={"ward_type": "Ward type", "short_stay_per_week": "Stays of 7 days or less, a week",
                                            "extra_short_stay_per_week": "Extra such discharges a week"}),
                         hide_index=True, width="stretch", column_config={
                             "Stays of 7 days or less, a week": st.column_config.NumberColumn(format="%.1f"),
                             "Extra such discharges a week": st.column_config.NumberColumn(format="%.1f")})
            note("As in the original model, an operational target: the extra discharges a week after stays of 7 days or "
                 "less implied by shortening each finished spell's stay by the scenario's share. Each whole-day stay is "
                 "treated as continuous within its day, so a 5% cut moves some 8-day stays under the threshold; "
                 "rounding, as the original did, moves none.")

# ---------------------------------------------------------------- projection
with tab_proj:
    c1, c2, c3 = st.columns(3)
    growth = c1.number_input("Admissions growth a year (%)", 0.0, 10.0, 1.5, 0.5, key="growth") / 100
    improve = c2.number_input("Length-of-stay improvement a year (%)", 0.0, 10.0, 1.0, 0.5, key="improve") / 100
    years = c3.slider("Years ahead", 1, 10, 5, key="years")
    run_proj = st.toggle(f"Run the projection (about {years + 1} seconds the first time)", key="run_proj")
    if run_proj:
        rows = []
        for t in range(years + 1):
            o = occ if t == 0 else get_occupancy(hosp, lag, mt, sims, "Baseline", t, growth, improve)
            tt = total(o)
            rows.append({"Year": f"{dates[0].year + t}/{(dates[-1].year + t) % 100:02d}",
                         "Core beds": beds_for(tt, core_q), "Escalation beds": beds_for(tt, esc_q),
                         "Admissions a day": sum(w.arrivals.per_day for w in model.wards.values()) * (1 + growth) ** t})
        proj = pd.DataFrame(rows)
        long = proj.melt(id_vars="Year", value_vars=["Core beds", "Escalation beds"], var_name="Measure", value_name="Beds")
        chart = alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("Year:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Beds:Q", scale=alt.Scale(zero=False), title="Beds, all ward types"),
            color=alt.Color("Measure:N", scale=alt.Scale(range=[ORANGE, CRITICAL]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Year", "Measure", "Beds"]).properties(height=300)
        st.altair_chart(chart, width="stretch")
        st.dataframe(proj, hide_index=True, width="stretch",
                     column_config={"Admissions a day": st.column_config.NumberColumn(format="%.1f")})
        note("Admissions × (1 + growth)^t; every stay × (1 − improvement)^t. The mix of ward types, the weekly pattern and "
             "the season stay as in the data year.")

# ---------------------------------------------------------------- long stays
with tab_long:
    fin = extract.spells[extract.spells["Finished"]]["Length of Stay"]
    over = fin > 60
    st.markdown(
        f'<div class="callout">In hospital {hosp}, <b>{over.mean():.1%} of finished spells</b> lasted more than 60 days, '
        f'but they account for <b>{fin[over].sum() / fin.sum():.0%} of bed-days</b>. The original extract kept stays of '
        "1 to 60 days only, so those patients vanished from the admissions as well as the stays. Its fit then left out "
        "Tukey outliers (in effect, stays beyond about a week for elective ward types and two to four weeks for emergency ones), it capped stays at 90 days, and it started "
        "each simulated year with an empty hospital 30 days before it began.</div>", unsafe_allow_html=True)
    run_path = st.toggle("Compare each step from the original method to the recommended one (about 15 seconds the first "
                         "time)", key="run_path")
    if run_path:
        with st.spinner("Fitting and simulating each step…"):
            path = get_path(hosp, lag, sims, core_q)
        t_core = beds_for(total(get_truth(hosp, lag, sims)), core_q)
        st.altair_chart(path_chart(path, t_core), width="stretch")
        covered = ", ".join(f"{r['Nights covered (truth)']:.0%}" for _, r in path.iterrows())
        note(f"Each step adds one change to those above it, so a step's effect depends on the order. Share of the true "
             f"year's nights each number of beds covers, top to bottom (aim: {g['core_pct']}%): {covered}.")
    if METHODS_FILE.exists():
        df = pd.read_csv(METHODS_FILE)
        st.markdown(f"**Across {df['hospital'].nunique()} synthetic hospitals** (`scripts/run_methods.py`)")
        summ = (df.groupby("variant", sort=False)
                .agg(error=("core_error_pct", "mean"), covered=("core_days_covered", "mean"),
                     ward_err=("ward_core_abs_error_pct", "mean")).reset_index())
        summ.columns = ["Method", "Core beds vs truth (%)", "True nights covered", "Ward types: mean error (%)"]
        st.dataframe(summ, hide_index=True, width="stretch", height=35 * (len(summ) + 1) + 3, column_config={
            "Core beds vs truth (%)": st.column_config.NumberColumn(format="%+.1f"),
            "True nights covered": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Ward types: mean error (%)": st.column_config.NumberColumn(format="%.1f")})
        note("Means across the hospitals. The core number aims to cover 85% of nights. Ward types: the average size of "
             "each ward type's error in core beds, whichever way it goes.")

with tab_about:
    st.markdown(METHODOLOGY)
