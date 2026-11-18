"""Fit a bed demand model to an extract, simulate the year ahead, and run scenarios.

A ``Method`` says how the extract is turned into a model. Two presets:

* ``ORIGINAL``: the method of the model this project rebuilds. Finished spells of 1-60 days only
  (the extract's WHERE clause), Tukey outliers excluded from a multi-distribution fit, sampled
  stays capped at 90 days, day-of-week pattern but no seasonality or surges, and a 30-day
  warm-up from an empty hospital.
* ``RECOMMENDED``: every spell, including those still in hospital (Kaplan-Meier), resampled
  rather than fitted; shared seasonality and emergency surges; a 365-day warm-up.

Every difference between the two can be switched on or off by itself, which is how
``scripts/run_methods.py`` measures what each one is worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .arrivals import Arrivals, Seasonality, Surges, fit_arrivals, fit_seasonality, fit_surges
from .los import FittedLoS, SpellPool, fit_original, make_pool
from .simulate import extend_back, occupancy
from .synthetic import EMERGENCY, WARD_TYPES, Extract, Hospital, daily_rate, sample_stays, surge_paths

WARD_NAMES = {w.name for w in WARD_TYPES}


@dataclass(frozen=True)
class Method:
    name: str = "Recommended"
    los: str = "pool"  # "pool" (resample spells) or "fitted" (the original multi-distribution fit)
    max_los: int | None = None  # drop finished spells longer than this, from arrivals as well as stays
    tukey: bool = False  # exclude stays above Q3 + 3 x IQR from the length-of-stay model
    cap: int | None = None  # cap sampled stays at this many days
    in_progress: bool = True  # use spells still in hospital when the extract was taken
    seasonality: bool = True
    surges: bool = True
    warmup: int = 365  # days simulated before the year, starting from an empty hospital


RECOMMENDED = Method()
ORIGINAL = Method("Original", los="fitted", max_los=60, tukey=True, cap=90, in_progress=False,
                  seasonality=False, surges=False, warmup=30)


@dataclass(frozen=True)
class Scenario:
    """A change to demand. Each part acts on individual simulated patients (with the spell-pool
    method), or, with the original fitted method, as the original did: as an equivalent cut in
    the admission rate or in the average stay."""
    name: str = "Baseline"
    los_reduction: float = 0.0  # every stay shorter by this share
    divert: tuple[tuple[str, float], ...] = ()  # (flag column, share of flagged admissions removed)
    discharge_ready: float = 0.0  # share of days spent waiting after being ready to leave removed
    virtual_ward_days: float = 0.0  # days saved by each eligible patient sent to a virtual ward...
    virtual_ward_share: float = 0.0  # ...for this share of eligible patients
    long_stay: tuple[int, int, float] | None = None  # (threshold days, target days, share achieved)
    admissions_growth: float = 0.0  # admissions up by this share


BASELINE = Scenario()

# The original model's scenarios (its configuration file), plus the three it had switched off.
STANDARD_SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("Stays 5% shorter", los_reduction=0.05),
    Scenario("Stays 10% shorter", los_reduction=0.10),
    Scenario("SDEC takes half its eligible patients", divert=(("sdec_eligible", 0.5),)),
    Scenario("SDEC takes all its eligible patients", divert=(("sdec_eligible", 1.0),)),
    Scenario("Planned day cases go home the same day", divert=(("intended_daycase", 1.0),)),
    Scenario("All day-case-eligible patients go home the same day", divert=(("daycase_eligible", 1.0),)),
    Scenario("Readmissions down 20%", divert=(("is_readmission", 0.2),)),
    Scenario("No waiting once ready to leave", discharge_ready=1.0),
    Scenario("Virtual ward saves 3 days for eligible patients", virtual_ward_days=3, virtual_ward_share=1.0),
    Scenario("Stays of 60+ days cut to 45", long_stay=(60, 45, 1.0)),
]}
DEFAULT_SCENARIOS = list(STANDARD_SCENARIOS)[:8]


@dataclass
class WardModel:
    name: str
    kind: str
    arrivals: Arrivals
    los: FittedLoS | SpellPool
    used: pd.DataFrame = field(repr=False)  # the spells the model was fitted to


@dataclass
class BedModel:
    method: Method
    wards: dict[str, WardModel]
    season: Seasonality | None
    surges: Surges | None
    plan_dates: pd.DatetimeIndex  # the year being planned for

    @property
    def names(self) -> list[str]:
        return list(self.wards)


def ward_kinds(spells: pd.DataFrame) -> dict[str, str]:
    """Emergency or elective, from the point of delivery of most of a ward type's spells."""
    nel = spells.groupby("WardTypeRequired")["PoD"].apply(lambda p: (p == "NEL").mean())
    return {k: (EMERGENCY if v >= 0.5 else "elective") for k, v in nel.items()}


def usable_spells(extract: Extract, method: Method) -> pd.DataFrame:
    s = extract.spells
    if not method.in_progress:
        s = s[s["Finished"]]
    if method.max_los is not None:
        s = s[~(s["Finished"] & (s["Length of Stay"] > method.max_los))]
    return s


def fit(extract: Extract, method: Method = RECOMMENDED, plan_days: int = 365) -> BedModel:
    spells = usable_spells(extract, method)
    kinds = ward_kinds(extract.spells)
    data_dates = pd.date_range(extract.start, periods=extract.days, freq="D")
    wards = {}
    order = [w.name for w in WARD_TYPES if w.name in kinds] + sorted(k for k in kinds if k not in WARD_NAMES)
    for name in order:
        s = spells[spells["WardTypeRequired"] == name]
        arr = fit_arrivals(s, extract.start, extract.days, kinds[name])
        if method.los == "fitted":
            los = fit_original(s["Length of Stay"].to_numpy(), tukey=method.tukey, cap=method.cap)
        else:
            los = make_pool(s, in_progress=method.in_progress, tukey=method.tukey, cap=method.cap)
        wards[name] = WardModel(name, kinds[name], arr, los, s)
    arrivals = {k: w.arrivals for k, w in wards.items()}
    season = fit_seasonality(spells, arrivals, data_dates) if method.seasonality else None
    surges = fit_surges(spells, arrivals, season, data_dates) if method.surges else None
    plan = pd.date_range(extract.end + pd.Timedelta(days=1), periods=plan_days, freq="D")
    return BedModel(method, wards, season, surges, plan)


# ---------------------------------------------------------------- scenarios as samplers
def _flag(pool: SpellPool, col: str, idx: np.ndarray) -> np.ndarray:
    if col not in pool.flags.columns:
        return np.zeros(len(idx), dtype=bool)
    return pool.flags[col].to_numpy()[idx].astype(bool)


def pool_sampler(pool: SpellPool, sc: Scenario, los_scale: float = 1.0):
    """Stays for simulated patients, with the scenario applied to each patient."""
    stretch = pool.stretch_for((1 - sc.los_reduction) * los_scale)

    def sample(n: int, rng: np.random.Generator) -> np.ndarray:
        if n == 0:
            return np.zeros(0, dtype=int)
        idx = pool.draw(n, rng)
        if stretch != 1.0:
            stays = np.maximum(np.ceil(pool.continuous(idx, rng) * stretch), 1)
        else:
            stays = pool.los[idx].astype(float)
        if sc.discharge_ready > 0 and "days_excess_discharge" in pool.flags:
            excess = pool.flags["days_excess_discharge"].to_numpy()[idx]
            stays = np.maximum(stays - np.round(sc.discharge_ready * excess), 1)
        if sc.virtual_ward_days > 0 and sc.virtual_ward_share > 0:
            go = _flag(pool, "virtual_ward_eligible", idx) & (rng.random(n) < sc.virtual_ward_share)
            stays = np.where(go, np.maximum(stays - sc.virtual_ward_days, 1), stays)
        if sc.long_stay is not None:
            threshold, target, share = sc.long_stay
            go = (stays >= threshold) & (rng.random(n) < share)
            stays = np.where(go, np.minimum(stays, target), stays)
        if pool.cap is not None:
            stays = np.minimum(stays, pool.cap)
        for col, share in sc.divert:
            stays = np.where(_flag(pool, col, idx) & (rng.random(n) < share), 0, stays)
        return stays.astype(int)
    return sample


def original_adjustments(w: WardModel, sc: Scenario) -> tuple[float, float]:
    """The original model's version of a scenario: (admission multiplier, stay scale), from the
    spells the ward type's model was fitted to."""
    s, mult, red = w.used, 1.0, sc.los_reduction
    los = s["Length of Stay"].astype(float)
    mean_los = float(los.mean()) if len(s) else 0.0
    for col, share in sc.divert:
        if col in s:
            mult *= 1 - float(s[col].astype(bool).mean()) * share
    extra = 0.0
    if mean_los > 0:
        if sc.discharge_ready > 0 and "days_excess_discharge" in s:
            extra += float((s["days_excess_discharge"].clip(lower=0) * sc.discharge_ready).mean()) / mean_los
        if sc.virtual_ward_days > 0 and "virtual_ward_eligible" in s:
            extra += float(s["virtual_ward_eligible"].astype(bool).mean()) * sc.virtual_ward_days \
                * sc.virtual_ward_share / mean_los
        if sc.long_stay is not None:
            threshold, target, share = sc.long_stay
            over = (los.clip(lower=threshold) - target).clip(lower=0)
            extra += float((over * share).mean()) / mean_los
    scale = (1 - red) * (1 - min(extra, 1.0))
    return mult, scale


# ---------------------------------------------------------------- simulation
def _seed(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def simulate(model: BedModel, scenario: Scenario = BASELINE, n_sims: int = 1000, seed: int = 0,
             years_ahead: float = 0.0, growth: float = 0.0, los_improvement: float = 0.0,
             wards: list[str] | None = None) -> dict[str, np.ndarray]:
    """Simulated occupancy for each ward type, shape (n_sims, days in the planning year).

    Ward types share the emergency surge paths, so their occupancies move together as they would.
    ``years_ahead``, ``growth`` and ``los_improvement`` give the multi-year projection: admissions
    x (1 + growth)^t and stays x (1 - los_improvement)^t.
    """
    m = model.method
    dates = model.plan_dates
    warm = m.warmup
    surge = None
    if model.surges is not None and model.surges.sd > 0:
        all_days = len(dates) + warm
        surge = surge_paths(n_sims, all_days, model.surges.persistence, model.surges.sd, _seed(seed, 999))
    out = {}
    names = wards or model.names
    for i, name in enumerate(model.names):
        if name not in names:
            continue
        w = model.wards[name]
        season = model.season.factor(dates, w.kind) if model.season is not None else None
        rate = w.arrivals.rate(dates, season) * (1 + scenario.admissions_growth) * (1 + growth) ** years_ahead
        los_scale = (1 - los_improvement) ** years_ahead
        if isinstance(w.los, SpellPool):
            sampler = pool_sampler(w.los, scenario, los_scale)
        else:
            mult, scale = original_adjustments(w, scenario)
            rate = rate * mult
            fitted, s_total = w.los, scale * los_scale

            def sampler(n, rng, fitted=fitted, s_total=s_total):
                return fitted.sample(n, rng, s_total)
        mult_paths = surge if (surge is not None and w.kind == EMERGENCY) else None
        out[name] = occupancy(rate, sampler, n_sims, _seed(seed, i), warm, mult_paths)
    return out


def total(occ: dict[str, np.ndarray], names: list[str] | None = None) -> np.ndarray:
    """Occupancy summed over ward types (simulation by simulation, day by day)."""
    names = names or list(occ)
    return np.sum([occ[n] for n in names if n in occ], axis=0)


# ---------------------------------------------------------------- the true answer (synthetic only)
def true_occupancy(hospital: Hospital, start: pd.Timestamp, days: int = 365, n_sims: int = 1000,
                   seed: int = 0, warmup: int = 730) -> dict[str, np.ndarray]:
    """Occupancy under the process that generated the data: what a perfect model would simulate."""
    dates = pd.date_range(start, periods=days, freq="D")
    surge = surge_paths(n_sims, days + warmup, hospital.surge_persistence, hospital.surge_sd, _seed(seed, 7, 999))
    out = {}
    for i, w in enumerate(hospital.ward_types):
        rate = daily_rate(w, dates, hospital.season_amplitude)

        def sampler(n, rng, w=w):
            return sample_stays(w, n, rng)[0]
        out[w.name] = occupancy(rate, sampler, n_sims, _seed(seed, 7, i), warmup,
                                surge if w.kind == EMERGENCY else None)
    return out


# ---------------------------------------------------------------- uncertainty from the data
def resample_year(extract: Extract, rng: np.random.Generator, reach_weeks: int = 4) -> Extract:
    """A bootstrap copy of the data year, for the uncertainty that comes from having one year.

    The year is cut into weeks. Each week is replaced by a week drawn at random from within
    ``reach_weeks`` weeks of it (a 'local block bootstrap'), moved whole: its admissions, their
    stays and flags, and its day-of-week pattern. Drawing nearby weeks keeps the season in place;
    moving whole weeks keeps busy spells together."""
    s = extract.spells
    day = (s["Arrival Date"] - extract.start).dt.days.to_numpy()
    week = day // 7
    n_weeks = int(np.ceil(extract.days / 7))
    by_week = {k: np.flatnonzero(week == k) for k in range(n_weeks)}
    parts = []
    for k in range(n_weeks):
        src = int(np.clip(k + rng.integers(-reach_weeks, reach_weeks + 1), 0, n_weeks - 1))
        if src == n_weeks - 1 and k != n_weeks - 1 and extract.days % 7:
            src = n_weeks - 2  # the short final week can't stand in for a full one
        rows = s.iloc[by_week[src]].copy()
        shift = pd.Timedelta(days=7 * (k - src))
        rows["Arrival Date"] = rows["Arrival Date"] + shift
        rows = rows[(rows["Arrival Date"] >= extract.start) & (rows["Arrival Date"] <= extract.end)]
        parts.append(rows)
    spells = pd.concat(parts, ignore_index=True)
    return Extract(spells, extract.start, extract.end, extract.taken, extract.truth)


@dataclass
class DataInterval:
    per_ward: pd.DataFrame  # ward type, core beds low/high, escalation beds low/high
    total: dict  # same for all ward types together
    draws: pd.DataFrame  # one row per bootstrap copy


def year_level_sd(surges: Surges, days: int) -> float:
    """How far the data year's average surge level could be from its long-run value of 1, as a
    log-scale standard deviation: the standard deviation of the mean of ``days`` consecutive values of
    the AR(1) process x_t, which is sd x sqrt((1 + 2 x sum over lags l of (1 - l/days) x rho^l) / days).
    With rho = 0.92 and sd = 0.10 over 365 days it is about 0.026: a year's emergency admissions can
    easily run 3% above or below the long-run rate because of surges alone."""
    rho, sd = surges.persistence, surges.sd
    lags = np.arange(1, days)
    return float(sd * np.sqrt((1 + 2 * np.sum((1 - lags / days) * rho ** lags)) / days))


def data_interval(extract: Extract, method: Method, n_boot: int = 40, n_sims: int = 200, seed: int = 0,
                  core: float = 0.85, escalation: float = 0.95, level: float = 0.95,
                  year_level: bool = True) -> DataInterval:
    """Refit the model to ``n_boot`` bootstrap copies of the year and see how far the bed numbers move.

    Redrawing weeks from nearby weeks can't show that the whole year may have been busier or quieter
    than usual, because a surge lasts longer than a week. So when the method models surges and
    ``year_level`` is on, each copy's emergency admission rates are also multiplied by
    exp(z s - s^2/2), with z standard normal and s = ``year_level_sd``.
    """
    from .metrics import beds_for
    rows = []
    base = fit(extract, method)
    level_sd = year_level_sd(base.surges, extract.days) if (year_level and base.surges is not None) else 0.0
    for b in range(n_boot):
        ex = resample_year(extract, _seed(seed, 31, b))
        m = fit(ex, method)
        if level_sd > 0:
            factor = float(np.exp(_seed(seed, 32, b).normal(0, level_sd) - level_sd ** 2 / 2))
            for w in m.wards.values():
                if w.kind == EMERGENCY:
                    w.arrivals.per_day *= factor
        occ = simulate(m, n_sims=n_sims, seed=seed + 1000 + b)
        row = {"copy": b}
        for name, o in occ.items():
            row[(name, "core")] = beds_for(o, core)
            row[(name, "escalation")] = beds_for(o, escalation)
        t = total(occ)
        row[("All ward types", "core")] = beds_for(t, core)
        row[("All ward types", "escalation")] = beds_for(t, escalation)
        rows.append(row)
    draws = pd.DataFrame(rows).set_index("copy")
    a = (1 - level) / 2
    out = []
    for name in dict.fromkeys(k[0] for k in draws.columns):
        out.append({"ward_type": name,
                    "core_low": draws[(name, "core")].quantile(a), "core_high": draws[(name, "core")].quantile(1 - a),
                    "escalation_low": draws[(name, "escalation")].quantile(a),
                    "escalation_high": draws[(name, "escalation")].quantile(1 - a)})
    per = pd.DataFrame(out)
    return DataInterval(per[per.ward_type != "All ward types"].reset_index(drop=True),
                        per[per.ward_type == "All ward types"].iloc[0].to_dict(), draws)


# ---------------------------------------------------------------- discharge targets (as the original)
def discharge_targets(spells: pd.DataFrame, los_reduction: float, short_stay_days: int = 7) -> pd.DataFrame:
    """For each ward type: discharges a week after stays of ``short_stay_days`` or fewer in the data year,
    and how many more a week a given cut in length of stay implies, as an operational target.

    Each finished stay of L whole days is taken to be somewhere in (L - 1, L], uniformly (as in
    ``SpellPool.continuous``), and shortened by the cut. The chance it then counts as short is the
    chance that (L - U) x (1 - cut) <= ``short_stay_days``, with U uniform on (0, 1). The original model
    instead rounded L x (1 - cut), which for a 5% cut and a 7-day threshold adds no short stays at all:
    an 8-day stay becomes 7.6 days, which rounds back to 8.
    """
    s = spells[spells["Finished"]] if "Finished" in spells else spells
    keep = 1 - los_reduction
    rows = []
    for name, g in s.groupby("WardTypeRequired", sort=False):
        los = g["Length of Stay"].to_numpy(float)
        weeks = max(1.0, (g["Arrival Date"].max() - g["Arrival Date"].min()).days / 7)
        base = float((los <= short_stay_days).sum())
        p_short = 1 - np.clip(los - short_stay_days / keep, 0, 1)
        rows.append({"ward_type": name, "short_stay_per_week": base / weeks,
                     "extra_short_stay_per_week": (p_short.sum() - base) / weeks})
    return pd.DataFrame(rows)


def with_method(method: Method, **changes) -> Method:
    return replace(method, **changes)


__all__ = ["Method", "ORIGINAL", "RECOMMENDED", "Scenario", "BASELINE", "BedModel", "fit", "simulate", "total",
           "true_occupancy", "data_interval", "discharge_targets", "extend_back", "resample_year"]
