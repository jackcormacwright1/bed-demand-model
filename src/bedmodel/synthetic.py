"""A synthetic hospital: surgical ward types, their admissions and lengths of stay.

Everything here is invented. Admission rates, lengths of stay and the shares of long stays are
plausible illustrative values, not figures from any NHS organisation. Because the process that
generates the data is known, any estimate of bed demand made from the data can be checked
against the true answer.

How a year of admissions is generated, for ward type s on day t:

    admissions ~ Poisson( lambda_s x dow_s(t) x season_kind(t) x surge(t) )

* ``lambda_s``      average admissions a day over the year.
* ``dow_s(t)``      day-of-week pattern: elective admissions on weekdays only, emergency
                    admissions busiest on Mondays. Averages 1 over a week.
* ``season(t)``     emergency admissions peak in mid-January (+10%) and dip in mid-July;
                    elective admissions dip in August and stop over Christmas.
* ``surge(t)``      a slowly wandering pressure factor shared by every emergency ward type
                    (busy and quiet spells lasting a week or two). Elective admissions don't surge.

Length of stay mixes two groups of patients:

* most patients: a lognormal stay with a typical (median) length of a few days;
* a small share of **long stays** (complex discharges, waits for care packages): a lognormal
  stay with a median of 30-50 days, so some of them stay more than 60 days.

A stay of L days (L >= 1) means the patient occupies a bed on L midnights.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

EMERGENCY, ELECTIVE = "emergency", "elective"


@dataclass(frozen=True)
class WardType:
    name: str
    kind: str  # "emergency" or "elective"
    per_day: float  # average admissions a day
    median_days: float  # typical stay of most patients (lognormal median, before rounding up)
    spread: float  # log-scale standard deviation of those stays
    long_share: float  # share of admissions who are long stays
    long_median_days: float = 45.0
    long_spread: float = 0.45
    sdec_share: float = 0.0  # of emergency stays of 1-2 days, share who could be managed in SDEC
    virtual_ward_share: float = 0.0  # of emergency stays of 3+ days, share eligible for a virtual ward


# Fifteen ward types, invented values (see module docstring).
WARD_TYPES: tuple[WardType, ...] = (
    WardType("General surgery - emergency - male", EMERGENCY, 4.0, 3.0, 0.85, 0.025, sdec_share=0.35, virtual_ward_share=0.10),
    WardType("General surgery - emergency - female", EMERGENCY, 3.8, 2.8, 0.85, 0.025, sdec_share=0.35, virtual_ward_share=0.10),
    WardType("General surgery - elective - male", ELECTIVE, 1.6, 1.8, 0.70, 0.010, long_median_days=35),
    WardType("General surgery - elective - female", ELECTIVE, 1.5, 1.8, 0.70, 0.010, long_median_days=35),
    WardType("Trauma and orthopaedics", EMERGENCY, 5.0, 4.5, 0.80, 0.060, long_median_days=50, virtual_ward_share=0.05),
    WardType("Gynaecology - emergency", EMERGENCY, 1.8, 1.5, 0.70, 0.005, long_median_days=30, sdec_share=0.30),
    WardType("Gynaecology - elective", ELECTIVE, 1.2, 1.5, 0.60, 0.005, long_median_days=30),
    WardType("Urology - emergency - male", EMERGENCY, 1.6, 2.5, 0.80, 0.020, long_median_days=40, sdec_share=0.30),
    WardType("Urology - emergency - female", EMERGENCY, 0.6, 2.2, 0.80, 0.020, long_median_days=40, sdec_share=0.30),
    WardType("Urology - elective", ELECTIVE, 1.4, 1.5, 0.70, 0.005, long_median_days=30),
    WardType("Gastroenterology - male", EMERGENCY, 2.2, 4.0, 0.85, 0.040, virtual_ward_share=0.15),
    WardType("Gastroenterology - female", EMERGENCY, 2.0, 4.0, 0.85, 0.040, virtual_ward_share=0.15),
    WardType("Colorectal surgery - male", EMERGENCY, 0.9, 6.0, 0.80, 0.040, long_median_days=50, virtual_ward_share=0.05),
    WardType("Colorectal surgery - female", EMERGENCY, 0.8, 6.0, 0.80, 0.040, long_median_days=50, virtual_ward_share=0.05),
    WardType("Breast surgery", ELECTIVE, 0.5, 1.3, 0.50, 0.000),
)

# Ward design groups: ward types that could share a ward (as in the original model's configuration).
WARD_GROUPS: dict[str, tuple[str, ...]] = {
    "Female emergency surgery and gynaecology": ("Gynaecology - emergency", "General surgery - emergency - female"),
    "Female elective surgery and gynaecology": ("Gynaecology - elective", "General surgery - elective - female"),
    "Male emergency surgery and urology": ("General surgery - emergency - male", "Urology - emergency - male"),
    "Colorectal and gastroenterology": ("Colorectal surgery - male", "Colorectal surgery - female",
                                         "Gastroenterology - male", "Gastroenterology - female"),
}

# Day-of-week patterns, Monday first (scaled to average 1 over the week).
DOW = {
    EMERGENCY: np.array([1.12, 1.04, 1.00, 1.00, 1.02, 0.88, 0.94]),
    ELECTIVE: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
}
DOW = {k: v * 7 / v.sum() for k, v in DOW.items()}

SEASON_AMPLITUDE = 0.10  # emergency admissions: +10% in mid-January, -10% in mid-July
SURGE_PERSISTENCE = 0.92  # day-to-day persistence of the shared emergency pressure (AR(1) coefficient)
SURGE_SD = 0.10  # its typical size: about +/-10%


@dataclass(frozen=True)
class Hospital:
    ward_types: tuple[WardType, ...]
    seed: int
    surge_persistence: float = SURGE_PERSISTENCE
    surge_sd: float = SURGE_SD
    season_amplitude: float = SEASON_AMPLITUDE

    def get(self, name: str) -> WardType:
        return next(w for w in self.ward_types if w.name == name)

    @property
    def names(self) -> list[str]:
        return [w.name for w in self.ward_types]


def make_hospital(seed: int = 0, vary: bool = True) -> Hospital:
    """The reference ward types, each varied at random (unless ``vary`` is False) so that different
    seeds give different hospitals: admissions +/-15%, typical stays +/-10%, long-stay share x0.6-1.4,
    long-stay median +/-15%."""
    r = np.random.default_rng(np.random.SeedSequence([seed, 11]))
    wards = []
    for w in WARD_TYPES:
        if vary:
            w = replace(w, per_day=w.per_day * r.uniform(0.85, 1.15),
                        median_days=w.median_days * r.uniform(0.9, 1.1),
                        long_share=w.long_share * r.uniform(0.6, 1.4),
                        long_median_days=w.long_median_days * r.uniform(0.85, 1.15))
        wards.append(w)
    return Hospital(tuple(wards), seed)


# ---------------------------------------------------------------- calendar and intensities
def calendar(start: str | pd.Timestamp, days: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), periods=days, freq="D")


def season_factor(dates: pd.DatetimeIndex, kind: str, amplitude: float = SEASON_AMPLITUDE) -> np.ndarray:
    """Seasonal multiplier for each date, averaging about 1 over a year."""
    if kind == EMERGENCY:
        doy = dates.dayofyear.to_numpy()
        return 1 + amplitude * np.cos(2 * np.pi * (doy - 15) / 365.25)
    f = np.ones(len(dates))
    f[dates.month == 8] = 0.75  # summer leave
    christmas = ((dates.month == 12) & (dates.day >= 22)) | ((dates.month == 1) & (dates.day <= 2))
    f[christmas] = 0.4
    return f / 0.97  # keeps the yearly average close to 1 (August and Christmas lose about 3%)


def surge_paths(n: int, days: int, persistence: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """``n`` paths of the shared emergency pressure: exp(x_t - sd^2/2), with x an AR(1) process,
    x_t = persistence * x_{t-1} + e_t, started at its long-run distribution (standard deviation ``sd``).
    The -sd^2/2 makes the multiplier average 1."""
    if sd <= 0:
        return np.ones((n, days))
    e_sd = sd * np.sqrt(1 - persistence ** 2)
    x = np.empty((n, days))
    x[:, 0] = rng.normal(0, sd, n)
    eps = rng.normal(0, e_sd, (n, days))
    for t in range(1, days):
        x[:, t] = persistence * x[:, t - 1] + eps[:, t]
    return np.exp(x - sd ** 2 / 2)


def daily_rate(w: WardType, dates: pd.DatetimeIndex, amplitude: float = SEASON_AMPLITUDE) -> np.ndarray:
    """Expected admissions on each date, before any surge."""
    return w.per_day * DOW[w.kind][dates.dayofweek.to_numpy()] * season_factor(dates, w.kind, amplitude)


# ---------------------------------------------------------------- lengths of stay and flags
def sample_stays(w: WardType, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``n`` true lengths of stay in whole days (at least 1), and which of them are long stays."""
    long = rng.random(n) < w.long_share
    body = rng.lognormal(np.log(w.median_days), w.spread, n)
    tail = rng.lognormal(np.log(w.long_median_days), w.long_spread, n)
    days = np.ceil(np.where(long, tail, body)).astype(int)
    return np.maximum(days, 1), long


def spell_flags(w: WardType, los: np.ndarray, long: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Flags used by the demand-reduction scenarios, generated plausibly (all invented):

    * ``days_excess_discharge``: days spent in hospital after being ready to leave. Long stays spend
      30-70% of their stay waiting; 6% of other stays wait 1-3 days.
    * ``is_readmission``: 8% of emergency admissions.
    * ``sdec_eligible``: of emergency stays of 1-2 days, the ward type's SDEC share.
    * ``intended_daycase`` (planned as a day case but stayed overnight): 30% of 1-day elective stays.
    * ``daycase_eligible``: 40% of elective stays of 1-2 days.
    * ``virtual_ward_eligible``: of emergency stays of 3+ days, the ward type's virtual-ward share.
    """
    n = len(los)
    em = w.kind == EMERGENCY
    excess = np.where(long, np.floor(los * rng.uniform(0.3, 0.7, n)), 0)
    short_wait = (~long) & (rng.random(n) < 0.06)
    excess = np.where(short_wait, np.minimum(los - 1, rng.integers(1, 4, n)), excess).astype(int)
    return {
        "days_excess_discharge": np.maximum(excess, 0),
        "is_readmission": em & (rng.random(n) < 0.08),
        "sdec_eligible": em & (los <= 2) & (rng.random(n) < w.sdec_share),
        "intended_daycase": (not em) & (los == 1) & (rng.random(n) < 0.30),
        "daycase_eligible": (not em) & (los <= 2) & (rng.random(n) < 0.40),
        "virtual_ward_eligible": em & (los >= 3) & (rng.random(n) < w.virtual_ward_share),
    }


# ---------------------------------------------------------------- one observed year
@dataclass
class Extract:
    """What a spell extract would show: every spell admitted between ``start`` and ``end`` (inclusive),
    as it stood on ``taken``. A spell still in hospital on that date shows its stay so far, with
    ``Finished`` False. ``truth`` keeps every spell's true length of stay, for checking only."""
    spells: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    taken: pd.Timestamp
    truth: pd.DataFrame = field(repr=False)

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def finished_only(self, max_los: int | None = None) -> pd.DataFrame:
        """The spells a 'finished spells' query returns, optionally only those up to ``max_los`` days
        (the original extract kept 1-60)."""
        s = self.spells[self.spells["Finished"]]
        return s if max_los is None else s[s["Length of Stay"] <= max_los]


def observe_year(hospital: Hospital, start: str = "2025-04-01", days: int = 365, lag_days: int = 30,
                 seed: int | None = None) -> Extract:
    """Simulate one year of admissions and take an extract ``lag_days`` after the year ends."""
    seed = hospital.seed if seed is None else seed
    rng = np.random.default_rng(np.random.SeedSequence([seed, 21]))
    dates = calendar(start, days)
    surge = surge_paths(1, days, hospital.surge_persistence, hospital.surge_sd, rng)[0]
    taken = dates[-1] + pd.Timedelta(days=lag_days)
    rows = []
    for w in hospital.ward_types:
        mult = surge if w.kind == EMERGENCY else 1.0
        counts = rng.poisson(daily_rate(w, dates, hospital.season_amplitude) * mult)
        day = np.repeat(np.arange(days), counts)
        los, long = sample_stays(w, len(day), rng)
        flags = spell_flags(w, los, long, rng)
        arrival = dates[day]
        df = pd.DataFrame({"WardTypeRequired": w.name, "Arrival Date": arrival, "true_los": los,
                           "long_stay": long, "PoD": "NEL" if w.kind == EMERGENCY else "EL", **flags})
        rows.append(df)
    truth = pd.concat(rows, ignore_index=True).sort_values("Arrival Date", kind="stable").reset_index(drop=True)
    discharge = truth["Arrival Date"] + pd.to_timedelta(truth["true_los"], unit="D")
    finished = discharge <= taken
    so_far = (taken - truth["Arrival Date"]).dt.days
    spells = truth.drop(columns=["true_los", "long_stay"]).copy()
    spells.insert(2, "Length of Stay", np.where(finished, truth["true_los"], so_far))
    spells.insert(3, "Finished", finished.to_numpy())
    # Excess days are only known once a spell has finished
    spells.loc[~spells["Finished"], "days_excess_discharge"] = 0
    return Extract(spells, dates[0], dates[-1], taken, truth)
