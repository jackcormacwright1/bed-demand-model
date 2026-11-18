"""Admissions: how many patients arrive each day.

From the extract, for each ward type:

* ``lambda``: admissions a day over the year (total admitted / days in the year);
* day-of-week weights: share of admissions on each weekday x 7, so they average 1.

Optionally, shared across ward types of the same kind (emergency or elective), because one
ward type's year is too noisy to estimate these well on its own:

* **seasonality**: for emergency admissions, one smooth yearly wave (a cosine and a sine fitted by
  least squares); for elective admissions, a factor for each month;
* **surges**: busy and quiet spells in emergency admissions beyond what chance and the season
  explain, as a slowly wandering multiplier (see ``fit_surges``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .synthetic import ELECTIVE, EMERGENCY


@dataclass
class Arrivals:
    per_day: float
    dow: np.ndarray  # (7,), Monday first, averages 1
    kind: str

    def rate(self, dates: pd.DatetimeIndex, season: np.ndarray | None = None) -> np.ndarray:
        r = self.per_day * self.dow[dates.dayofweek.to_numpy()]
        return r if season is None else r * season


@dataclass
class Seasonality:
    """Emergency: 1 + a cos(2 pi d / 365.25) + b sin(2 pi d / 365.25), d the day of the year.
    Elective: one factor per month."""
    emergency: tuple[float, float] = (0.0, 0.0)
    elective: np.ndarray | None = None  # (12,), January first

    def factor(self, dates: pd.DatetimeIndex, kind: str) -> np.ndarray:
        if kind == EMERGENCY:
            a, b = self.emergency
            ang = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
            return 1 + a * np.cos(ang) + b * np.sin(ang)
        if self.elective is None:
            return np.ones(len(dates))
        return self.elective[dates.month.to_numpy() - 1]

    @property
    def emergency_peak(self) -> tuple[float, int]:
        """Size of the emergency wave and the day of the year it peaks."""
        a, b = self.emergency
        amp = float(np.hypot(a, b))
        return amp, int(round((np.arctan2(b, a) / (2 * np.pi) * 365.25) % 365.25))


@dataclass
class Surges:
    """A shared emergency multiplier exp(x_t - sd^2/2) with x_t = persistence x_{t-1} + e_t."""
    persistence: float = 0.0
    sd: float = 0.0


def fit_arrivals(spells: pd.DataFrame, start: pd.Timestamp, days: int, kind: str) -> Arrivals:
    dates = pd.to_datetime(spells["Arrival Date"])
    per_day = len(spells) / days
    counts = dates.dt.dayofweek.value_counts().reindex(range(7), fill_value=0).to_numpy(float)
    # Weekdays occur 52 or 53 times in a year: divide by the number of each weekday in the period
    n_wd = pd.date_range(start, periods=days, freq="D").dayofweek.value_counts().reindex(range(7)).to_numpy(float)
    per_wd = counts / n_wd
    dow = per_wd / per_wd.mean() if per_wd.sum() > 0 else np.ones(7)
    return Arrivals(per_day, dow, kind)


def daily_counts(spells: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    return pd.to_datetime(spells["Arrival Date"]).value_counts().reindex(dates, fill_value=0).to_numpy(float)


def fit_seasonality(spells: pd.DataFrame, arrivals: dict[str, Arrivals], dates: pd.DatetimeIndex) -> Seasonality:
    """Fit the shared seasonal patterns to the daily totals of each kind of ward type."""
    out = Seasonality()
    for kind in (EMERGENCY, ELECTIVE):
        names = [k for k, a in arrivals.items() if a.kind == kind]
        if not names:
            continue
        sub = spells[spells["WardTypeRequired"].isin(names)]
        obs = daily_counts(sub, dates)
        exp = sum(arrivals[k].rate(dates) for k in names)
        if kind == EMERGENCY:
            ang = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
            X = np.column_stack([np.cos(ang), np.sin(ang)])
            ok = exp > 0
            y = obs[ok] / exp[ok] - 1
            coef, *_ = np.linalg.lstsq(X[ok], y, rcond=None)
            out.emergency = (float(coef[0]), float(coef[1]))
        else:
            f = np.ones(12)
            months = dates.month.to_numpy()
            for m in range(1, 13):
                sel = (months == m) & (exp > 0)
                if sel.any() and exp[sel].sum() > 0:
                    f[m - 1] = obs[sel].sum() / exp[sel].sum()
            out.elective = f
    return out


def block_factor(persistence: float, k: int = 7) -> float:
    """Variance of the average of k consecutive days of an AR(1) process, as a share of the daily
    variance: (1/k^2) x sum over pairs of days i, j of persistence^|i - j|. It is 1/k when days are
    independent and 1 when they move together."""
    lags = np.arange(1, k)
    return float((k + 2 * np.sum((k - lags) * persistence ** lags)) / k ** 2)


def fit_surges(spells: pd.DataFrame, arrivals: dict[str, Arrivals], season: Seasonality | None,
               dates: pd.DatetimeIndex) -> Surges:
    """Estimate the shared emergency surge from the daily emergency totals by the method of moments.

    With e_t the expected admissions on day t (rate x season) and o_t those observed, the relative
    excess d_t = (o_t - e_t) / e_t has variance 1/e_t from chance (Poisson) plus s^2 from the surge.
    So s^2 = mean(d_t^2 - 1/e_t).

    How long surges last is estimated from whole weeks, which is steadier than from one day to the
    next: for each week, D_w = (sum of o - sum of e) / sum of e, whose surge part has variance
    s^2 x B(rho), where B (``block_factor``) runs from 1/7 (no persistence) to 1 (surges last for
    weeks). Solving mean(D_w^2 - 1/sum of e) = s^2 B(rho) for rho gives the persistence. The surge
    multiplier's log-scale standard deviation is sqrt(log(1 + s^2)).

    One year holds only a dozen or so independent surges, so these estimates are rough; the
    bootstrap interval (``model.data_interval``) carries that uncertainty into the bed numbers.
    """
    names = [k for k, a in arrivals.items() if a.kind == EMERGENCY]
    if not names:
        return Surges()
    sub = spells[spells["WardTypeRequired"].isin(names)]
    obs = daily_counts(sub, dates)
    exp = sum(arrivals[k].rate(dates, season.factor(dates, EMERGENCY) if season else None) for k in names)
    ok = exp > 0
    d = (obs[ok] - exp[ok]) / exp[ok]
    s2 = float(np.mean(d ** 2 - 1 / exp[ok]))
    if s2 <= 0:
        return Surges()
    n_w = len(obs) // 7
    o_w = obs[: 7 * n_w].reshape(n_w, 7).sum(axis=1)
    e_w = exp[: 7 * n_w].reshape(n_w, 7).sum(axis=1)
    v7 = float(np.mean(((o_w - e_w) / e_w) ** 2 - 1 / e_w))
    ratio = float(np.clip(v7 / s2, 1 / 7, block_factor(0.98)))
    lo, hi = 0.0, 0.98
    for _ in range(50):  # bisection: block_factor rises with persistence
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if block_factor(mid) < ratio else (lo, mid)
    return Surges(persistence=float((lo + hi) / 2), sd=float(np.sqrt(np.log1p(s2))))
