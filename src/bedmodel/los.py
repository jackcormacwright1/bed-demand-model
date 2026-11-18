"""Length-of-stay models.

Two ways of turning an extract into a model of how long the next patients will stay.

``FittedLoS`` (the original method). Excludes Tukey far-fence outliers (stays above
Q3 + 3 x IQR), fits five distributions (lognormal, exponential, gamma, Weibull, log-logistic),
each anchored at 0 and at 1 day, and keeps the one with the smallest Kolmogorov-Smirnov
statistic. If even that one has KS >= 0.10, it falls back to resampling the observed stays.
Sampled stays are rounded and capped at 90 days.

``SpellPool`` (the recommended method). Keeps every finished spell, with its flags, and draws
new patients' stays by resampling them. Spells still in hospital when the extract was taken are
used through Kaplan-Meier weights rather than dropped (see ``kaplan_meier_weights``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

ORIGINAL_CAP = 90
CANDIDATES = ["lognorm", "expon", "gamma", "weibull_min", "fisk"]
FLAG_COLUMNS = ["days_excess_discharge", "is_readmission", "sdec_eligible", "intended_daycase",
                "daycase_eligible", "virtual_ward_eligible"]


def tukey_fence(los: np.ndarray) -> float:
    """Tukey's far fence: Q3 + 3 x IQR."""
    q1, q3 = np.percentile(los, [25, 75])
    return float(q3 + 3 * (q3 - q1))


# ---------------------------------------------------------------- the original method
@dataclass
class FittedLoS:
    dist: str  # scipy name, or "empirical"
    params: tuple
    n: int  # spells given to the fit
    n_outliers: int  # of those, excluded from the fit by the Tukey fence
    ks: float
    quality: str  # good / marginal / empirical / insufficient
    cap: int | None = ORIGINAL_CAP
    values: np.ndarray | None = field(default=None, repr=False)  # for "empirical"

    @property
    def mean(self) -> float:
        if self.dist == "empirical":
            return float(np.mean(self.values))
        return float(getattr(stats, self.dist).mean(*self.params))

    def sample(self, n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """``n`` whole-day stays. ``scale`` < 1 shortens stays: the distribution's scale parameter
        is multiplied by it (so a 1-day anchor stays at 1 day), or, for the empirical fallback,
        each stay is multiplied by it and rounded."""
        if n == 0:
            return np.zeros(0, dtype=int)
        if self.dist == "empirical":
            raw = rng.choice(self.values, n).astype(float)
            if scale != 1.0:
                raw = np.maximum(1.0, np.round(raw * scale))
        else:
            p = list(self.params)
            p[-1] *= scale
            raw = getattr(stats, self.dist).rvs(*p, size=n, random_state=rng)
        out = np.maximum(np.round(raw), 1)
        if self.cap is not None:
            out = np.minimum(out, self.cap)
        return out.astype(int)


def fit_original(los: np.ndarray, tukey: bool = True, cap: int | None = ORIGINAL_CAP) -> FittedLoS:
    """The original multi-distribution fit, ported faithfully (with the Tukey exclusion optional)."""
    los = np.asarray(los, dtype=float)
    los = los[los > 0]
    if len(los) == 0:
        return FittedLoS("empirical", (), 0, 0, np.nan, "insufficient", cap, np.array([1.0]))
    fence = tukey_fence(los) if tukey else np.inf
    data = los[los <= fence]
    n_out = int(len(los) - len(data))
    if len(data) < 10:
        m = float(np.mean(data)) if len(data) else 1.0
        return FittedLoS("lognorm", (0.5, 0.0, m), len(los), n_out, np.nan, "insufficient", cap)
    best = (np.inf, None, None)
    for name in CANDIDATES:
        for loc in (0, 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = getattr(stats, name).fit(data, floc=loc)
                    mean = float(getattr(stats, name).mean(*p))
                    if not np.isfinite(mean) or mean <= 0:
                        continue
                    ks = float(stats.kstest(data, name, args=p).statistic)
            except Exception:  # noqa: BLE001 - a failed candidate is simply skipped, as in the original
                continue
            if ks < best[0]:
                best = (ks, name, p)
    ks, name, p = best
    if name is None or ks >= 0.10:
        # The fallback resamples every stay, outliers included, as the original did
        return FittedLoS("empirical", (), len(los), n_out, ks, "empirical", cap, los.copy())
    return FittedLoS(name, tuple(p), len(los), n_out, ks, "good" if ks < 0.05 else "marginal", cap)


# ---------------------------------------------------------------- the recommended method
def kaplan_meier_weights(los: np.ndarray, finished: np.ndarray) -> np.ndarray:
    """Probability weights on the finished spells that estimate the distribution of stays, using
    spells still in hospital as censored observations (Kaplan-Meier, 'redistribute to the right').

    Work through the stays from shortest to longest. Every spell starts with weight 1/n. When a
    spell still in hospital is reached (it has stayed at least this long, but we don't know how
    much longer), its weight is shared equally among all the spells with longer stays. The weight
    ends up on finished spells only, and adds up to 1. Weight left over after the longest finished
    stay (only possible if the longest stays are all still in hospital) stays on those unfinished
    spells, valued at their stay so far: a lower bound.

    At a tie, finished spells count as shorter than spells still in hospital, as is usual.
    """
    los = np.asarray(los, dtype=float)
    finished = np.asarray(finished, dtype=bool)
    n = len(los)
    order = np.lexsort((~finished, los))  # by stay; finished first within a tie
    w = np.full(n, 1.0 / n)
    out = np.zeros(n)
    remaining = n
    for pos, i in enumerate(order):
        remaining -= 1
        if finished[i] or remaining == 0:
            out[i] = w[i]
        else:  # pass this spell's weight to everyone after it
            w[order[pos + 1:]] += w[i] / remaining
    return out


@dataclass
class SpellPool:
    """Spells to resample, with probability weights. ``los`` is in whole days; the flag columns
    travel with each spell so the scenarios can act on individual patients."""
    los: np.ndarray
    weights: np.ndarray
    flags: pd.DataFrame
    n_spells: int  # spells in the extract for this ward type
    n_in_progress: int  # of those, still in hospital when it was taken
    n_excluded: int  # excluded by trimming (cutoff or Tukey fence)
    cap: int | None = None

    @property
    def mean(self) -> float:
        return float(np.sum(self.weights * self.los))

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Indices of ``n`` resampled spells."""
        return rng.choice(len(self.los), size=n, p=self.weights)

    def continuous(self, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A continuous stay for each drawn spell: a stay recorded as L whole days is somewhere in
        (L - 1, L], taken as uniform. Scaling these and rounding up again shortens stays by the
        intended share on average; scaling whole days and rounding leaves stays of up to 4 days
        unchanged (a 10% cut takes 4 days to 3.6, which rounds back to 4), so it cuts too little."""
        return self.los[idx] - rng.random(len(idx))

    def stretch_for(self, factor: float) -> float:
        """The multiplier s to apply to continuous stays so that the average whole-day stay becomes
        ``factor`` x what it is now. A stay can't fall below one night, so s is a little below
        ``factor``: the longer stays shorten a little more to make up for the 1-day stays that can't.
        Found by bisection, with the uniform position within each day averaged over 64 points."""
        if factor == 1.0:
            return 1.0
        u = (np.arange(64) + 0.5) / 64
        cont = self.los[:, None] - u[None, :]
        target = factor * self.mean

        def mean_at(s: float) -> float:
            return float(np.sum(self.weights * np.maximum(np.ceil(cont * s), 1).mean(axis=1)))
        lo, hi = 0.0, max(2.0, 2 * factor)
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if mean_at(mid) < target else (lo, mid)
        return (lo + hi) / 2

    def sample(self, n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """``n`` whole-day stays; ``scale`` multiplies the average whole-day stay (see ``stretch_for``)."""
        if n == 0:
            return np.zeros(0, dtype=int)
        idx = self.draw(n, rng)
        if scale == 1.0:
            out = self.los[idx]
        else:
            out = np.maximum(np.ceil(self.continuous(idx, rng) * self.stretch_for(scale)), 1)
        if self.cap is not None:
            out = np.minimum(out, self.cap)
        return out.astype(int)


def make_pool(spells: pd.DataFrame, in_progress: bool = True, max_los: int | None = None,
              tukey: bool = False, cap: int | None = None) -> SpellPool:
    """Build the pool for one ward type's spells.

    ``in_progress``: use spells still in hospital via Kaplan-Meier (otherwise they are dropped).
    ``max_los``: drop finished spells longer than this (the original extract kept 1-60 days).
    ``tukey``: drop stays above the Tukey far fence.
    """
    s = spells if in_progress else spells[spells["Finished"]]
    n_all = len(s)
    keep = np.ones(len(s), dtype=bool)
    if max_los is not None:
        keep &= ~(s["Finished"].to_numpy() & (s["Length of Stay"].to_numpy() > max_los))
    if tukey and keep.any():
        fin = s["Finished"].to_numpy() & keep
        fence = tukey_fence(s.loc[fin, "Length of Stay"].to_numpy()) if fin.any() else np.inf
        keep &= ~(s["Finished"].to_numpy() & (s["Length of Stay"].to_numpy() > fence))
    s = s[keep]
    los = s["Length of Stay"].to_numpy().astype(int)
    fin = s["Finished"].to_numpy().astype(bool)
    if len(s) == 0:
        return SpellPool(np.array([1]), np.array([1.0]), pd.DataFrame({c: [0] for c in FLAG_COLUMNS}), 0, 0, n_all, cap)
    w = kaplan_meier_weights(los, fin) if in_progress else np.full(len(s), 1.0 / len(s))
    use = w > 0
    flags = s.loc[use, [c for c in FLAG_COLUMNS if c in s.columns]].reset_index(drop=True)
    return SpellPool(los[use], w[use] / w[use].sum(), flags, n_all, int((~fin).sum()), n_all - len(s), cap)
