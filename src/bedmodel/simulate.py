"""The Monte Carlo engine: simulated bed occupancy, day by day.

For each simulated year and each day t, admissions are Poisson with mean rate(t) x surge(t),
and each admitted patient gets a length of stay L from the ward type's model. The patient
occupies a bed on the midnights of days t, t + 1, ..., t + L - 1, so

    occupancy(t) = admissions up to and including day t - discharges up to and including day t,

with the discharge on day t + L. That is two running totals, computed for every simulated year at
once with ``numpy.bincount`` and ``cumsum``, with no loop over patients.

The simulation starts ``warmup`` days before the year being measured, with no one in hospital,
and the warm-up days are discarded. The warm-up repeats the end of the measured year's pattern.
It has to be at least as long as the longest stays that matter, or the year starts with too few
long-stay patients in hospital. The original model used 30 days.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# sampler(n, rng, idx_kind) -> stays (whole days, 0 = not admitted after all); see model.py
Sampler = Callable[[int, np.random.Generator], np.ndarray]


def extend_back(x: np.ndarray, warmup: int) -> np.ndarray:
    """Prepend ``warmup`` days to a daily series by repeating it cyclically (the end of the year
    stands in for the days before it)."""
    if warmup <= 0:
        return np.asarray(x)
    reps = int(np.ceil(warmup / len(x)))
    tail = np.tile(x, reps)[-warmup:]
    return np.concatenate([tail, x])


def occupancy(rate: np.ndarray, sampler: Sampler, n_sims: int, rng: np.random.Generator,
              warmup: int = 365, multiplier: np.ndarray | None = None) -> np.ndarray:
    """Simulated midnight occupancy, shape (n_sims, len(rate)).

    ``rate``: expected admissions on each day of the measured year.
    ``multiplier``: optional (n_sims, warmup + len(rate)) surge paths, shared between ward types.
    """
    T = len(rate)
    D = warmup + T
    lam = np.broadcast_to(extend_back(rate, warmup), (n_sims, D))
    if multiplier is not None:
        lam = lam * multiplier
    counts = rng.poisson(lam)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((n_sims, T), dtype=np.int32)
    flat_day = np.repeat(np.arange(n_sims * D), counts.ravel())  # sim * D + day, one per admission
    stays = sampler(total, rng)
    sim, day = np.divmod(flat_day, D)
    return census(sim, day, stays, n_sims, D)[:, warmup:]


def census(sim: np.ndarray, day: np.ndarray, stays: np.ndarray, n_sims: int, days: int) -> np.ndarray:
    """Midnight occupancy, shape (n_sims, days), from each admission's simulation, day and stay.
    A stay of 0 (an admission removed by a scenario) occupies nothing."""
    keep = stays > 0
    sim, day, stays = sim[keep], day[keep], stays[keep]
    out_day = np.minimum(day + stays, days)  # a discharge after the horizon is never counted
    width = days + 1
    arr = np.bincount(sim * width + day, minlength=n_sims * width).reshape(n_sims, width)
    dep = np.bincount(sim * width + out_day, minlength=n_sims * width).reshape(n_sims, width)
    return np.cumsum(arr - dep, axis=1)[:, :days].astype(np.int32)
