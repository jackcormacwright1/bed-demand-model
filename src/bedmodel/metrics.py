"""From simulated occupancy to bed numbers.

**Core** and **escalation** demand are the 85th and 95th percentiles of occupancy across every
simulated day of every simulated year. They are reported as whole beds: the smallest number of
beds b with occupancy at or below b on at least 85% (or 95%) of days, together with the share
of days b actually covers. The original model interpolated between whole numbers (and, for
ward types needing fewer than 5 beds, interpolated each day's occupancy to hourly steps), which
gives fractional beds that no ward can open.
"""

from __future__ import annotations

import numpy as np


def beds_for(occ: np.ndarray, q: float) -> int:
    """Smallest whole number of beds b with P(occupancy <= b) >= q (q between 0 and 1)."""
    return int(np.quantile(np.asarray(occ).ravel(), q, method="inverted_cdf"))


def coverage(occ: np.ndarray, beds: float) -> float:
    """Share of simulated days on which occupancy is at or below ``beds``."""
    return float(np.mean(np.asarray(occ).ravel() <= beds))


def interpolated(occ: np.ndarray, q: float) -> float:
    """The original model's percentile: linear interpolation between whole numbers."""
    return float(np.percentile(np.asarray(occ).ravel(), 100 * q))


def summary(occ: np.ndarray, core: float = 0.85, escalation: float = 0.95) -> dict:
    b_core, b_esc = beds_for(occ, core), beds_for(occ, escalation)
    return {
        "core_beds": b_core, "core_coverage": coverage(occ, b_core),
        "escalation_beds": b_esc, "escalation_coverage": coverage(occ, b_esc),
        "mean_occupancy": float(np.mean(occ)), "peak": int(np.max(occ)),
        "core_interpolated": interpolated(occ, core), "escalation_interpolated": interpolated(occ, escalation),
    }


def simulation_interval(occ: np.ndarray, q: float, n_boot: int = 200, rng: np.random.Generator | None = None,
                        level: float = 0.95) -> tuple[float, float]:
    """How much the interpolated percentile would move if the simulation were rerun with new random
    numbers: resample the simulated years with replacement. This is Monte Carlo noise only; it says
    nothing about how well one year of data pins down the hospital (see ``model.data_interval``)."""
    rng = rng or np.random.default_rng(0)
    n = occ.shape[0]
    vals = [interpolated(occ[rng.integers(0, n, n)], q) for _ in range(n_boot)]
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
