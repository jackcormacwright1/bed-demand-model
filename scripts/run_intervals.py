"""Are the uncertainty ranges honest? Checked against the truth on many synthetic hospitals.

For each hospital, the recommended model gives two 95% ranges for core beds (all ward types):

* the Monte Carlo range: how far the number moves if the simulation is rerun with new random
  numbers (the original model's confidence interval);
* the data range: how far it moves when the model is refitted to bootstrap copies of the data year
  (``model.data_interval``), with and without the uncertainty in the year's average surge level.

An honest 95% range should contain the true number in about 95% of hospitals.

    python scripts/run_intervals.py --hospitals 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bedmodel import RECOMMENDED, data_interval, fit, make_hospital, observe_year, simulate, total, true_occupancy  # noqa: E402
from bedmodel.metrics import beds_for, simulation_interval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospitals", type=int, default=20)
    ap.add_argument("--boot", type=int, default=30)
    args = ap.parse_args()
    rows, t0 = [], time.time()
    for h in range(args.hospitals):
        hosp = make_hospital(h)
        ex = observe_year(hosp)
        truth = beds_for(total(true_occupancy(hosp, ex.end + pd.Timedelta(days=1), n_sims=1000, seed=h + 500)), 0.85)
        occ = total(simulate(fit(ex, RECOMMENDED), n_sims=1000, seed=h))
        est = beds_for(occ, 0.85)
        mc_lo, mc_hi = simulation_interval(occ, 0.85, rng=np.random.default_rng(h))
        weeks = data_interval(ex, RECOMMENDED, n_boot=args.boot, n_sims=200, seed=h, year_level=False).total
        di = data_interval(ex, RECOMMENDED, n_boot=args.boot, n_sims=200, seed=h).total
        rows.append({"hospital": h, "truth": truth, "estimate": est, "mc_low": mc_lo, "mc_high": mc_hi,
                     "weeks_low": weeks["core_low"], "weeks_high": weeks["core_high"],
                     "data_low": di["core_low"], "data_high": di["core_high"]})
        r = rows[-1]
        print(f"hospital {h}: truth {truth}, estimate {est}, MC {mc_lo:.1f}-{mc_hi:.1f}, "
              f"weeks only {r['weeks_low']:.0f}-{r['weeks_high']:.0f}, "
              f"data {r['data_low']:.0f}-{r['data_high']:.0f} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["mc_contains"] = (df.mc_low <= df.truth) & (df.truth <= df.mc_high)
    df["weeks_contains"] = (df.weeks_low <= df.truth) & (df.truth <= df.weeks_high)
    df["data_contains"] = (df.data_low <= df.truth) & (df.truth <= df.data_high)
    out = ROOT / "results"
    df.to_csv(out / "intervals.csv", index=False)
    summary = (f"# Uncertainty ranges against the truth\n\n{len(df)} synthetic hospitals, recommended method, core beds "
               f"for all ward types.\n\n"
               f"| Range | Average width (beds) | Contains the true number |\n|---|---|---|\n"
               f"| Monte Carlo only (the original's interval) | {(df.mc_high - df.mc_low).mean():.1f} | "
               f"{df.mc_contains.sum()} of {len(df)} |\n"
               f"| Data year bootstrap: weeks redrawn from nearby weeks | {(df.weeks_high - df.weeks_low).mean():.1f} | "
               f"{df.weeks_contains.sum()} of {len(df)} |\n"
               f"| ...plus the uncertainty in the year's surge level (the recommended range) | "
               f"{(df.data_high - df.data_low).mean():.1f} | {df.data_contains.sum()} of {len(df)} |\n")
    (out / "intervals_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
