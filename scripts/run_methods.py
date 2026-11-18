"""What each of the original method's choices does to the bed numbers, against the true answer.

For each of ``--hospitals`` synthetic hospitals: one year of data is observed, every method
variant is fitted to it and simulated, and the results are compared with the true occupancy of the
following year (simulated from the process that generated the data).

The variants go from the original method to the recommended one, one change at a time, so each
row shows what that change adds on top of the ones before it. Two more rows put the original's
60-day cutoff, and its Tukey exclusion, back into the recommended method on their own.

Outputs: results/methods.csv (one row per hospital and variant) and results/methods_summary.md.

    python scripts/run_methods.py --hospitals 20 --sims 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bedmodel import ORIGINAL, RECOMMENDED, fit, make_hospital, observe_year, simulate, total, true_occupancy  # noqa: E402
from bedmodel.metrics import beds_for, coverage  # noqa: E402

STEPS = [
    ("Original method", ORIGINAL),
    ("+ keep stays over 60 days", replace(ORIGINAL, max_los=None)),
    ("+ no Tukey exclusion", replace(ORIGINAL, max_los=None, tukey=False)),
    ("+ no 90-day cap", replace(ORIGINAL, max_los=None, tukey=False, cap=None)),
    ("+ 365-day warm-up", replace(ORIGINAL, max_los=None, tukey=False, cap=None, warmup=365)),
    ("+ resample spells instead of fitting", replace(RECOMMENDED, in_progress=False, seasonality=False, surges=False)),
    ("+ use spells still in hospital (Kaplan-Meier)", replace(RECOMMENDED, seasonality=False, surges=False)),
    ("+ seasonality", replace(RECOMMENDED, surges=False)),
    ("+ surges (recommended method)", RECOMMENDED),
]
EXTRA = [
    ("Recommended with the 60-day cutoff", replace(RECOMMENDED, max_los=60)),
    ("Recommended with the Tukey exclusion", replace(RECOMMENDED, tukey=True)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hospitals", type=int, default=20)
    ap.add_argument("--sims", type=int, default=1000)
    args = ap.parse_args()
    rows = []
    t0 = time.time()
    for hseed in range(args.hospitals):
        h = make_hospital(hseed)
        ex = observe_year(h)
        truth = true_occupancy(h, ex.end + pd.Timedelta(days=1), n_sims=args.sims, seed=hseed + 500)
        t_tot = total(truth)
        true_core, true_esc = beds_for(t_tot, 0.85), beds_for(t_tot, 0.95)
        true_ward = {k: (beds_for(v, 0.85), beds_for(v, 0.95)) for k, v in truth.items()}
        for label, method in STEPS + EXTRA:
            occ = simulate(fit(ex, method), n_sims=args.sims, seed=hseed)
            tot = total(occ)
            core, esc = beds_for(tot, 0.85), beds_for(tot, 0.95)
            ward_err = [100 * (beds_for(occ[k], 0.85) - true_ward[k][0]) / max(true_ward[k][0], 1) for k in occ]
            ward_cov = [coverage(truth[k], beds_for(occ[k], 0.85)) for k in occ]
            rows.append({
                "hospital": hseed, "variant": label,
                "core": core, "true_core": true_core, "escalation": esc, "true_escalation": true_esc,
                "core_error_pct": 100 * (core - true_core) / true_core,
                "escalation_error_pct": 100 * (esc - true_esc) / true_esc,
                "core_days_covered": coverage(t_tot, core), "escalation_days_covered": coverage(t_tot, esc),
                "ward_core_abs_error_pct": float(np.mean(np.abs(ward_err))),
                "ward_core_days_covered": float(np.mean(ward_cov)),
                "mean_occupancy": float(tot.mean()), "true_mean_occupancy": float(t_tot.mean()),
            })
        print(f"hospital {hseed}: truth {true_core}/{true_esc}, original {rows[-len(STEPS + EXTRA)]['core']}, "
              f"recommended {rows[-len(EXTRA) - 1]['core']} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "methods.csv", index=False)
    write_summary(df, out / "methods_summary.md", args)


def write_summary(df: pd.DataFrame, path: Path, args) -> None:
    order = [label for label, _ in STEPS + EXTRA]
    g = df.groupby("variant")
    n = df["hospital"].nunique()

    def ms(col: str, scale: float = 1.0, dec: int = 1) -> pd.Series:
        m, se = g[col].mean() * scale, g[col].std() * scale / np.sqrt(n)
        return (m.round(dec).astype(str) + " ± " + (2 * se).round(dec).astype(str)).reindex(order)

    tab = pd.DataFrame({
        "Core beds, all ward types: error vs truth (%)": ms("core_error_pct"),
        "Days the core number covers (target 85%)": ms("core_days_covered", 100),
        "Escalation beds: error vs truth (%)": ms("escalation_error_pct"),
        "Days the escalation number covers (target 95%)": ms("escalation_days_covered", 100),
        "Ward types: mean absolute error in core beds (%)": ms("ward_core_abs_error_pct"),
        "Ward types: days their core number covers": ms("ward_core_days_covered", 100),
    })
    lines = ["# Methods against the truth\n",
             f"{n} synthetic hospitals, {args.sims} simulated years each for the models and the truth. Each figure is the "
             "mean across hospitals ± two standard errors. Truth: the following year simulated from the process that "
             "generated the data. 'Days covered': the share of the true year's days on which occupancy is within the "
             "number of beds the model recommends.\n",
             tab.to_markdown(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(tab.to_string())


if __name__ == "__main__":
    main()
