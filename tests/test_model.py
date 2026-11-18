from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bedmodel import ORIGINAL, RECOMMENDED, WARD_GROUPS, Scenario, fit, make_hospital, observe_year, simulate, total
from bedmodel.los import fit_original
from bedmodel.metrics import beds_for
from bedmodel.model import pool_sampler, resample_year, usable_spells


@pytest.fixture(scope="module")
def extract():
    return observe_year(make_hospital(0))


def test_the_extract_shows_spells_as_they_stood_when_it_was_taken(extract):
    s, t = extract.spells, extract.truth
    assert len(s) == len(t)
    assert s["Arrival Date"].between(extract.start, extract.end).all()
    unfinished = ~s["Finished"]
    assert unfinished.any()
    assert (s.loc[unfinished, "Length of Stay"] < t.loc[unfinished, "true_los"]).all()
    assert (s.loc[~unfinished, "Length of Stay"] == t.loc[~unfinished, "true_los"]).all()
    assert (t["true_los"] > 60).any()  # long stays exist, so the cutoff matters


def test_the_original_method_sees_finished_spells_of_60_days_or_less(extract):
    used = usable_spells(extract, ORIGINAL)
    assert used["Finished"].all() and used["Length of Stay"].max() <= 60
    assert len(usable_spells(extract, RECOMMENDED)) == len(extract.spells)


def test_the_original_fit_and_its_cap():
    rng = np.random.default_rng(1)
    los = np.ceil(rng.lognormal(np.log(4), 0.6, 3000))
    f = fit_original(los)
    assert f.quality in ("good", "marginal", "empirical")
    assert f.sample(10000, rng).max() <= 90
    assert fit_original(los, tukey=False).n_outliers == 0


def test_scenarios_act_on_individual_patients(extract):
    pool = fit(extract).wards["General surgery - emergency - male"].los
    rng = np.random.default_rng(2)
    base = pool_sampler(pool, Scenario())(50000, np.random.default_rng(3))
    divert = pool_sampler(pool, Scenario(divert=(("sdec_eligible", 1.0),)))(50000, np.random.default_rng(3))
    removed = (divert == 0).mean()
    expected = np.sum(pool.weights * pool.flags["sdec_eligible"].to_numpy())
    assert removed == pytest.approx(expected, abs=0.01)
    assert (base[divert == 0] <= 2).all()  # SDEC-eligible patients are short stays
    ready = pool_sampler(pool, Scenario(discharge_ready=1.0))(50000, np.random.default_rng(3))
    assert ready.mean() < base.mean() and ready.min() >= 1
    capped = pool_sampler(pool, Scenario(long_stay=(30, 21, 1.0)))(50000, rng)
    assert capped.max() <= max(29, 21)


def test_ward_groups_need_fewer_beds_than_their_members_separately(extract):
    occ = simulate(fit(extract), n_sims=300, seed=1)
    for members in WARD_GROUPS.values():
        assert beds_for(total(occ, list(members)), 0.85) <= sum(beds_for(occ[m], 0.85) for m in members)


def test_same_seed_same_answer(extract):
    m = fit(extract, RECOMMENDED)
    a, b = simulate(m, n_sims=50, seed=4), simulate(m, n_sims=50, seed=4)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_seasonality_and_surges_are_recovered_from_a_long_run_of_data():
    ex = observe_year(make_hospital(5), days=5 * 365)
    m = fit(ex, RECOMMENDED)
    amp, peak = m.season.emergency_peak
    assert amp == pytest.approx(0.10, abs=0.04)
    assert min(abs(peak - 15), 365 - abs(peak - 15)) < 30
    assert m.surges.persistence > 0.6 and 0.05 < m.surges.sd < 0.15
    # elective: August and Christmas dips
    assert m.season.elective[7] < 0.9 and m.season.elective[11] < m.season.elective[10]


def test_a_bootstrap_year_keeps_weekdays_and_dates_in_place(extract):
    ex = resample_year(extract, np.random.default_rng(6))
    s = ex.spells
    assert s["Arrival Date"].between(extract.start, extract.end).all()
    assert len(s) == pytest.approx(len(extract.spells), rel=0.05)
    orig_dow = extract.spells["Arrival Date"].dt.dayofweek.value_counts(normalize=True).sort_index()
    new_dow = s["Arrival Date"].dt.dayofweek.value_counts(normalize=True).sort_index()
    assert np.allclose(orig_dow, new_dow, atol=0.01)
    elective = s[s["PoD"] == "EL"]
    assert (elective["Arrival Date"].dt.dayofweek < 5).all()  # elective admissions still on weekdays


def test_the_recommended_method_lands_near_the_truth_and_the_original_well_below(extract):
    from bedmodel import true_occupancy
    h = make_hospital(0)
    truth = total(true_occupancy(h, extract.end + pd.Timedelta(days=1), n_sims=300, seed=9))
    rec = total(simulate(fit(extract, RECOMMENDED), n_sims=300, seed=1))
    orig = total(simulate(fit(extract, ORIGINAL), n_sims=300, seed=1))
    assert beds_for(rec, 0.85) == pytest.approx(beds_for(truth, 0.85), rel=0.08)
    assert beds_for(orig, 0.85) < 0.9 * beds_for(truth, 0.85)
    # keeping the long stays alone moves the original up
    keep = total(simulate(fit(extract, replace(ORIGINAL, max_los=None)), n_sims=300, seed=1))
    assert beds_for(keep, 0.85) > beds_for(orig, 0.85)


def test_discharge_targets_treat_whole_days_as_continuous():
    from bedmodel import discharge_targets
    s = pd.DataFrame({"WardTypeRequired": "W", "Length of Stay": [3, 8, 8, 20], "Finished": True,
                      "Arrival Date": pd.to_datetime(["2025-04-01", "2025-04-08", "2025-04-15", "2025-04-22"])})
    t = discharge_targets(s, 0.05).iloc[0]
    weeks = 3.0
    # each 8-day stay becomes 7 days or less with probability 1 - (8 - 7/0.95) = 0.368; the 20-day stay never does
    assert t["extra_short_stay_per_week"] == pytest.approx(2 * (1 - (8 - 7 / 0.95)) / weeks)
    assert t["short_stay_per_week"] == pytest.approx(1 / weeks)


def test_the_data_interval_brackets_the_estimate(extract):
    from bedmodel import data_interval
    di = data_interval(extract, RECOMMENDED, n_boot=6, n_sims=40, seed=1)
    assert di.total["core_low"] <= di.total["core_high"]
    assert len(di.per_ward) == len(extract.spells["WardTypeRequired"].unique())
    est = beds_for(total(simulate(fit(extract), n_sims=200, seed=1)), 0.85)
    assert di.total["core_low"] - 10 <= est <= di.total["core_high"] + 10


def test_the_projection_grows_with_admissions_and_shrinks_with_shorter_stays(extract):
    m = fit(extract)
    base = total(simulate(m, n_sims=100, seed=2)).mean()
    grown = total(simulate(m, n_sims=100, seed=2, years_ahead=5, growth=0.02)).mean()
    shorter = total(simulate(m, n_sims=100, seed=2, years_ahead=5, los_improvement=0.02)).mean()
    assert grown == pytest.approx(base * 1.02 ** 5, rel=0.02)
    assert shorter == pytest.approx(base * 0.98 ** 5, rel=0.02)
