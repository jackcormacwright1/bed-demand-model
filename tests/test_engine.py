import numpy as np
import pytest

from bedmodel.los import kaplan_meier_weights, make_pool, tukey_fence
from bedmodel.metrics import beds_for, coverage, interpolated
from bedmodel.simulate import census, extend_back, occupancy


def test_a_stay_of_three_days_occupies_three_midnights():
    occ = census(np.array([0]), np.array([2]), np.array([3]), n_sims=1, days=8)
    assert occ[0].tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    # a stay of 0 (an admission a scenario removed) occupies nothing; one running past the end is still in
    occ = census(np.array([0, 0]), np.array([1, 6]), np.array([0, 5]), n_sims=1, days=8)
    assert occ[0].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]


def test_mean_occupancy_follows_littles_law():
    # admissions a day x stay = average beds occupied, once the warm-up is long enough
    rate = np.full(365, 4.0)
    occ = occupancy(rate, lambda n, rng: np.full(n, 6), 400, np.random.default_rng(1), warmup=60)
    assert occ.mean() == pytest.approx(24.0, rel=0.01)
    # Poisson admissions and fixed stays: occupancy on a day is Poisson(24), so its variance is 24 too
    assert occ[:, 100].var() == pytest.approx(24.0, rel=0.15)


def test_a_short_warm_up_starts_the_year_with_too_few_long_stays():
    rate = np.full(365, 1.0)
    stay = lambda n, rng: np.full(n, 90)  # noqa: E731
    short = occupancy(rate, stay, 300, np.random.default_rng(2), warmup=30)
    full = occupancy(rate, stay, 300, np.random.default_rng(2), warmup=365)
    assert short[:, 0].mean() == pytest.approx(31, rel=0.05)  # only the 30 warm-up days' admissions (+ day 0)
    assert full[:, 0].mean() == pytest.approx(90, rel=0.05)
    assert short[:, 200].mean() == pytest.approx(90, rel=0.05)  # it catches up after 90 days


def test_extend_back_repeats_the_end_of_the_year():
    assert extend_back(np.array([1, 2, 3]), 4).tolist() == [3, 1, 2, 3, 1, 2, 3]


def test_kaplan_meier_weights_match_the_hand_calculation():
    # stays 1, 2, 2 (still in hospital), 3: the censored spell's quarter passes to the 3-day stay
    w = kaplan_meier_weights(np.array([1, 2, 2, 3]), np.array([True, True, False, True]))
    assert w.tolist() == pytest.approx([0.25, 0.25, 0.0, 0.5])
    # no one still in hospital: equal weights
    assert kaplan_meier_weights(np.arange(1, 5), np.ones(4, bool)).tolist() == pytest.approx([0.25] * 4)


def test_kaplan_meier_recovers_stays_that_finished_only_data_misses():
    rng = np.random.default_rng(3)
    true = np.ceil(rng.lognormal(np.log(5), 0.9, 20000)).astype(int)
    follow_up = rng.uniform(5, 60, len(true))  # admitted at different times before the extract
    finished = true <= follow_up
    seen = np.where(finished, true, np.floor(follow_up)).astype(int)
    w = kaplan_meier_weights(seen, finished)
    assert np.sum(w * seen) == pytest.approx(true.mean(), rel=0.02)
    assert seen[finished].mean() < 0.9 * true.mean()


def test_tukey_fence_and_trimming():
    assert tukey_fence(np.array([1, 2, 3, 4, 5, 6, 7, 8])) == pytest.approx(6.25 + 3 * 3.5)
    import pandas as pd
    s = pd.DataFrame({"Length of Stay": [1, 2, 3, 4, 90], "Finished": True, "days_excess_discharge": 0})
    assert make_pool(s, tukey=True).los.max() == 4
    assert make_pool(s).los.max() == 90


def test_whole_beds_and_coverage():
    occ = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
    assert beds_for(occ, 0.5) == 2 and coverage(occ, 2) == 0.5
    assert beds_for(occ, 0.85) == 4 and interpolated(occ, 0.85) == pytest.approx(3.95)


def test_shortening_continuous_stays_cuts_the_average_by_the_intended_share():
    import pandas as pd
    rng = np.random.default_rng(4)
    los = np.ceil(rng.lognormal(np.log(3), 0.7, 5000)).astype(int)
    pool = make_pool(pd.DataFrame({"Length of Stay": los, "Finished": True, "days_excess_discharge": 0}))
    base = pool.sample(200000, np.random.default_rng(5)).mean()
    cut = pool.sample(200000, np.random.default_rng(5), scale=0.9).mean()
    assert cut / base == pytest.approx(0.9, abs=0.005)
    # a 1-day stay can't get shorter, so continuous stays are cut a little more than 10%
    assert pool.stretch_for(0.9) < 0.9
    # rounding whole days instead leaves stays of up to 4 days as they were, so it cuts too little
    rounded = np.maximum(np.round(pool.los * 0.9), 1)
    assert np.all(rounded[pool.los <= 4] == pool.los[pool.los <= 4])
    assert np.sum(pool.weights * rounded) / pool.mean > 0.91
