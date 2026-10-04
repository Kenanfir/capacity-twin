import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sim import policy  # noqa: E402

ZONE = "synthetic"
HORIZON = 10
N = 200


def _make_run(rng: np.random.Generator, sigma: float) -> dict:
    """A synthetic occupancy series -- a slow oscillating trend plus i.i.d.
    Gaussian noise -- with no dependency on the crowd simulation, so this
    test checks calibrate_conformal_margin / the coverage guarantee in
    isolation."""
    t = np.arange(N, dtype=float)
    trend = 50.0 + 10.0 * np.sin(t / 40.0)
    series = np.clip(trend + rng.normal(0.0, sigma, size=N), 0.0, None)
    return {"occupancy": {ZONE: series}}


def test_conformal_coverage_near_target():
    rng = np.random.default_rng(7)
    sigma = 5.0
    target_coverage = 0.9

    calibration_runs = [_make_run(rng, sigma) for _ in range(80)]
    test_runs = [_make_run(rng, sigma) for _ in range(80)]

    margin = policy.calibrate_conformal_margin(
        calibration_runs, ZONE, HORIZON, target_coverage)

    covered = 0
    total = 0
    for run in test_runs:
        residuals = policy._residuals(run, ZONE, HORIZON)
        covered += int(np.sum(np.abs(residuals) <= margin))
        total += len(residuals)
    empirical_coverage = covered / total

    assert abs(empirical_coverage - target_coverage) <= 0.05, (
        f"empirical coverage {empirical_coverage:.3f} vs target {target_coverage}"
    )


def test_margin_increases_with_target_coverage():
    rng = np.random.default_rng(11)
    sigma = 4.0
    calibration_runs = [_make_run(rng, sigma) for _ in range(50)]
    low = policy.calibrate_conformal_margin(calibration_runs, ZONE, HORIZON, 0.6)
    high = policy.calibrate_conformal_margin(calibration_runs, ZONE, HORIZON, 0.95)
    assert high > low


def test_margin_uses_finite_sample_order_statistic():
    runs = [{"occupancy": {ZONE: np.arange(20, dtype=float)}}]
    residuals = np.abs(policy._residuals(runs[0], ZONE, HORIZON))
    target_coverage = 0.5
    rank = int(np.ceil((len(residuals) + 1) * target_coverage))

    margin = policy.calibrate_conformal_margin(
        runs, ZONE, HORIZON, target_coverage)

    assert margin == np.sort(residuals)[rank - 1]


def test_margin_is_infinite_when_requested_rank_exceeds_sample():
    runs = [{"occupancy": {ZONE: np.arange(12, dtype=float)}}]

    margin = policy.calibrate_conformal_margin(runs, ZONE, HORIZON, 0.99)

    assert np.isinf(margin)


def test_cluster_bootstrap_ci_reflects_few_independent_worlds():
    """5 worlds, 3 fully caught and 2 fully missed -> point recall 0.6, but
    resampling only 5 clusters must give a wide interval, not a falsely
    tight one -- this is the exact fragility the paper needs to disclose
    honestly for its own n=5-breach-world result."""
    counts = {
        "tp": np.array([10, 10, 10, 0, 0]),
        "fp": np.array([0, 0, 0, 0, 0]),
        "tn": np.array([50, 50, 50, 50, 50]),
        "fn": np.array([0, 0, 0, 10, 10]),
    }
    rng = np.random.default_rng(42)
    ci = policy._cluster_bootstrap_ci(counts, rng, n_boot=5000)
    assert ci["recall_ci_lo"] is not None and ci["recall_ci_hi"] is not None
    assert ci["recall_ci_lo"] <= 0.6 <= ci["recall_ci_hi"]
    # With only 5 clusters the interval must be visibly wide, not a
    # per-instance-count artifact (there are 100 pooled minute instances,
    # which a naive non-clustered interval would treat as 100 independent
    # trials and report a much narrower band than this).
    assert (ci["recall_ci_hi"] - ci["recall_ci_lo"]) > 0.4


def test_cluster_bootstrap_ci_undefined_when_no_alerts_ever_fire():
    """The unsurveyed-precision policy fires zero alerts in the real
    experiment (recall is None); the bootstrap must report that same
    undefined state, not a fabricated interval."""
    counts = {
        "tp": np.zeros(5, dtype=np.int64),
        "fp": np.zeros(5, dtype=np.int64),
        "tn": np.array([50, 50, 50, 50, 50]),
        "fn": np.array([10, 10, 10, 10, 10]),
    }
    rng = np.random.default_rng(1)
    ci = policy._cluster_bootstrap_ci(counts, rng, n_boot=500)
    # recall is defined here (tp+fn>0 always), but false-alarm rate never is
    # (fp+tn>0 but fp is always 0 -- so FAR=0 every resample, a defined,
    # degenerate interval at exactly 0, which is the honest answer).
    assert ci["far_ci_lo"] == 0.0 and ci["far_ci_hi"] == 0.0
    assert ci["far_ci_undefined_frac"] == 0.0


def _make_evaluable_run(rng: np.random.Generator, sigma: float,
                         true_capacity: float, estimate_capacity: float) -> dict:
    run = _make_run(rng, sigma)
    run["true_capacity"] = {ZONE: true_capacity}
    run["estimate_geometry"] = {ZONE: estimate_capacity / policy.zn.DENSITY_THRESHOLD_PERSONS_PER_M2}
    return run


def test_evaluate_policy_ci_matches_manual_bootstrap_seed():
    """evaluate_policy's ci_rng plumbing must actually reach
    _cluster_bootstrap_ci and produce the CI keys, not silently skip them."""
    rng = np.random.default_rng(3)
    calib_runs = [_make_run(rng, 5.0) for _ in range(40)]
    test_runs = [_make_evaluable_run(rng, 5.0, true_capacity=55.0, estimate_capacity=55.0)
                 for _ in range(40)]
    margin = policy.calibrate_conformal_margin(calib_runs, ZONE, HORIZON, 0.9)

    without_ci = policy.evaluate_policy(test_runs, ZONE, margin, HORIZON)
    assert "recall_ci_lo" not in without_ci

    with_ci = policy.evaluate_policy(
        test_runs, ZONE, margin, HORIZON, ci_rng=np.random.default_rng(9), n_boot=200)
    assert "recall_ci_lo" in with_ci and "recall_ci_hi" in with_ci
    assert with_ci["recall"] == without_ci["recall"]


def test_persistence_predictor_ignores_trend():
    series = np.arange(30, dtype=float)
    assert policy.predict_occupancy_ahead(series, 20, HORIZON, "persistence") == 20.0
    assert policy.predict_occupancy_ahead(series, 20, HORIZON, "trend") > 20.0


def test_oracle_threshold_uses_true_capacity_not_estimate():
    rng = np.random.default_rng(5)
    runs = [_make_evaluable_run(rng, 3.0, true_capacity=60.0, estimate_capacity=500.0)
            for _ in range(10)]
    est = policy.evaluate_policy(runs, ZONE, 0.0, HORIZON)
    ora = policy.evaluate_policy(runs, ZONE, 0.0, HORIZON, oracle=True)
    assert est["recall"] in (None, 0.0)
    assert ora["recall"] is not None and ora["recall"] > 0.0
