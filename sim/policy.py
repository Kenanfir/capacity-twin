"""A capacity-alert policy, and split-conformal calibration of its margin.

The policy's belief about "safe occupancy" comes from whichever geometry
precision level it was calibrated on (sim.zones.estimate_geometry). This
module measures how well that belief predicts real (true-geometry) safety-
threshold violations -- the paper's central question.
"""
from __future__ import annotations

import numpy as np

from . import zones as zn


def alert_threshold(estimate_geometry: dict[str, float]) -> dict[str, float]:
    return zn.capacity_from_geometry(estimate_geometry)


def predict_occupancy_ahead(occupancy_series: np.ndarray, t_idx: int, horizon_steps: int,
                            predictor: str = "trend") -> float:
    """current occupancy + horizon_steps * (slope over the last 5 samples).
    predictor="persistence" drops the trend term (forecast = current
    occupancy), the no-trend comparator."""
    current = float(occupancy_series[t_idx])
    if predictor == "persistence":
        return current
    lo = max(0, t_idx - 4)
    window = occupancy_series[lo:t_idx + 1]
    if len(window) < 2:
        slope = 0.0
    else:
        x = np.arange(len(window))
        slope = float(np.polyfit(x, window, 1)[0])
    return current + horizon_steps * slope


def _residuals(run: dict, zone: str, horizon_steps: int, predictor: str = "trend") -> np.ndarray:
    series = run["occupancy"][zone]
    n = len(series)
    out = []
    for t_idx in range(0, n - horizon_steps):
        pred = predict_occupancy_ahead(series, t_idx, horizon_steps, predictor)
        actual = float(series[t_idx + horizon_steps])
        out.append(actual - pred)
    return np.asarray(out)


def calibrate_conformal_margin(calibration_runs: list[dict], zone: str,
                                horizon_steps: int, target_coverage: float = 0.9,
                                predictor: str = "trend") -> float:
    """Return the finite-sample split-conformal absolute-residual margin.

    Scores are pooled across every (run, t) pair in ``calibration_runs`` for
    ``zone``.  The residual is a property of the naive predictor's error
    against this zone's own crowd dynamics, not of any particular geometry
    estimate, so pooling across calibration runs is legitimate even though
    each run carries its own true/estimate geometry pair (see
    ``run_experiment.py``'s per-run threshold pairing in ``evaluate_policy``).

    ``target_coverage`` is the desired marginal coverage, i.e. 1 - alpha.
    The order statistic uses ceil((n + 1) * target_coverage), the standard
    finite-sample split-conformal choice.  If the requested order is larger
    than the available calibration sample, returning infinity is the honest
    finite-sample result: no finite residual margin can provide that bound.
    """
    all_residuals = np.concatenate([
        _residuals(run, zone, horizon_steps, predictor) for run in calibration_runs
    ])
    abs_residuals = np.abs(all_residuals)
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be strictly between 0 and 1")
    rank = int(np.ceil((len(abs_residuals) + 1) * target_coverage))
    if rank > len(abs_residuals):
        return float("inf")
    return float(np.sort(abs_residuals)[rank - 1])


def _per_world_counts(test_runs: list[dict], zone: str, margin: float,
                       horizon_steps: int, predictor: str = "trend",
                       oracle: bool = False) -> dict[str, np.ndarray]:
    """One (tp, fp, tn, fn) count per test world, not pooled -- the unit a
    cluster bootstrap must resample, since minute-level instances within the
    same world's breach episode are correlated, not independent draws."""
    tp = np.zeros(len(test_runs), dtype=np.int64)
    fp = np.zeros(len(test_runs), dtype=np.int64)
    tn = np.zeros(len(test_runs), dtype=np.int64)
    fn = np.zeros(len(test_runs), dtype=np.int64)
    for i, run in enumerate(test_runs):
        series = run["occupancy"][zone]
        true_capacity = run["true_capacity"][zone]
        if oracle:
            threshold = true_capacity
        else:
            threshold = run["estimate_geometry"][zone] * zn.DENSITY_THRESHOLD_PERSONS_PER_M2
        n = len(series)
        for t_idx in range(0, n - horizon_steps):
            pred = predict_occupancy_ahead(series, t_idx, horizon_steps, predictor)
            alert = (pred + margin) > threshold
            actual_violation = float(series[t_idx + horizon_steps]) > true_capacity
            if alert and actual_violation:
                tp[i] += 1
            elif alert and not actual_violation:
                fp[i] += 1
            elif not alert and actual_violation:
                fn[i] += 1
            else:
                tn[i] += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _cluster_bootstrap_ci(counts: dict[str, np.ndarray], rng: np.random.Generator,
                           n_boot: int = 2000, alpha: float = 0.05) -> dict:
    """Resample WORLDS (not minute-level instances) with replacement, so the
    reported interval reflects how few independent breach worlds the test
    split actually contains -- a per-instance interval on the pooled
    tp/fp/tn/fn counts would understate this, since instances inside one
    world's breach episode are not independent trials."""
    n = len(counts["tp"])
    if n == 0:
        return {"recall_ci_lo": None, "recall_ci_hi": None, "recall_ci_undefined_frac": 1.0,
                "far_ci_lo": None, "far_ci_hi": None, "far_ci_undefined_frac": 1.0}
    idx = rng.integers(0, n, size=(n_boot, n))
    btp = counts["tp"][idx].sum(axis=1)
    bfp = counts["fp"][idx].sum(axis=1)
    btn = counts["tn"][idx].sum(axis=1)
    bfn = counts["fn"][idx].sum(axis=1)

    def _percentile_ci(numer: np.ndarray, denom: np.ndarray) -> tuple[float | None, float | None, float]:
        defined = denom > 0
        if not np.any(defined):
            return None, None, 1.0
        rate = numer[defined] / denom[defined]
        lo, hi = np.percentile(rate, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi), float(1.0 - defined.mean())

    r_lo, r_hi, r_undef = _percentile_ci(btp, btp + bfn)
    f_lo, f_hi, f_undef = _percentile_ci(bfp, bfp + btn)
    return {
        "recall_ci_lo": r_lo, "recall_ci_hi": r_hi, "recall_ci_undefined_frac": r_undef,
        "far_ci_lo": f_lo, "far_ci_hi": f_hi, "far_ci_undefined_frac": f_undef,
    }


def evaluate_policy(test_runs: list[dict], zone: str, margin: float,
                     horizon_steps: int, ci_rng: np.random.Generator | None = None,
                     n_boot: int = 2000, predictor: str = "trend",
                     oracle: bool = False) -> dict:
    """Each test run supplies its OWN alert threshold, from its OWN
    estimate_geometry -- a policy calibrated on a level-L survey of THIS
    world is only ever evaluated against that same world's true dynamics,
    never against a different run's true capacity. `run['estimate_geometry']`
    must be present (run_experiment.py attaches it).

    If ``ci_rng`` is given, also returns a 95% cluster-bootstrap confidence
    interval on recall and false-alarm rate, resampling whole worlds (see
    ``_cluster_bootstrap_ci``) rather than pooled minute-level instances."""
    counts = _per_world_counts(test_runs, zone, margin, horizon_steps, predictor, oracle)
    tp, fp, tn, fn = (int(counts[k].sum()) for k in ("tp", "fp", "tn", "fn"))

    n_violations = tp + fn
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else None
    result = {
        "recall": recall,
        "precision": precision,
        "false_alarm_rate": false_alarm_rate,
        "n_violations": n_violations,
    }
    if ci_rng is not None:
        result.update(_cluster_bootstrap_ci(counts, ci_rng, n_boot=n_boot))
    return result
