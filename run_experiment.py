#!/usr/bin/env python3
"""Run the full geometry-precision x seed experiment grid.

    python3 run_experiment.py [--n-runs 600] [--out results/]

Writes results/raw_runs.{parquet,csv}, results/manifest.json, and
results/policy_evaluation.csv. Deterministic: the same --n-runs and
MASTER_SEED always produce byte-identical raw_runs content, because every
stochastic draw in this file and in sim/ is threaded through an explicitly
seeded numpy Generator, never a module-level default RNG.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sim import engine, metrics, policy, zones as zn  # noqa: E402

MASTER_SEED = 20260918
HORIZON_STEPS = 10
TARGET_COVERAGE = 0.9


def _rng(*parts: int) -> np.random.Generator:
    return np.random.default_rng([MASTER_SEED, *parts])


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                            capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _one_seed(seed: int):
    """One independent world: depends only on (MASTER_SEED, seed), so seeds
    can run in any order or in parallel with identical results."""
    split = "calibration" if seed % 2 == 0 else "test"
    true_geometry = zn.draw_true_geometry(_rng(seed, 0))
    run = engine.run_single(true_geometry, seed=int(_rng(seed, 1).integers(0, 2**31 - 1)))
    summary = metrics.summarize_run(run)
    rows, runs = [], {}
    for level_idx, level in enumerate(zn.PRECISION_LEVELS):
        est_rng = _rng(seed, 2, level_idx)
        estimate = zn.estimate_geometry(true_geometry, level, est_rng)
        row = {"seed": seed, "level": level, "split": split, **summary}
        for z, cap in zn.capacity_from_geometry(estimate).items():
            row[f"estimate_capacity_{z}"] = cap
        rows.append(row)
        runs[(seed, level)] = {**run, "estimate_geometry": estimate}
    return rows, runs


def run_grid(n_runs: int, jobs: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (raw_runs DataFrame, {(seed, level): run_dict}) -- the run
    dict is kept in memory (not all serialized) so policy evaluation can
    reuse the full occupancy series without re-simulating. Seeds are
    independent, so they run in a process pool; results are collected in
    seed order, identical to the serial path (--jobs 1)."""
    import multiprocessing as mp
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    rows: list = []
    runs_by_seed_level: dict[tuple[int, str], dict] = {}
    if jobs == 1:
        results = map(_one_seed, range(n_runs))
        pool = None
    else:
        pool = mp.get_context("fork").Pool(jobs)
        results = pool.imap(_one_seed, range(n_runs), chunksize=4)
    for r, runs in results:
        rows.extend(r)
        runs_by_seed_level.update(runs)
    if pool is not None:
        pool.close(); pool.join()
    return pd.DataFrame(rows), runs_by_seed_level


_EVAL_CTX: dict = {}


def _eval_cell(task):
    """One (block, level, zone) unit of policy evaluation. Blocks: 'A' = the
    paper's conformal policy + the zero-margin ablation; 'B' = the no-trend
    (persistence) comparator and, at the last level, the oracle comparator.
    Each unit uses only its own seeded bootstrap RNG, so units are
    independent and can run in parallel with identical results."""
    block, level, z = task
    calib_runs = _EVAL_CTX["calib"][level]
    test_runs = _EVAL_CTX["test"][level]
    li, zi = zn.PRECISION_LEVELS.index(level), zn.ZONES.index(z)
    est_thr = float(np.mean([r["estimate_geometry"][z] * zn.DENSITY_THRESHOLD_PERSONS_PER_M2
                             for r in test_runs]))
    rows = []
    if block == "A":
        margin = policy.calibrate_conformal_margin(calib_runs, z, HORIZON_STEPS, TARGET_COVERAGE)
        for policy_type, m in (("conformal", margin), ("naive_zero_margin", 0.0)):
            # Distinct namespace (leading 2_000_000_000) from the per-world
            # seeds (integer world indices 0..n_runs-1), so the bootstrap RNG
            # never collides with a simulation draw.
            policy_idx = 0 if policy_type == "conformal" else 1
            ci_rng = _rng(2_000_000_000, li, zi, policy_idx)
            result = policy.evaluate_policy(test_runs, z, m, HORIZON_STEPS, ci_rng=ci_rng)
            rows.append({"policy_type": policy_type, "level": level, "zone": z,
                         "mean_threshold": est_thr, "margin": m, **result})
    else:
        m_p = policy.calibrate_conformal_margin(
            calib_runs, z, HORIZON_STEPS, TARGET_COVERAGE, predictor="persistence")
        ci_rng = _rng(2_000_000_000, li, zi, 2)
        result = policy.evaluate_policy(test_runs, z, m_p, HORIZON_STEPS,
                                        ci_rng=ci_rng, predictor="persistence")
        rows.append({"policy_type": "persistence_conformal", "level": level, "zone": z,
                     "mean_threshold": est_thr, "margin": m_p, **result})
        if level == zn.PRECISION_LEVELS[-1]:
            m_t = policy.calibrate_conformal_margin(calib_runs, z, HORIZON_STEPS, TARGET_COVERAGE)
            ci_rng = _rng(2_000_000_000, 99, zi, 3)
            result = policy.evaluate_policy(test_runs, z, m_t, HORIZON_STEPS,
                                            ci_rng=ci_rng, oracle=True)
            rows.append({"policy_type": "oracle_conformal", "level": "P_oracle", "zone": z,
                         "mean_threshold": float(np.mean([r["true_capacity"][z] for r in test_runs])),
                         "margin": m_t, **result})
    return rows


def evaluate_policies(runs_by_seed_level: dict[tuple[int, str], dict],
                       raw_runs: pd.DataFrame, jobs: int | None = None) -> pd.DataFrame:
    """One row per (policy_type, level, zone). policy_type='conformal' is
    the paper's calibrated policy (margin from calibrate_conformal_margin).
    'naive_zero_margin' is the same predictor and threshold with the margin
    forced to 0 (ablation isolating what split-conformal calibration
    contributes). 'persistence_conformal' drops the trend term from the
    predictor; 'oracle_conformal' gives the policy the TRUE geometry (an
    upper bound on what any survey could deliver). Row order is identical
    to the serial implementation (units are collected in submission order)."""
    import multiprocessing as mp
    for level in zn.PRECISION_LEVELS:
        calib_seeds = sorted(s for s in raw_runs.loc[
            (raw_runs.level == level) & (raw_runs.split == "calibration"), "seed"])
        test_seeds = sorted(s for s in raw_runs.loc[
            (raw_runs.level == level) & (raw_runs.split == "test"), "seed"])
        _EVAL_CTX.setdefault("calib", {})[level] = [runs_by_seed_level[(s, level)] for s in calib_seeds]
        _EVAL_CTX.setdefault("test", {})[level] = [runs_by_seed_level[(s, level)] for s in test_seeds]
    tasks = []
    for level in zn.PRECISION_LEVELS:
        tasks += [("A", level, z) for z in zn.ZONES]
        tasks += [("B", level, z) for z in zn.ZONES]
    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    if jobs == 1:
        parts = map(_eval_cell, tasks)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(jobs)
        parts = pool.imap(_eval_cell, tasks, chunksize=1)
    rows = [r for part in parts for r in part]
    if pool is not None:
        pool.close(); pool.join()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: cpus-2; 1 = serial)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    raw_runs, runs_by_seed_level = run_grid(args.n_runs, args.jobs)

    fmt = "csv"
    out_path = os.path.join(args.out, "raw_runs.csv")
    try:
        import pyarrow  # noqa: F401
        out_path = os.path.join(args.out, "raw_runs.parquet")
        raw_runs.to_parquet(out_path, index=False)
        fmt = "parquet"
    except Exception:
        raw_runs.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({fmt}, {len(raw_runs)} rows)")

    policy_eval = evaluate_policies(runs_by_seed_level, raw_runs, args.jobs)
    policy_eval_path = os.path.join(args.out, "policy_evaluation.csv")
    policy_eval.to_csv(policy_eval_path, index=False)
    print(f"wrote {policy_eval_path} ({len(policy_eval)} rows)")

    manifest = {
        "n_runs": args.n_runs,
        "master_seed": MASTER_SEED,
        "precision_levels": list(zn.PRECISION_LEVELS),
        "raw_runs_format": fmt,
        "horizon_steps": HORIZON_STEPS,
        "target_coverage": TARGET_COVERAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
