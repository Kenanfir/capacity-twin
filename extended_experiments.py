#!/usr/bin/env python3
"""Robustness experiments added for the IEEE Access version of the paper.

    python3 extended_experiments.py [--n-runs 1800] [--jobs N]

Reuses the headline experiment's worlds (same MASTER_SEED, same per-world
seeds, so the 1,800 worlds and the calibration/test split are identical) and
adds, for the performance zone RJK-Z02 only:

  E1  continuous survey-error sweep (sigma 0.025 .. 1.0, plus the oracle)
  E2  alert-level sweep: target coverage 1-alpha in {0.5 .. 0.99}
  E3  empirical coverage of the conformal forecast interval on test worlds
      (all ten zones; 1-alpha = 0.90)
  E4  forecast-horizon sweep h in {5, 10, 15, 20} minutes
  E5  demand scenarios: peak arrival rate 2.5 / 3.0 / 3.5 groups/min

Survey draws use their own RNG namespace (3_000_000_000), so nothing here can
perturb the headline results. Writes results/extended_*.csv.
Per-world predictions are computed once with a vectorised closed-form OLS
slope that is checked against policy.predict_occupancy_ahead.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sim import arrivals as arr, engine, policy, zones as zn  # noqa: E402
from run_experiment import MASTER_SEED, _rng  # noqa: E402

ZONE = "RJK-Z02"
SIGMAS = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
COVERAGES = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
HORIZONS = [5, 10, 15, 20]
N_SURVEY_DRAWS = 5
N_BOOT = 2000


def trend_predictions(series: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised equivalent of policy.predict_occupancy_ahead for every
    t in [0, n-h): returns (prediction, actual-at-t+h)."""
    n = len(series)
    t = np.arange(0, n - h)
    pred = np.empty(len(t))
    for i in range(min(4, len(t))):          # short windows at the day start
        pred[i] = policy.predict_occupancy_ahead(series, i, h)
    if len(t) > 4:
        s = series.astype(float)
        j = t[4:]
        slope = (-2 * s[j - 4] - s[j - 3] + s[j - 1] + 2 * s[j]) / 10.0
        pred[4:] = s[j] + h * slope
    return pred, series[t + h].astype(float)


def _world(args):
    seed, peak, zones_wanted = args
    arr.PEAK_RATE = peak
    true_geo = zn.draw_true_geometry(_rng(seed, 0))
    run = engine.run_single(true_geo, seed=int(_rng(seed, 1).integers(0, 2**31 - 1)))
    out = {"seed": seed, "true_area": true_geo[ZONE], "split": "calibration" if seed % 2 == 0 else "test"}
    out["series"] = {z: np.asarray(run["occupancy"][z], dtype=float) for z in zones_wanted}
    # survey draws for E1: independent namespace, one per (sigma, draw)
    est = np.empty((len(SIGMAS), N_SURVEY_DRAWS))
    for si, sg in enumerate(SIGMAS):
        r = np.random.default_rng([MASTER_SEED, 3_000_000_000, seed, si])
        est[si] = true_geo[ZONE] * r.lognormal(0.0, sg, size=N_SURVEY_DRAWS)
    out["est_area"] = est
    return out


def simulate(n_runs, peak, jobs, zones_wanted):
    arr.PEAK_RATE = peak
    with mp.get_context("fork").Pool(jobs) as pool:
        return pool.map(_world, [(s, peak, zones_wanted) for s in range(n_runs)], chunksize=4)


def margin_for(worlds, zone, h, cov):
    res = []
    for w in worlds:
        if w["split"] != "calibration":
            continue
        p, a = trend_predictions(w["series"][zone], h)
        res.append(np.abs(a - p))
    s = np.sort(np.concatenate(res))
    rank = int(np.ceil((len(s) + 1) * cov))
    return float("inf") if rank > len(s) else float(s[rank - 1])


def counts(worlds, zone, h, margin, area_fn):
    """per-test-world (tp, fp, tn, fn) summed over the survey draws."""
    rows = []
    for w in worlds:
        if w["split"] != "test":
            continue
        p, a = trend_predictions(w["series"][zone], h)
        cap_true = zn.DENSITY_THRESHOLD_PERSONS_PER_M2 * w["true_area"]
        viol = a > cap_true
        tp = fp = tn = fn = 0
        for area in area_fn(w):
            thr = zn.DENSITY_THRESHOLD_PERSONS_PER_M2 * area
            al = (p + margin) > thr
            tp += int((al & viol).sum()); fp += int((al & ~viol).sum())
            tn += int((~al & ~viol).sum()); fn += int((~al & viol).sum())
        rows.append((tp, fp, tn, fn))
    return np.array(rows, dtype=np.int64)


def summarise(c, rng):
    tp, fp, tn, fn = c.sum(axis=0)
    out = {"n_breach_worlds": int(((c[:, 0] + c[:, 3]) > 0).sum()),
           "n_violation_instances": int(tp + fn),
           "recall": tp / (tp + fn) if tp + fn else None,
           "far": fp / (fp + tn) if fp + tn else None}
    idx = rng.integers(0, len(c), size=(N_BOOT, len(c)))
    b = c[idx].sum(axis=1)
    for name, num, den in (("recall", b[:, 0], b[:, 0] + b[:, 3]), ("far", b[:, 1], b[:, 1] + b[:, 2])):
        ok = den > 0
        if ok.any():
            lo, hi = np.percentile(num[ok] / den[ok], [2.5, 97.5])
            out[f"{name}_lo"], out[f"{name}_hi"] = float(lo), float(hi)
        else:
            out[f"{name}_lo"] = out[f"{name}_hi"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=1800)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- self-check of the vectorised predictor against the library ----
    rs = np.random.default_rng(1).integers(0, 150, size=480).astype(float)
    p, _ = trend_predictions(rs, 10)
    ref = np.array([policy.predict_occupancy_ahead(rs, i, 10) for i in range(len(p))])
    assert np.allclose(p, ref), "vectorised predictor disagrees with sim.policy"

    all_zones = list(zn.ZONES)
    worlds = simulate(args.n_runs, arr.PEAK_RATE, args.jobs, all_zones)
    print(f"simulated {len(worlds)} worlds (peak {arr.PEAK_RATE})")

    # ---- E1: sigma sweep (+ oracle, + unsurveyed even split) ----
    q = margin_for(worlds, ZONE, 10, 0.9)
    rows = []
    for si, sg in enumerate(SIGMAS):
        c = counts(worlds, ZONE, 10, q, lambda w, si=si: w["est_area"][si])
        rows.append({"sigma": sg, **summarise(c, _rng(3_000_000_001, si))})
    c = counts(worlds, ZONE, 10, q, lambda w: [w["true_area"]])
    rows.append({"sigma": 0.0, **summarise(c, _rng(3_000_000_001, 99))})
    even = sum(zn.nominal_areas().values()) / len(all_zones)
    c = counts(worlds, ZONE, 10, q, lambda w: [even])
    rows.append({"sigma": float("nan"), **summarise(c, _rng(3_000_000_001, 98))})
    pd.DataFrame(rows).assign(margin=q).to_csv(os.path.join(args.out, "extended_sigma_sweep.csv"), index=False)
    print("E1 done, margin", q)

    # ---- E2: alert-level sweep at three survey errors ----
    rows = []
    for cov in COVERAGES:
        qc = margin_for(worlds, ZONE, 10, cov)
        for si, sg in enumerate(SIGMAS):
            if sg not in (0.05, 0.2, 0.5):
                continue
            c = counts(worlds, ZONE, 10, qc, lambda w, si=si: w["est_area"][si])
            rows.append({"coverage": cov, "sigma": sg, "margin": qc,
                         **summarise(c, _rng(3_000_000_002, int(cov * 1000), si))})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "extended_alert_level_sweep.csv"), index=False)
    print("E2 done")

    # ---- E3: empirical coverage of the forecast interval, every zone ----
    rows = []
    for z in all_zones:
        qz = margin_for(worlds, z, 10, 0.9)
        hit = tot = 0
        per_world = []
        for w in worlds:
            if w["split"] != "test":
                continue
            p, a = trend_predictions(w["series"][z], 10)
            ok = np.abs(a - p) <= qz
            hit += int(ok.sum()); tot += len(ok); per_world.append(ok.mean())
        pw = np.array(per_world)
        idx = _rng(3_000_000_003, all_zones.index(z)).integers(0, len(pw), size=(N_BOOT, len(pw)))
        lo, hi = np.percentile(pw[idx].mean(axis=1), [2.5, 97.5])
        rows.append({"zone": z, "margin": qz, "coverage": hit / tot, "cov_lo": lo, "cov_hi": hi})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "extended_coverage.csv"), index=False)
    print("E3 done")

    # ---- E4: horizon sweep ----
    rows = []
    for h in HORIZONS:
        qh = margin_for(worlds, ZONE, h, 0.9)
        for label, fn in (("oracle", lambda w: [w["true_area"]]),
                          ("sigma0.05", lambda w: w["est_area"][SIGMAS.index(0.05)]),
                          ("sigma0.2", lambda w: w["est_area"][SIGMAS.index(0.2)]),
                          ("sigma0.5", lambda w: w["est_area"][SIGMAS.index(0.5)])):
            c = counts(worlds, ZONE, h, qh, fn)
            rows.append({"horizon_min": h, "survey": label, "margin": qh,
                         **summarise(c, _rng(3_000_000_004, h, len(label)))})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "extended_horizon_sweep.csv"), index=False)
    print("E4 done")

    # ---- E5: demand scenarios (3.0 is the headline scenario) ----
    rows = []
    for peak in (2.5, 3.0, 3.5):
        wl = worlds if peak == 3.0 else simulate(args.n_runs, peak, args.jobs, [ZONE])
        arr.PEAK_RATE = 3.0
        qd = margin_for(wl, ZONE, 10, 0.9)
        for label, fn in (("oracle", lambda w: [w["true_area"]]),
                          ("sigma0.05", lambda w: w["est_area"][SIGMAS.index(0.05)]),
                          ("sigma0.2", lambda w: w["est_area"][SIGMAS.index(0.2)]),
                          ("sigma0.5", lambda w: w["est_area"][SIGMAS.index(0.5)]),
                          ("unsurveyed", lambda w: [even])):
            c = counts(wl, ZONE, 10, qd, fn)
            rows.append({"peak_rate": peak, "survey": label, "margin": qd,
                         **summarise(c, _rng(3_000_000_005, int(peak * 10), len(label)))})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "extended_demand_scenarios.csv"), index=False)
    print("E5 done")


if __name__ == "__main__":
    main()
