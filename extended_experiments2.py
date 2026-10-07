#!/usr/bin/env python3
"""Second batch of robustness experiments (IEEE Access revision).

    python3 extended_experiments2.py [--jobs N]

E6  large replicate: 9,000 worlds (the first 1,800 are the headline worlds),
    sigma sweep, ~5x the breach worlds
E7  systematic survey bias (area over/under-estimated by a constant factor)
E8  heavy-tailed survey error (Student-t, 3 dof, matched standard deviation)
E9  forecaster robustness: trend-OLS (paper), persistence, Holt linear
    smoothing, ridge autoregression fitted on calibration worlds
E10 calibration shift: margin calibrated at peak 3.0 groups/min, applied at
    other demand levels, versus recalibrating

Zone RJK-Z02 only, h = 10 min, 1-alpha = 0.90, five survey draws per world.
Writes results/extended2_*.csv. Survey draws use their own RNG namespace.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import arrivals as arr, zones as zn  # noqa: E402
from run_experiment import MASTER_SEED, _rng  # noqa: E402
import extended_experiments as E1  # noqa: E402
from extended_experiments import ZONE, simulate, trend_predictions, summarise  # noqa: E402

H = 10
RHO = zn.DENSITY_THRESHOLD_PERSONS_PER_M2
EVEN = sum(zn.nominal_areas().values()) / len(zn.ZONES)


def split(worlds, which):
    return [w for w in worlds if w["split"] == which]


def noise(seed, tag, size=5, sd=0.05, kind="normal"):
    r = np.random.default_rng([MASTER_SEED, 3_000_000_010, tag, seed])
    if kind == "t3":  # Student-t, 3 dof, scaled to standard deviation sd
        return r.standard_t(3, size=size) * sd / np.sqrt(3.0)
    return r.normal(0.0, sd, size=size)


def margin_from(preds_actual, cov=0.9):
    s = np.sort(np.concatenate([np.abs(a - p) for p, a in preds_actual]))
    rank = int(np.ceil((len(s) + 1) * cov))
    return float("inf") if rank > len(s) else float(s[rank - 1])


def count(test, preds, q, area_fn):
    rows = []
    for w, (p, a) in zip(test, preds):
        cap_true = RHO * w["true_area"]
        viol = a > cap_true
        tp = fp = tn = fn = 0
        for area in area_fn(w):
            al = (p + q) > RHO * area
            tp += int((al & viol).sum()); fp += int((al & ~viol).sum())
            tn += int((~al & ~viol).sum()); fn += int((~al & viol).sum())
        rows.append((tp, fp, tn, fn))
    return np.array(rows, dtype=np.int64)


def trend_pa(worlds):
    return [trend_predictions(w["series"][ZONE], H) for w in worlds]


# ---- forecasters (each returns (pred, actual) for t in [0, n-H)) ----
def persistence_pa(w):
    s = w["series"][ZONE]; t = np.arange(0, len(s) - H)
    return s[t], s[t + H]


def holt_pa(w, a=0.5, b=0.3):
    s = w["series"][ZONE]; n = len(s); lvl, tr = s[0], 0.0
    pred = np.empty(n - H)
    for t in range(n - H):
        if t > 0:
            new = a * s[t] + (1 - a) * (lvl + tr)
            tr = b * (new - lvl) + (1 - b) * tr
            lvl = new
        pred[t] = lvl + H * tr
    return pred, s[H:n]


def lag_matrix(s, k=6):
    pad = np.concatenate([np.full(k - 1, s[0]), s])
    n = len(s)
    return np.stack([pad[k - 1 - j: k - 1 - j + n] for j in range(k)], axis=1)  # O_t, O_{t-1}, ...


def ar_fit(cal, lam=1.0):
    X = np.concatenate([lag_matrix(w["series"][ZONE])[: -H] for w in cal])
    y = np.concatenate([w["series"][ZONE][H:] for w in cal])
    X1 = np.hstack([X, np.ones((len(X), 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[-1, -1] -= lam
    return np.linalg.solve(A, X1.T @ y)


def ar_pa(w, beta):
    s = w["series"][ZONE]
    X1 = np.hstack([lag_matrix(s)[:-H], np.ones((len(s) - H, 1))])
    return X1 @ beta, s[H:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- E6: large replicate ----
    big = simulate(9000, 3.0, args.jobs, [ZONE])
    cal, test = split(big, "calibration"), split(big, "test")
    pa_cal = trend_pa(cal); pa_test = trend_pa(test); q = margin_from(pa_cal)
    rows = []
    for si, sg in enumerate(E1.SIGMAS):
        c = count(test, pa_test, q, lambda w, si=si: w["est_area"][si])
        rows.append({"sigma": sg, **summarise(c, _rng(3_000_000_020, si))})
    rows.append({"sigma": 0.0, **summarise(count(test, pa_test, q, lambda w: [w["true_area"]]), _rng(3_000_000_020, 98))})
    rows.append({"sigma": float("nan"), **summarise(count(test, pa_test, q, lambda w: [EVEN]), _rng(3_000_000_020, 99))})
    d = pd.DataFrame(rows).assign(margin=q, n_worlds=9000, n_test=len(test))
    d.to_csv(os.path.join(args.out, "extended2_large_replicate.csv"), index=False)
    print("E6 done; breach worlds", int(d.n_breach_worlds.iloc[0]), "margin", q)

    # work on the headline 1,800 worlds for the rest
    worlds = [w for w in big if w["seed"] < 1800]
    cal, test = split(worlds, "calibration"), split(worlds, "test")
    pa_cal = trend_pa(cal); pa_test = trend_pa(test); q = margin_from(pa_cal)

    # ---- E7: systematic bias ----
    rows = []
    for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0):
        c = count(test, pa_test, q, lambda w, f=f: w["true_area"] * f * np.exp(noise(w["seed"], 7)))
        rows.append({"area_factor": f, **summarise(c, _rng(3_000_000_021, int(f * 100)))})
    pd.DataFrame(rows).assign(margin=q).to_csv(os.path.join(args.out, "extended2_bias.csv"), index=False)
    print("E7 done")

    # ---- E8: heavy-tailed error ----
    rows = []
    for sg in (0.1, 0.2, 0.5):
        for kind in ("normal", "t3"):
            c = count(test, pa_test, q, lambda w, sg=sg, kind=kind: w["true_area"] * np.exp(noise(w["seed"], 8, sd=sg, kind=kind)))
            rows.append({"sigma": sg, "error": kind, **summarise(c, _rng(3_000_000_022, int(sg * 100), len(kind)))})
    pd.DataFrame(rows).assign(margin=q).to_csv(os.path.join(args.out, "extended2_heavy_tail.csv"), index=False)
    print("E8 done")

    # ---- E9: forecasters ----
    beta = ar_fit(cal)
    fcs = {"trend OLS (paper)": (None, None),
           "persistence": (persistence_pa, None),
           "Holt linear": (holt_pa, None),
           "ridge AR(6)": (lambda w: ar_pa(w, beta), None)}
    rows = []
    for name, (fn, _) in fcs.items():
        pc = pa_cal if fn is None else [fn(w) for w in cal]
        pt = pa_test if fn is None else [fn(w) for w in test]
        qf = margin_from(pc)
        mae = float(np.mean(np.concatenate([np.abs(a - p) for p, a in pt])))
        for label, afn in (("oracle", lambda w: [w["true_area"]]),
                           ("sigma 0.05", lambda w: w["true_area"] * np.exp(noise(w["seed"], 9, sd=0.05))),
                           ("sigma 0.2", lambda w: w["true_area"] * np.exp(noise(w["seed"], 9, sd=0.2))),
                           ("sigma 0.5", lambda w: w["true_area"] * np.exp(noise(w["seed"], 9, sd=0.5)))):
            c = count(test, pt, qf, afn)
            rows.append({"forecaster": name, "survey": label, "margin": qf, "test_mae": mae,
                         **summarise(c, _rng(3_000_000_023, len(name), len(label)))})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "extended2_forecasters.csv"), index=False)
    bb = []
    for name, (fn, _) in fcs.items():
        pt = pa_test if fn is None else [fn(w) for w in test]
        act, prd = [], []
        for w, (p, a) in zip(test, pt):
            v = a > RHO * w["true_area"]
            if v.any():
                act.append(a[v]); prd.append(p[v])
        act, prd = np.concatenate(act), np.concatenate(prd)
        bb.append({"forecaster": name, "breach_minutes": len(act), "mean_actual": act.mean(),
                   "mean_forecast": prd.mean(), "mean_error": (act - prd).mean()})
    pd.DataFrame(bb).to_csv(os.path.join(args.out, "extended2_breach_bias.csv"), index=False)
    print("E9 done")

    # ---- E10: calibration shift ----
    rows = []
    for peak in (2.0, 2.5, 3.5, 4.0, 5.0):
        sh = simulate(1800, peak, args.jobs, [ZONE]); arr.PEAK_RATE = 3.0
        tst = split(sh, "test"); cl = split(sh, "calibration")
        pt = trend_pa(tst); q_re = margin_from(trend_pa(cl))
        cover_stale = float(np.mean(np.concatenate([np.abs(a - p) <= q for p, a in pt])))
        cover_re = float(np.mean(np.concatenate([np.abs(a - p) <= q_re for p, a in pt])))
        for label, qq in (("stale margin (calibrated at 3.0)", q), ("recalibrated", q_re)):
            c = count(tst, pt, qq, lambda w: w["true_area"] * np.exp(noise(w["seed"], 10, sd=0.2)))
            rows.append({"peak_rate": peak, "margin_kind": label, "margin": qq,
                         "coverage": cover_stale if "stale" in label else cover_re,
                         **summarise(c, _rng(3_000_000_024, int(peak * 10), len(label)))})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "extended2_shift.csv"), index=False)
    print("E10 done")


if __name__ == "__main__":
    main()
