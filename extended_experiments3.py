#!/usr/bin/env python3
"""Third batch (IEEE Access revision, answering reviewer-style objections).

    python3 extended_experiments3.py [--jobs N]

Uses the 9,000-world replicate (4,500 test worlds, 63 breach worlds), zone
RJK-Z02, h = 10 min, five survey draws per world and survey error.

Policies compared at each survey error sigma:
  conformal   alert when forecast + q > estimated capacity (q: 0.90 split-conformal margin)
  safety      alert when forecast > kappa * estimated capacity; kappa chosen on the
              calibration worlds so its minute-level false-alarm rate equals the
              conformal policy's calibration false-alarm rate (matched operating point)
  crc         alert when forecast > kappa * estimated capacity; kappa chosen on the
              calibration worlds, with their own survey draws, by conformal risk
              control so that the expected fraction of violation minutes missed in a
              breach world is <= beta = 0.10 (finite-sample corrected)
  prior       conformal policy but with the design area A0 instead of a survey (sigma n/a)
  even        conformal policy with the even-split area (the paper's unsurveyed level)
Metrics (test worlds): minute-level recall and FAR; event detection (any alert in
the 40 min before the first violation minute) with exact Clopper-Pearson interval;
median lead time; share of breach-free days with >=1 alert; mean alert minutes per
breach-free day; and the same with a 3-consecutive-minute debounce.
Writes results/extended3_event_metrics.csv and extended3_breach_population.csv.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import beta as beta_dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import arrivals as arr, zones as zn  # noqa: E402
import extended_experiments as E1  # noqa: E402
from extended_experiments import ZONE, simulate, trend_predictions  # noqa: E402

H, RHO, W, BETA = 10, zn.DENSITY_THRESHOLD_PERSONS_PER_M2, 40, 0.10
KAPPAS = np.round(np.arange(0.30, 1.30, 0.01), 2)
EVEN = sum(zn.nominal_areas().values()) / len(zn.ZONES)
A0 = zn.nominal_areas()[ZONE]
SIG = [0.05, 0.1, 0.2, 0.3, 0.5]


def clopper_pearson(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta_dist.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def debounce(al, k=3):
    if k <= 1:
        return al
    c = np.convolve(al.astype(int), np.ones(k, dtype=int), mode="full")[: len(al)]
    return c >= k


class WorldData:
    def __init__(self, w):
        p, a = trend_predictions(w["series"][ZONE], H)
        self.p, self.a = p, a
        self.cap = RHO * w["true_area"]
        self.viol = a > self.cap
        self.breach = bool(self.viol.any())
        # first violation minute in alert-time index (alert at t refers to t+H)
        self.V = int(np.argmax(self.viol)) if self.breach else -1
        self.est = w["est_area"]            # [len(SIGMAS), 5]
        self.true_area = w["true_area"]


def minute_counts(wd, al):
    v = wd.viol
    return (int((al & v).sum()), int((al & ~v).sum()), int((~al & ~v).sum()), int((~al & v).sum()))


def evaluate(test, alert_fn, k):
    """alert_fn(wd, draw) -> bool mask. returns metrics dict."""
    tp = fp = tn = fn = 0
    det, lead, nb, fday, fmin = [], [], 0, 0, 0
    det1 = []
    n_draw = 5
    for wd in test:
        for d in range(n_draw):
            al = debounce(alert_fn(wd, d), k)
            a, b, c, e = minute_counts(wd, al)
            tp += a; fp += b; tn += c; fn += e
            if wd.breach:
                lo = max(0, wd.V - W)
                hit = np.flatnonzero(al[lo:wd.V])
                det.append(bool(len(hit)))
                if len(hit):
                    lead.append(wd.V - (lo + hit[0]))
                if d == 0:
                    det1.append(bool(len(hit)))
            else:
                nb += 1; fday += int(al.any()); fmin += int(al.sum())
    k1, n1 = sum(det1), len(det1)
    lo, hi = clopper_pearson(k1, n1)
    return {"recall_min": tp / (tp + fn) if tp + fn else np.nan, "far_min": fp / (fp + tn),
            "detect_rate": float(np.mean(det)), "detect_k": k1, "detect_n": n1,
            "detect_cp_lo": lo, "detect_cp_hi": hi,
            "median_lead_min": float(np.median(lead)) if lead else np.nan,
            "free_days_alerted": fday / nb, "alert_min_per_free_day": fmin / nb}


def kappa_counts(cal, getarea, d):
    """per kappa: minute FAR and per-breach-world miss-fraction mean (for CRC)."""
    fp = np.zeros(len(KAPPAS)); tn = np.zeros(len(KAPPAS))
    miss_sum = np.zeros(len(KAPPAS)); nb = 0
    for wd in cal:
        thr = RHO * getarea(wd, d)
        r = wd.p / thr
        rv, rn = np.sort(r[wd.viol]), np.sort(r[~wd.viol])
        # alert iff r > kappa
        fpk = len(rn) - np.searchsorted(rn, KAPPAS, side="right")
        fp += fpk; tn += len(rn) - fpk
        if wd.breach:
            nb += 1
            hit = len(rv) - np.searchsorted(rv, KAPPAS, side="right")
            miss_sum += 1.0 - hit / len(rv)
    return fp / (fp + tn), miss_sum, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    big = simulate(9000, 3.0, args.jobs, [ZONE])
    wds = [(w["split"], WorldData(w)) for w in big]
    cal = [x for s, x in wds if s == "calibration"]
    test = [x for s, x in wds if s == "test"]

    # conformal margin from calibration worlds
    res = np.concatenate([np.abs(x.a - x.p) for x in cal]); res.sort()
    q = float(res[int(np.ceil((len(res) + 1) * 0.9)) - 1])

    # breach-population facts
    tb = [x for x in test if x.breach]
    pop = pd.DataFrame([{"n_test_worlds": len(test), "n_breach_worlds": len(tb),
                         "mean_true_capacity_all": float(np.mean([x.cap for x in test])),
                         "mean_true_capacity_breach": float(np.mean([x.cap for x in tb])),
                         "min_true_capacity_breach": float(min(x.cap for x in tb)),
                         "max_true_capacity_breach": float(max(x.cap for x in tb)),
                         "median_breach_minutes": float(np.median([x.viol.sum() for x in tb])),
                         "margin_q": q}])
    pop.to_csv(os.path.join(args.out, "extended3_breach_population.csv"), index=False)

    rows = []
    def add(name, sg, k, m):
        rows.append({"policy": name, "sigma": sg, "debounce_k": k, **m})

    for k in (1, 3):
        add("conformal (oracle geometry)", 0.0, k, evaluate(test, lambda wd, d: (wd.p + q) > RHO * wd.true_area, k))
        add("prior design area A0", np.nan, k, evaluate(test, lambda wd, d: (wd.p + q) > RHO * A0, k))
        add("even split (paper P0)", np.nan, k, evaluate(test, lambda wd, d: (wd.p + q) > RHO * EVEN, k))

    for sg in SIG:
        si = E1.SIGMAS.index(sg)
        area = lambda wd, d, si=si: wd.est[si][d]
        # calibration FAR of the conformal policy
        fp = tn = 0
        for wd in cal:
            for d in range(5):
                al = (wd.p + q) > RHO * area(wd, d)
                fp += int((al & ~wd.viol).sum()); tn += int((~al & ~wd.viol).sum())
        far_target = fp / (fp + tn)
        # matched-FAR safety factor and CRC kappa, averaged over the five calibration draws
        far_k = np.zeros(len(KAPPAS)); miss_k = np.zeros(len(KAPPAS)); nb = 0
        for d in range(5):
            f, m, nb_d = kappa_counts(cal, area, d)
            far_k += f / 5; miss_k += m; nb += nb_d
        kap_safety = float(KAPPAS[np.argmin(np.abs(far_k - far_target))])
        # CRC: smallest-alert kappa (largest) s.t. (sum miss + 1)/(nb + 1) <= BETA
        crc_ok = (miss_k + 1.0) / (nb + 1.0) <= BETA
        kap_crc = float(KAPPAS[crc_ok][-1]) if crc_ok.any() else float(KAPPAS[0])
        for k in (1, 3):
            add("conformal margin", sg, k, evaluate(test, lambda wd, d: (wd.p + q) > RHO * area(wd, d), k))
            add("safety factor (matched FAR)", sg, k, evaluate(test, lambda wd, d: wd.p > kap_safety * RHO * area(wd, d), k) | {"kappa": kap_safety})
            add("risk-controlled kappa (CRC, beta=0.10)", sg, k, evaluate(test, lambda wd, d: wd.p > kap_crc * RHO * area(wd, d), k) | {"kappa": kap_crc})
        print("sigma", sg, "kappa safety", kap_safety, "crc", kap_crc, flush=True)
    pd.DataFrame(rows).assign(margin_q=q).to_csv(os.path.join(args.out, "extended3_event_metrics.csv"), index=False)
    print("E11 done")


if __name__ == "__main__":
    main()
