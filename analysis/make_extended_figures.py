#!/usr/bin/env python3
"""Figures for the robustness experiments (reads results/extended_*.csv)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})
C = ["#1f4e79", "#c0504d", "#4f8f4f"]

# ---- sigma sweep ----
d = pd.read_csv(os.path.join(RES, "extended_sigma_sweep.csv"))
sw = d[d.sigma > 0].sort_values("sigma")
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.5))
ax[0].fill_between(sw.sigma, sw.recall_lo, sw.recall_hi, color=C[0], alpha=0.2)
ax[0].plot(sw.sigma, sw.recall, "o-", color=C[0], ms=3, label="survey estimate")
ax[0].axhline(1.0, color="k", ls="--", lw=0.8, label="true geometry (oracle)")
ax[0].axhline(0.0, color=C[1], ls=":", lw=1.0, label="unsurveyed (even split)")
ax[0].set_xlabel(r"survey error $\sigma$ (log-area standard deviation)")
ax[0].set_ylabel("alert recall"); ax[0].set_ylim(-0.05, 1.08); ax[0].legend(loc="center left")
ax[0].set_title("(a) Recall falls as the survey gets worse")
ax[1].fill_between(sw.sigma, sw.far_lo, sw.far_hi, color=C[0], alpha=0.2)
ax[1].plot(sw.sigma, sw.far, "o-", color=C[0], ms=3, label="survey estimate")
orc = d[d.sigma == 0].far.iloc[0]
ax[1].axhline(orc, color="k", ls="--", lw=0.8, label="true geometry (oracle)")
ax[1].set_xlabel(r"survey error $\sigma$ (log-area standard deviation)")
ax[1].set_ylabel("false-alarm rate"); ax[1].legend(loc="upper left")
ax[1].set_title("(b) False alarms rise as the survey gets worse")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext_sigma_sweep.png"), dpi=300); plt.close(fig)

# ---- operating curves ----
a = pd.read_csv(os.path.join(RES, "extended_alert_level_sweep.csv"))
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for col, sg in zip(C, (0.05, 0.2, 0.5)):
    s = a[a.sigma == sg].sort_values("coverage")
    ax.plot(s.far, s.recall, "o-", color=col, ms=3, label=rf"$\sigma={sg}$")
    for _, r in s.iterrows():
        if sg == 0.5 or r.coverage in (0.5, 0.99):
            pass
        if sg == 0.2:
            ax.annotate(f"{r.coverage:.2f}", (r.far, r.recall), textcoords="offset points",
                        xytext=(4, -9), fontsize=6)
ax.set_xscale("symlog", linthresh=0.01)
ax.set_xlabel("false-alarm rate (symlog)"); ax.set_ylabel("alert recall")
ax.set_xlim(right=1.8)
ax.set_title("Operating curves: target coverage swept 0.50 to 0.99")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext_operating_curves.png"), dpi=300); plt.close(fig)

# ---- coverage + demand scenarios ----
cv = pd.read_csv(os.path.join(RES, "extended_coverage.csv"))
dm = pd.read_csv(os.path.join(RES, "extended_demand_scenarios.csv"))
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw={"width_ratios": [1.1, 1]})
x = range(len(cv))
ax[0].errorbar(list(x), cv.coverage, yerr=[cv.coverage - cv.cov_lo, cv.cov_hi - cv.coverage],
               fmt="o", color=C[0], capsize=2, ms=4)
ax[0].axhline(0.9, color="k", ls="--", lw=0.8, label="target 0.90")
ax[0].set_xticks(list(x)); ax[0].set_xticklabels([z[-3:] for z in cv.zone], rotation=0)
ax[0].set_ylim(0.89, 0.92); ax[0].set_xlabel("zone (RJK-Z..)"); ax[0].set_ylabel("empirical coverage")
ax[0].set_title("(a) Forecast-interval coverage, test worlds"); ax[0].legend(loc="upper left")
labs = [("oracle", "oracle"), ("sigma0.05", r"$\sigma$=0.05"), ("sigma0.2", r"$\sigma$=0.2"),
        ("sigma0.5", r"$\sigma$=0.5"), ("unsurveyed", "unsurveyed")]
w = 0.26
for i, (peak, col) in enumerate(zip((2.5, 3.0, 3.5), C)):
    sub = dm[dm.peak_rate == peak].set_index("survey").loc[[l for l, _ in labs]]
    n = int(sub.n_breach_worlds.iloc[0])
    ax[1].bar([j + (i - 1) * w for j in range(5)], sub.recall, w, color=col, alpha=0.85,
              label=f"peak {peak} groups/min ({n} breach world{'s' if n != 1 else ''})")
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([l for _, l in labs], fontsize=6.5)
ax[1].set_ylabel("alert recall"); ax[1].set_ylim(0, 1.5); ax[1].legend(loc="upper right", fontsize=6)
ax[1].set_title("(b) Recall under three demand levels")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext_coverage_demand.png"), dpi=300); plt.close(fig)
print("ok")

# ---- headline comparison figure (recall by level, four policies) ----
pe = pd.read_csv(os.path.join(RES, "policy_evaluation.csv"))
pe = pe[pe.zone == "RJK-Z02"]
levels = ["P0_unsurveyed", "P1_low", "P2_medium", "P3_high"]
names = ["P0\nunsurveyed", "P1\nlow", "P2\nmedium", "P3\nhigh"]
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.5))
spec = [("conformal", "trend + conformal (paper)", C[0], "o", -0.08),
        ("naive_zero_margin", "zero margin", C[1], "s", 0.0),
        ("persistence_conformal", "no trend + conformal", C[2], "^", 0.08)]
for pt, lab, col, mk, dx in spec:
    s = pe[pe.policy_type == pt].set_index("level").loc[levels]
    x = [i + dx for i in range(4)]
    rec = s.recall.astype(float)
    yerr = None
    if s.recall_ci_lo.notna().any():
        yerr = [(rec - s.recall_ci_lo.astype(float)).fillna(0), (s.recall_ci_hi.astype(float) - rec).fillna(0)]
    ax[0].errorbar(x, rec, yerr=yerr, fmt=mk + "-", color=col, ms=4, lw=1, capsize=2, label=lab)
    ax[1].plot(x, s.false_alarm_rate.astype(float), mk + "-", color=col, ms=4, lw=1, label=lab)
orc = pe[pe.policy_type == "oracle_conformal"].iloc[0]
ax[0].axhline(orc.recall, color="k", ls="--", lw=0.8, label="oracle (true geometry)")
ax[1].axhline(orc.false_alarm_rate, color="k", ls="--", lw=0.8, label="oracle (true geometry)")
for a, yl, t in ((ax[0], "alert recall", "(a) Recall"), (ax[1], "false-alarm rate", "(b) False-alarm rate")):
    a.set_xticks(range(4)); a.set_xticklabels(names); a.set_ylabel(yl); a.set_title(t)
ax[1].legend(loc="upper right", fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext_policy_comparison.png"), dpi=300); plt.close(fig)

# ---- second batch: large replicate, bias, forecasters ----
lr = pd.read_csv(os.path.join(RES, "extended2_large_replicate.csv"))
lr = lr[lr.sigma > 0].sort_values("sigma")
bi = pd.read_csv(os.path.join(RES, "extended2_bias.csv")).sort_values("area_factor")
fc = pd.read_csv(os.path.join(RES, "extended2_forecasters.csv"))
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.45))
s1 = d[d.sigma > 0].sort_values("sigma")
ax[0].fill_between(lr.sigma, lr.recall_lo, lr.recall_hi, color=C[0], alpha=0.2)
ax[0].plot(lr.sigma, lr.recall, "o-", color=C[0], ms=3, label="9,000 worlds (63 breach)")
ax[0].plot(s1.sigma, s1.recall, "s--", color=C[1], ms=3, lw=0.9, label="1,800 worlds (13 breach)")
ax[0].set_xlabel(r"survey error $\sigma$"); ax[0].set_ylabel("alert recall"); ax[0].set_ylim(0.4, 1.03)
ax[0].legend(loc="lower left", fontsize=6); ax[0].set_title("(a) Larger replicate")
ax[1].plot(bi.area_factor, bi.recall, "o-", color=C[0], ms=3, label="recall")
ax[1].fill_between(bi.area_factor, bi.recall_lo, bi.recall_hi, color=C[0], alpha=0.2)
ax[1].plot(bi.area_factor, bi.far * 5, "s-", color=C[1], ms=3, label="false-alarm rate (x5)")
ax[1].axvline(1.0, color="k", ls=":", lw=0.8)
ax[1].set_xlabel("estimated / true area"); ax[1].set_title("(b) Systematic bias"); ax[1].legend(loc="center right", fontsize=6)
order = ["trend OLS (paper)", "Holt linear", "persistence", "ridge AR(6)"]
lab = ["trend\nOLS", "Holt", "persist.", "ridge\nAR(6)"]
w = 0.2
for i, (sv, col) in enumerate(zip(("oracle", "sigma 0.05", "sigma 0.2", "sigma 0.5"), (*C, "#888888"))):
    v = [fc[(fc.forecaster == f) & (fc.survey == sv)].recall.iloc[0] for f in order]
    ax[2].bar([j + (i - 1.5) * w for j in range(4)], v, w, color=col, alpha=0.85, label=sv.replace("sigma", r"$\sigma$"))
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(lab, fontsize=6.5); ax[2].set_ylim(0, 1.3)
ax[2].set_title("(c) Forecaster"); ax[2].legend(loc="upper right", fontsize=5.5, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ext2_replicate_bias_forecasters.png"), dpi=300); plt.close(fig)
