#!/usr/bin/env python3
"""Publication-size versions of the pipeline diagram and the case-study trace
(larger fonts, no hand-synced citation markers), written as ieee_*.png."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sim import engine, policy, zones as zn  # noqa: E402
from run_experiment import _rng, HORIZON_STEPS  # noqa: E402

FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
plt.rcParams.update({"font.family": "serif", "font.size": 8})

# ---- case study (single column) ----
pe = pd.read_csv(os.path.join(RES, "policy_evaluation.csv"))
seed, level = 395, "P3_high"
geo = zn.draw_true_geometry(_rng(seed, 0))
run = engine.run_single(geo, seed=int(_rng(seed, 1).integers(0, 2**31 - 1)))
s = run["occupancy"]["RJK-Z02"]; cap = run["true_capacity"]["RJK-Z02"]
grid = [i * run["record_interval"] for i in range(len(s))]
est = zn.estimate_geometry(geo, level, _rng(seed, 2, list(zn.PRECISION_LEVELS).index(level)))
thr = est["RJK-Z02"] * zn.DENSITY_THRESHOLD_PERSONS_PER_M2
q = float(pe[(pe.policy_type == "conformal") & (pe.level == level) & (pe.zone == "RJK-Z02")].iloc[0]["margin"])
pred = [policy.predict_occupancy_ahead(s, i, HORIZON_STEPS) for i in range(len(s) - HORIZON_STEPS)]
al = [i for i, p in enumerate(pred) if p + q > thr]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(grid, s, color="#1f4e79", lw=1.2, label="occupancy")
ax.axhline(cap, color="k", ls=":", label=f"true limit ({cap:.1f})")
ax.axhline(thr, color="#4f8f4f", ls="--", label=f"estimated limit, P3 ({thr:.1f})")
ax.scatter([grid[i] for i in al], [s[i] for i in al], s=6, color="#c0504d", zorder=5, label="alert fires")
for t in (90, 330):
    ax.axvspan(t, t + 45, color="grey", alpha=0.15)
ax.set_ylim(0, 245); ax.set_xlabel("minutes after opening"); ax.set_ylabel("occupancy (persons)")
ax.legend(loc="upper left", fontsize=6.5, ncol=1, framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ieee_case_study.png"), dpi=300); plt.close(fig)

# ---- pipeline diagram (full width) ----
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
def column(ax, title, steps, fc, ec):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    n = len(steps); ys = [1 - (i + 0.5) / n for i in range(n)]; h = 0.9 / n * 0.72
    for y, t in zip(ys, steps):
        ax.add_patch(plt.Rectangle((0.03, y - h / 2), 0.94, h, fc=fc, ec=ec, lw=1.1, transform=ax.transAxes))
        ax.text(0.5, y, t, ha="center", va="center", fontsize=7.4, transform=ax.transAxes)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.annotate("", xy=(0.5, y1 + h / 2 + 0.01), xytext=(0.5, y0 - h / 2 - 0.01),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="grey", lw=1.1))
column(axes[0], "Current practice",
       ["Capacity set once: static occupant-load\nlookup or a fixed per-ticket seat quota",
        "Crowding checked, if at all, against a\nfixed density guideline",
        "Reactive alert only: no predictive lead\ntime and no survey-precision axis"], "#f2f2f2", "#888888")
column(axes[1], "Proposed module (this paper)",
       ["Geometry-uncertainty variable: true geometry\nvs. survey estimate at several precisions",
        "Hybrid agent-based + discrete-event simulation:\narrivals, queues, routed crowd flow",
        "Occupancy forecast (current + trend) with a\nsplit-conformal alert margin",
        "Alert evaluated against each world's own\ntrue dynamics: recall and false-alarm rate"], "#fdecec", "#c0504d")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "ieee_pipeline.png"), dpi=300); plt.close(fig)
print("ok")
