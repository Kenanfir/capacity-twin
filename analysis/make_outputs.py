#!/usr/bin/env python3
"""Read results/raw_runs.* and results/policy_evaluation.csv, write the
figures and summary tables the manuscript cites. Every number in
summary_table.{csv,md} is an aggregation (mean/std/quantile) over
raw_runs/policy_evaluation -- nothing here is recomputed from scratch.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sim import engine, policy, zones as zn  # noqa: E402
from run_experiment import MASTER_SEED, HORIZON_STEPS, TARGET_COVERAGE, _rng  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")


def _load_raw_runs() -> pd.DataFrame:
    parquet_path = os.path.join(RESULTS, "raw_runs.parquet")
    csv_path = os.path.join(RESULTS, "raw_runs.csv")
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ModuleNotFoundError, ValueError):
            # CSV is the committed interchange format; parquet is an
            # optional acceleration path when an engine is installed.
            pass
    return pd.read_csv(csv_path)


def fig_occupancy_by_precision(raw_runs: pd.DataFrame) -> str:
    test = raw_runs[raw_runs.split == "test"]
    fig, axes = plt.subplots(1, len(zn.PRECISION_LEVELS), figsize=(15, 2.3), sharey=True)
    for ax, level in zip(axes, zn.PRECISION_LEVELS):
        sub = test[test.level == level]
        cols = [f"peak_occupancy_{z}" for z in zn.ZONES]
        data = [sub[c].values for c in cols]
        bp = ax.boxplot(data, tick_labels=[z.replace("RJK-", "") for z in zn.ZONES], showfliers=False)
        z02_idx = zn.ZONES.index("RJK-Z02")
        for patch in [bp["boxes"][z02_idx]]:
            patch.set(color="tab:red", linewidth=2)
        ax.set_title(level, fontsize=9)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].set_ylabel("peak occupancy (persons)")
    fig.tight_layout()
    out = os.path.join(FIGURES, "occupancy_by_precision.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_policy_recall_vs_precision(policy_eval: pd.DataFrame) -> str:
    conformal = policy_eval[policy_eval.policy_type == "conformal"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    level_x = {level: i for i, level in enumerate(zn.PRECISION_LEVELS)}
    z02 = conformal[conformal.zone == "RJK-Z02"].copy()
    z02["x"] = z02.level.map(level_x)
    z02 = z02.sort_values("x")
    # Asymmetric error bars from the cluster-bootstrap CI columns (may be
    # NaN, e.g. at P0_unsurveyed where recall is 0 with no defined
    # bootstrap replicates); clip at 0 so a lower bound never goes negative.
    recall = z02["recall"].to_numpy(dtype=float)
    lo = z02["recall_ci_lo"].to_numpy(dtype=float)
    hi = z02["recall_ci_hi"].to_numpy(dtype=float)
    has_ci = ~np.isnan(lo) & ~np.isnan(hi)
    yerr_lo = np.where(has_ci, np.clip(recall - lo, 0, None), 0.0)
    yerr_hi = np.where(has_ci, np.clip(hi - recall, 0, None), 0.0)
    ax.errorbar(z02["x"], recall, yerr=[yerr_lo, yerr_hi], fmt="o-",
                color="tab:red", linewidth=2.5, capsize=3,
                label="RJK-Z02 recall (95% cluster-bootstrap CI)")
    ax.plot(z02["x"], z02["false_alarm_rate"], marker="s", color="tab:orange",
            linewidth=2.5, linestyle="--", label="RJK-Z02 false-alarm rate")
    ax.set_xticks(list(level_x.values()))
    ax.set_xticklabels(list(level_x.keys()))
    ax.set_ylabel("rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGURES, "policy_recall_vs_precision.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_case_study_timeseries(policy_eval: pd.DataFrame, seed: int = 395,
                              level: str = "P3_high") -> str:
    """Re-simulate one specific test-split world (the most sustained real
    breach in the test split) and plot its full-day RJK-Z02 occupancy trace
    against the true safety threshold and the conformal policy's alert
    threshold at `level`, so a reader can see what a real breach actually
    looks like, not just its summary statistics. The margin is read from
    the same policy-evaluation output used for the manuscript tables, so
    the visual cannot silently fall back to a zero-margin alert rule."""
    true_geometry = zn.draw_true_geometry(_rng(seed, 0))
    engine_seed = int(_rng(seed, 1).integers(0, 2**31 - 1))
    run = engine.run_single(true_geometry, seed=engine_seed)
    series = run["occupancy"]["RJK-Z02"]
    true_capacity = run["true_capacity"]["RJK-Z02"]
    grid = [i * run["record_interval"] for i in range(len(series))]

    level_idx = list(zn.PRECISION_LEVELS).index(level)
    estimate = zn.estimate_geometry(true_geometry, level, _rng(seed, 2, level_idx))
    est_threshold = estimate["RJK-Z02"] * zn.DENSITY_THRESHOLD_PERSONS_PER_M2
    margin = float(policy_eval[
        (policy_eval.policy_type == "conformal")
        & (policy_eval.level == level)
        & (policy_eval.zone == "RJK-Z02")
    ].iloc[0]["margin"])

    predicted = [policy.predict_occupancy_ahead(series, i, HORIZON_STEPS)
                 for i in range(len(series) - HORIZON_STEPS)]
    alert_t = [grid[i] for i, p in enumerate(predicted)
               if p + margin > est_threshold]

    fig, ax = plt.subplots(figsize=(9, 2.9))
    ax.plot(grid, series, color="tab:blue", linewidth=1.5, label="RJK-Z02 occupancy")
    ax.axhline(true_capacity, color="black", linestyle=":", label="true safety threshold")
    ax.axhline(est_threshold, color="tab:green", linestyle="--",
               label=f"{level} estimated threshold")
    if alert_t:
        ax.scatter(alert_t, [series[grid.index(t)] for t in alert_t],
               color="tab:red", s=10, zorder=5,
               label="alert fires (prediction + conformal margin > threshold)")
    for show_t in (90, 330):
        ax.axvspan(show_t, show_t + 45, color="grey", alpha=0.15)
    ax.set_xlabel("minutes after opening")
    ax.set_ylabel("occupancy (persons)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIGURES, "case_study_timeseries.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_methodology_diagram() -> str:
    """Schematic contrast: how capacity limits are set in current practice
    (a static occupant-load-factor lookup, MUSERA's own real per-ticket
    quota today) versus this paper's proposed module (geometry-uncertainty
    axis -> simulation -> split-conformal calibrated alert). Purely a
    diagram of the pipeline already implemented elsewhere in this
    prototype (sim/zones.py, sim/engine.py, sim/policy.py) and of facts
    already cited in the manuscript (MUSERA's real seat-quota state,
    IBC/NFPA occupant-load-factor standard, Fruin/Purple Guide density
    threshold) -- no new claim or number is introduced here. The [N]
    citation markers baked into the box text below are hand-synced with
    the manuscript's reference list; if a reference is ever added,
    removed, or reordered, re-check these against the current numbering
    before re-rendering, the same way the manuscript's own in-text
    citations must be re-checked."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    box_style = dict(boxstyle="round,pad=0.35", linewidth=1.1)

    def _column(ax, title, steps, box_color, edge_color):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=8)
        n = len(steps)
        y_positions = [1 - (i + 0.5) / n for i in range(n)]
        box_h = 0.9 / n * 0.62
        for y, text in zip(y_positions, steps):
            ax.add_patch(plt.Rectangle(
                (0.05, y - box_h / 2), 0.9, box_h,
                facecolor=box_color, edgecolor=edge_color, linewidth=1.1,
                transform=ax.transAxes))
            ax.text(0.5, y, text, ha="center", va="center", fontsize=6.6,
                     transform=ax.transAxes, wrap=True)
        for y0, y1 in zip(y_positions[:-1], y_positions[1:]):
            ax.annotate("", xy=(0.5, y1 + box_h / 2 + 0.015),
                        xytext=(0.5, y0 - box_h / 2 - 0.015),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="grey", linewidth=1.2))

    _column(
        axes[0],
        "Current practice (MUSERA today; industry-standard baseline)",
        ["Capacity set once: static occupant-load-factor\nlookup (IBC/NFPA 101) [23] or a fixed\nper-ticket-type seat quota",
         "Crowding monitored (if at all) by manual\nheadcount or CCTV/IoT people-counting against\na fixed density guideline [24, 25]",
         "Alert: reactive only -- an operator\nnotices a crowded zone after the fact,\nno predictive lead time, no precision axis"],
        box_color="#f2f2f2", edge_color="#888888",
    )
    _column(
        axes[1],
        "Proposed module (this paper)",
        ["Geometry-uncertainty axis: true geometry\nvs. survey estimate at 4 precision levels\n(unsurveyed / low / medium / high)",
         "Hybrid ABM+DES simulation: NHPP visitor\narrivals, gate/vendor queues, BFS-routed\ncrowd flow -> per-zone occupancy trace",
         "Naive occupancy predictor (current + trend),\nmargin set by split-conformal calibration\nat the world's survey precision level",
         "Alert policy evaluated against that world's\nown true dynamics: recall, precision,\nfalse-alarm rate reported per precision level"],
        box_color="#fdecec", edge_color="tab:red",
    )
    fig.tight_layout()
    out = os.path.join(FIGURES, "methodology_diagram.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def summary_table(raw_runs: pd.DataFrame, policy_eval: pd.DataFrame) -> pd.DataFrame:
    conformal = policy_eval[policy_eval.policy_type == "conformal"]
    test = raw_runs[raw_runs.split == "test"]
    rows = []
    for level in zn.PRECISION_LEVELS:
        sub = test[test.level == level]
        z02_peak = sub["peak_occupancy_RJK-Z02"]
        overall_peak = sub[[f"peak_occupancy_{z}" for z in zn.ZONES]].max(axis=1)
        viol_minutes = sub["safety_violation_minutes_RJK-Z02"]
        vendor_wait = pd.concat([sub["vendor_wait_mean_RJK-Z07"], sub["vendor_wait_mean_RJK-Z08"]])
        pol_z02 = conformal[(conformal.level == level) & (conformal.zone == "RJK-Z02")].iloc[0]
        rows.append({
            "level": level,
            "peak_occ_Z02_mean": z02_peak.mean(), "peak_occ_Z02_sd": z02_peak.std(),
            "peak_occ_overall_mean": overall_peak.mean(), "peak_occ_overall_sd": overall_peak.std(),
            "gate_wait_p95_mean": sub["gate_wait_p95"].mean(),
            "vendor_wait_mean": vendor_wait.mean(),
            "safety_violation_minutes_Z02_mean": viol_minutes.mean(),
            "safety_violation_minutes_Z02_sd": viol_minutes.std(),
            "policy_recall_Z02": pol_z02["recall"],
            "policy_recall_Z02_ci_lo": pol_z02.get("recall_ci_lo"),
            "policy_recall_Z02_ci_hi": pol_z02.get("recall_ci_hi"),
            "policy_precision_Z02": pol_z02["precision"],
            "policy_false_alarm_rate_Z02": pol_z02["false_alarm_rate"],
            "policy_false_alarm_rate_Z02_ci_lo": pol_z02.get("far_ci_lo"),
            "policy_false_alarm_rate_Z02_ci_hi": pol_z02.get("far_ci_hi"),
        })
    return pd.DataFrame(rows)


def policy_comparison_table(policy_eval: pd.DataFrame) -> pd.DataFrame:
    """Conformal-calibrated vs. naive zero-margin policy, RJK-Z02, at every
    precision level -- isolates what split-conformal calibration itself
    contributes, holding survey precision fixed."""
    z02 = policy_eval[policy_eval.zone == "RJK-Z02"]
    rows = []
    for level in zn.PRECISION_LEVELS:
        conf = z02[(z02.level == level) & (z02.policy_type == "conformal")].iloc[0]
        naive = z02[(z02.level == level) & (z02.policy_type == "naive_zero_margin")].iloc[0]
        rows.append({
            "level": level,
            "conformal_recall": conf["recall"], "naive_recall": naive["recall"],
            "conformal_recall_ci_lo": conf.get("recall_ci_lo"),
            "conformal_recall_ci_hi": conf.get("recall_ci_hi"),
            "conformal_false_alarm_rate": conf["false_alarm_rate"],
            "naive_false_alarm_rate": naive["false_alarm_rate"],
            "conformal_margin": conf["margin"],
        })
    return pd.DataFrame(rows)


def comparator_table(policy_eval: pd.DataFrame) -> pd.DataFrame:
    """RJK-Z02: the paper's policy vs. the no-trend (persistence) predictor at
    every survey level, plus the oracle row (true geometry, trend+conformal)."""
    z02 = policy_eval[policy_eval.zone == "RJK-Z02"]
    rows = []
    for level in zn.PRECISION_LEVELS:
        c = z02[(z02.level == level) & (z02.policy_type == "conformal")].iloc[0]
        pz = z02[(z02.level == level) & (z02.policy_type == "persistence_conformal")].iloc[0]
        rows.append({"policy": "trend + conformal (paper)", "level": level,
                     "recall": c["recall"], "recall_ci_lo": c.get("recall_ci_lo"),
                     "recall_ci_hi": c.get("recall_ci_hi"), "false_alarm_rate": c["false_alarm_rate"],
                     "margin": c["margin"]})
        rows.append({"policy": "no-trend + conformal", "level": level,
                     "recall": pz["recall"], "recall_ci_lo": pz.get("recall_ci_lo"),
                     "recall_ci_hi": pz.get("recall_ci_hi"), "false_alarm_rate": pz["false_alarm_rate"],
                     "margin": pz["margin"]})
    o = z02[z02.policy_type == "oracle_conformal"].iloc[0]
    rows.append({"policy": "oracle (true geometry) + conformal", "level": "P_oracle",
                 "recall": o["recall"], "recall_ci_lo": o.get("recall_ci_lo"),
                 "recall_ci_hi": o.get("recall_ci_hi"), "false_alarm_rate": o["false_alarm_rate"],
                 "margin": o["margin"]})
    return pd.DataFrame(rows)


def per_zone_table(raw_runs: pd.DataFrame, policy_eval: pd.DataFrame,
                    level: str = "P3_high") -> pd.DataFrame:
    """All ten zones, test split, at one precision level -- shows why the
    manuscript's narrative focuses on RJK-Z02: it is not cherry-picked, it
    is the only zone with any real safety-threshold breach in the test
    split at any precision level."""
    test = raw_runs[(raw_runs.split == "test") & (raw_runs.level == level)]
    conformal = policy_eval[(policy_eval.policy_type == "conformal") & (policy_eval.level == level)]
    rows = []
    for z in zn.ZONES:
        peak = test[f"peak_occupancy_{z}"]
        viol = test[f"safety_violation_minutes_{z}"]
        pol = conformal[conformal.zone == z].iloc[0]
        rows.append({
            "zone": z, "role": zn.ZONE_ROLES[z],
            "peak_occ_mean": peak.mean(), "peak_occ_sd": peak.std(),
            "violation_minutes_total": viol.sum(),
            "n_seeds_with_violation": int((viol > 0).sum()),
            "policy_recall": pol["recall"], "policy_false_alarm_rate": pol["false_alarm_rate"],
        })
    return pd.DataFrame(rows)


def _write_table(df: pd.DataFrame, name: str) -> None:
    csv_path = os.path.join(RESULTS, f"{name}.csv")
    md_path = os.path.join(RESULTS, f"{name}.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def main() -> None:
    os.makedirs(FIGURES, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    raw_runs = _load_raw_runs()
    policy_eval = pd.read_csv(os.path.join(RESULTS, "policy_evaluation.csv"))

    for p in (fig_methodology_diagram(),
              fig_occupancy_by_precision(raw_runs),
              fig_policy_recall_vs_precision(policy_eval),
              fig_case_study_timeseries(policy_eval)):
        print(f"wrote {p}")

    _write_table(summary_table(raw_runs, policy_eval), "summary_table")
    _write_table(policy_comparison_table(policy_eval), "policy_comparison_table")
    _write_table(per_zone_table(raw_runs, policy_eval), "per_zone_table")
    _write_table(comparator_table(policy_eval), "comparator_table")


if __name__ == "__main__":
    main()
