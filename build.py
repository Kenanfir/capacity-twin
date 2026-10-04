#!/usr/bin/env python3
"""Build and verify the capacity-twin simulation prototype.

    python3 build.py            rerun the experiment + analysis
    python3 build.py --check    rebuild, then run every gate; exit 0 only if
                                 all of them pass

A say(ok, name, detail) gate tracker; exit 0 only on an all-PASS run. A gate
that cannot currently run counts as a FAIL, never a silent skip.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
FAILS: list[str] = []


def say(ok: bool, name: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def run(cmd: list[str], cwd: str = HERE) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def build(n_runs: int = 600) -> None:
    print("== experiment ==")
    r = run([sys.executable, "run_experiment.py", "--n-runs", str(n_runs)])
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-3000:])
        sys.exit("run_experiment.py failed")
    print(r.stdout.strip())

    print("\n== analysis ==")
    r = run([sys.executable, "make_outputs.py"], cwd=os.path.join(HERE, "analysis"))
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-3000:])
        sys.exit("analysis/make_outputs.py failed")
    print(r.stdout.strip())


EXPECTED_FIGURES = ["methodology_diagram.png", "occupancy_by_precision.png",
                     "policy_recall_vs_precision.png", "case_study_timeseries.png"]
EXPECTED_TABLES = ["summary_table.csv", "policy_comparison_table.csv", "per_zone_table.csv"]


def _load_raw_runs(directory: str) -> pd.DataFrame:
    parquet_path = os.path.join(directory, "raw_runs.parquet")
    csv_path = os.path.join(directory, "raw_runs.csv")
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ModuleNotFoundError, ValueError):
            # CSV is the committed interchange format; parquet is an
            # optional acceleration path when an engine is installed.
            pass
    return pd.read_csv(csv_path)


def check(n_runs: int = 600) -> None:
    print(f"\n== checks ==")

    # 1 -- pytest suite
    r = run([sys.executable, "-m", "pytest", "tests/", "-q"])
    say(r.returncode == 0, "pytest tests/ passes",
        r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:])
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-2000:])

    # 2 -- reproducibility: re-run into a temp dir, compare to committed results
    with tempfile.TemporaryDirectory() as tmp:
        r = run([sys.executable, "run_experiment.py", "--n-runs", str(n_runs), "--out", tmp])
        if r.returncode != 0:
            say(False, "reproducibility re-run", "run_experiment.py failed on rerun")
            print(r.stdout[-3000:], r.stderr[-2000:])
        else:
            try:
                committed = _load_raw_runs(RESULTS).sort_values(
                    list(_load_raw_runs(RESULTS).columns)).reset_index(drop=True)
                rerun = _load_raw_runs(tmp).sort_values(
                    list(_load_raw_runs(tmp).columns)).reset_index(drop=True)
                identical = committed.equals(rerun)
                say(identical, "reproducibility: identical raw_runs across two runs",
                    "DataFrame.equals() after sorting" if identical else "mismatch found")
            except Exception as e:
                say(False, "reproducibility re-run", f"comparison failed: {e}")

    # 3 -- finite, non-negative occupancy values
    try:
        raw = _load_raw_runs(RESULTS)
        occ_cols = [c for c in raw.columns if c.startswith("peak_occupancy_")]
        vals = raw[occ_cols].to_numpy(dtype=float)
        finite = np.isfinite(vals).all()
        nonneg = (vals >= 0).all()
        say(finite and nonneg, "occupancy values finite and non-negative",
            f"{vals.size} values checked")
    except Exception as e:
        say(False, "occupancy values finite and non-negative", f"could not check: {e}")

    # 4 -- every expected figure exists and is non-empty
    for fname in EXPECTED_FIGURES:
        p = os.path.join(FIGURES, fname)
        exists = os.path.exists(p) and os.path.getsize(p) > 0
        say(exists, f"figure exists: {fname}", p)

    # 5 -- summary table exists, non-empty, no *unexplained* NaN.
    # policy_precision_Z02 is a legitimate 0/0 when a level's policy issues
    # zero alerts across the whole test split (e.g. P0_unsurveyed's
    # threshold is so miscalibrated it never fires) -- that is the finding,
    # not a defect, and sim/policy.py already reports it as None rather than
    # a fabricated 0 or a crash. Every other numeric column must be finite.
    summary_path = os.path.join(RESULTS, "summary_table.csv")
    NULLABLE_WHEN_ZERO_ALERTS = {"policy_precision_Z02"}
    try:
        summary = pd.read_csv(summary_path)
        non_empty = len(summary) > 0
        numeric = summary.select_dtypes(include=[float, int])
        strict_cols = [c for c in numeric.columns if c not in NULLABLE_WHEN_ZERO_ALERTS]
        no_unexplained_nan = not numeric[strict_cols].isna().any().any()
        explained = numeric.columns.intersection(NULLABLE_WHEN_ZERO_ALERTS)
        nan_rows = summary[numeric[explained].isna().any(axis=1)] if len(explained) else summary.iloc[0:0]
        detail = f"{len(summary)} rows"
        if len(nan_rows):
            detail += "; undefined precision (0 alerts fired) at: " + ", ".join(nan_rows["level"])
        say(non_empty and no_unexplained_nan, "summary_table.csv non-empty, no unexplained NaN", detail)
    except Exception as e:
        say(False, "summary_table.csv non-empty, no unexplained NaN", f"could not read: {e}")

    # 6 -- the other tables exist and are non-empty (policy_comparison_table
    # and per_zone_table can legitimately carry the same explained NaN as
    # summary_table for the same reason; only check non-emptiness here)
    for fname in EXPECTED_TABLES[1:]:
        p = os.path.join(RESULTS, fname)
        try:
            df = pd.read_csv(p)
            say(len(df) > 0, f"table non-empty: {fname}", f"{len(df)} rows")
        except Exception as e:
            say(False, f"table non-empty: {fname}", f"could not read: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n-runs", type=int, default=600)
    args = ap.parse_args()

    build(n_runs=args.n_runs)
    if args.check:
        check(n_runs=args.n_runs)
        print(f"\n== {'PASS' if not FAILS else 'FAIL'} ==")
        if FAILS:
            print("failed gates: " + ", ".join(FAILS))
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
