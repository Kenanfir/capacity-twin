"""Flatten a single run() output into scalar summary statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import zones as zn


def summarize_run(run: dict) -> dict:
    out: dict = {}
    for z in zn.ZONES:
        occ = run["occupancy"][z]
        out[f"peak_occupancy_{z}"] = float(occ.max())
        viol_minutes = float(run["safety_violations"][z].sum() * run["record_interval"])
        out[f"safety_violation_minutes_{z}"] = viol_minutes

    gate_wait = np.asarray(run["gate_wait"]) if run["gate_wait"] else np.asarray([0.0])
    out["gate_wait_mean"] = float(np.mean(gate_wait))
    out["gate_wait_p95"] = float(np.quantile(gate_wait, 0.95))

    for z in ("RJK-Z07", "RJK-Z08"):
        waits = run["vendor_wait"][z]
        w = np.asarray(waits) if waits else np.asarray([0.0])
        out[f"vendor_wait_mean_{z}"] = float(np.mean(w))
        out[f"vendor_completions_{z}"] = int(run["vendor_completions"][z])

    out["total_admissions"] = float(run["admissions_per_hour"].sum())
    out["n_arrivals"] = run["n_arrivals"]
    out["n_admitted"] = run["n_admitted"]
    out["n_departures"] = run["n_departures"]
    out["n_still_inside"] = run["n_still_inside"]
    return out


def aggregate_runs(summaries: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(summaries)
