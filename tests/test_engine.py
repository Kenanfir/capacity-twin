import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sim import engine, zones as zn  # noqa: E402

FIXED_SEEDS = [0, 1, 2, 3, 4]


def _true_geometry(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return zn.draw_true_geometry(rng)


@pytest.mark.parametrize("seed", FIXED_SEEDS)
def test_conservation(seed):
    run = engine.run_single(_true_geometry(seed), seed=seed)
    assert run["n_admitted"] <= run["n_arrivals"]
    assert run["n_departures"] <= run["n_admitted"]
    assert run["n_departures"] + run["n_still_inside"] == run["n_admitted"]


@pytest.mark.parametrize("seed", FIXED_SEEDS)
def test_occupancy_never_negative(seed):
    run = engine.run_single(_true_geometry(seed), seed=seed)
    for z in zn.ZONES:
        assert (run["occupancy"][z] >= 0).all(), f"zone {z} went negative"


@pytest.mark.parametrize("seed", FIXED_SEEDS)
def test_bounded_runtime(seed):
    start = time.time()
    engine.run_single(_true_geometry(seed), seed=seed)
    elapsed = time.time() - start
    assert elapsed < 30.0, f"seed {seed} took {elapsed:.1f}s -- possible infinite loop"


def test_deterministic_given_same_inputs():
    geometry = _true_geometry(42)
    run_a = engine.run_single(geometry, seed=42)
    run_b = engine.run_single(geometry, seed=42)
    for z in zn.ZONES:
        assert (run_a["occupancy"][z] == run_b["occupancy"][z]).all()
    assert run_a["n_admitted"] == run_b["n_admitted"]
    assert run_a["n_departures"] == run_b["n_departures"]
