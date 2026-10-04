import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sim import arrivals as arr  # noqa: E402


def test_rate_never_below_baseline():
    ts = np.linspace(0.0, arr.DAY_LENGTH, 2000)
    rates = [arr.arrival_rate(float(t)) for t in ts]
    assert min(rates) >= arr.BASELINE_RATE - 1e-9


def test_rate_never_exceeds_peak():
    ts = np.linspace(0.0, arr.DAY_LENGTH, 2000)
    rates = [arr.arrival_rate(float(t)) for t in ts]
    assert max(rates) <= arr.PEAK_RATE + 1e-9


def test_thinning_events_within_day():
    rng = np.random.default_rng(1)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        events = arr.simulate_arrivals(rng)
        for e in events:
            assert 0.0 <= e["t"] < arr.DAY_LENGTH


def test_empirical_rate_matches_rate_function():
    """Over many simulated days, the empirical arrival rate in each time bin
    should match the average of arrival_rate(t) over that bin, within a
    generous statistical tolerance."""
    n_days = 60
    n_bins = 20
    bin_width = arr.DAY_LENGTH / n_bins
    counts = np.zeros(n_bins)
    for seed in range(n_days):
        rng = np.random.default_rng(10_000 + seed)
        events = arr.simulate_arrivals(rng)
        for e in events:
            b = min(int(e["t"] // bin_width), n_bins - 1)
            counts[b] += 1

    empirical_rate = counts / (n_days * bin_width)

    for b in range(n_bins):
        bin_start = b * bin_width
        sample_ts = np.linspace(bin_start, bin_start + bin_width, 20, endpoint=False)
        expected_rate = np.mean([arr.arrival_rate(float(t)) for t in sample_ts])
        # Wide tolerance: this is a statistical sanity check, not an exact match.
        assert empirical_rate[b] == pytest.approx(expected_rate, rel=0.35, abs=0.15), (
            f"bin {b} ({bin_start:.0f}-{bin_start + bin_width:.0f} min): "
            f"empirical={empirical_rate[b]:.3f} expected={expected_rate:.3f}"
        )
