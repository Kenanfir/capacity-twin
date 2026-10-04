"""Non-homogeneous Poisson arrival process, modulated by a gamelan-show
schedule, via Lewis-Shedler thinning.
"""
from __future__ import annotations

import numpy as np

SHOW_TIMES: tuple[float, ...] = (90.0, 330.0)   # minutes after open: 10:30, 14:30
SHOW_DURATION = 45.0
DAY_LENGTH = 480.0                              # 09:00-17:00
BASELINE_RATE = 0.5                             # groups/min
PEAK_RATE = 3.0                                 # groups/min, in the 60 min before a show
RAMP_MINUTES = 60.0
DECAY_MINUTES = 20.0

_GROUP_SIZES = np.array([1, 2, 3, 4, 5])
_GROUP_SIZE_PROBS = np.array([0.25, 0.35, 0.20, 0.12, 0.08])


def arrival_rate(t: float) -> float:
    """BASELINE_RATE, ramping linearly to PEAK_RATE over the RAMP_MINUTES
    before each show time, then decaying linearly back to BASELINE_RATE over
    DECAY_MINUTES after the show starts. Never below BASELINE_RATE."""
    rate = BASELINE_RATE
    for show_t in SHOW_TIMES:
        ramp_start = show_t - RAMP_MINUTES
        decay_end = show_t + DECAY_MINUTES
        if ramp_start <= t <= show_t:
            frac = (t - ramp_start) / RAMP_MINUTES
            rate = max(rate, BASELINE_RATE + frac * (PEAK_RATE - BASELINE_RATE))
        elif show_t < t <= decay_end:
            frac = (t - show_t) / DECAY_MINUTES
            rate = max(rate, PEAK_RATE - frac * (PEAK_RATE - BASELINE_RATE))
    return rate


def _in_pre_show_window(t: float) -> bool:
    return any(show_t - RAMP_MINUTES <= t <= show_t for show_t in SHOW_TIMES)


def simulate_arrivals(rng: np.random.Generator) -> list[dict]:
    """Lewis-Shedler thinning against arrival_rate, upper-bounded by
    PEAK_RATE. Returns a time-ordered list of
    {'t': float, 'group_size': int, 'performance_goer': bool}."""
    events: list[dict] = []
    t = 0.0
    while t < DAY_LENGTH:
        t += float(rng.exponential(1.0 / PEAK_RATE))
        if t >= DAY_LENGTH:
            break
        if rng.uniform(0.0, PEAK_RATE) <= arrival_rate(t):
            group_size = int(rng.choice(_GROUP_SIZES, p=_GROUP_SIZE_PROBS))
            p_perf = 0.6 if _in_pre_show_window(t) else 0.15
            performance_goer = bool(rng.uniform() < p_perf)
            events.append({"t": t, "group_size": group_size, "performance_goer": performance_goer})
    events.sort(key=lambda e: e["t"])
    return events
