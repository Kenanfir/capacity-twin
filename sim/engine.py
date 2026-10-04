"""Hand-rolled discrete-event simulation of a single operating day.

Runs entirely against the *true* (still-synthetic) geometry -- the policy's
belief about capacity, built from an estimate geometry at some survey
precision level, is evaluated separately (see sim/policy.py and
run_experiment.py) against this run's actual occupancy trace.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np

from . import arrivals as arr
from . import zones as zn

DAY_LENGTH = arr.DAY_LENGTH
RECORD_INTERVAL_DEFAULT = 1.0

GATE_SERVERS = 3
GATE_SERVICE_MEAN = 1.5
GATE_SERVICE_SIGMA = 0.4
EXHIBIT_DWELL_MEAN = 8.0
EXHIBIT_DWELL_SIGMA = 0.5
VENDOR_SERVICE_MEAN = 4.0
VENDOR_SERVICE_SIGMA = 0.5
VENDOR_SERVERS = 2
MAX_OVERFLOW_RETRIES = 3


@dataclass
class Group:
    gid: int
    size: int
    performance_goer: bool
    entered_site_t: float
    visit_budget: int
    time_cap: float
    zones_visited: int = 0
    current_zone: str | None = None


def _weighted_choice(rng: np.random.Generator, candidates: list[str], weights: list[float]) -> str:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return str(rng.choice(candidates, p=w))


def _show_active(t: float) -> str | None:
    for show_t in arr.SHOW_TIMES:
        if show_t <= t < show_t + arr.SHOW_DURATION:
            return "active"
    return None


def _in_pre_show_or_show(t: float) -> bool:
    for show_t in arr.SHOW_TIMES:
        if show_t - arr.RAMP_MINUTES <= t < show_t + arr.SHOW_DURATION:
            return True
    return False


def run_single(true_geometry: dict[str, float], seed: int,
                record_interval: float = RECORD_INTERVAL_DEFAULT) -> dict:
    rng = np.random.default_rng(seed)
    adjacency = zn.adjacency()
    true_capacity = zn.capacity_from_geometry(true_geometry)
    hard_capacity = zn.hard_capacity_from_geometry(true_geometry)
    dist_to_z02 = zn.distances_from("RJK-Z02")

    # ---- Stage 1: arrivals + gate queue (exact GATE_SERVERS-server FIFO,
    # earliest-available-server assignment) ----
    raw_arrivals = arr.simulate_arrivals(rng)
    gate_wait: list[float] = []
    groups: list[Group] = []
    gate_servers_free_until = [0.0] * GATE_SERVERS
    for gid, a in enumerate(raw_arrivals):
        service_time = float(rng.lognormal(mean=np.log(GATE_SERVICE_MEAN), sigma=GATE_SERVICE_SIGMA))
        idx = min(range(GATE_SERVERS), key=lambda i: gate_servers_free_until[i])
        service_start = max(a["t"], gate_servers_free_until[idx])
        service_end = service_start + service_time
        gate_wait.append(service_start - a["t"])
        gate_servers_free_until[idx] = service_end
        k = int(rng.integers(3, 7))
        time_cap = float(rng.triangular(60.0, 120.0, 180.0))
        groups.append(Group(
            gid=gid, size=a["group_size"], performance_goer=a["performance_goer"],
            entered_site_t=service_end, visit_budget=k, time_cap=time_cap,
        ))

    # ---- Stage 2: interior routing + zone dwell, event-driven ----
    seq = itertools.count()
    heap: list[tuple[float, int, str, dict]] = []

    def push(t: float, etype: str, payload: dict) -> None:
        heapq.heappush(heap, (t, next(seq), etype, payload))

    occupancy = {z: 0 for z in zn.ZONES}
    occ_events: list[tuple[float, str, int]] = []  # (t, zone, occupancy_after)
    vendor_servers_free_until = {z: [0.0] * VENDOR_SERVERS for z in ("RJK-Z07", "RJK-Z08")}
    vendor_wait: dict[str, list[float]] = {"RJK-Z07": [], "RJK-Z08": []}
    vendor_completions = {"RJK-Z07": 0, "RJK-Z08": 0}
    admitted_gids: set[int] = set()
    departed_gids: set[int] = set()

    def record(t: float, zone: str) -> None:
        occ_events.append((t, zone, occupancy[zone]))

    def choose_zone(rng, g: Group, exclude: set[str]) -> str | None:
        current = g.current_zone or "RJK-Z01"
        candidates = [z for z in adjacency[current] if z not in exclude]
        if not candidates:
            return None
        weights = []
        pulled = g.performance_goer and _in_pre_show_or_show(t_now[0])
        cur_dist = dist_to_z02.get(current, 99)
        for z in candidates:
            w = 1.0
            if pulled:
                if z == "RJK-Z02":
                    w = 6.0
                elif dist_to_z02.get(z, 99) < cur_dist:
                    w = 3.0
                elif dist_to_z02.get(z, 99) > cur_dist:
                    w = 0.3
            weights.append(w)
        return _weighted_choice(rng, candidates, weights)

    t_now = [0.0]  # mutable box so choose_zone can see "current event time"

    def try_enter(g: Group, t: float) -> str:
        """Weighted routing with overflow diversion. Diversion is checked
        against the *hard physical* limit, not the safety-density threshold
        -- crowds are allowed to exceed the safety threshold (that is the
        event a capacity policy needs to catch), they are only physically
        prevented from exceeding the much higher crush limit. Returns the
        zone actually entered (may fall back to the hub)."""
        t_now[0] = t
        exclude: set[str] = set()
        for _ in range(MAX_OVERFLOW_RETRIES):
            z = choose_zone(rng, g, exclude)
            if z is None:
                break
            if occupancy[z] + g.size <= hard_capacity[z]:
                return z
            exclude.add(z)
        return "RJK-Z09"

    def schedule_zone_enter(g: Group, t: float) -> None:
        z = try_enter(g, t)
        g.current_zone = z
        g.zones_visited += 1
        occupancy[z] += g.size
        record(t, z)

        if z in vendor_servers_free_until:
            servers = vendor_servers_free_until[z]
            idx = min(range(VENDOR_SERVERS), key=lambda i: servers[i])
            service_start = max(t, servers[idx])
            service_time = float(rng.lognormal(mean=np.log(VENDOR_SERVICE_MEAN), sigma=VENDOR_SERVICE_SIGMA))
            service_end = service_start + service_time
            servers[idx] = service_end
            vendor_wait[z].append(service_start - t)
            push(service_end, "ZONE_EXIT", {"gid": g.gid, "zone": z, "is_completion": True})
        elif z == "RJK-Z02" and _show_active(t):
            show_end = next(st + arr.SHOW_DURATION for st in arr.SHOW_TIMES
                             if st <= t < st + arr.SHOW_DURATION)
            push(show_end, "ZONE_EXIT", {"gid": g.gid, "zone": z, "is_completion": False})
        else:
            dwell = float(rng.lognormal(mean=np.log(EXHIBIT_DWELL_MEAN), sigma=EXHIBIT_DWELL_SIGMA))
            push(t + dwell, "ZONE_EXIT", {"gid": g.gid, "zone": z, "is_completion": False})

    for g in groups:
        push(g.entered_site_t, "FIRST_ENTER", {"gid": g.gid})

    by_gid = {g.gid: g for g in groups}

    while heap:
        t, _, etype, payload = heapq.heappop(heap)
        if t > DAY_LENGTH:
            g = by_gid[payload["gid"]]
            if etype == "ZONE_EXIT" and g.current_zone is not None:
                occupancy[g.current_zone] = max(0, occupancy[g.current_zone] - g.size)
                record(min(t, DAY_LENGTH), g.current_zone)
            continue
        g = by_gid[payload["gid"]]

        if etype == "FIRST_ENTER":
            admitted_gids.add(g.gid)
            schedule_zone_enter(g, t)
            continue

        # ZONE_EXIT
        zone = payload["zone"]
        occupancy[zone] = max(0, occupancy[zone] - g.size)
        record(t, zone)
        if payload.get("is_completion") and zone in vendor_completions:
            vendor_completions[zone] += 1

        if g.zones_visited >= g.visit_budget or (t - g.entered_site_t) >= g.time_cap:
            departed_gids.add(g.gid)
            continue
        schedule_zone_enter(g, t)

    grid = np.arange(0.0, DAY_LENGTH + record_interval, record_interval)
    occupancy_series: dict[str, np.ndarray] = {}
    for z in zn.ZONES:
        events_z = sorted((t, occ) for (t, zz, occ) in occ_events if zz == z)
        series = np.zeros(len(grid))
        idx = 0
        current = 0
        for i, gt in enumerate(grid):
            while idx < len(events_z) and events_z[idx][0] <= gt:
                current = events_z[idx][1]
                idx += 1
            series[i] = current
        occupancy_series[z] = series

    safety_violations = {
        z: occupancy_series[z] > true_capacity[z] for z in zn.ZONES
    }

    admissions_per_hour = np.zeros(8)
    for g in groups:
        if g.gid in admitted_gids:
            bucket = min(int(g.entered_site_t // 60), 7)
            admissions_per_hour[bucket] += 1

    still_inside = admitted_gids - departed_gids

    return {
        "occupancy": occupancy_series,
        "gate_wait": gate_wait,
        "vendor_wait": vendor_wait,
        "admissions_per_hour": admissions_per_hour,
        "vendor_completions": vendor_completions,
        "safety_violations": safety_violations,
        "true_capacity": true_capacity,
        "n_arrivals": len(groups),
        "n_admitted": len(admitted_gids),
        "n_departures": len(departed_gids),
        "n_still_inside": len(still_inside),
        "seed": seed,
        "record_interval": record_interval,
    }
