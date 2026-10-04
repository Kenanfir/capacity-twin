"""RJK-shaped synthetic zone geometry and the geometry-precision axis.

MUSERA's real ten zones (RJK-Z01..RJK-Z10) carry provisional, null-coordinate
geometry -- no survey exists. Everything below is a synthetic stand-in
recognizable as "ten RJK-shaped zones" (same codes, same rough roles: a
gate, a gamelan performance zone, exhibits, two UMKM vendor clusters, a
plaza hub, an exit) without claiming to be the real surveyed site.
"""
from __future__ import annotations

import numpy as np

ZONE_ROLES: dict[str, str] = {
    "RJK-Z01": "gate",
    "RJK-Z02": "performance",
    "RJK-Z03": "exhibit",
    "RJK-Z04": "exhibit",
    "RJK-Z05": "exhibit",
    "RJK-Z06": "exhibit",
    "RJK-Z07": "vendor",
    "RJK-Z08": "vendor",
    "RJK-Z09": "hub",
    "RJK-Z10": "exit",
}

ZONES: tuple[str, ...] = tuple(ZONE_ROLES)

_NOMINAL_AREA_M2: dict[str, float] = {
    "RJK-Z01": 150.0,
    "RJK-Z02": 100.0,
    "RJK-Z03": 180.0,
    "RJK-Z04": 180.0,
    "RJK-Z05": 180.0,
    "RJK-Z06": 180.0,
    "RJK-Z07": 90.0,
    "RJK-Z08": 90.0,
    "RJK-Z09": 400.0,
    "RJK-Z10": 120.0,
}

# Undirected adjacency, RJK-Z09 is the hub reachable from/to every other zone.
_ADJACENCY_EDGES: tuple[tuple[str, str], ...] = (
    ("RJK-Z01", "RJK-Z03"), ("RJK-Z01", "RJK-Z04"),
    ("RJK-Z02", "RJK-Z03"), ("RJK-Z02", "RJK-Z05"),
    ("RJK-Z03", "RJK-Z04"), ("RJK-Z04", "RJK-Z05"), ("RJK-Z05", "RJK-Z06"),
    ("RJK-Z06", "RJK-Z03"),
    ("RJK-Z09", "RJK-Z01"), ("RJK-Z09", "RJK-Z02"), ("RJK-Z09", "RJK-Z03"),
    ("RJK-Z09", "RJK-Z04"), ("RJK-Z09", "RJK-Z05"), ("RJK-Z09", "RJK-Z06"),
    ("RJK-Z09", "RJK-Z07"), ("RJK-Z09", "RJK-Z08"), ("RJK-Z09", "RJK-Z10"),
)


def adjacency() -> dict[str, list[str]]:
    """Undirected adjacency as a symmetric neighbor-list dict."""
    adj: dict[str, list[str]] = {z: [] for z in ZONES}
    for a, b in _ADJACENCY_EDGES:
        adj[a].append(b)
        adj[b].append(a)
    return adj


# Crowd-safety density threshold; cited in the manuscript, not derived here.
DENSITY_THRESHOLD_PERSONS_PER_M2 = 2.0

PRECISION_LEVELS: tuple[str, ...] = ("P0_unsurveyed", "P1_low", "P2_medium", "P3_high")
PRECISION_SIGMA: dict[str, float] = {"P1_low": 0.5, "P2_medium": 0.2, "P3_high": 0.05}


def nominal_areas() -> dict[str, float]:
    return dict(_NOMINAL_AREA_M2)


def draw_true_geometry(rng: np.random.Generator) -> dict[str, float]:
    """One Monte-Carlo draw of the actual (still-synthetic) site geometry:
    nominal area x lognormal(mean=0, sigma=0.15) per zone, independent per
    zone. Represents "what the real survey might actually find," never a
    known ground truth."""
    nominal = _NOMINAL_AREA_M2
    return {
        z: nominal[z] * float(rng.lognormal(mean=0.0, sigma=0.15))
        for z in ZONES
    }


def estimate_geometry(
    true_geometry: dict[str, float], level: str, rng: np.random.Generator
) -> dict[str, float]:
    """A survey of the same real site at a given precision level.

    P0_unsurveyed: zero site-specific information -- the total nominal
    footprint split evenly across the ten zones, matching MUSERA's real
    null-coordinate state.

    P1/P2/P3: true_geometry[z] x lognormal(mean=0, sigma=PRECISION_SIGMA
    [level]) per zone, independent per zone -- a noisier or cleaner survey
    of the same real site, not a different site.
    """
    if level == "P0_unsurveyed":
        even_share = sum(_NOMINAL_AREA_M2.values()) / len(ZONES)
        return {z: even_share for z in ZONES}
    if level not in PRECISION_SIGMA:
        raise ValueError(f"unknown precision level: {level!r}")
    sigma = PRECISION_SIGMA[level]
    return {
        z: true_geometry[z] * float(rng.lognormal(mean=0.0, sigma=sigma))
        for z in ZONES
    }


def capacity_from_geometry(geometry: dict[str, float]) -> dict[str, float]:
    return {z: area * DENSITY_THRESHOLD_PERSONS_PER_M2 for z, area in geometry.items()}


# The safety-density threshold above is a *soft* limit -- crowds can and do
# exceed it in reality, which is the whole reason a capacity-alert policy is
# useful. HARD_LIMIT_MULTIPLIER is a much higher, genuinely physical crush
# limit used only to keep the simulation's overflow-diversion rule from
# stacking people without bound; it is never the value a policy is judged
# against.
HARD_LIMIT_MULTIPLIER = 2.25


def hard_capacity_from_geometry(geometry: dict[str, float]) -> dict[str, float]:
    return {z: cap * HARD_LIMIT_MULTIPLIER for z, cap in capacity_from_geometry(geometry).items()}


def distances_from(source: str) -> dict[str, int]:
    """BFS shortest-hop distance from `source` to every zone."""
    adj = adjacency()
    dist = {source: 0}
    frontier = [source]
    while frontier:
        nxt = []
        for z in frontier:
            for n in adj[z]:
                if n not in dist:
                    dist[n] = dist[z] + 1
                    nxt.append(n)
        frontier = nxt
    return dist
