#!/usr/bin/env python3
r"""Exact in-zone solver (opt-in; v0's greedy-descent path is unchanged).

Agreed in docs/plan.md, "Amendment after Opus run 3", P2:
  - zones of <= 12 stops: Held–Karp dynamic programming (provably optimal);
  - larger zones: OR-Tools GUIDED_LOCAL_SEARCH with a fixed 1 s limit (v0_decode.solve_zone_path).
No other knobs.

Objective: IDENTICAL to v0_decode.solve_zone_path. A path starts at `start` (the last stop already
driven; not part of the zone), visits every stop in `stops` exactly once, and pays
    costs[start][s1] + costs[s1][s2] + ... + costs[s_{k-1}][s_k] + exit_cost[s_k]
where costs are the same integer travel times (tenths of a second) and exit_cost[s] is the cost of
leaving s for the next zone (the imaginary END node in solve_zone_path; start -> END is forbidden
there, and here the path always has >= 1 stop, so the same).

Held–Karp in one sentence: best[S, j] = the cheapest way to leave `start`, visit exactly the set S
of zone stops and stand at stop j. It is built from smaller sets: best[S, j] = min over i in S\{j}
of best[S\{j}, i] + cost(i, j). The answer is min over j of best[all, j] + exit_cost[j].
Work: 2^k * k^2 steps (k = 12: about 0.6 million), instead of k! orders (12! = 479 million).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v0_decode import solve_zone_path  # noqa: E402
from v0_model import zone_membership  # noqa: E402

DP_MAX = 12
GLS_SECONDS = 1.0


def path_cost(costs, start, path, exit_cost):
    """The solve_zone_path objective of a given path."""
    total, prev = 0, start
    for s in path:
        total += costs[prev][s]
        prev = s
    return total + exit_cost[path[-1]]


def held_karp(costs, start, stops, exit_cost):
    """Optimal open path (see module docstring). Deterministic: ties go to the lowest index."""
    k = len(stops)
    C = np.array([[costs[a][b] for b in stops] for a in stops], dtype=np.int64)
    first = np.array([costs[start][s] for s in stops], dtype=np.int64)
    ex = np.array([exit_cost[s] for s in stops], dtype=np.int64)
    INF = np.iinfo(np.int64).max // 4
    full = (1 << k) - 1
    best = np.full((1 << k, k), INF, dtype=np.int64)
    parent = np.full((1 << k, k), -1, dtype=np.int8)
    for j in range(k):
        best[1 << j, j] = first[j]
    for mask in range(1, full + 1):
        row = best[mask]
        if row.min() >= INF:
            continue
        cand = row[:, None] + C                     # cand[i, j]: be at i (in mask), then go to j
        arg = cand.argmin(axis=0)
        val = cand[arg, np.arange(k)]
        for j in range(k):
            if mask >> j & 1:
                continue
            nm = mask | (1 << j)
            if val[j] < best[nm, j]:
                best[nm, j] = val[j]
                parent[nm, j] = arg[j]
    end = int(np.argmin(best[full] + ex))
    order, mask, j = [], full, end
    while j != -1:
        order.append(j)
        pj = int(parent[mask, j])
        mask ^= 1 << j
        j = pj
    return [stops[i] for i in reversed(order)]


def solve_zone_path_exact(costs, start, stops, exit_cost):
    if len(stops) == 1:
        return list(stops)
    if len(stops) <= DP_MAX:
        return held_karp(costs, start, list(stops), exit_cost)
    return solve_zone_path(costs, start, stops, exit_cost, time_limit_s=GLS_SECONDS)


def stops_from_zone_order_exact(route, zone_order):
    """Same as v0_decode.stops_from_zone_order, with the exact in-zone solver."""
    names, zone_of = zone_membership(route)
    if sorted(zone_order) != names:
        raise ValueError("zone_order must list every zone once")
    index = {z: k for k, z in enumerate(names)}
    members = [np.where(zone_of == k)[0] for k in range(len(names))]
    costs = np.rint(route["travel_time"].astype(np.float64) * 10).astype(np.int64).tolist()
    order = [0]
    for k, name in enumerate(zone_order):
        nxt = members[index[zone_order[k + 1]]] if k + 1 < len(zone_order) else [0]
        stops = [int(i) for i in members[index[name]]]
        exit_cost = {s: min(costs[s][int(j)] for j in nxt) for s in stops}
        order += solve_zone_path_exact(costs, order[-1], stops, exit_cost)
    ids = route["stop_ids"]
    assert sorted(order) == list(range(len(ids)))
    return [str(ids[i]) for i in order]
