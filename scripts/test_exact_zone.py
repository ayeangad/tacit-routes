#!/usr/bin/env python3
"""Unit test for scripts/exact_zone.py (TRAIN routes only; light).

For real zones taken from train routes (driver zone order, filled zone by zone with v0's greedy
solver to get realistic start stops), check:
  1. Held–Karp cost == brute-force optimum for every zone of 2..8 stops;
  2. Held–Karp cost <= greedy-descent cost (v0_decode.solve_zone_path) for every zone of 2..12 stops;
  3. Held–Karp returns a permutation of the zone's stops.
  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/test_exact_zone.py [n_routes]
"""
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exact_zone import held_karp, path_cost  # noqa: E402
from v0_decode import solve_zone_path  # noqa: E402
from v0_model import SPLITS, driver_zone_order, load_npz, prepare_route  # noqa: E402


def zones_of_route(route):
    P = prepare_route(route)
    names, zone_of = P["names"], P["zone_of"]
    order = driver_zone_order(P, route)
    costs = np.rint(route["travel_time"].astype(np.float64) * 10).astype(np.int64).tolist()
    members = [list(map(int, np.where(zone_of == k)[0])) for k in range(len(names))]
    cur = 0
    for t, z in enumerate(order):
        nxt = members[order[t + 1]] if t + 1 < len(order) else [0]
        stops = members[z]
        exit_cost = {s: min(costs[s][j] for j in nxt) for s in stops}
        yield costs, cur, stops, exit_cost
        cur = solve_zone_path(costs, cur, stops, exit_cost)[-1] if len(stops) > 1 else stops[0]


def main():
    n_routes = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    splits = json.load(open(SPLITS))
    ids = sorted(r["route_id"] for r in splits["train"])
    random.Random(7).shuffle(ids)
    n_brute = n_cmp = n_better = 0
    for rid in ids[:n_routes]:
        for costs, start, stops, ex in zones_of_route(load_npz(rid)):
            k = len(stops)
            if k < 2 or k > 12:
                continue
            hk = held_karp(costs, start, stops, ex)
            assert sorted(hk) == sorted(stops)
            c_hk = path_cost(costs, start, hk, ex)
            c_gd = path_cost(costs, start, solve_zone_path(costs, start, stops, ex), ex)
            assert c_hk <= c_gd, (rid, k, c_hk, c_gd)
            n_cmp += 1
            n_better += c_hk < c_gd
            if k <= 8:
                c_bf = min(path_cost(costs, start, list(p), ex) for p in itertools.permutations(stops))
                assert c_hk == c_bf, (rid, k, c_hk, c_bf)
                n_brute += 1
    print(f"OK: {n_brute} zones matched brute force exactly (2-8 stops); {n_cmp} zones (2-12 stops) "
          f"had DP cost <= greedy descent; DP strictly cheaper on {n_better}.")
    assert n_brute >= 200, "need >= 200 brute-force zones; pass more routes"


if __name__ == "__main__":
    main()
