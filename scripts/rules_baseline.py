#!/usr/bin/env python3
"""Hand-written zone rules vs the plain shortest route, scored with the official scorer.

Run from tacit-routes/ (see results/rules_baseline.md for the full story):

  # 1. choose R2's ID rule and R3's hybrid on 300 random High *train* routes (never val, never pilot)
  .venv/bin/python -I scripts/rules_baseline.py tune --workers 3
  # 2. score S, R1, R2, R3 once on the 300 val routes
  .venv/bin/python -I scripts/rules_baseline.py val --workers 3

Arms
  S   plain shortest route: OR-Tools, closed tour from the station, PATH_CHEAPEST_ARC +
      GUIDED_LOCAL_SEARCH, 10 s per route (identical to run_plain_baseline.py).
  R1  zones as blocks, zone order = TSP over zone "centroids" (mean stop-to-stop travel time
      between two zones), starting and ending at the station.
  R2  zones as blocks, zone order read off the zone IDs (rule chosen on train, see RULES below).
  R3  a hybrid of R1 and R2, chosen on train.
Every rule arm then solves the stops inside each zone with OR-Tools as an open path that starts
at the last stop of the previous zone and ends at a dummy node whose cost is the travel time to
the nearest stop of the next zone (the station after the last zone).

Scoring: the official, unmodified rc-cli score() (scripts/official_rc_cli/score.py), called per
route exactly as its evaluate() does (isinvalid check first, invalid score if it fails). The cost
matrix is rebuilt from the npz (float32) rounded to 0.1 s, which reproduces the raw JSON values.
evaluate() itself needs the whole cost-matrix JSON in memory (too big for 300 routes on this
laptop), so the val run checks on 3 routes that evaluate() and this path give identical scores.
"""
import argparse
import copy
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import re
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCORER = HERE / "official_rc_cli" / "score.py"
ROUTES = ROOT / "data" / "processed" / "routes"
SPLITS = ROOT / "data" / "processed" / "splits.json"
INVALID = ROOT / "data" / "raw" / "invalid_sequence_scores.json"
RESULTS = ROOT / "results"

SEED = 20261111
TUNE_N = 300
N_BOOT = 10_000

# Solver settings (all recorded in the output JSON).
S_TIME_LIMIT_S = 10            # plain shortest route, per route, same as run_plain_baseline.py
ZONE_TSP_TIME_LIMIT_MS = 1000  # zone-order TSP (about 20 zone nodes + station), per route
INNER_TIME_LIMIT_MS = 100      # open path inside one zone, per zone (see INNER note in the .md)

ZONE_RE = re.compile(r"^([A-Z])-(\d+)\.(\d+)([A-Z])$")

# R4-hist: history-weighted zone TSP (a counting control for later neural models).
#   cost(a->b) = D(a,b) + alpha * scale * mult(a) * (-log P(b | a)),  cost(zone -> station) = D only
#   scale = median zone-to-zone mean travel time of the route (so alpha is unit-free)
#   mult(station) = HIST_START_MULT, mult(zone) = 1
#   P(b | a) = (c(a,b) + K) / sum over b' in the route's candidate zones of (c(a,b') + K)
#   c = this station's counts; if this station's row total over the candidates is < BACKOFF_MIN,
#   c = station counts + counts pooled over all stations (same zone-ID pair).
HIST_START = "__START__"
HIST_K = 1.0
HIST_START_MULT = 3.0
HIST_BACKOFF_MIN = 5
HIST_ALPHAS = [0.25, 0.5, 1, 2, 4, 8]
# Round 2, added because round 1's best alpha (0.25) was the smallest tried (edge of the grid).
HIST_ALPHAS_ROUND2 = [0.05, 0.1, 0.15]

# Candidate rules. R2 candidates are pure ID rules; R3 candidates are hybrids.
R2_CANDIDATES = ["id_snake", "id_lex"]
R3_CANDIDATES = ["adj_pen_0.5", "adj_pen_1", "adj_pen_2", "group_tsp", "group_tsp+adj_pen_1"]
# Round 2, added after round 1 showed every round-1 hybrid losing to the pure snake rule on
# train: blends of R1 and the snake rule. Same 300 train routes; val still untouched.
R3_CANDIDATES_ROUND2 = ["snake_pen_0.25", "snake_pen_0.5", "snake_pen_1", "snake_pen_2"]


# ----------------------------------------------------------------------------------- data

def load_route(rid):
    z = np.load(ROUTES / f"{rid}.npz", allow_pickle=False)
    ids = [str(s) for s in z["stop_ids"]]
    tt = np.round(z["travel_time"].astype(np.float64), 1)
    zones = [str(s) for s in z["zone_ids"]]
    if str(z["stop_type"][0]) != "Station":
        raise SystemExit(f"{rid}: index 0 is not the station")
    return {"rid": rid, "ids": ids, "tt": tt, "raw_zones": zones,
            "order": [int(i) for i in z["actual_order"]], "station": str(z["station"])}


def assign_zones(route):
    """Zone per stop index (station gets None). Stops whose zone is missing or not of the form
    A-2.1C get the zone of their nearest zoned stop (by round-trip travel time)."""
    tt, raw = route["tt"], route["raw_zones"]
    n = len(raw)
    ok = [i for i in range(1, n) if ZONE_RE.match(raw[i])]
    zone = [None] * n
    if not ok:  # no usable zone at all: one block
        for i in range(1, n):
            zone[i] = "X-0.0A"
        return zone, n - 1
    for i in ok:
        zone[i] = raw[i]
    attached = 0
    for i in range(1, n):
        if zone[i] is None:
            j = min(ok, key=lambda k: tt[i, k] + tt[k, i])
            zone[i] = raw[j]
            attached += 1
    return zone, attached


# --------------------------------------------------------------------------------- solver

def solve_ortools(cost, start, end, time_limit_ms):
    """Shortest path over all nodes from start to end (start == end gives a closed tour).
    PATH_CHEAPEST_ARC first solution, then GUIDED_LOCAL_SEARCH until the time limit."""
    n = len(cost)
    manager = pywrapcp.RoutingIndexManager(n, 1, [start], [end])
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(
        lambda i, j: cost[manager.IndexToNode(i)][manager.IndexToNode(j)])
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_ms))
    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("OR-Tools found no solution")
    out, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        out.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    if start != end:
        out.append(manager.IndexToNode(idx))
    return out


def as_int(m):
    """OR-Tools needs integers: tenths of a second, as in run_plain_baseline.py."""
    return [[int(round(x * 10)) for x in row] for row in m]


def shortest_route(route, time_limit_s=S_TIME_LIMIT_S):
    return solve_ortools(as_int(route["tt"]), 0, 0, time_limit_s * 1000)


# ---------------------------------------------------------------------------- zone layer

def zone_matrix(route, zone, zlist):
    """Node 0 = station, node k = zlist[k-1]. Entry (a, b) = mean travel time from the stops
    of a to the stops of b (the 'centroid' distance in travel-time terms)."""
    tt = route["tt"]
    members = [[0]] + [[i for i in range(len(zone)) if zone[i] == z] for z in zlist]
    m = len(members)
    D = np.zeros((m, m))
    for a in range(m):
        rows = tt[members[a]]
        for b in range(m):
            if a != b:
                D[a, b] = rows[:, members[b]].mean()
    return D, members


def parse(z):
    s, n1, n2, l = ZONE_RE.match(z).groups()
    return s, int(n1), int(n2), l


def grid_neighbours(zlist):
    """adj[a][b] = True if zones a and b differ in exactly one ID part by one step: num2 by 1,
    num1 by 1, or the final letter by one place among the letters this route uses."""
    P = [parse(z) for z in zlist]
    letters = sorted({p[3] for p in P})
    rank = {l: i for i, l in enumerate(letters)}
    k = len(P)
    adj = np.zeros((k, k), dtype=bool)
    for a in range(k):
        for b in range(k):
            sa, n1a, n2a, la = P[a]
            sb, n1b, n2b, lb = P[b]
            if sa != sb or a == b:
                continue
            diffs = [abs(n1a - n1b), abs(n2a - n2b), abs(rank[la] - rank[lb])]
            if sorted(diffs) == [0, 0, 1]:
                adj[a, b] = True
    return adj


def tour_cost(D, seq):
    """Closed cost station -> seq -> station on the zone matrix (seq holds node numbers >= 1)."""
    path = [0] + list(seq) + [0]
    return sum(D[a, b] for a, b in zip(path, path[1:]))


def order_tsp(D, extra=None):
    C = D if extra is None else D + extra
    tour = solve_ortools(as_int(C), 0, 0, ZONE_TSP_TIME_LIMIT_MS)
    return tour[1:]


def id_snake_variants(zlist):
    """Boustrophedon ('snake') walk through the zone-ID grid, as drivers do in train.
    Group = letter-num1 (e.g. 'D-23'); inside a group, columns = final letter, rows = num2.
    Walk the groups in ID order; inside a group walk columns in letter order, reversing the
    letter direction in each new group; inside a column walk rows, reversing direction at
    every new column so consecutive zones touch. 2 x 2 x 2 = 8 variants (group order, first
    letter direction, first row direction); each variant's reverse is also in the set."""
    P = {z: parse(z) for z in zlist}
    groups = sorted({(p[0], p[1]) for p in P.values()})
    out = []
    for g_dir in (1, -1):
        for l_dir0 in (1, -1):
            for r_dir0 in (1, -1):
                seq, l_dir, r_dir = [], l_dir0, r_dir0
                for g in groups[::g_dir]:
                    zs = [z for z in zlist if (P[z][0], P[z][1]) == g]
                    letters = sorted({P[z][3] for z in zs})[::l_dir]
                    for l in letters:
                        col = sorted((z for z in zs if P[z][3] == l), key=lambda z: P[z][2])
                        seq += col[::r_dir]
                        r_dir = -r_dir
                    l_dir = -l_dir
                out.append(seq)
    return out


def id_lex_variants(zlist):
    """Plain sort by (letter, num1, num2, final letter), ascending or descending."""
    key = sorted(zlist, key=parse)
    return [key, key[::-1]]


def driver_zone_sequence(rid):
    """Driver's zones in order: station dropped, zone-less/odd IDs dropped, repeats collapsed."""
    z = np.load(ROUTES / f"{rid}.npz", allow_pickle=False)
    zs = [str(x) for x in z["zone_ids"][z["actual_order"]]]
    zs = [x for x in zs if ZONE_RE.match(x)]
    return str(z["station"]), [x for i, x in enumerate(zs) if i == 0 or x != zs[i - 1]]


def build_hist(route_ids):
    """Transition counts from driver routes. st[station][a][b] and pool[a][b]; a can be START."""
    st, pool = {}, {}
    for rid in route_ids:
        station, seq = driver_zone_sequence(rid)
        for a, b in zip([HIST_START] + seq, seq):
            row = st.setdefault(station, {}).setdefault(a, {})
            row[b] = row.get(b, 0) + 1
            prow = pool.setdefault(a, {})
            prow[b] = prow.get(b, 0) + 1
    return {"st": st, "pool": pool, "n_routes": len(route_ids)}


def hist_penalty(route, zlist, hist):
    """H[a, b] = -log P(b | a) over the zone-matrix nodes (0 = station). H[:, 0] = 0."""
    k = len(zlist)
    H = np.zeros((k + 1, k + 1))
    st = hist["st"].get(route["station"], {})
    names = [HIST_START] + zlist
    backed_off = 0
    for a in range(k + 1):
        cands = [b for b in range(1, k + 1) if b != a]
        srow = st.get(names[a], {})
        c = np.array([srow.get(names[b], 0) for b in cands], dtype=float)
        if c.sum() < HIST_BACKOFF_MIN:
            prow = hist["pool"].get(names[a], {})
            c = c + np.array([prow.get(names[b], 0) for b in cands], dtype=float)
            backed_off += 1
        p = (c + HIST_K) / (c + HIST_K).sum()
        H[a, cands] = -np.log(p)
    return H, backed_off


def zone_order(route, zone, rule):
    zlist = sorted(set(z for z in zone[1:]))
    D, members = zone_matrix(route, zone, zlist)
    k = len(zlist)
    node = {z: i + 1 for i, z in enumerate(zlist)}
    info = {"n_zones": k}
    if k == 1:
        return [1], members, info
    if rule == "tsp":
        seq = order_tsp(D)
    elif rule.startswith("hist_"):
        alpha = float(rule.split("_")[1])
        scale = float(np.median(D[1:, 1:][~np.eye(k, dtype=bool)]))
        H, backed_off = hist_penalty(route, zlist, _hist)
        mult = np.ones((k + 1, 1))
        mult[0] = HIST_START_MULT
        seq = order_tsp(D, alpha * scale * mult * H)
        info["hist_rows_backed_off"] = backed_off
    elif rule in ("id_snake", "id_lex"):
        variants = id_snake_variants(zlist) if rule == "id_snake" else id_lex_variants(zlist)
        seqs = [[node[z] for z in v] for v in variants]
        costs = [tour_cost(D, s) for s in seqs]
        best = int(np.argmin(costs))
        seq = seqs[best]
        info["variant"] = best
    else:
        extra = np.zeros_like(D)
        scale = float(np.median(D[1:, 1:][~np.eye(k, dtype=bool)]))
        if rule.startswith("group_tsp"):
            # transitions between different letter-num1 groups cost a large constant, so each
            # group is visited as one block; the TSP still picks the group order.
            grp = [(parse(z)[0], parse(z)[1]) for z in zlist]
            big = 10.0 * float(D.max()) + 1.0
            for a in range(k):
                for b in range(k):
                    if grp[a] != grp[b]:
                        extra[a + 1, b + 1] += big
        m = re.search(r"adj_pen_([\d.]+)$", rule)
        s = re.fullmatch(r"snake_pen_([\d.]+)", rule)
        if s:
            # R1 x R2 blend: start from the cheapest snake order; a TSP step that does not
            # follow the snake (forwards or backwards) costs lam x the median zone distance.
            # lam = 0 is R1, a very large lam is R2.
            lam = float(s.group(1))
            variants = [[node[z] for z in v] for v in id_snake_variants(zlist)]
            snake = min(variants, key=lambda v: tour_cost(D, v))
            ok = np.eye(k + 1, dtype=bool)
            for x, y in zip(snake, snake[1:]):
                ok[x, y] = ok[y, x] = True
            ok[0, :] = ok[:, 0] = True
            extra += lam * scale * ~ok
        elif m:
            lam = float(m.group(1))
            adj = grid_neighbours(zlist)
            extra[1:, 1:] += lam * scale * (~adj & ~np.eye(k, dtype=bool))
        elif not rule.startswith("group_tsp") and not s:
            raise ValueError(rule)
        seq = order_tsp(D, extra)
    info["zone_tour_seconds_zone_matrix"] = float(tour_cost(D, seq))
    return seq, members, info


def blocks_route(route, rule):
    """Full stop order (indices, starting with the station) for a zones-as-blocks rule."""
    zone, attached = assign_zones(route)
    seq, members, info = zone_order(route, zone, rule)
    tt = route["tt"]
    order, prev = [0], 0
    for pos, zn in enumerate(seq):
        stops = members[zn]
        nxt = members[seq[pos + 1]] if pos + 1 < len(seq) else [0]
        if len(stops) == 1:
            order += stops
            prev = stops[0]
            continue
        # local nodes: 0 = previous stop, 1..m = this zone's stops, m+1 = exit dummy
        loc = [prev] + stops
        m = len(stops)
        exit_cost = tt[np.ix_(stops, nxt)].min(axis=1)
        cost = np.zeros((m + 2, m + 2))
        cost[:m + 1, :m + 1] = tt[np.ix_(loc, loc)]
        cost[1:m + 1, m + 1] = exit_cost
        cost[0, m + 1] = 1e6  # never leave without visiting the zone
        path = solve_ortools(as_int(cost), 0, m + 1, INNER_TIME_LIMIT_MS)
        inside = [loc[i] for i in path[1:-1]]
        order += inside
        prev = inside[-1]
    info["zoneless_stops_attached"] = attached
    return order, info


# --------------------------------------------------------------------------------- scoring

_official = None
_invalid = None
_hist = None


def worker_init(hist=None):
    global _official, _invalid, _hist
    _hist = hist
    spec = importlib.util.spec_from_file_location("official_score", SCORER)
    _official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_official)
    with open(INVALID) as f:
        _invalid = json.load(f)


def cost_dict(route):
    ids, tt = route["ids"], route["tt"]
    return {a: {b: float(tt[i, j]) for j, b in enumerate(ids)} for i, a in enumerate(ids)}


def official_score(route, order):
    """Same steps as the official evaluate() for one route: lists closed at the station,
    isinvalid() check, then score(). A fresh cost dict each time (score() mutates it)."""
    ids = route["ids"]
    actual = [ids[i] for i in route["order"]]
    sub = [ids[i] for i in order]
    actual, sub = actual + [actual[0]], sub + [sub[0]]
    if _official.isinvalid(actual, sub):
        return float(_invalid[route["rid"]]), False
    return float(_official.score(actual, sub, cost_dict(route))), True


def zone_revisits(route, order):
    raw = route["raw_zones"]
    zs = [raw[i] for i in order[1:] if raw[i]]
    visits = [z for i, z in enumerate(zs) if i == 0 or z != zs[i - 1]]
    return len(visits) - len(set(zs))


def tour_seconds(route, order):
    tt = route["tt"]
    return float(sum(tt[a, b] for a, b in zip(order, order[1:] + order[:1])))


def run_route(job):
    rid, arms = job
    route = load_route(rid)
    out = {"route_id": rid, "station_code": route["station"], "n_stops": len(route["ids"]),
           "driver_tour_seconds": tour_seconds(route, route["order"]),
           "driver_zone_revisits": zone_revisits(route, route["order"]), "arms": {}}
    for arm, rule in arms.items():
        t0 = time.time()
        if rule == "shortest":
            order, info = shortest_route(route), {}
        else:
            order, info = blocks_route(route, rule)
        solve_s = time.time() - t0
        if sorted(order) != list(range(len(route["ids"]))) or order[0] != 0:
            raise RuntimeError(f"{rid} {arm}: not a permutation starting at the station")
        t0 = time.time()
        sc, feasible = official_score(route, order)
        out["arms"][arm] = {"score": sc, "feasible": feasible, "solve_seconds": solve_s,
                            "score_seconds": time.time() - t0,
                            "tour_seconds": tour_seconds(route, order),
                            "zone_revisits": zone_revisits(route, order),
                            "order": [route["ids"][i] for i in order], **info}
    return out


# ----------------------------------------------------------------------------------- stats

def bootstrap_ci(values, seed=SEED):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_test(diffs):
    """Exact two-sided sign test on non-zero differences."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def summarize(scores):
    return {"n": len(scores), "mean": float(np.mean(scores)), "median": float(np.median(scores)),
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "bootstrap_95_ci": bootstrap_ci(scores)}


def paired(a, b):
    """a minus b, per route. Negative = a is better (lower)."""
    d = [x - y for x, y in zip(a, b)]
    return {"mean_diff": float(np.mean(d)), "median_diff": float(np.median(d)),
            "bootstrap_95_ci": bootstrap_ci(d),
            "first_better": sum(1 for x in d if x < 0), "first_worse": sum(1 for x in d if x > 0),
            "tied": sum(1 for x in d if x == 0), "sign_test_p_two_sided": sign_test(d)}


# ------------------------------------------------------------------------------------ main

def settings():
    return {
        "S": {"solver": "OR-Tools routing (constraint_solver), one vehicle, closed tour from the station",
              "first_solution": "PATH_CHEAPEST_ARC", "metaheuristic": "GUIDED_LOCAL_SEARCH",
              "time_limit_seconds_per_route": S_TIME_LIMIT_S,
              "cost_units": "travel time in tenths of a second (integers)"},
        "zone_order_tsp": {"solver": "same OR-Tools settings, closed tour station -> zones -> station",
                           "time_limit_ms_per_route": ZONE_TSP_TIME_LIMIT_MS,
                           "zone_distance": "mean travel time over all (stop in zone a, stop in zone b) pairs; "
                                            "station row/column the same with the station as a one-stop zone"},
        "inside_zone": {"solver": "same OR-Tools settings, open path with fixed start and end node",
                        "start": "last stop of the previous zone (the station for the first zone)",
                        "end": "dummy node; cost from a stop = travel time to the nearest stop of the next "
                               "zone (to the station after the last zone)",
                        "time_limit_ms_per_zone": INNER_TIME_LIMIT_MS,
                        "one_stop_zones": "placed directly, no solver call"},
        "zoneless_stops": "stops with a missing zone or one not matching A-2.1C take the zone of the zoned "
                          "stop with the smallest round-trip travel time",
        "id_rule_direction": "every ID-rule variant is costed as a closed zone tour on the zone matrix; the "
                             "cheapest variant is used (this includes ascending vs descending)",
        "bootstrap": f"{N_BOOT} percentile resamples of routes, numpy default_rng({SEED})",
        "sign_test": "exact two-sided binomial test on routes with a non-zero difference",
        "workers": None,
    }


def run(route_ids, arms, workers, hist=None):
    jobs = [(rid, arms) for rid in route_ids]
    t0 = time.time()
    out = []
    with mp.Pool(workers, initializer=worker_init, initargs=(hist,), maxtasksperchild=50) as pool:
        for k, r in enumerate(pool.imap(run_route, jobs, chunksize=1)):
            out.append(r)
            if (k + 1) % 25 == 0:
                means = {a: np.mean([x["arms"][a]["score"] for x in out]) for a in arms}
                print(f"{k + 1}/{len(jobs)} routes, {time.time() - t0:.0f}s, running means "
                      + ", ".join(f"{a} {m:.4f}" for a, m in means.items()), flush=True)
    return out, time.time() - t0


def write_json(path, obj):
    tmp = path.with_suffix(".json.part")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def tune(a):
    splits = json.load(open(SPLITS))
    forbidden = {r["route_id"] for r in splits["val"]} | {r["route_id"] for r in splits["pilot"]}
    pool = sorted(r["route_id"] for r in splits["train"] if r["quality"] == "High")
    assert not forbidden & set(pool)
    ids = random.Random(SEED).sample(pool, TUNE_N)
    if a.phase == "tune-hist":
        return tune_hist(a, ids, [r["route_id"] for r in splits["train"]], a.round)
    if a.round == 2:
        return tune_round2(a, ids)
    arms = {"R1_tsp": "tsp", **{f"R2_{c}": c for c in R2_CANDIDATES},
            **{f"R3_{c}": c for c in R3_CANDIDATES}}
    rows, wall = run(ids, arms, a.workers)
    rows.sort(key=lambda r: r["route_id"])
    summary = {arm: summarize([r["arms"][arm]["score"] for r in rows]) for arm in arms}
    for arm in arms:
        summary[arm]["mean_zone_revisits"] = float(np.mean([r["arms"][arm]["zone_revisits"] for r in rows]))
    r2 = min(R2_CANDIDATES, key=lambda c: summary[f"R2_{c}"]["mean"])
    r3 = min(R3_CANDIDATES, key=lambda c: summary[f"R3_{c}"]["mean"])
    st = settings()
    st["workers"] = a.workers
    res = {"description": "Rule selection on train routes only (High quality, random 300). Val and pilot untouched.",
           "selection": f"random.Random({SEED}).sample(sorted High train route IDs, {TUNE_N})",
           "chosen": {"R2": r2, "R3": r3,
                      "rule": "lowest mean official score on these train routes, within each family"},
           "summary": summary,
           "paired_vs_R1": {arm: paired([r["arms"][arm]["score"] for r in rows],
                                        [r["arms"]["R1_tsp"]["score"] for r in rows]) for arm in arms if arm != "R1_tsp"},
           "settings": st,
           "run": {"wall_seconds": wall,
                   "peak_rss_mb_main": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "peak_rss_mb_largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024},
           "per_route": [{k: v for k, v in r.items() if k != "arms"} |
                         {arm: {k: v for k, v in x.items() if k != "order"} for arm, x in r["arms"].items()}
                         for r in rows]}
    write_json(RESULTS / "rules_baseline_tuning.json", res)
    print(json.dumps({"chosen": res["chosen"], "summary": summary}, indent=1))


def tune_round2(a, ids):
    r1 = json.load(open(RESULTS / "rules_baseline_tuning.json"))
    r1_rows = {r["route_id"]: r for r in r1["per_route"]}
    assert set(r1_rows) == set(ids), "round 2 must use the round-1 train routes"
    arms = {f"R3_{c}": c for c in R3_CANDIDATES_ROUND2}
    rows, wall = run(ids, arms, a.workers)
    rows.sort(key=lambda r: r["route_id"])
    summary = {arm: summarize([r["arms"][arm]["score"] for r in rows]) for arm in arms}
    for arm in arms:
        summary[arm]["mean_zone_revisits"] = float(np.mean([r["arms"][arm]["zone_revisits"] for r in rows]))
    r1_scores = [r1_rows[r["route_id"]]["R1_tsp"]["score"] for r in rows]
    r2_scores = [r1_rows[r["route_id"]]["R2_id_snake"]["score"] for r in rows]
    res = {"description": "Round 2 of rule selection: R1 x snake blends, same 300 train routes as round 1.",
           "why": "added after round 1, where every round-1 hybrid scored worse than the pure snake ID rule",
           "summary": summary,
           "paired_vs_R1": {arm: paired([r["arms"][arm]["score"] for r in rows], r1_scores) for arm in arms},
           "paired_vs_R2_id_snake": {arm: paired([r["arms"][arm]["score"] for r in rows], r2_scores) for arm in arms},
           "run": {"wall_seconds": wall,
                   "peak_rss_mb_largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024},
           "per_route": [{"route_id": r["route_id"]} |
                         {arm: {k: v for k, v in x.items() if k != "order"} for arm, x in r["arms"].items()}
                         for r in rows]}
    write_json(RESULTS / "rules_baseline_tuning_round2.json", res)
    print(json.dumps(summary, indent=1))


def hist_settings(counts_from):
    return {"formula": "cost(a->b) = D(a,b) + alpha * scale * mult(a) * (-log P(b|a)); cost(zone->station) = D",
            "D": "mean stop-to-stop travel time between zones (same zone matrix as R1)",
            "scale": "median off-diagonal zone-to-zone D of the route",
            "mult": {"station_to_first_zone": HIST_START_MULT, "zone_to_zone": 1, "zone_to_station": 0},
            "P": "(c + k) / sum over the route's other zones of (c + k); candidates are only the zones "
                 "present in this route",
            "k_add_smoothing": HIST_K,
            "counts": "consecutive distinct zones in driver order (station and zone-less stops dropped, "
                      "repeats collapsed), plus station -> first zone; keyed by station and full zone ID",
            "backoff": f"if this station's counts over the candidates sum to < {HIST_BACKOFF_MIN}, add the "
                       "counts pooled over all stations for the same zone-ID pair",
            "counts_from": counts_from}


def tune_hist(a, ids, train_all, rnd=1):
    """Tune alpha for R4-hist. Counts exclude the 300 tuning routes (held out from the table)."""
    held = set(ids)
    count_ids = sorted(r for r in train_all if r not in held)
    hist = build_hist(count_ids)
    alphas = HIST_ALPHAS if rnd == 1 else HIST_ALPHAS_ROUND2
    arms = {f"R4_hist_{x}": f"hist_{x}" for x in alphas}
    rows, wall = run(ids, arms, a.workers, hist)
    rows.sort(key=lambda r: r["route_id"])
    summary = {arm: summarize([r["arms"][arm]["score"] for r in rows]) for arm in arms}
    for arm in arms:
        summary[arm]["mean_zone_revisits"] = float(np.mean([r["arms"][arm]["zone_revisits"] for r in rows]))
    best = min(alphas, key=lambda x: summary[f"R4_hist_{x}"]["mean"])
    t1 = json.load(open(RESULTS / "rules_baseline_tuning.json"))
    r1_rows = {r["route_id"]: r for r in t1["per_route"]}
    res = {"description": "R4-hist alpha tuning on the same 300 High train routes; their own transitions "
                          "are excluded from the count table while tuning.",
           "round": rnd, "best_alpha_this_round": best, "alphas_tried": alphas,
           "settings": hist_settings(f"{len(count_ids)} train routes (all quality labels; = all train "
                                     "routes minus the 300 tuning routes)"),
           "summary": summary,
           "paired_vs_R1": {arm: paired([r["arms"][arm]["score"] for r in rows],
                                        [r1_rows[r["route_id"]]["R1_tsp"]["score"] for r in rows]) for arm in arms},
           "paired_vs_R2_id_snake": {arm: paired([r["arms"][arm]["score"] for r in rows],
                                                 [r1_rows[r["route_id"]]["R2_id_snake"]["score"] for r in rows])
                                     for arm in arms},
           "run": {"wall_seconds": wall,
                   "peak_rss_mb_main": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "peak_rss_mb_largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024},
           "per_route": [{"route_id": r["route_id"]} |
                         {arm: {k: v for k, v in x.items() if k != "order"} for arm, x in r["arms"].items()}
                         for r in rows]}
    write_json(RESULTS / ("rules_baseline_tuning_hist.json" if rnd == 1 else
                          "rules_baseline_tuning_hist_round2.json"), res)
    print(json.dumps({"best_alpha_this_round": best, "summary": summary}, indent=1))


def choose_alpha():
    """alpha with the lowest train mean across both R4-hist rounds (both files required)."""
    means = {}
    for name in ("rules_baseline_tuning_hist.json", "rules_baseline_tuning_hist_round2.json"):
        t = json.load(open(RESULTS / name))
        means |= {x: t["summary"][f"R4_hist_{x}"]["mean"] for x in t["alphas_tried"]}
    return min(means, key=means.get), means


def choose_rules():
    """R2 = best pure ID rule on train; R3 = best hybrid on train across both rounds."""
    t1 = json.load(open(RESULTS / "rules_baseline_tuning.json"))
    means = {c: t1["summary"][f"R3_{c}"]["mean"] for c in R3_CANDIDATES}
    p2 = RESULTS / "rules_baseline_tuning_round2.json"
    if p2.exists():
        t2 = json.load(open(p2))
        means |= {c: t2["summary"][f"R3_{c}"]["mean"] for c in R3_CANDIDATES_ROUND2}
    return t1["chosen"]["R2"], min(means, key=means.get), means


def check_evaluate(rows, n=3):
    """Score a few routes through the official evaluate() on real JSON files and compare."""
    worker_init()
    tmp = RESULTS / "_rules_baseline_check"
    tmp.mkdir(exist_ok=True)
    try:
        sel = rows[:n]
        actual, costs, invalid = {}, {}, {}
        for r in sel:
            route = load_route(r["route_id"])
            ids = route["ids"]
            actual[r["route_id"]] = {"actual": {ids[i]: p for p, i in enumerate(route["order"])}}
            costs[r["route_id"]] = cost_dict(route)
            invalid[r["route_id"]] = _invalid[r["route_id"]]
        for name, obj in (("actual", actual), ("costs", costs), ("invalid", invalid)):
            json.dump(obj, open(tmp / f"{name}.json", "w"))
        mism = []
        for arm in sel[0]["arms"]:
            prop = {r["route_id"]: {"proposed": {s: p for p, s in enumerate(r["arms"][arm]["order"])}}
                    for r in sel}
            json.dump(prop, open(tmp / "proposed.json", "w"))
            ev = _official.evaluate(str(tmp / "actual.json"), str(tmp / "proposed.json"),
                                    str(tmp / "costs.json"), str(tmp / "invalid.json"))
            for r in sel:
                if ev["route_scores"][r["route_id"]] != r["arms"][arm]["score"]:
                    mism.append((arm, r["route_id"], ev["route_scores"][r["route_id"]], r["arms"][arm]["score"]))
        if mism:
            raise SystemExit(f"evaluate() disagrees with per-route scoring: {mism}")
        # the driver's own order must score exactly 0
        for r in sel:
            route = load_route(r["route_id"])
            if official_score(route, route["order"])[0] != 0:
                raise SystemExit(f"{r['route_id']}: driver order did not score 0")
        return {"routes_checked": [r["route_id"] for r in sel], "arms": list(sel[0]["arms"]),
                "evaluate_matches_per_route_scoring": True, "driver_order_scores_zero": True}
    finally:
        shutil.rmtree(tmp)


def worked_example(rows, best_arm):
    ranked = sorted(rows, key=lambda r: r["arms"]["S"]["score"])
    r = ranked[len(ranked) // 2]
    route = load_route(r["route_id"])
    ids, raw = route["ids"], route["raw_zones"]
    zone_of = {ids[i]: (raw[i] or "(none)") for i in range(len(ids))}
    driver = [ids[i] for i in route["order"]]

    def oriented(order):
        # the metric ignores direction; show each order in the direction that matches the driver
        # better (Spearman-style: sum of |position differences|), and say which one was used
        pos = {s: p for p, s in enumerate(driver)}
        fwd = order
        rev = [order[0]] + order[1:][::-1]
        cost = lambda o: sum(abs(pos[s] - p) for p, s in enumerate(o))
        return (fwd, "as produced") if cost(fwd) <= cost(rev) else (rev, "reversed for display")

    s_order, s_dir = oriented(r["arms"]["S"]["order"])
    b_order, b_dir = oriented(r["arms"][best_arm]["order"])
    return {"route_id": r["route_id"], "selection": "val route with the median S score (rank n//2)",
            "station_code": r["station_code"], "n_stops_including_station": r["n_stops"],
            "best_arm": best_arm, "S_display": s_dir, "best_display": b_dir,
            "rows": [{"k": k + 1, "driver": driver[k + 1], "driver_zone": zone_of[driver[k + 1]],
                      "S": s_order[k + 1], "S_zone": zone_of[s_order[k + 1]],
                      "best": b_order[k + 1], "best_zone": zone_of[b_order[k + 1]]} for k in range(15)],
            "scores": {arm: x["score"] for arm, x in r["arms"].items()},
            "zone_revisits": {"driver": r["driver_zone_revisits"],
                              **{arm: x["zone_revisits"] for arm, x in r["arms"].items()}}}


def val(a):
    r2, r3, r3_train_means = choose_rules()
    print(f"R2 = {r2}, R3 = {r3} (train means {r3_train_means})", flush=True)
    splits = json.load(open(SPLITS))
    ids = sorted(r["route_id"] for r in splits["val"])
    assert len(ids) == 300 and not set(ids) & {r["route_id"] for r in splits["pilot"]}
    alpha, alpha_means = choose_alpha()
    print(f"R4 alpha = {alpha} (train means {alpha_means})", flush=True)
    pilot = {r["route_id"] for r in splits["pilot"]}
    count_ids = sorted(r["route_id"] for r in splits["train"])
    assert not (set(count_ids) & (set(ids) | pilot))
    hist = build_hist(count_ids)  # all train routes; val and pilot never counted
    arms = {"S": "shortest", "R1": "tsp", "R2": r2, "R3": r3, "R4": f"hist_{alpha}"}
    rows, wall = run(ids, arms, a.workers, hist)
    rows.sort(key=lambda r: r["route_id"])
    check = check_evaluate(rows)
    scores = {arm: [r["arms"][arm]["score"] for r in rows] for arm in arms}
    summary = {arm: summarize(scores[arm]) for arm in arms}
    for arm in arms:
        summary[arm]["all_feasible"] = all(r["arms"][arm]["feasible"] for r in rows)
        summary[arm]["mean_zone_revisits"] = float(np.mean([r["arms"][arm]["zone_revisits"] for r in rows]))
        summary[arm]["mean_tour_over_driver_tour"] = float(np.mean(
            [r["arms"][arm]["tour_seconds"] / r["driver_tour_seconds"] for r in rows]))
        summary[arm]["mean_solve_seconds"] = float(np.mean([r["arms"][arm]["solve_seconds"] for r in rows]))
    summary["driver_mean_zone_revisits"] = float(np.mean([r["driver_zone_revisits"] for r in rows]))
    pairs = {f"{arm}_minus_S": paired(scores[arm], scores["S"]) for arm in ("R1", "R2", "R3", "R4")}
    best123 = min(("R1", "R2", "R3"), key=lambda x: summary[x]["mean"])
    pairs[f"R4_minus_{best123}"] = paired(scores["R4"], scores[best123])
    pairs["R3_minus_R2"] = paired(scores["R3"], scores["R2"])
    best = min(("R1", "R2", "R3", "R4"), key=lambda x: summary[x]["mean"])
    st = settings()
    st["workers"] = a.workers
    source_commit = (HERE / "official_rc_cli" / "SOURCE_COMMIT.txt").read_text().strip()
    res = {
        "description": "Rules baselines vs plain shortest route on the 300 val routes (High quality).",
        "split": "data/processed/splits.json 'val' (seed 20261111); pilot routes never used",
        "arms": {"S": "plain shortest route", "R1": "zones as blocks, zone order by TSP (rule 'tsp')",
                 "R2": f"zones as blocks, zone order by zone ID (rule '{r2}', chosen on train)",
                 "R3": f"hybrid (rule '{r3}', chosen on train)",
                 "R4": f"R4-hist: zone TSP on travel time + history penalty (rule 'hist_{alpha}', alpha chosen on train)"},
        "best_of_R1_R3_on_val": best123,
        "r4_alpha_train_means": {str(k): v for k, v in alpha_means.items()},
        "R4_hist_settings": hist_settings(f"all {len(count_ids)} train routes (all quality labels)"),
        "tuning_files": ["results/rules_baseline_tuning.json", "results/rules_baseline_tuning_round2.json",
                         "results/rules_baseline_tuning_hist.json",
                         "results/rules_baseline_tuning_hist_round2.json"],
        "r3_candidate_train_means": r3_train_means,
        "metric": {"scorer": "official, unmodified rc-cli score.py",
                   "scorer_source": f"https://github.com/MIT-CAVE/rc-cli/blob/{source_commit}/scoring/score.py",
                   "direction": "lower is better; 0 = driver's exact order (or its exact reverse)",
                   "numpy": np.__version__},
        "summary": summary, "paired": pairs, "best_rule_on_val": best,
        "scorer_check": check, "settings": st,
        "worked_example": worked_example(rows, best),
        "run": {"wall_seconds": wall,
                "peak_rss_mb_main": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "peak_rss_mb_largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024},
        "per_route": rows,
    }
    write_json(RESULTS / "rules_baseline.json", res)
    print(json.dumps({"summary": summary, "paired": pairs, "check": check, "run": res["run"]}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["tune", "tune-hist", "val"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--round", type=int, default=1, choices=[1, 2], help="tune only")
    a = ap.parse_args()
    if a.workers > 3:
        raise SystemExit("at most 3 workers on this laptop")
    (val if a.phase == "val" else tune)(a)


if __name__ == "__main__":
    sys.exit(main())
