#!/usr/bin/env python3
"""Measure driver habits on the TRAIN split (plus descriptive pilot comparisons). Sealed output.

Run from tacit-routes/:
  .venv/bin/python -I scripts/habits.py --data-dir data/processed --results-dir results

Writes results/sealed/habits.json and results/sealed/habits.md. The solver tours for the train
subsample are cached in results/habits_shortest_train.json (a tour list, nothing to predict), so a
rerun skips the solving.

Data rules: only TRAIN routes are measured. VALIDATION routes are never opened. The 50 PILOT routes
are used only for descriptive comparisons against results/proposed_shortest.json (10 s solver);
nothing is fitted on any data here (every "rule" below is fixed by hand, nothing is tuned).

Solver comparison on train: OR-Tools (same settings as run_plain_baseline.py: PATH_CHEAPEST_ARC +
GUIDED_LOCAL_SEARCH, closed tour from the station, integer tenths of a second) on a fixed random
subsample of train routes: all Low routes plus SOLVER_N High and SOLVER_N Medium, 5 s each,
2 worker processes. The tour direction a solver returns is arbitrary (the reverse is equally short),
so where direction matters we say how it is handled.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import itertools
import json
import multiprocessing as mp
import random
import re
import resource
import time
from pathlib import Path

import numpy as np

SEED = 20261111
N_BOOT = 10_000
SOLVER_N = 150
SOLVER_SECONDS = 5
LABELS = ["High", "Medium", "Low"]
ZONE_RE = re.compile(r"([A-Z])-(\d+)\.(\d+)([A-Z])")
BINS = [(0, 30, "under 30 s"), (30, 60, "30-60 s"), (60, 180, "1-3 min"), (180, 1e9, "over 3 min")]
MAX_EXACT = 9  # exact within-zone shortest path for runs of 3..9 stops (<= 7! interior orders)


# ----------------------------------------------------------------------------- helpers

def load(data_dir, rid):
    with np.load(data_dir / "routes" / f"{rid}.npz", allow_pickle=False) as d:
        return {k: d[k] for k in ("stop_ids", "zone_ids", "travel_time", "tw_start", "tw_end",
                                  "service_time", "actual_order", "quality")}


def parse_zone(z):
    m = ZONE_RE.fullmatch(z)
    return None if m is None else (m[1], int(m[2]), int(m[3]), m[4])


def zkey(p):
    """Sort key: letter, number, sub-number, sub-letter (alphabetical)."""
    return (p[0], p[1], p[2], p[3])


def tour_legs(order):
    """Closed tour legs (includes the return to the station)."""
    o = list(order) + [order[0]]
    return list(zip(o[:-1], o[1:]))


def visits_of(order, zones):
    """Collapsed zone visits (station and zoneless stops dropped), as in zone_revisits.py."""
    zs = [zones[i] for i in order if i != 0 and zones[i] != ""]
    return [z for k, z in enumerate(zs) if k == 0 or z != zs[k - 1]]


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def boot_ratio(num, den, seed=SEED):
    """Percentile 95% CI of sum(num)/sum(den), resampling routes with replacement (chunked)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    n = len(num)
    if n == 0 or den.sum() == 0:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT // 500):
        idx = rng.integers(0, n, size=(500, n), dtype=np.int32)
        d = den[idx].sum(1)
        out.append(num[idx].sum(1) / np.where(d == 0, np.nan, d))
    out = np.concatenate(out)
    return float(num.sum() / den.sum()), [float(np.nanpercentile(out, 2.5)),
                                          float(np.nanpercentile(out, 97.5))]


def stat(recs, numf, denf=None):
    """Pooled ratio (or mean when denf is None) with CI, for All and each label."""
    res = {}
    for lab in ["All"] + LABELS:
        rs = [r for r in recs if lab == "All" or r["quality"] == lab]
        rs = [r for r in rs if numf(r) is not None and (denf is None or denf(r) is not None)]
        num = [numf(r) for r in rs]
        den = [1.0] * len(rs) if denf is None else [denf(r) for r in rs]
        v, ci = boot_ratio(num, den)
        res[lab] = {"value": v, "ci95": ci, "routes": len(rs)}
    return res


# ----------------------------------------------------------------------------- solver

def solve(job):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    data_dir, rid, seconds = job
    tt = load(Path(data_dir), rid)["travel_time"]
    costs = np.rint(tt.astype(np.float64) * 10).astype(np.int64).tolist()
    n = len(costs)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(lambda i, j: costs[manager.IndexToNode(i)][manager.IndexToNode(j)])
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromSeconds(seconds)
    sol = routing.SolveWithParameters(p)
    if sol is None:
        return rid, None
    order, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    return rid, order


def solver_tours(data_dir, results_dir, train):
    rng = random.Random(SEED)
    by = {lab: sorted(r["route_id"] for r in train if r["quality"] == lab) for lab in LABELS}
    chosen = by["Low"] + rng.sample(by["High"], SOLVER_N) + rng.sample(by["Medium"], SOLVER_N)
    cache = results_dir / "habits_shortest_train.json"
    tours = json.load(open(cache))["tours"] if cache.exists() else {}
    todo = [r for r in chosen if r not in tours]
    if todo:
        print(f"solving {len(todo)} train routes, {SOLVER_SECONDS} s each, 2 workers", flush=True)
        t0 = time.time()
        with mp.Pool(2) as pool:
            for k, (rid, order) in enumerate(pool.imap_unordered(
                    solve, [(str(data_dir), r, SOLVER_SECONDS) for r in todo], chunksize=1)):
                if order is None:
                    raise SystemExit(f"solver found no tour for {rid}")
                tours[rid] = order
                if (k + 1) % 50 == 0:
                    print(f"  {k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
                    json.dump({"note": "OR-Tools tours (indices into stop_ids, station first), "
                               f"{SOLVER_SECONDS} s each, habits.py", "tours": tours}, open(cache, "w"))
        json.dump({"note": "OR-Tools tours (indices into stop_ids, station first), "
                   f"{SOLVER_SECONDS} s each, habits.py", "tours": tours}, open(cache, "w"))
    return {r: tours[r] for r in chosen}


# ----------------------------------------------------------------------------- per-route measures

SUBLETTERS = None  # global ordered list of sub-letters, filled in main


def zone_transitions(visits, rng):
    """ID-order features of consecutive zone visits."""
    pv = [parse_zone(z) for z in visits]
    distinct = sorted({p for p in pv if p is not None}, key=zkey)
    rank = {p: i for i, p in enumerate(distinct)}
    c = dict(pairs=0, same_letter=0, same_letter_number=0, same_lns=0, grid_neighbour=0,
             sorted_adjacent=0, ascending=0)
    for a, b in zip(pv, pv[1:]):
        if a is None or b is None:
            continue
        c["pairs"] += 1
        c["same_letter"] += a[0] == b[0]
        c["same_letter_number"] += a[:2] == b[:2]
        c["same_lns"] += a[:3] == b[:3]
        if a[:2] == b[:2]:
            ds = abs(a[2] - b[2])
            dt = abs(SUBLETTERS.index(a[3]) - SUBLETTERS.index(b[3]))
            c["grid_neighbour"] += (ds + dt) == 1
        c["sorted_adjacent"] += abs(rank[a] - rank[b]) == 1
        c["ascending"] += zkey(b) > zkey(a)
    return c


def zone_predict(order, zones, tt):
    """Teacher-forced next-zone prediction with fixed hand rules, first visits only.

    At each change of zone in the driver's order: candidates = parsed zones of this route not yet
    entered. Rules pick one candidate; a hit means it is the zone the driver entered next.
    """
    seq = [i for i in order if i != 0 and parse_zone(zones[i]) is not None]
    zstops = {}
    for i in seq:
        zstops.setdefault(zones[i], []).append(i)
    allz = sorted(zstops, key=lambda z: zkey(parse_zone(z)))
    rank = {z: k for k, z in enumerate(allz)}
    zst = {z: np.array(v) for z, v in zstops.items()}
    out = dict(n=0, revisit=0, chance=0.0, id_next=0, prefix_id=0, nearest=0, prefix_nearest=0)
    seen, cur, last_stop, direction = set(), None, None, +1
    for i in seq:
        z = zones[i]
        if cur is None:
            cur, last_stop = z, i
            seen.add(z)
            continue
        if z == cur:
            last_stop = i
            continue
        cand = [c for c in allz if c not in seen]
        if cand:
            out["n"] += 1
            out["chance"] += 1 / len(cand)
            if z in seen:
                out["revisit"] += 1
            pc = parse_zone(cur)
            r0 = rank[cur]

            def id_dist(c):
                d = rank[c] - r0
                return (0 if d * direction > 0 else 1, abs(d))

            def shared(c):
                p = parse_zone(c)
                k = 0
                while k < 3 and p[k] == pc[k]:
                    k += 1
                return k

            def tdist(c):
                return float(tt[last_stop, zst[c]].min())

            out["id_next"] += min(cand, key=id_dist) == z
            out["prefix_id"] += min(cand, key=lambda c: (-shared(c),) + id_dist(c)) == z
            out["nearest"] += min(cand, key=tdist) == z
            out["prefix_nearest"] += min(cand, key=lambda c: (-shared(c), tdist(c))) == z
            if z not in seen:
                direction = 1 if rank[z] > r0 else -1
        seen.add(z)
        cur, last_stop = z, i
    return out


def arrivals(order, tt, svc):
    t, arr = 0.0, np.zeros(len(order))
    for k in range(1, len(order)):
        t += float(tt[order[k - 1], order[k]])
        arr[k] = t
        t += float(svc[order[k]])
    a = np.empty(len(order))
    a[np.asarray(order)] = arr
    return a  # arrival time by stop index


def windows(order, d):
    tw_e, tw_s = d["tw_end"], d["tw_start"]
    w = np.where(np.isfinite(tw_e))[0]
    w = w[w != 0]
    out = dict(n_stops=len(order) - 1, n_window=len(w))
    if len(w) == 0:
        return out
    out["width_sum_h"] = float(((tw_e[w] - tw_s[w]) / 3600).sum())
    arr = arrivals(order, d["travel_time"], d["service_time"])
    out["on_time"] = int((arr[w] <= tw_e[w]).sum())
    out["early"] = int((arr[w] < tw_s[w]).sum())
    pos = np.empty(len(order))
    pos[np.asarray(order)] = np.arange(len(order))
    out["mean_rel_pos"] = float(((pos[w] - 1) / max(len(order) - 2, 1)).mean())
    if len(w) >= 3:
        out["rho"] = spearman(tw_e[w], pos[w])
    return out


def leg_bins(order, d):
    tt, zones = d["travel_time"], d["zone_ids"]
    b = np.zeros(len(BINS))
    within = between = 0.0
    nb = np.zeros(len(BINS), int)
    for a, c in tour_legs(order):
        x = float(tt[a, c])
        k = next(k for k, (lo, hi, _) in enumerate(BINS) if lo <= x < hi)
        b[k] += x
        nb[k] += 1
        if a != 0 and c != 0 and zones[a] != "" and zones[a] == zones[c]:
            within += x
        else:
            between += x
    return dict(total=float(b.sum()), bins=b.tolist(), bin_legs=nb.tolist(), within=within, between=between)


def first_stop(order, d, closer_end=False):
    """Station distance of the first stop and its zone. closer_end: use whichever tour end is
    closer to the station (for solver tours, whose direction is arbitrary)."""
    tt, zones = d["travel_time"], d["zone_ids"]
    st = tt[0].astype(float)
    first, last = order[1], order[-1]
    if closer_end and st[last] < st[first]:
        first, last = last, first
    others = np.arange(1, len(st))
    out = dict(first_s=float(st[first]), last_s=float(st[last]), closest_s=float(st[others].min()),
               median_s=float(np.median(st[others])),
               first_pct=float((st[others] < st[first]).mean()))
    zs = {}
    for i in others:
        if zones[i] != "":
            zs.setdefault(zones[i], []).append(st[i])
    if zones[first] in zs and len(zs) >= 2:
        zmin = {z: min(v) for z, v in zs.items()}
        zmean = {z: float(np.mean(v)) for z, v in zs.items()}
        out["n_zones"] = len(zs)
        out["first_zone_closest_min"] = int(min(zmin, key=zmin.get) == zones[first])
        out["first_zone_closest_mean"] = int(min(zmean, key=zmean.get) == zones[first])
        out["first_zone_rank_pct"] = float(sum(v < zmin[zones[first]] for v in zmin.values()) / (len(zs) - 1))
    return out


PERMS = {m: np.array(list(itertools.permutations(range(m))), dtype=np.int16) for m in range(1, MAX_EXACT - 1)}


def within_zone(order, d):
    tt, zones = d["travel_time"], d["zone_ids"]
    zstops = {}
    for i in order[1:]:
        if zones[i] != "":
            zstops.setdefault(zones[i], set()).add(i)
    visited = set()
    out = dict(trans=0, nn=0, chance=0.0, runs=0, exact_runs=0, drv=0.0, opt=0.0, optimal_runs=0,
               runs_ge3=0)
    seq = list(order[1:])
    k = 0
    while k < len(seq):
        z = zones[seq[k]]
        j = k
        while j + 1 < len(seq) and z != "" and zones[seq[j + 1]] == z:
            j += 1
        run = seq[k:j + 1]
        if z != "" and len(run) >= 3:
            out["runs_ge3"] += 1
            for a_i in range(len(run) - 1):
                a = run[a_i]
                visited.add(a)
                cand = [c for c in zstops[z] if c not in visited]
                out["trans"] += 1
                out["chance"] += 1 / len(cand)
                best = min(cand, key=lambda c: (float(tt[a, c]), c))
                out["nn"] += float(tt[a, run[a_i + 1]]) <= float(tt[a, best])  # ties count as nearest
            if len(run) <= MAX_EXACT:
                s, e, mid = run[0], run[-1], np.array(run[1:-1])
                P = mid[PERMS[len(mid)]]
                cost = tt[s, P[:, 0]] + tt[P[:, -1], e]
                for q in range(P.shape[1] - 1):
                    cost = cost + tt[P[:, q], P[:, q + 1]]
                opt = float(cost.min())
                drv = float(sum(tt[a, b] for a, b in zip(run, run[1:])))
                out["exact_runs"] += 1
                out["drv"] += drv
                out["opt"] += opt
                out["optimal_runs"] += drv <= opt + 0.05
        visited.update(run)
        k = j + 1
    return out


def route_record(rid, d, order, kind, rng):
    rec = {"route_id": rid, "quality": str(d["quality"]), "kind": kind, "n_stops": len(order)}
    v = visits_of(order, d["zone_ids"])
    rec["zones"] = len(set(v))
    rec["revisits"] = len(v) - len(set(v))
    rec["zt"] = zone_transitions(v, rng)
    shuffled = list(v)
    # chance reference: the same visits in random order
    rng.shuffle(shuffled)
    rec["zt_random"] = zone_transitions(shuffled, rng)
    rec["zp"] = zone_predict(order, d["zone_ids"], d["travel_time"])
    rec["tw"] = windows(order, d)
    rec["legs"] = leg_bins(order, d)
    rec["first"] = first_stop(order, d, closer_end=(kind != "driver"))
    rec["wz"] = within_zone(order, d)
    return rec


def best_direction_windows(order, d):
    a = windows(order, d)
    if a["n_window"] == 0:
        return a
    b = windows([order[0]] + list(order[1:])[::-1], d)
    return a if a["on_time"] >= b["on_time"] else b


# ----------------------------------------------------------------------------- main

def fmt(s, pct=False, nd=2):
    if s["value"] is None:
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.{nd}f}")
    return f"{f(s['value'])} [{f(s['ci95'][0])}, {f(s['ci95'][1])}]"


def table(rows, pct_flags, header):
    L = ["| Measure | " + " | ".join(header) + " |", "|---" * (len(header) + 1) + "|"]
    for (name, st), pct in zip(rows, pct_flags):
        L.append(f"| {name} | " + " | ".join(
            (fmt(st[h], pct) + f" (n={st[h]['routes']})") if st[h]["routes"] else "n/a" for h in header) + " |")
    return L


def main():
    global SUBLETTERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    a = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    t0 = time.time()
    splits = json.load(open(a.data_dir / "splits.json"))
    train = splits["train"]
    val_ids = {r["route_id"] for r in splits["val"]}
    pilot_ids = [r["route_id"] for r in splits["pilot"]]
    assert not val_ids & {r["route_id"] for r in train}

    # Zone-ID grammar (train only)
    pat, subl, bad = {}, set(), {}
    for r in train:
        zones = load(a.data_dir, r["route_id"])["zone_ids"][1:]
        for z in zones:
            if z == "":
                bad["(missing)"] = bad.get("(missing)", 0) + 1
                continue
            p = parse_zone(z)
            if p is None:
                bad[z] = bad.get(z, 0) + 1
            else:
                subl.add(p[3])
                for part, val in zip(("letter", "number", "sub_number", "sub_letter"), p):
                    pat.setdefault(part, {}).setdefault(str(val), 0)
                    pat[part][str(val)] += 1
    SUBLETTERS = sorted(subl)
    grammar = {"regex": ZONE_RE.pattern, "parts": {k: dict(sorted(v.items(), key=lambda kv: (len(kv[0]), kv[0])))
                                                   for k, v in pat.items()},
               "unparsed_non_station_stops": bad, "sub_letter_order": SUBLETTERS}
    print(f"grammar done {time.time() - t0:.0f}s", flush=True)

    tours = solver_tours(a.data_dir, a.results_dir, train)
    print(f"solver done {time.time() - t0:.0f}s", flush=True)

    rng = random.Random(SEED)
    drv, sol_drv, sol = [], [], []
    for k, r in enumerate(train):
        rid = r["route_id"]
        d = load(a.data_dir, rid)
        order = [int(x) for x in d["actual_order"]]
        rec = route_record(rid, d, order, "driver", rng)
        drv.append(rec)
        if rid in tours:
            s = tours[rid]
            assert sorted(s) == list(range(len(order))) and s[0] == 0
            srec = route_record(rid, d, s, "solver", rng)
            srec["tw"] = best_direction_windows(s, d)
            srec["rho_both"] = [windows(s, d).get("rho"), windows([0] + s[1:][::-1], d).get("rho")]
            sol.append(srec)
            sol_drv.append(rec)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(train)} routes {time.time() - t0:.0f}s, peak RSS "
                  f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)

    # pilot: descriptive only, against the 10 s shortest tours already produced
    shortest = json.load(open(a.results_dir / "proposed_shortest.json"))
    pil_d, pil_s = [], []
    for rid in pilot_ids:
        d = load(a.data_dir, rid)
        pos = {s: i for i, s in enumerate(d["stop_ids"])}
        prop = shortest[rid]["proposed"]
        s = [pos[x] for x in sorted(prop, key=prop.get)]
        assert sorted(s) == list(range(len(s))) and s[0] == 0
        order = [int(x) for x in d["actual_order"]]
        pil_d.append({"route_id": rid, "quality": "High", "n_stops": len(order),
                      "legs": leg_bins(order, d), "first": first_stop(order, d)})
        pil_s.append({"route_id": rid, "quality": "High", "n_stops": len(order),
                      "legs": leg_bins(s, d), "first": first_stop(s, d, closer_end=True)})
    print(f"measures done {time.time() - t0:.0f}s", flush=True)

    R = {}
    G = lambda f: (lambda r: f(r))  # noqa: E731
    # 1. zone-ID order
    zt_keys = ["same_letter", "same_letter_number", "same_lns", "grid_neighbour", "sorted_adjacent", "ascending"]
    R["zone_id_order"] = {
        "driver": {k: stat(drv, G(lambda r, k=k: r["zt"][k]), lambda r: r["zt"]["pairs"]) for k in zt_keys},
        "driver_visits_shuffled": {k: stat(drv, G(lambda r, k=k: r["zt_random"][k]), lambda r: r["zt_random"]["pairs"]) for k in zt_keys},
        "solver_subsample": {k: stat(sol, G(lambda r, k=k: r["zt"][k]), lambda r: r["zt"]["pairs"]) for k in zt_keys},
        "driver_on_solver_subsample": {k: stat(sol_drv, G(lambda r, k=k: r["zt"][k]), lambda r: r["zt"]["pairs"]) for k in zt_keys},
        "route_direction_consistency": stat(drv, lambda r: max(r["zt"]["ascending"], r["zt"]["pairs"] - r["zt"]["ascending"]) / r["zt"]["pairs"] if r["zt"]["pairs"] else None),
        "route_direction_consistency_shuffled": stat(drv, lambda r: max(r["zt_random"]["ascending"], r["zt_random"]["pairs"] - r["zt_random"]["ascending"]) / r["zt_random"]["pairs"] if r["zt_random"]["pairs"] else None),
        "next_zone_prediction": {k: stat(drv, G(lambda r, k=k: r["zp"][k]), lambda r: r["zp"]["n"])
                                 for k in ["chance", "id_next", "prefix_id", "nearest", "prefix_nearest", "revisit"]},
    }
    # 2. time windows
    R["time_windows"] = {
        "share_stops_with_window": stat(drv, lambda r: r["tw"]["n_window"], lambda r: r["tw"]["n_stops"]),
        "share_routes_with_any_window": stat(drv, lambda r: float(r["tw"]["n_window"] > 0)),
        "driver_on_time": stat(drv, lambda r: r["tw"].get("on_time"), lambda r: r["tw"]["n_window"] or None),
        "driver_early": stat(drv, lambda r: r["tw"].get("early"), lambda r: r["tw"]["n_window"] or None),
        "driver_mean_relative_position": stat(drv, lambda r: r["tw"].get("mean_rel_pos")),
        "driver_rho_window_end_vs_position": stat(drv, lambda r: r["tw"].get("rho")),
        "driver_on_time_subsample": stat(sol_drv, lambda r: r["tw"].get("on_time"), lambda r: r["tw"]["n_window"] or None),
        "solver_on_time_best_direction": stat(sol, lambda r: r["tw"].get("on_time"), lambda r: r["tw"]["n_window"] or None),
        "driver_abs_rho_subsample": stat(sol_drv, lambda r: abs(r["tw"]["rho"]) if r["tw"].get("rho") is not None else None),
        "solver_abs_rho": stat(sol, lambda r: abs(r["rho_both"][0]) if r["rho_both"][0] is not None else None),
    }
    R["time_windows"]["window_length_hours"] = stat(
        drv, lambda r: r["tw"].get("width_sum_h"), lambda r: r["tw"]["n_window"] or None)
    # 3. park and walk
    def pw(dl, sl):
        m = {x["route_id"]: x for x in sl}
        pairs = [(x, m[x["route_id"]]) for x in dl]
        recs = [{"quality": x["quality"], "dx": x["legs"]["total"] - y["legs"]["total"],
                 "dt": x["legs"]["total"], "st": y["legs"]["total"],
                 "bx": [p - q for p, q in zip(x["legs"]["bins"], y["legs"]["bins"])],
                 "dl": x["legs"]["bin_legs"], "sl": y["legs"]["bin_legs"],
                 "wx": x["legs"]["within"] - y["legs"]["within"],
                 "btx": x["legs"]["between"] - y["legs"]["between"]} for x, y in pairs]
        out = {"extra_seconds_per_route": stat(recs, lambda r: r["dx"]),
               "driver_over_solver_time": stat(recs, lambda r: r["dt"], lambda r: r["st"]),
               "share_of_extra_by_leg_length": {lab: stat(recs, lambda r, k=k: r["bx"][k], lambda r: r["dx"])
                                                for k, (_, _, lab) in enumerate(BINS)},
               "extra_seconds_by_leg_length": {lab: stat(recs, lambda r, k=k: r["bx"][k])
                                               for k, (_, _, lab) in enumerate(BINS)},
               "driver_legs_by_length": {lab: stat(recs, lambda r, k=k: r["dl"][k]) for k, (_, _, lab) in enumerate(BINS)},
               "solver_legs_by_length": {lab: stat(recs, lambda r, k=k: r["sl"][k]) for k, (_, _, lab) in enumerate(BINS)},
               "share_of_extra_within_zone": stat(recs, lambda r: r["wx"], lambda r: r["dx"]),
               "share_of_extra_between_zones": stat(recs, lambda r: r["btx"], lambda r: r["dx"])}
        return out
    R["park_and_walk"] = {"train_subsample": pw(sol_drv, sol), "pilot_descriptive": pw(pil_d, pil_s)}

    # 4. first stop
    def fs(rs):
        return {"first_stop_seconds_from_station": stat(rs, lambda r: r["first"]["first_s"]),
                "last_stop_seconds_from_station": stat(rs, lambda r: r["first"]["last_s"]),
                "closest_stop_seconds": stat(rs, lambda r: r["first"]["closest_s"]),
                "median_stop_seconds": stat(rs, lambda r: r["first"]["median_s"]),
                "first_stop_percentile_by_station_distance": stat(rs, lambda r: r["first"]["first_pct"]),
                "first_is_closer_than_last": stat(rs, lambda r: float(r["first"]["first_s"] < r["first"]["last_s"])),
                "first_zone_is_closest_zone_min": stat(rs, lambda r: r["first"].get("first_zone_closest_min")),
                "first_zone_is_closest_zone_mean": stat(rs, lambda r: r["first"].get("first_zone_closest_mean")),
                "first_zone_rank_percentile": stat(rs, lambda r: r["first"].get("first_zone_rank_pct")),
                "chance_first_zone_closest": stat(rs, lambda r: 1 / r["first"]["n_zones"] if "n_zones" in r["first"] else None)}
    R["first_stop"] = {"driver_train": fs(drv), "driver_subsample": fs(sol_drv),
                       "solver_subsample_closer_end": fs(sol),
                       "pilot_driver": fs(pil_d), "pilot_solver_closer_end": fs(pil_s)}

    # 5. within zone
    def wz(rs):
        return {"share_next_is_nearest_unvisited_in_zone": stat(rs, lambda r: r["wz"]["nn"], lambda r: r["wz"]["trans"] or None),
                "chance": stat(rs, lambda r: r["wz"]["chance"], lambda r: r["wz"]["trans"] or None),
                "path_time_over_exact_shortest": stat(rs, lambda r: r["wz"]["drv"], lambda r: r["wz"]["opt"] or None),
                "share_runs_exactly_shortest": stat(rs, lambda r: r["wz"]["optimal_runs"], lambda r: r["wz"]["exact_runs"] or None),
                "share_runs_covered_by_exact": stat(rs, lambda r: r["wz"]["exact_runs"], lambda r: r["wz"]["runs_ge3"] or None)}
    R["within_zone"] = {"driver_train": wz(drv), "driver_subsample": wz(sol_drv), "solver_subsample": wz(sol)}

    # 6. quality labels
    m = {x["route_id"]: x for x in sol}
    R["quality"] = {
        "routes": {lab: sum(r["quality"] == lab for r in drv) for lab in LABELS},
        "revisits": stat(drv, lambda r: r["revisits"]),
        "share_routes_zero_revisits": stat(drv, lambda r: float(r["revisits"] == 0)),
        "stops": stat(drv, lambda r: r["n_stops"] - 1),
        "zones": stat(drv, lambda r: r["zones"]),
        "solver_revisits_subsample": stat(sol, lambda r: r["revisits"]),
        "driver_over_solver_time_subsample": stat(sol_drv, lambda r: r["legs"]["total"] / m[r["route_id"]]["legs"]["total"]),
    }

    out = {"description": "Driver habits on the TRAIN split; pilot only descriptive; validation never opened.",
           "script": "scripts/habits.py", "seed": SEED,
           "bootstrap": f"percentile 95% CI, {N_BOOT} resamples of routes; pooled ratios are sum/sum",
           "train_routes": len(drv), "solver_subsample": {"routes": len(sol), "seconds_per_route": SOLVER_SECONDS,
                                                          "by_quality": {lab: sum(r["quality"] == lab for r in sol) for lab in LABELS}},
           "zone_grammar": grammar, "results": R,
           "runtime_seconds": time.time() - t0,
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    sealed = a.results_dir / "sealed"
    sealed.mkdir(exist_ok=True)
    json.dump(out, open(sealed / "habits.json", "w"), indent=1)
    write_md(out, sealed / "habits.md")
    print(f"done {time.time() - t0:.0f}s peak RSS {out['peak_rss_mb']:.0f} MB")


def write_md(o, path):
    R = o["results"]
    H = ["All"] + LABELS
    L = ["# Driver habits (SEALED: predict before you read past the questions)", "",
         "Produced by `scripts/habits.py`; every number is in `results/sealed/habits.json`.", "",
         "## Predict first", "",
         "Write down a number for each before scrolling. No answers in this section.", "",
         "1. **Zone-ID order.** Zone IDs look like `A-2.1C`. When a driver leaves one zone, how often is the next "
         "zone in the same `A-2` group? And if you guessed the next zone purely from the ID (the next one in "
         "sorted order), how often would you be right? Would \"the nearest zone by driving time\" do better or worse?",
         "2. **Time windows.** What share of stops have a delivery time window? Do drivers visit stops with "
         "earlier deadlines earlier (a correlation near +1, 0 or −1)? What share of window stops would be on time?",
         "3. **Park and walk.** Drivers take longer than the shortest route. What share of that extra time is on "
         "very short hops (under 30 s of driving) between neighbouring stops?",
         "4. **First stop.** Is the driver's first stop closer to or farther from the station than the solver's? "
         "How often is the first zone the zone closest to the station?",
         "5. **Within-zone order.** Inside one zone, how often does the driver go to the nearest stop not yet "
         "visited? How much longer is the driver's path through a zone than the shortest possible path?",
         "6. **Quality labels.** Do High, Medium and Low routes differ in zone revisits and in how much longer they "
         "are than the solver's route?", "",
         "---", "",
         "## Setup", "",
         f"- Measured on the **train split only**: {o['train_routes']} routes "
         f"(High {R['quality']['routes']['High']}, Medium {R['quality']['routes']['Medium']}, "
         f"Low {R['quality']['routes']['Low']}). Validation routes were never opened.",
         f"- Solver comparison: OR-Tools shortest closed tour, {o['solver_subsample']['seconds_per_route']} s per route, on a "
         f"fixed random train subsample of {o['solver_subsample']['routes']} routes "
         f"({', '.join(f'{k} {v}' for k, v in o['solver_subsample']['by_quality'].items())}). "
         "\"Driver (subsample)\" rows use the same routes, so they compare like with like.",
         "- Pilot-route rows (50 High routes, 10 s solver tours from `results/proposed_shortest.json`) are descriptive "
         "only. Nothing in this file is fitted or tuned; every rule is fixed by hand.",
         f"- Brackets are bootstrap 95% confidence intervals ({o['bootstrap']}). A pooled share like "
         "\"62%\" means: add up the hits over all routes, divide by all the chances. n = routes used.",
         "- Travel times are the challenge's planned driving times, not what the van really took.", ""]

    v = lambda s: s["All"]["value"]  # noqa: E731
    P = lambda s: f"{100 * v(s):.0f}%"  # noqa: E731
    zo, tw, pw = R["zone_id_order"], R["time_windows"], R["park_and_walk"]["train_subsample"]
    fsr, w, q = R["first_stop"], R["within_zone"], R["quality"]
    L += ["## In plain words (the reveal)", "",
          f"1. **Zone IDs carry geography.** {P(zo['driver']['same_letter_number'])} of zone changes stay in the same "
          f"group (random order: {P(zo['driver_visits_shuffled']['same_letter_number'])}), and "
          f"{P(zo['driver']['grid_neighbour'])} move one step on the group's little grid (random: "
          f"{P(zo['driver_visits_shuffled']['grid_neighbour'])}; shortest route: {P(zo['solver_subsample']['grid_neighbour'])}). "
          f"But drivers have no favourite direction ({P(zo['driver']['ascending'])} of steps go up the alphabet). "
          f"Guessing the next zone from the ID alone is right {P(zo['next_zone_prediction']['prefix_id'])} of the time "
          f"(chance {P(zo['next_zone_prediction']['chance'])}); simply picking the nearest unvisited zone is right "
          f"{P(zo['next_zone_prediction']['nearest'])}.",
          f"2. **Time windows barely matter.** Only {P(tw['share_stops_with_window'])} of stops have one, and the "
          f"windows are long (about {v(tw['window_length_hours']):.0f} hours). The driver's order is on time for "
          f"{P(tw['driver_on_time'])} of them, but so is the shortest route ({P(tw['solver_on_time_best_direction'])}). "
          f"Deadline order and visit order are only weakly linked (correlation {v(tw['driver_rho_window_end_vs_position']):+.2f}).",
          f"3. **Park and walk does not explain the extra time.** Drivers' tours take "
          f"{v(pw['driver_over_solver_time']):.2f}× the shortest route, about {v(pw['extra_seconds_per_route']) / 60:.0f} "
          f"extra minutes. On hops under 30 s the driver actually spends *less* than the solver "
          f"({P(pw['share_of_extra_by_leg_length']['under 30 s'])} of the extra); the extra sits in hops of 1 minute or more.",
          f"4. **Drivers do not start at the nearest zone.** The driver's first zone is the closest zone to the station "
          f"{P(fsr['driver_train']['first_zone_is_closest_zone_min'])} of the time (the solver's nearer end: "
          f"{P(fsr['solver_subsample_closer_end']['first_zone_is_closest_zone_min'])}). The first stop is on average "
          f"{v(fsr['driver_train']['first_stop_seconds_from_station']) / 60:.0f} min from the station, about "
          f"{(v(fsr['driver_subsample']['first_stop_seconds_from_station']) - v(fsr['solver_subsample_closer_end']['first_stop_seconds_from_station'])) / 60:.1f} "
          "min farther than the solver's.",
          f"5. **Inside a zone, drivers are close to shortest.** {P(w['driver_train']['share_next_is_nearest_unvisited_in_zone'])} "
          f"of steps go to the nearest unvisited stop (chance {P(w['driver_train']['chance'])}), and the path through a "
          f"zone is only {100 * (v(w['driver_train']['path_time_over_exact_shortest']) - 1):.0f}% longer than the best "
          f"path with the same entry and exit.",
          f"6. **Quality labels hardly differ.** Revisits per route: High {q['revisits']['High']['value']:.2f}, Medium "
          f"{q['revisits']['Medium']['value']:.2f}, Low {q['revisits']['Low']['value']:.2f}; driver/solver time: High "
          f"{q['driver_over_solver_time_subsample']['High']['value']:.2f}, Medium "
          f"{q['driver_over_solver_time_subsample']['Medium']['value']:.2f}, Low "
          f"{q['driver_over_solver_time_subsample']['Low']['value']:.2f}. Intervals overlap.", "",
          "What it suggests for the model (a hypothesis to test, not a result): the driver's knowledge is mostly about "
          "*which zone next*; the solver can handle the stops inside each zone.", ""]

    g = o["zone_grammar"]
    L += ["## 1. Zone-ID order", "",
          "**The grammar.** Every real zone ID on the train routes matches `letter-number.digit letter`, e.g. "
          "`A-2.1C`: a letter (`A`), a number (`2`), a sub-number (`1`) and a sub-letter (`C`). Call `A-2` the "
          "*group* and `A-2.1` the *row*.",
          f"- Letters seen: {', '.join(g['parts']['letter'])}. Numbers: {min(map(int, g['parts']['number']))} to "
          f"{max(map(int, g['parts']['number']))}. Sub-numbers: {', '.join(g['parts']['sub_number'])}. "
          f"Sub-letters: {', '.join(g['parts']['sub_letter'])} (no F or I).",
          f"- Stops that do not fit: {', '.join(f'`{k}` ×{v}' for k, v in g['unparsed_non_station_stops'].items())}. "
          "Those are left out of the ID measures.",
          "- \"Sorted ID order\" sorts by letter, then number, then sub-number, then sub-letter.", "",
          "**Definitions.** Take the driver's zone *visits* (consecutive stops in the same zone count as one visit, "
          "as in `zone_revisits.md`) and look at each pair of consecutive visits.",
          "- *same letter / group / row*: the next zone shares the first 1 / 2 / 3 parts.",
          "- *grid neighbour*: same group, and the sub-number or the sub-letter (not both) moves one step "
          "(e.g. `A-2.1C` → `A-2.1D` or `A-2.2C`).",
          "- *sorted neighbour*: the next zone is right before or right after the current one in this route's "
          "sorted list of zones.",
          "- *ascending*: the next ID sorts after the current one.",
          "- *shuffled*: the same visits put in random order, as a chance reference.", ""]
    zo = R["zone_id_order"]
    names = [("same letter", "same_letter"), ("same group (A-2)", "same_letter_number"),
             ("same row (A-2.1)", "same_lns"), ("grid neighbour", "grid_neighbour"),
             ("sorted neighbour", "sorted_adjacent"), ("ascending", "ascending")]
    L += ["| Share of consecutive visits | Driver (train) | Shuffled | Driver (subsample) | Solver (subsample) |",
          "|---|---|---|---|---|"]
    for lab, k in names:
        L.append(f"| {lab} | {fmt(zo['driver'][k]['All'], True)} | {fmt(zo['driver_visits_shuffled'][k]['All'], True)} | "
                 f"{fmt(zo['driver_on_solver_subsample'][k]['All'], True)} | {fmt(zo['solver_subsample'][k]['All'], True)} |")
    L += ["", "By quality label (driver, train):", ""]
    L += table([(lab, zo["driver"][k]) for lab, k in names], [True] * len(names), H)
    L += ["", f"**Direction.** Per route, the share of steps that go in the route's main direction (ascending or "
          f"descending, whichever is more common), averaged over routes: driver "
          f"{fmt(zo['route_direction_consistency']['All'], True)}, shuffled "
          f"{fmt(zo['route_direction_consistency_shuffled']['All'], True)}.", "",
          "**Guessing the next zone.** Each time the driver leaves a zone, guess which zone comes next among the "
          "zones not yet entered; score a hit if the guess is right. Fixed rules, nothing learned:",
          "- *chance*: pick at random (expected hit rate).",
          "- *ID next*: the nearest zone in sorted ID order, continuing in the direction of the last step "
          "(ascending at the start); if none is left that way, the nearest the other way.",
          "- *same prefix, then ID*: prefer zones sharing the most leading parts with the current one; ties by ID next.",
          "- *nearest*: the zone with the stop closest (driving time) to the driver's last stop.",
          "- *same prefix, then nearest*: prefer zones sharing the most leading parts; ties by driving time.",
          "- *revisit*: the driver went back to an already-visited zone (every rule misses these).", ""]
    pn = [("chance", "chance"), ("ID next", "id_next"), ("same prefix, then ID", "prefix_id"),
          ("nearest", "nearest"), ("same prefix, then nearest", "prefix_nearest"), ("(revisit, all rules miss)", "revisit")]
    L += table([(lab, zo["next_zone_prediction"][k]) for lab, k in pn], [True] * len(pn), H)

    tw = R["time_windows"]
    L += ["", "## 2. Time windows", "",
          "**Definitions.** A stop *has a window* if any of its packages has a delivery time window. The stop's "
          "window runs from the earliest package start to the latest package end (a lenient choice). "
          "*Arrival time* under an order = sum of planned driving times plus the planned service time at each "
          "earlier stop, starting at departure; no waiting, no breaks. *On time* = arrival before the window "
          "ends; *early* = arrival before it opens (the driver would have to wait). *Rank correlation* (Spearman, "
          "−1 to +1): per route with at least 3 window stops, does a later window end go with a later visit "
          "position? +1 = perfectly in deadline order; 0 = no relation. The solver ignores windows and its "
          "direction is arbitrary, so for the solver we use whichever direction is on time more often, and the "
          "size of its correlation (|ρ|) rather than its sign.", ""]
    rows = [("stops with a window", tw["share_stops_with_window"], True),
            ("routes with any window stop", tw["share_routes_with_any_window"], True),
            ("mean window length, hours", tw["window_length_hours"], False),
            ("window stops on time (driver)", tw["driver_on_time"], True),
            ("window stops early (driver)", tw["driver_early"], True),
            ("mean position of window stops (0 = first, 1 = last)", tw["driver_mean_relative_position"], False),
            ("rank corr. window end vs position (driver)", tw["driver_rho_window_end_vs_position"], False)]
    L += table([(a, b) for a, b, _ in rows], [c for _, _, c in rows], H)
    L += ["", "Driver against the solver on the subsample:", ""]
    rows = [("on time, driver", tw["driver_on_time_subsample"], True),
            ("on time, solver (better direction)", tw["solver_on_time_best_direction"], True),
            ("|rank corr.|, driver", tw["driver_abs_rho_subsample"], False),
            ("|rank corr.|, solver", tw["solver_abs_rho"], False)]
    L += table([(a, b) for a, b, _ in rows], [c for _, _, c in rows], H)

    L += ["", "## 3. Park and walk", "",
          "**Definitions.** Split each closed tour (station → stops → station) into legs and put each leg in a "
          "bin by its planned driving time. *Extra time* = driver tour time − solver tour time for the same "
          "route. The extra in a bin = driver's time on legs in that bin − solver's time on legs in that bin, so "
          "the bins add up to the whole extra. A share can be negative (the driver spends *less* time than the "
          "solver in that bin) or above 100%. Pooled over routes.", ""]
    for key, title in [("train_subsample", "Train subsample (5 s solver)"), ("pilot_descriptive", "Pilot routes, descriptive (10 s solver)")]:
        p = R["park_and_walk"][key]
        hh = ["All"] if key == "pilot_descriptive" else H
        L += [f"**{title}**", ""]
        rows = [("driver time / solver time", p["driver_over_solver_time"], False),
                ("extra seconds per route", p["extra_seconds_per_route"], False)]
        rows += [(f"share of extra on legs {lab}", p["share_of_extra_by_leg_length"][lab], True) for _, _, lab in BINS]
        rows += [(f"extra seconds per route, legs {lab}", p["extra_seconds_by_leg_length"][lab], False) for _, _, lab in BINS]
        rows += [(f"driver legs {lab} per route", p["driver_legs_by_length"][lab], False) for _, _, lab in BINS]
        rows += [(f"solver legs {lab} per route", p["solver_legs_by_length"][lab], False) for _, _, lab in BINS]
        rows += [("share of extra on legs inside one zone", p["share_of_extra_within_zone"], True),
                 ("share of extra on legs between zones (or to/from station)", p["share_of_extra_between_zones"], True)]
        L += table([(a, b) for a, b, _ in rows], [c for _, _, c in rows], hh) + [""]

    L += ["## 4. First stop and direction", "",
          "**Definitions.** Distance = planned driving seconds from the station. *Percentile* of the first stop = "
          "share of the route's stops that are closer to the station than it (0% = it is the closest stop). A "
          "zone's distance = its closest stop (\"min\") or the average of its stops (\"mean\"). For the solver, whose "
          "direction is arbitrary, the \"first\" stop is whichever tour end is closer to the station, which "
          "flatters the solver on these measures.", ""]
    fsr = R["first_stop"]
    keys = [("first stop, seconds from station", "first_stop_seconds_from_station", False),
            ("last stop, seconds from station", "last_stop_seconds_from_station", False),
            ("closest stop, seconds", "closest_stop_seconds", False),
            ("median stop, seconds", "median_stop_seconds", False),
            ("first-stop percentile", "first_stop_percentile_by_station_distance", True),
            ("first stop closer than last stop", "first_is_closer_than_last", True),
            ("first zone = closest zone (min)", "first_zone_is_closest_zone_min", True),
            ("first zone = closest zone (mean)", "first_zone_is_closest_zone_mean", True),
            ("first-zone percentile among zones", "first_zone_rank_percentile", True),
            ("chance that a random zone is the closest", "chance_first_zone_closest", True)]
    L += ["Driver, all train routes:", ""]
    L += table([(a, fsr["driver_train"][k]) for a, k, _ in keys], [c for _, _, c in keys], H)
    L += ["", "Driver against solver, same routes (overall):", "",
          "| Measure | Driver (subsample) | Solver (subsample) | Driver (pilot) | Solver (pilot, 10 s) |",
          "|---|---|---|---|---|"]
    for a, k, c in keys:
        L.append(f"| {a} | {fmt(fsr['driver_subsample'][k]['All'], c)} | {fmt(fsr['solver_subsample_closer_end'][k]['All'], c)} | "
                 f"{fmt(fsr['pilot_driver'][k]['All'], c)} | {fmt(fsr['pilot_solver_closer_end'][k]['All'], c)} |")

    w = R["within_zone"]
    L += ["", "## 5. Within-zone order", "",
          "**Definitions.** A *run* = consecutive driver stops in the same zone; only runs of 3+ stops count. For "
          "each step inside a run: is the next stop the nearest (driving time) stop of that zone not yet visited? "
          "Chance = 1 / number of such stops. *Exact shortest path*: for runs of 3 to 9 stops, keep the driver's "
          "first and last stop of the run fixed and try every order of the stops in between; compare the driver's "
          "driving time through the run to the best one.", ""]
    rows = [("next = nearest unvisited stop in zone", "share_next_is_nearest_unvisited_in_zone", True),
            ("chance", "chance", True),
            ("driver path time / exact shortest", "path_time_over_exact_shortest", False),
            ("runs where driver's path is exactly shortest", "share_runs_exactly_shortest", True),
            ("runs short enough for the exact check", "share_runs_covered_by_exact", True)]
    L += table([(a, w["driver_train"][k]) for a, k, _ in rows], [c for _, _, c in rows], H)
    L += ["", "Same measures for the solver's own runs (subsample), for reference:", "",
          "| Measure | Driver (subsample) | Solver (subsample) |", "|---|---|---|"]
    for a, k, c in rows:
        L.append(f"| {a} | {fmt(w['driver_subsample'][k]['All'], c)} | {fmt(w['solver_subsample'][k]['All'], c)} |")

    q = R["quality"]
    L += ["", "## 6. Quality labels", "",
          "**Definitions.** Revisits as in `zone_revisits.md` (visits − distinct zones). Driver / solver time = "
          "per-route ratio of closed-tour driving times, averaged over routes (1.10 = driver 10% longer).", ""]
    rows = [("driver zone revisits per route", q["revisits"], False),
            ("routes with zero revisits", q["share_routes_zero_revisits"], True),
            ("stops per route", q["stops"], False), ("zones per route", q["zones"], False),
            ("solver zone revisits (subsample)", q["solver_revisits_subsample"], False),
            ("driver time / solver time (subsample)", q["driver_over_solver_time_subsample"], False)]
    L += table([(a, b) for a, b, _ in rows], [c for _, _, c in rows], H)

    L += ["", "## Caveats", "",
          "- Exploratory and descriptive. No hypothesis was pre-registered; with this many measures, some "
          "differences will look real by chance. Treat label differences with overlapping intervals as no evidence.",
          "- Bootstrap intervals resample routes but treat routes as independent; routes from the same station and "
          "day share streets, so true uncertainty is somewhat wider.",
          "- The 5 s solver is a heuristic, not a proven optimum; a better solver would make the driver's extra "
          "time look slightly larger.",
          "- Arrival times use planned driving and service times; real arrivals differ. Window on-time shares are "
          "a model, not what happened.",
          "- Low has only ~100 routes; its intervals are wide.", ""]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
