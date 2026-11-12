#!/usr/bin/env python3
"""v0 decoding and scoring: turn the trained zone model into full stop orders and score them.

Two jobs, run from tacit-routes/:

  1. The plain shortest route on the 300 validation routes (same settings as
     run_plain_baseline.py: OR-Tools, PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH, 10 s per route,
     closed tour from the station). This is the opponent every v0 variant is compared with.
       .venv/bin/python -I scripts/v0_decode.py shortest --workers 3

  2. Decode every v0 variant on val, fill in the stops inside each zone with OR-Tools,
     score everything with the official scorer and write results/v0.json.
       .venv/bin/python -I scripts/v0_decode.py decode --workers 3

  3. Check whether each decoded zone order runs in the driver's direction (Kendall tau),
     because the official score hardly punishes a reversed route. Adds to results/v0.json.
       .venv/bin/python -I scripts/v0_decode.py direction

The validation split is used only for the final numbers here. The one tuned knob (the
travel-time weight) is tuned on the held-out 10% of train, never on val, never on pilot.
"""
import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))       # -I leaves the script folder off the path; add it back
from v0_model import (FEATURE_NAMES, ZonePointer, build_examples, driver_zone_order,  # noqa: E402
                      evaluate, features, prepare_route, split_train, zone_membership)
ROOT = HERE.parent
SCORER = HERE / "official_rc_cli" / "score.py"
ROUTES_DIR = ROOT / "data" / "processed" / "routes"
SPLITS = ROOT / "data" / "processed" / "splits.json"
INVALID = ROOT / "data" / "raw" / "invalid_sequence_scores.json"
RUN_DIR = ROOT / "runs" / "v0"
RESULTS = ROOT / "results"
SHORTEST_VAL = RESULTS / "v0_shortest_val.json"


# ----------------------------------------------------------------------------------------------
# Small helpers shared by both jobs
# ----------------------------------------------------------------------------------------------

def load_scorer():
    """Import the official, unmodified rc-cli score.py from its file path."""
    spec = importlib.util.spec_from_file_location("official_score", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_npz(route_id):
    """One route as saved by build_routes.py (station is index 0)."""
    with np.load(ROUTES_DIR / f"{route_id}.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def cost_dict(route):
    """Travel times in the challenge's JSON layout: {stop: {stop: seconds}}.

    The npz stores float32; the raw data has one decimal place, so rounding to 0.1 s
    gives back the original numbers exactly."""
    ids = [str(s) for s in route["stop_ids"]]
    tt = route["travel_time"]
    return {a: {b: round(float(tt[i, j]), 1) for j, b in enumerate(ids)} for i, a in enumerate(ids)}


def actual_dict(route):
    """The driver's order in the challenge's JSON layout: {"actual": {stop: position}}."""
    ids = route["stop_ids"]
    return {"actual": {str(ids[i]): pos for pos, i in enumerate(route["actual_order"])}}


def as_proposal(order):
    """Our order in the challenge's JSON layout: {"proposed": {stop: position}}."""
    return {"proposed": {stop: position for position, stop in enumerate(order)}}


_SCORER = None


def score_chunk(job):
    """Score a few routes with the official evaluate(), the way the challenge did:
    write the four JSON files it expects, call it, read back each route's score.

    evaluate() loads whole files into Python dicts; travel times for 300 routes would not fit
    in our memory budget, so callers hand us chunks of about 20 routes. Per-route scores are
    exactly what a single big call would give."""
    global _SCORER
    if _SCORER is None:
        _SCORER = load_scorer()
    orders, invalid = job
    tmp = RUN_DIR / "score_tmp" / str(os.getpid())
    tmp.mkdir(parents=True, exist_ok=True)
    part = sorted(orders)
    routes = {r: load_npz(r) for r in part}
    files = {
        "actual": {r: actual_dict(routes[r]) for r in part},
        "proposed": {r: as_proposal(orders[r]) for r in part},
        "costs": {r: cost_dict(routes[r]) for r in part},
        "invalid": {r: invalid[r] for r in part},
    }
    for name, data in files.items():
        with open(tmp / f"{name}.json", "w") as f:
            json.dump(data, f)
    out = _SCORER.evaluate(str(tmp / "actual.json"), str(tmp / "proposed.json"),
                           str(tmp / "costs.json"), str(tmp / "invalid.json"))
    for name in files:
        (tmp / f"{name}.json").unlink(missing_ok=True)
    return {r: (float(out["route_scores"][r]), bool(out["route_feasibility"][r])) for r in part}


def official_scores(orders, invalid, pool=None, chunk=20):
    """{route_id: [stop ids, station first]} -> ({route_id: score}, {route_id: feasible})."""
    ids = sorted(orders)
    jobs = [({r: orders[r] for r in ids[k:k + chunk]}, {r: invalid[r] for r in ids[k:k + chunk]})
            for k in range(0, len(ids), chunk)]
    parts = pool.map(score_chunk, jobs, chunksize=1) if pool else [score_chunk(j) for j in jobs]
    merged = {r: v for p in parts for r, v in p.items()}
    return {r: v[0] for r, v in merged.items()}, {r: v[1] for r, v in merged.items()}


def bootstrap_ci(values, seed, resamples=10000):
    """Percentile 95% CI of the mean, resampling routes with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_test(wins, losses):
    """Exact two-sided sign test: if wins and losses were really 50/50, how surprising is
    a split at least this lopsided? Ties are dropped, as is standard."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


# ----------------------------------------------------------------------------------------------
# Job 1: the plain shortest route on val
# ----------------------------------------------------------------------------------------------

def solve_shortest(job):
    """Closed shortest tour from the station; identical settings to run_plain_baseline.py."""
    route_id, time_limit = job
    route = load_npz(route_id)
    ids = [str(s) for s in route["stop_ids"]]
    # Same integer costs as the baseline: tenths of a second.
    costs = np.rint(route["travel_time"].astype(np.float64) * 10).astype(np.int64).tolist()
    manager = pywrapcp.RoutingIndexManager(len(ids), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    callback = routing.RegisterTransitCallback(
        lambda i, j: costs[manager.IndexToNode(i)][manager.IndexToNode(j)])
    routing.SetArcCostEvaluatorOfAllVehicles(callback)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit)
    t0 = time.time()
    solution = routing.SolveWithParameters(params)
    if solution is None:
        return route_id, None, time.time() - t0
    order, index = [], routing.Start(0)
    while not routing.IsEnd(index):
        order.append(ids[manager.IndexToNode(index)])
        index = solution.Value(routing.NextVar(index))
    return route_id, order, time.time() - t0


def run_shortest(args):
    splits = json.load(open(SPLITS))
    val_ids = sorted(r["route_id"] for r in splits["val"])
    if args.limit:
        val_ids = val_ids[:args.limit]
    invalid = json.load(open(INVALID))
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        solved = pool.map(solve_shortest, [(r, args.time_limit) for r in val_ids], chunksize=1)
    wall = time.time() - t0
    if any(order is None for _, order, _ in solved):
        raise SystemExit("solver found no tour for some route")
    orders = {r: order for r, order, _ in solved}
    scores, feasible = official_scores(orders, invalid)
    out = {
        "description": "Plain shortest route on the 300 validation routes (opponent for v0)",
        "settings": {"solver": "OR-Tools routing, one vehicle, closed tour from the station",
                     "first_solution": "PATH_CHEAPEST_ARC", "metaheuristic": "GUIDED_LOCAL_SEARCH",
                     "time_limit_seconds_per_route": args.time_limit, "workers": args.workers,
                     "cost_units": "travel times in tenths of a second"},
        "mean": float(np.mean(list(scores.values()))),
        "bootstrap_95_ci": bootstrap_ci([scores[r] for r in val_ids], 20261111),
        "all_feasible": all(feasible.values()),
        "solve_wall_seconds": wall,
        "per_route": {r: {"score": scores[r], "order": orders[r]} for r in val_ids},
    }
    path = SHORTEST_VAL if not args.limit else RUN_DIR / "shortest_test.json"
    with open(path.with_suffix(".part"), "w") as f:
        json.dump(out, f)
    os.replace(path.with_suffix(".part"), path)
    print(f"shortest on val: {out['mean']:.4f} {out['bootstrap_95_ci']} in {wall:.0f}s solve")


# ----------------------------------------------------------------------------------------------
# Job 2, part A: from a zone order to a full stop order (OR-Tools inside each zone)
# ----------------------------------------------------------------------------------------------

def solve_zone_path(costs, start, stops, exit_cost, time_limit_s=None):
    """Shortest path that starts at `start` (the last stop already driven), visits every stop
    in `stops` once, and ends wherever leaving towards the next zone is cheapest.

    costs:     integer travel times (tenths of a second) between all stops of the route
    exit_cost: {stop: cost of going from that stop to the next zone}
    time_limit_s: None = greedy descent (default, used for every v0 number); a number = guided
                  local search for that many seconds on this zone
    We add an imaginary END node: reaching it from stop s costs exit_cost[s]. The solver
    must end at END, so it is pushed to finish near the next zone."""
    if len(stops) == 1:
        return list(stops)
    nodes = [start] + list(stops)                         # node 0 = start, last node = END
    n = len(nodes) + 1
    END = n - 1
    big = 10**9

    def arc(i, j):
        if i == END:                                      # nothing leaves END; cost unused
            return 0
        if j == END:                                      # leaving towards the next zone
            return big if i == 0 else exit_cost[nodes[i]]
        return costs[nodes[i]][nodes[j]]

    matrix = [[arc(i, j) for j in range(n)] for i in range(n)]
    manager = pywrapcp.RoutingIndexManager(n, 1, [0], [END])
    routing = pywrapcp.RoutingModel(manager)
    cb = routing.RegisterTransitCallback(
        lambda a, b: matrix[manager.IndexToNode(a)][manager.IndexToNode(b)])
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    if time_limit_s is None:
        # Greedy descent = keep applying small improving moves until none helps. Deterministic
        # and fast (milliseconds for a zone of about 7 stops); no time limit needed.
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
    else:
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromMilliseconds(max(1, int(time_limit_s * 1000)))
    solution = routing.SolveWithParameters(params)
    path, index = [], solution.Value(routing.NextVar(routing.Start(0)))
    while not routing.IsEnd(index):
        path.append(nodes[manager.IndexToNode(index)])
        index = solution.Value(routing.NextVar(index))
    return path


def stops_from_zone_order(route, zone_order, time_limit_s=None):
    """Zone order -> full stop order. Reusable on its own (e.g. for Opus's zone orders).

    route:        one route as returned by load_npz(route_id) (needs stop_ids, travel_time,
                  zone_ids)
    zone_order:   list of zone ID strings, e.g. ["A-2.2A", "A-2.1A", ...], containing every zone
                  of the route exactly once. Zones are defined as in v0_model.zone_membership:
                  stops with no zone ID belong to the zone of their nearest stop.
    time_limit_s: None (default, used for all v0 results) = OR-Tools PATH_CHEAPEST_ARC +
                  GREEDY_DESCENT per zone; a number = GUIDED_LOCAL_SEARCH for that many seconds
                  per zone.
    Returns the stop IDs, station first. Inside each zone the stops follow the shortest path
    that starts at the previous zone's last stop (the station for the first zone) and ends
    where leaving for the next zone is cheapest (the next zone's nearest stop; the station
    after the last zone). Raises ValueError if zone_order is not a permutation of the zones."""
    names, zone_of = zone_membership(route)
    if sorted(zone_order) != names:
        missing = sorted(set(names) - set(zone_order))
        extra = sorted(set(zone_order) - set(names))
        raise ValueError(f"zone_order must list every zone once; missing {missing}, unknown {extra}, "
                         f"{len(zone_order)} given for {len(names)} zones")
    index = {z: k for k, z in enumerate(names)}
    members = [np.where(zone_of == k)[0] for k in range(len(names))]
    costs = np.rint(route["travel_time"].astype(np.float64) * 10).astype(np.int64).tolist()
    order = [0]                                           # stop indices; 0 is the station
    for k, name in enumerate(zone_order):
        nxt = members[index[zone_order[k + 1]]] if k + 1 < len(zone_order) else [0]
        stops = [int(i) for i in members[index[name]]]
        exit_cost = {s: min(costs[s][int(j)] for j in nxt) for s in stops}
        order += solve_zone_path(costs, order[-1], stops, exit_cost, time_limit_s)
    ids = route["stop_ids"]
    assert sorted(order) == list(range(len(ids)))         # every stop exactly once
    return [str(ids[i]) for i in order]


# ----------------------------------------------------------------------------------------------
# Job 2, part B: from the model to a zone order
# ----------------------------------------------------------------------------------------------

def beam_decode(model, P, beam, lam):
    """Choose a zone order using the model's log-probabilities.

    beam = 1 is greedy decoding: at every step take the single most likely next zone.
    beam = 5 keeps the 5 best partial orders at every step, so one tempting but bad early
    choice can be undone later. A partial order's score is the sum of its log-probabilities
    (= log of the product of probabilities), minus lam * travel time (in units of the route's
    typical zone-to-zone time). lam = 0 is the pure model."""
    import torch
    Z = P["Z"]
    # Each state: (score, zone order so far, visited mask, current zone, previous zone)
    states = [(0.0, [], np.zeros(Z, dtype=bool), -1, -1)]
    for t in range(Z):
        x = np.stack([features(P, cur, prev, vis, t) for _, _, vis, cur, prev in states])
        m = np.stack([~vis for _, _, vis, _, _ in states])
        with torch.no_grad():
            logp = model(torch.from_numpy(x), torch.from_numpy(m)).numpy()
        expansions = []
        for s, (score, order, vis, cur, prev) in enumerate(states):
            secs = P["ztt"][cur if cur >= 0 else Z]
            for z in np.where(~vis)[0]:
                total = score + float(logp[s, z]) - lam * float(secs[z]) / P["tt_scale"]
                expansions.append((total, s, int(z)))
        expansions.sort(key=lambda e: -e[0])
        new_states = []
        for total, s, z in expansions[:beam]:
            _, order, vis, cur, _ = states[s]
            vis2 = vis.copy()
            vis2[z] = True
            new_states.append((total, order + [z], vis2, z, cur))
        states = new_states
    return states[0][1]


def nearest_zone_order(P):
    """No-model control: always drive to the nearest unvisited zone (average travel time)."""
    Z = P["Z"]
    vis, cur, order = np.zeros(Z, dtype=bool), -1, []
    for _ in range(Z):
        secs = P["ztt"][cur if cur >= 0 else Z].copy()
        secs[vis] = np.inf
        cur = int(np.argmin(secs))
        vis[cur] = True
        order.append(cur)
    return order


_MODELS = {}
_HISTORY = None


def load_model(tag):
    """Load a trained checkpoint (cached per worker process).
    Returns (model, history mode): "none" = v0, "all" = v0-hist, "past" = v0-hist-past."""
    import torch
    if tag not in _MODELS:
        torch.set_num_threads(1)                          # one thread per worker, 3 workers
        ck = torch.load(RUN_DIR / tag / "model.pt", weights_only=False)
        model = ZonePointer(ck["n_features"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        mode = ck["history"]
        _MODELS[tag] = (model, {True: "all", False: "none"}.get(mode, mode))
    return _MODELS[tag]


def get_history():
    """Train-only station history (v0-hist and v0-hist-past), loaded once per process."""
    global _HISTORY
    if _HISTORY is None:
        from v0_history import History
        _HISTORY = History()
    return _HISTORY


def prepare_for(route, tag):
    """prepare_route plus, for history models, the history arrays. Val routes are not in the
    train table, so "all" mode subtracts nothing for them; "past" mode keeps only train
    routes from days before the route's date."""
    P = prepare_route(route)
    mode = load_model(tag)[1] if tag is not None else "none"
    if mode != "none":
        get_history().attach(P, str(route["station"]), str(route["route_id"]),
                             mode=mode, date=str(route["date"]))
    return P


def decode_job(job):
    """One route, one method -> full stop order. Runs inside a worker process."""
    route_id, method, tag, beam, lam = job
    if tag is not None:
        load_model(tag)                                   # one-off load, not counted per route
        if load_model(tag)[1] != "none":
            get_history()
    route = load_npz(route_id)
    # CPU time (this process only), split into "zone order" (features, history lookups and
    # the model) and "solver" (OR-Tools inside zones). Wall time is also kept.
    w0, c0 = time.time(), time.process_time()
    P = prepare_for(route, tag)
    if method == "model":
        zones = beam_decode(load_model(tag)[0], P, beam, lam)
    elif method == "nearest_zone":
        zones = nearest_zone_order(P)
    elif method == "driver_zones":                        # oracle: the driver's own zone order
        zones = driver_zone_order(P, route)
    else:
        raise ValueError(method)
    c1 = time.process_time()
    zone_names = [P["names"][z] for z in zones]
    order = stops_from_zone_order(route, zone_names)
    c2 = time.process_time()
    timing = {"zone_order_cpu_s": c1 - c0, "solver_cpu_s": c2 - c1, "total_cpu_s": c2 - c0,
              "total_wall_s": time.time() - w0}
    return job, order, zone_names, timing


def run_variant(pool, route_ids, method, tag, beam, lam, invalid):
    """Decode and score one variant on a list of routes."""
    jobs = [(r, method, tag, beam, lam) for r in route_ids]
    done = pool.map(decode_job, jobs, chunksize=4)
    orders = {job[0]: order for job, order, _, _ in done}
    zone_orders = {job[0]: z for job, _, z, _ in done}
    timing = {job[0]: s for job, _, _, s in done}
    scores, feasible = official_scores(orders, invalid, pool)
    return {"scores": scores, "feasible": feasible, "orders": orders,
            "zone_orders": zone_orders, "timing": timing}


def timing_summary(timing):
    """Mean and median per-route seconds for each timing field."""
    out = {}
    for key in ("zone_order_cpu_s", "solver_cpu_s", "total_cpu_s", "total_wall_s"):
        v = [t[key] for t in timing.values()]
        out[key] = {"mean": float(np.mean(v)), "median": float(np.median(v))}
    return out


# ----------------------------------------------------------------------------------------------
# Job 2, part C: everything together
# ----------------------------------------------------------------------------------------------

def paired(variant_scores, shortest_scores, ids, seed):
    """Variant minus shortest, route by route (negative = variant closer to the driver)."""
    diff = [variant_scores[r] - shortest_scores[r] for r in ids]
    better = sum(d < 0 for d in diff)
    worse = sum(d > 0 for d in diff)
    return {"mean_diff": float(np.mean(diff)), "diff_95_ci": bootstrap_ci(diff, seed),
            "better": better, "worse": worse, "tied": len(diff) - better - worse,
            "sign_test_p": sign_test(better, worse)}


def teacher_forced_accuracy(model, data):
    """Next-zone top-1 accuracy when the model is shown the driver's real history at each step,
    plus the same for the no-model rule "go to the nearest zone" for comparison."""
    loss, acc, steps = evaluate(model, data)
    near = int(sum((X[np.arange(len(Y)), Y, FEATURE_NAMES.index("is_nearest_candidate")] == 1).sum()
                   for _, X, _, Y in data))
    return {"top1": acc, "cross_entropy": loss, "steps": steps, "nearest_zone_top1": near / steps}


def worked_example(tag, route_id, steps=5):
    """Top-3 next-zone probabilities along the driver's real route, first few steps."""
    import torch
    model = load_model(tag)[0]
    route = load_npz(route_id)
    P = prepare_for(route, tag)
    order = driver_zone_order(P, route)
    Z, rows = P["Z"], []
    vis, cur, prev = np.zeros(Z, dtype=bool), -1, -1
    for t in range(min(steps, Z - 1)):
        x = features(P, cur, prev, vis, t)[None]
        with torch.no_grad():
            p = model(torch.from_numpy(x), torch.from_numpy(~vis[None])).exp().numpy()[0]
        top = np.argsort(-p)[:3]
        actual = order[t]
        secs = P["ztt"][cur if cur >= 0 else Z]
        cand = np.where(~vis)[0]
        nearest = cand[np.argmin(secs[cand])]
        rows.append({
            "step": t + 1, "current_zone": P["names"][cur] if cur >= 0 else "station",
            "candidates": int((~vis).sum()),
            "top3": [{"zone": P["names"][z], "prob": float(p[z]),
                      "minutes": float(secs[z] / 60)} for z in top],
            "driver_choice": P["names"][actual], "driver_choice_prob": float(p[actual]),
            "driver_choice_rank": int((p > p[actual]).sum()) + 1,
            "nearest_zone": P["names"][nearest],
        })
        vis[actual] = True
        prev, cur = cur, actual
    return {"route_id": route_id, "n_stops": len(route["stop_ids"]), "n_zones": Z,
            "driver_zone_order": [P["names"][z] for z in order], "steps": rows}


def summarize(scores, ids, seed):
    v = [scores[r] for r in ids]
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "bootstrap_95_ci": bootstrap_ci(v, seed)}


def write_curve_svg(summaries, path):
    """Training curve as a plain SVG (no plotting library needed): cross-entropy per epoch,
    solid = training routes, dashed = held-out High routes. Lower is better."""
    W, H, L, R, T, B = 640, 360, 60, 150, 20, 45
    curves = {t: s["curve"] for t, s in summaries.items()}
    ys = [c[k] for cs in curves.values() for c in cs for k in ("train_loss", "heldout_loss")]
    xmax = max(len(cs) for cs in curves.values())
    lo, hi = min(ys) * 0.97, max(ys) * 1.03
    X = lambda e: L + (e - 1) / max(xmax - 1, 1) * (W - L - R)
    Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    colours = ["#1f77b4", "#d62728", "#2ca02c"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="black"/>',
             f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="black"/>',
             f'<text x="{(L + W - R) / 2}" y="{H - 10}" text-anchor="middle">epoch</text>',
             f'<text x="14" y="{(T + H - B) / 2}" transform="rotate(-90 14 {(T + H - B) / 2})" '
             f'text-anchor="middle">cross-entropy (lower = better)</text>']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        parts.append(f'<text x="{L - 5}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.3f}</text>')
    for e in range(1, xmax + 1, max(1, xmax // 10)):
        parts.append(f'<text x="{X(e):.1f}" y="{H - B + 15}" text-anchor="middle">{e}</text>')
    for i, (tag, cs) in enumerate(curves.items()):
        col = colours[i % len(colours)]
        for key, dash in (("train_loss", ""), ("heldout_loss", ' stroke-dasharray="5,4"')):
            pts = " ".join(f"{X(c['epoch']):.1f},{Y(c[key]):.1f}" for c in cs)
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        best = summaries[tag]["best_epoch"]
        bc = next(c for c in cs if c["epoch"] == best)
        parts.append(f'<circle cx="{X(best):.1f}" cy="{Y(bc["heldout_loss"]):.1f}" r="4" fill="{col}"/>')
        y0 = T + 10 + i * 36
        parts.append(f'<line x1="{W - R + 10}" y1="{y0}" x2="{W - R + 30}" y2="{y0}" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{W - R + 35}" y="{y0 + 4}">{tag} train</text>')
        parts.append(f'<line x1="{W - R + 10}" y1="{y0 + 14}" x2="{W - R + 30}" y2="{y0 + 14}" stroke="{col}" '
                     f'stroke-width="2" stroke-dasharray="5,4"/>')
        parts.append(f'<text x="{W - R + 35}" y="{y0 + 18}">{tag} held-out</text>')
    parts.append(f'<text x="{W - R + 10}" y="{T + 10 + len(curves) * 36 + 4}">dot = kept epoch</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def family(tag):
    """Model family from its tag: v0 (no history), v0-hist, v0-hist-past."""
    return "v0-hist-past" if tag.startswith("past_") else "v0-hist" if tag.startswith("hist_") else "v0"


def history_coverage(val_ids):
    """How much history each target gets. For v0-hist-past, early-dated routes see little or no
    past; this counts them. Uses only the train table (and val routes' own zone lists)."""
    H = get_history()
    out = {}
    for mode in ("all", "past"):
        rows = []
        for r in val_ids:
            route = load_npz(r)
            names, _ = zone_membership(route)
            c = H.cutoff(str(route["date"]) if mode == "past" else None)
            shared, _, allowed = H.reference(str(route["station"]), set(names), r, c)
            rows.append((allowed, shared / len(names)))
        train_rows = []
        for r in H.by_id.values():                       # train routes, as used in training
            c = H.cutoff(r["date"] if mode == "past" else None)
            shared, _, allowed = H.reference(r["station"], set(r["zones"]), r["route_id"], c)
            train_rows.append((allowed, shared / max(len(r["zones"]), 1)))
        def describe(rs):
            a = np.array([x[0] for x in rs])
            sh = np.array([x[1] for x in rs])
            return {"routes": len(rs), "no_same_station_history": int((a == 0).sum()),
                    "under_10_same_station_routes": int((a < 10).sum()),
                    "same_station_routes_median": float(np.median(a)),
                    "reference_shares_under_half_of_zones": int((sh < 0.5).sum()),
                    "reference_zone_share_median": float(np.median(sh))}
        out[mode] = {"val": describe(rows), "train": describe(train_rows)}
    return out


def cached(name, fn):
    """Re-use a finished variant from runs/v0/cache/ if the run is restarted."""
    path = RUN_DIR / "cache" / (name.replace("/", "__") + ".json")
    if path.exists():
        return json.load(open(path))
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(path, "w"))
    return res


def run_decode(args):
    t_start = time.time()
    splits = json.load(open(SPLITS))
    val_ids = sorted(r["route_id"] for r in splits["val"])
    pilot = {r["route_id"] for r in splits["pilot"]}
    assert not pilot & set(val_ids)
    _, held_ids, quality = split_train(splits)
    tune_ids = sorted(r for r in held_ids if quality[r] == "High")
    if args.limit:
        val_ids, tune_ids = val_ids[:args.limit], tune_ids[:args.limit]
    assert not (set(val_ids) | set(tune_ids)) & pilot    # never score or tune on pilot
    invalid = json.load(open(INVALID))
    shortest = json.load(open(SHORTEST_VAL))["per_route"]
    shortest_scores = {r: shortest[r]["score"] for r in val_ids}
    tags = args.tags.split(",")
    lambdas = [float(x) for x in args.lambdas.split(",")]
    seed = 20261111

    out = {"tags": {}, "controls": {}, "tuning": {}}
    orders_dump = {}
    with mp.Pool(args.workers) as pool:
        # --- 1. tune the travel-time weight on held-out train routes (never val) -----------
        for tag in tags:
            t0 = time.time()
            grid = {}
            for lam in lambdas:
                res = cached(f"tune_{tag}_beam{args.beam}_lam{lam:g}_n{len(tune_ids)}",
                             lambda: run_variant(pool, tune_ids, "model", tag, args.beam, lam, invalid))
                grid[lam] = float(np.mean([res["scores"][r] for r in tune_ids]))
                print(f"[tune {tag}] lambda {lam}: held-out mean {grid[lam]:.4f}", flush=True)
            best = min(grid, key=grid.get)
            out["tuning"][tag] = {"routes": len(tune_ids), "beam": args.beam,
                                  "grid_mean_score": {str(k): v for k, v in grid.items()},
                                  "chosen_lambda": best, "seconds": time.time() - t0}

        # --- 2. val: every variant, scored once -------------------------------------------
        variants = []
        for tag in tags:
            lam = out["tuning"][tag]["chosen_lambda"]
            variants += [(f"{tag}/greedy", "model", tag, 1, 0.0),
                         (f"{tag}/beam{args.beam}", "model", tag, args.beam, 0.0),
                         (f"{tag}/beam{args.beam}+travel(lambda={lam:g})", "model", tag, args.beam, lam)]
        variants += [("control/nearest_zone", "nearest_zone", None, 0, 0.0),
                     ("oracle/driver_zone_order", "driver_zones", None, 0, 0.0)]
        val_results = {}
        for name, method, tag, beam, lam in variants:
            t0 = time.time()
            res = cached(f"val_{name}_n{len(val_ids)}",
                         lambda: run_variant(pool, val_ids, method, tag, beam, lam, invalid))
            val_results[name] = res
            entry = {"method": method, "model": tag, "family": family(tag) if tag else None,
                     "beam": beam, "lambda": lam,
                     **summarize(res["scores"], val_ids, seed),
                     "all_feasible": all(res["feasible"].values()),
                     "vs_shortest": paired(res["scores"], shortest_scores, val_ids, seed),
                     "per_route_seconds": timing_summary(res["timing"]),
                     "wall_seconds": time.time() - t0}
            (out["controls"] if method != "model" else out["tags"].setdefault(tag, {}))[name] = entry
            orders_dump[name] = {"zone_orders": res["zone_orders"], "stop_orders": res["orders"]}
            print(f"[val] {name}: {entry['mean']:.4f} {entry['bootstrap_95_ci']}  vs shortest "
                  f"{entry['vs_shortest']['mean_diff']:+.4f} ({entry['vs_shortest']['better']}/"
                  f"{entry['vs_shortest']['worse']}, p={entry['vs_shortest']['sign_test_p']:.3g})", flush=True)

    # --- 3. next-zone accuracy on val (teacher-forced) and the worked example -------------
    import torch
    torch.set_num_threads(3)
    val_data = {}                                        # one copy per feature set
    accuracy = {}
    for tag in tags:
        model, mode = load_model(tag)
        if mode not in val_data:
            val_data[mode] = build_examples(val_ids, log_every=0, mode=mode,
                                            history=get_history() if mode != "none" else None)
        accuracy[tag] = teacher_forced_accuracy(model, val_data[mode])
        print(f"[val] {tag} next-zone top-1 {accuracy[tag]['top1']:.3f} "
              f"(nearest-zone rule {accuracy[tag]['nearest_zone_top1']:.3f})", flush=True)

    # The worked example uses, per family, the model picked on held-out loss (train data only).
    # The route is the val route whose score under the v0 (no history) model's beam decoding is
    # the median: neither flattering nor unflattering. All families are shown on that route.
    summaries = {t: json.load(open(RUN_DIR / t / "summary.json")) for t in tags}
    fams = {}
    for t in tags:
        fams.setdefault(family(t), []).append(t)
    best_of = {f: min(ts, key=lambda t: summaries[t]["best_heldout_loss"]) for f, ts in fams.items()}
    main_tag = best_of["v0"]
    main_name = f"{main_tag}/beam{args.beam}"
    ranked = sorted(val_ids, key=lambda r: val_results[main_name]["scores"][r])
    example_id = ranked[len(ranked) // 2]
    example = worked_example(main_tag, example_id)
    example["scores"] = {name: res["scores"][example_id] for name, res in val_results.items()}
    example["scores"]["shortest"] = shortest_scores[example_id]
    example["model"] = main_tag
    example["decoded_zone_order_beam"] = val_results[main_name]["zone_orders"][example_id]
    example["other_families"] = {
        f: {"model": t, "steps": worked_example(t, example_id)["steps"],
            "decoded_zone_order_beam": val_results[f"{t}/beam{args.beam}"]["zone_orders"][example_id]}
        for f, t in best_of.items() if f != "v0"}

    # Families against each other, paired route by route: same training data, same decoding.
    def find(tag, kind):
        return next(n for n in val_results if n.startswith(f"{tag}/") and
                    (n.split("/")[1].startswith(f"beam{args.beam}+") if kind == "travel"
                     else n == f"{tag}/{kind}"))
    family_pairs = {}
    for quality_tag in [t for t in tags if family(t) == "v0"]:
        for a_tag, b_tag in ((f"hist_{quality_tag}", quality_tag), (f"past_{quality_tag}", quality_tag),
                             (f"past_{quality_tag}", f"hist_{quality_tag}")):
            if a_tag not in tags or b_tag not in tags:
                continue
            for kind in ("greedy", f"beam{args.beam}", "travel"):
                a, b = find(a_tag, kind), find(b_tag, kind)
                family_pairs[f"{a} minus {b}"] = paired(val_results[a]["scores"],
                                                        val_results[b]["scores"], val_ids, seed)
    coverage = history_coverage(val_ids) if any(family(t) != "v0" for t in tags) else {}

    write_curve_svg(summaries, RUN_DIR / "training_curve.svg")
    shortest_summary = summarize(shortest_scores, val_ids, seed)
    results = {
        "description": "v0 tiny zone model + OR-Tools inside zones, scored on the 300 val routes",
        "split": "val (300 High routes); pilot never used; travel weight tuned on held-out train",
        "metric": "official rc-cli score, lower is better, 0 = driver's exact order",
        "models": {t: {k: v for k, v in summaries[t].items() if k != "curve"} for t in tags},
        "main_model_for_example": main_tag,
        "best_model_per_family_by_heldout_loss": best_of,
        "shortest_val": shortest_summary,
        "variants": out["tags"],
        "controls": out["controls"],
        "tuning_on_heldout_train": out["tuning"],
        "next_zone_accuracy_val": accuracy,
        "family_comparisons_paired": family_pairs,
        "history_coverage": coverage,
        "training_cost": {t: {k: summaries[t].get(k) for k in (
            "feature_seconds", "train_seconds", "train_cpu_seconds",
            "total_cpu_seconds_features_plus_training", "total_wall_seconds_features_plus_training",
            "epochs_run", "peak_rss_mb")} for t in tags},
        "timing_note": "per-route CPU seconds of one worker process (1 torch thread): zone_order = "
                       "features + history lookup + model + beam search; solver = OR-Tools inside "
                       "zones. Scoring is not included. Training times are one-off, 3 torch threads.",
        "worked_example": example,
        "settings": {
            "in_zone_solver": "OR-Tools routing, start at previous stop, end at a dummy node whose "
                              "cost is the travel time to the nearest stop of the next zone (station "
                              "after the last zone); PATH_CHEAPEST_ARC + GREEDY_DESCENT",
            "zone_order_target": "first visit to each zone in the driver's order",
            "no_zone_stops": "attached to the zone of the nearest stop (travel time both ways)",
            "beam": args.beam, "lambda_grid": lambdas, "workers": args.workers,
            "bootstrap": "10000 percentile resamples of routes, seed 20261111",
            "sign_test": "exact two-sided binomial, ties dropped",
        },
        "run": {"wall_seconds": time.time() - t_start,
                "peak_rss_mb_main": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "peak_rss_mb_largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024},
    }
    if not args.limit:
        with open(RESULTS / "v0.json.part", "w") as f:
            json.dump(results, f, indent=1)
        os.replace(RESULTS / "v0.json.part", RESULTS / "v0.json")
        with open(RUN_DIR / "val_orders.json", "w") as f:
            json.dump(orders_dump, f)
    else:
        json.dump(results, open(RUN_DIR / "decode_test.json", "w"), indent=1)
    print(json.dumps(results["worked_example"], indent=1))
    print(f"done in {time.time() - t_start:.0f}s")


# ----------------------------------------------------------------------------------------------
# Job 3: does the decoded zone order run in the driver's direction?
# ----------------------------------------------------------------------------------------------

def kendall_tau(ours, driver):
    """+1 = same zone order, -1 = exactly reversed, about 0 = unrelated. Counts every pair of
    zones: +1 if we put them in the driver's order, -1 if the other way round, then averages."""
    pos = {z: i for i, z in enumerate(driver)}
    x = [pos[z] for z in ours]
    n = len(x)
    if n < 2:
        return 1.0
    agree = sum(1 if x[i] < x[j] else -1 for i in range(n) for j in range(i + 1, n))
    return agree / (n * (n - 1) / 2)


def run_direction(args):
    """The official score barely punishes a route driven backwards (the exact reverse scores 0),
    so we check direction separately, on the zone orders saved by the decode job."""
    orders = json.load(open(RUN_DIR / "val_orders.json"))
    val_ids = sorted(json.load(open(SHORTEST_VAL))["per_route"])
    driver = {}
    for r in val_ids:
        route = load_npz(r)
        P = prepare_route(route)
        driver[r] = [P["names"][z] for z in driver_zone_order(P, route)]
    out = {}
    for name, o in orders.items():
        t = np.array([kendall_tau(o["zone_orders"][r], driver[r]) for r in val_ids])
        out[name] = {"kendall_tau_mean": float(t.mean()), "kendall_tau_median": float(np.median(t)),
                     "routes_mostly_backwards_tau_below_0": int((t < 0).sum()),
                     "exact_zone_order": int(sum(o["zone_orders"][r] == driver[r] for r in val_ids)),
                     "exact_reverse_zone_order": int(sum(o["zone_orders"][r] == driver[r][::-1]
                                                         for r in val_ids))}
        print(f"{name:45s} tau {t.mean():+.3f}  backwards {out[name]['routes_mostly_backwards_tau_below_0']:3d}"
              f"  exact {out[name]['exact_zone_order']:3d}  reversed {out[name]['exact_reverse_zone_order']:3d}")
    results = json.load(open(RESULTS / "v0.json"))
    results["zone_order_direction_val"] = out
    with open(RESULTS / "v0.json.part", "w") as f:
        json.dump(results, f, indent=1)
    os.replace(RESULTS / "v0.json.part", RESULTS / "v0.json")


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="job", required=True)
    s = sub.add_parser("shortest")
    s.add_argument("--time-limit", type=int, default=10)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--limit", type=int, default=0, help="only the first N val routes (testing)")
    d = sub.add_parser("decode")
    d.add_argument("--tags", default="high,high_medium,hist_high,hist_high_medium,past_high,past_high_medium",
                   help="trained models under runs/v0/ (hist_* = v0-hist, past_* = v0-hist-past)")
    d.add_argument("--beam", type=int, default=5)
    d.add_argument("--lambdas", default="0,0.5,1,2", help="travel-time weights to try")
    d.add_argument("--workers", type=int, default=3)
    d.add_argument("--limit", type=int, default=0, help="only N routes per split (testing)")
    sub.add_parser("direction")
    args = ap.parse_args()
    if args.job == "direction":
        run_direction(args)
    elif args.job == "shortest":
        run_shortest(args)
    else:
        run_decode(args)


if __name__ == "__main__":
    sys.exit(main())
