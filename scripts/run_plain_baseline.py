#!/usr/bin/env python3
"""Score a plain shortest route and a random order against real driver routes.

  .venv/bin/python -I scripts/run_plain_baseline.py \
      --sample-dir data/raw/sample --results-dir results

Scoring is the unmodified official rc-cli score.py (scripts/official_rc_cli/).
"""
import argparse
import copy
import importlib.util
import json
import multiprocessing as mp
import os
import random
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

HERE = Path(__file__).resolve().parent
SCORER = HERE / "official_rc_cli" / "score.py"


def load_scorer():
    spec = importlib.util.spec_from_file_location("official_score", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path):
    with open(path) as f:
        return json.load(f)


def solve_shortest(job):
    """Closed shortest tour from the station. The scorer adds the return leg, so we do too."""
    route_id, station, matrix, time_limit = job
    stops = [station] + sorted(s for s in matrix if s != station)
    # Times have one decimal place; OR-Tools needs integers, so work in tenths of a second.
    costs = [[int(round(matrix[a][b] * 10)) for b in stops] for a in stops]
    manager = pywrapcp.RoutingIndexManager(len(stops), 1, 0)
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
    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order.append(stops[manager.IndexToNode(index)])
        index = solution.Value(routing.NextVar(index))
    return route_id, order, time.time() - t0


def as_proposal(order):
    return {"proposed": {stop: position for position, stop in enumerate(order)}}


def tour_seconds(order, matrix):
    return sum(matrix[a][b] for a, b in zip(order, order[1:] + order[:1]))


def bootstrap_ci(values, seed, resamples=10000):
    """Percentile 95% CI of the mean, resampling routes with replacement."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(scores, seed):
    return {"mean": float(np.mean(scores)), "median": float(np.median(scores)),
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "bootstrap_95_ci": bootstrap_ci(scores, seed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--time-limit", type=int, default=10, help="seconds per route")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    official = load_scorer()
    manifest = load_json(args.sample_dir / "sample_manifest.json")
    routes = load_json(args.sample_dir / "sample_route_data.json")
    actual = load_json(args.sample_dir / "sample_actual.json")
    travel = load_json(args.sample_dir / "sample_travel_times.json")
    invalid = load_json(args.sample_dir / "sample_invalid.json")
    route_ids = sorted(actual)

    station = {r: next(s for s, v in routes[r]["stops"].items() if v["type"] == "Station")
               for r in route_ids}
    driver_order = {r: sorted(actual[r]["actual"], key=actual[r]["actual"].get) for r in route_ids}

    jobs = [(r, station[r], travel[r], args.time_limit) for r in route_ids]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        solved = pool.map(solve_shortest, jobs, chunksize=1)
    solve_wall = time.time() - t0
    failed = [r for r, order, _ in solved if order is None]
    if failed:
        raise SystemExit(f"solver found no tour for {failed}")
    shortest = {r: order for r, order, _ in solved}
    solve_seconds = {r: t for r, _, t in solved}

    random_order = {}
    for r in route_ids:
        rest = sorted(s for s in travel[r] if s != station[r])
        random.Random(f"{args.seed}:{r}").shuffle(rest)
        random_order[r] = [station[r]] + rest

    # Score through the official evaluate(), reading the same files it would in the challenge.
    submissions = {"shortest": shortest, "random": random_order}
    output = {}
    for name, orders in submissions.items():
        path = args.results_dir / f"proposed_{name}.json"
        with open(path, "w") as f:
            json.dump({r: as_proposal(orders[r]) for r in route_ids}, f)
        t0 = time.time()
        output[name] = official.evaluate(
            str(args.sample_dir / "sample_actual.json"), str(path),
            str(args.sample_dir / "sample_travel_times.json"),
            str(args.sample_dir / "sample_invalid.json"))
        print(f"{name}: {output[name]['submission_score']:.4f} in {time.time() - t0:.0f}s", flush=True)

    # The driver's own route must score exactly 0 if the pipeline is wired correctly.
    identity = {r: driver_order[r] for r in route_ids[:3]}
    for r, order in identity.items():
        check = official.score(order + [order[0]], order + [order[0]], copy.deepcopy(travel[r]))
        if check != 0:
            raise SystemExit(f"{r}: driver route scored {check}, expected 0")

    per_route = []
    for r in route_ids:
        per_route.append({
            "route_id": r,
            "station_code": routes[r]["station_code"],
            "n_stops": len(travel[r]),
            "shortest_score": float(output["shortest"]["route_scores"][r]),
            "random_score": float(output["random"]["route_scores"][r]),
            "official_invalid_score": invalid[r],
            "shortest_feasible": bool(output["shortest"]["route_feasibility"][r]),
            "random_feasible": bool(output["random"]["route_feasibility"][r]),
            "driver_tour_seconds": tour_seconds(driver_order[r], travel[r]),
            "shortest_tour_seconds": tour_seconds(shortest[r], travel[r]),
            "random_tour_seconds": tour_seconds(random_order[r], travel[r]),
            "solve_seconds": solve_seconds[r],
        })

    shortest_scores = [p["shortest_score"] for p in per_route]
    random_scores = [p["random_score"] for p in per_route]
    ratios = [p["shortest_tour_seconds"] / p["driver_tour_seconds"] for p in per_route]

    # Worked example: the route with the median shortest score, so it is neither a
    # flattering nor an unflattering pick.
    ranked = sorted(per_route, key=lambda p: p["shortest_score"])
    example = ranked[len(ranked) // 2]
    r = example["route_id"]
    norm = official.normalize_matrix(copy.deepcopy(travel[r]))
    actual_closed = driver_order[r] + [station[r]]
    shortest_closed = shortest[r] + [station[r]]
    sd = official.seq_dev(actual_closed, shortest_closed)
    erp = official.erp_per_edit(actual_closed, shortest_closed, norm)
    zone = {s: v["zone_id"] for s, v in routes[r]["stops"].items()}
    worked = {
        "route_id": r,
        "selection": "route with the median shortest_score in the sample (rank n//2)",
        "station_code": routes[r]["station_code"],
        "station_stop": station[r],
        "n_stops_including_station": len(travel[r]),
        "driver_first_10_after_station": driver_order[r][1:11],
        "driver_first_10_zones": [zone[s] for s in driver_order[r][1:11]],
        "shortest_first_10_after_station": shortest[r][1:11],
        "shortest_first_10_zones": [zone[s] for s in shortest[r][1:11]],
        "sequence_deviation": float(sd),
        "erp_per_edit": float(erp),
        "score": example["shortest_score"],
        "score_recomputed_as_sd_times_erp_per_edit": float(sd * erp),
        "random_score": example["random_score"],
        "driver_tour_seconds": example["driver_tour_seconds"],
        "shortest_tour_seconds": example["shortest_tour_seconds"],
    }

    source_commit = (HERE / "official_rc_cli" / "SOURCE_COMMIT.txt").read_text().strip()
    results = {
        "description": "Plain shortest route vs random order, scored against real driver routes",
        "sample_size": len(route_ids),
        "seed": args.seed,
        "metric": {
            "name": "Amazon Last Mile Routing Research Challenge score",
            "formula": "sequence deviation * (normalized ERP / ERP edits), averaged over routes",
            "direction": "lower is better; 0 is a perfect match",
            "scorer": "official, unmodified",
            "scorer_source": f"https://github.com/MIT-CAVE/rc-cli/blob/{source_commit}/scoring/score.py",
            "numpy_note": f"score.py pins numpy 1.20.1, which does not install on Python 3.12; ran on numpy {np.__version__}",
        },
        "settings": {
            "data_side": manifest["side"],
            "quality_filter": manifest["quality_filter"],
            "quality_label_counts_all_training_routes": manifest["quality_label_counts_all_training_routes"],
            "sampling": manifest["sampling"],
            "stations": manifest["stations"],
            "stops_min_mean_max": manifest["stops_min_mean_max"],
            "solver": "OR-Tools routing, one vehicle, closed tour from the station",
            "ortools_first_solution": "PATH_CHEAPEST_ARC",
            "ortools_metaheuristic": "GUIDED_LOCAL_SEARCH",
            "time_limit_seconds_per_route": args.time_limit,
            "workers": args.workers,
            "cost_units": "supplied travel times, rounded to 0.1 s for the integer solver",
            "random_control": "uniform shuffle of non-station stops, random.Random(f'{seed}:{route_id}')",
            "bootstrap": "10000 percentile resamples of routes",
        },
        "summary": {
            "shortest": summarize(shortest_scores, args.seed),
            "random": summarize(random_scores, args.seed),
            "official_mean_invalid_score_for_these_routes": float(np.mean([p["official_invalid_score"] for p in per_route])),
            "random_minus_shortest_paired_95_ci": bootstrap_ci(
                [a - b for a, b in zip(random_scores, shortest_scores)], args.seed),
            "shortest_tour_time_over_driver_tour_time": {
                "mean": float(np.mean(ratios)), "median": float(statistics.median(ratios)),
                "routes_where_solver_tour_was_longer_than_driver": sum(1 for x in ratios if x > 1)},
            "all_routes_feasible": all(p["shortest_feasible"] and p["random_feasible"] for p in per_route),
        },
        "comparison_to_published": {
            "wu_et_al_arxiv_2205_04001": 0.0374,
            "caveat": "0.0374 is on the 3,052-route official evaluation set; this sample is 50 High "
                      "routes from the training side, so the numbers are not directly comparable.",
        },
        "worked_example": worked,
        "per_route": per_route,
        "run": {
            "solve_wall_seconds": solve_wall,
            "peak_rss_mb_main_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_rss_mb_largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
        },
    }
    with open(args.results_dir / "plain_baseline.json.part", "w") as f:
        json.dump(results, f, indent=1)
    os.replace(args.results_dir / "plain_baseline.json.part", args.results_dir / "plain_baseline.json")
    print(json.dumps({**results["summary"], "run": results["run"]}, indent=1))
    print(json.dumps(worked, indent=1))


if __name__ == "__main__":
    sys.exit(main())
