#!/usr/bin/env python3
"""Score the Opus fair-fight predictions on the 300 validation routes.

  .venv/bin/python -I scripts/score_opus_fight.py

Reads opus_arena/val_predictions.json (written by Opus's predict.py) and the true val driver
orders from data/processed/routes/ (kept outside the arena). Scoring is the official,
unmodified rc-cli evaluate(); an invalid prediction (not a list of every stop exactly once,
starting at the station) is left out of the proposal file, so evaluate() gives it the
challenge's invalid-sequence score, as in score_pilot.py.

Plain shortest route on val: reuses per-route shortest scores from results/rules_baseline.json
if it holds them for all 300 val routes; otherwise solves with run_plain_baseline.py's
solve_shortest (OR-Tools, PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH, 10 s per route, 3 workers).

Writes runs/opus_fight/scoring/* (official input files, per-route scores) and
results/opus_fight.json.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "scripts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official = load_module("official_score", HERE / "official_rc_cli" / "score.py")
baseline = load_module("run_plain_baseline", HERE / "run_plain_baseline.py")
bootstrap_ci, summarize, as_proposal = baseline.bootstrap_ci, baseline.summarize, baseline.as_proposal


def sign_test_p(wins, losses):
    """Exact two-sided binomial sign test, ties dropped, p = 0.5."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def paired(a, b, seed):
    d = [x - y for x, y in zip(a, b)]
    first, second = sum(1 for x in d if x < 0), sum(1 for x in d if x > 0)
    return {"n_routes": len(d), "mean_difference": float(np.mean(d)),
            "median_difference": float(np.median(d)), "bootstrap_95_ci": bootstrap_ci(d, seed),
            "routes_first_better": first, "routes_second_better": second,
            "ties": sum(1 for x in d if x == 0),
            "exact_sign_test_two_sided_p": sign_test_p(first, second)}


def check(seq, stop_ids):
    if not isinstance(seq, list) or not all(isinstance(x, str) for x in seq):
        return "not a list of strings"
    reasons = []
    if set(seq) - set(stop_ids):
        reasons.append("unknown IDs")
    if set(stop_ids) - set(seq):
        reasons.append(f"missing {len(set(stop_ids) - set(seq))} stops")
    if len(seq) != len(set(seq)):
        reasons.append("duplicates")
    if not seq or seq[0] != stop_ids[0]:
        reasons.append("does not start at station")
    return "; ".join(reasons) or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=ROOT / "opus_arena" / "val_predictions.json")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--tag", default="opus_fight", help="output names: runs/<tag>/scoring, results/<tag>.json")
    ap.add_argument("--time-limit", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    splits = json.loads((ROOT / "data" / "processed" / "splits.json").read_text())
    val = [x["route_id"] for x in splits["val"]]
    pilot = {x["route_id"] for x in splits["pilot"]}
    assert not pilot & set(val)
    invalid_all = json.loads((ROOT / "data" / "raw" / "invalid_sequence_scores.json").read_text())

    out = ROOT / "runs" / args.tag / "scoring"
    out.mkdir(parents=True, exist_ok=True)
    actual, costs, invalid, stops, meta = {}, {}, {}, {}, {}
    for r in val:
        with np.load(ROOT / "data" / "processed" / "routes" / f"{r}.npz", allow_pickle=False) as z:
            ids = [str(s) for s in z["stop_ids"]]
            order = [ids[i] for i in z["actual_order"]]
            tt = z["travel_time"]
            meta[r] = {"station": str(z["station"]), "n_stops": len(ids)}
        stops[r] = ids
        actual[r] = {"actual": {s: k for k, s in enumerate(order)}}
        # npz stores float32; the raw data has one decimal place, so round to 0.1 s (as v0_decode.cost_dict)
        costs[r] = {a: {b: round(float(tt[i, j]), 1) for j, b in enumerate(ids)} for i, a in enumerate(ids)}
        invalid[r] = invalid_all[r]
    for name, obj in (("val_actual.json", actual), ("val_costs.json", costs), ("val_invalid.json", invalid)):
        (out / name).write_text(json.dumps(obj))

    pred_raw = json.loads(args.predictions.read_text())
    proposal, validity = {}, {}
    for r in val:
        reason = "route missing from predictions" if r not in pred_raw else check(pred_raw[r], stops[r])
        validity[r] = reason
        if reason is None:
            proposal[r] = as_proposal(pred_raw[r])
    (out / "opus_proposal.json").write_text(json.dumps(proposal))
    t0 = time.time()
    res = official.evaluate(str(out / "val_actual.json"), str(out / "opus_proposal.json"),
                            str(out / "val_costs.json"), str(out / "val_invalid.json"))
    print(f"opus scored in {time.time() - t0:.0f}s: {res['submission_score']:.4f}", flush=True)
    opus = {r: float(res["route_scores"][r]) for r in val}
    for r in val:
        assert bool(res["route_feasibility"][r]) == (validity[r] is None), r

    # Plain shortest route on the same 300 routes.
    shortest_source, shortest = None, None
    rb = ROOT / "results" / "rules_baseline.json"
    if rb.exists():
        try:
            data = json.loads(rb.read_text())
            per = {p["route_id"]: p for p in data.get("per_route", [])}
            if all(r in per and "shortest_score" in per[r] for r in val):
                shortest = {r: float(per[r]["shortest_score"]) for r in val}
                shortest_source = "results/rules_baseline.json per_route shortest_score"
        except (json.JSONDecodeError, KeyError, TypeError):
            shortest = None
    v0 = ROOT / "results" / "v0_shortest_val.json"
    if shortest is None and v0.exists():
        data = json.loads(v0.read_text())
        st = data.get("settings", {})
        per = data.get("per_route", {})
        if (st.get("first_solution") == "PATH_CHEAPEST_ARC" and st.get("metaheuristic") == "GUIDED_LOCAL_SEARCH"
                and st.get("time_limit_seconds_per_route") == args.time_limit and all(r in per for r in val)):
            # Re-score the saved tours with the official scorer and require an exact match.
            (out / "shortest_proposal.json").write_text(json.dumps({r: as_proposal(per[r]["order"]) for r in val}))
            sres = official.evaluate(str(out / "val_actual.json"), str(out / "shortest_proposal.json"),
                                     str(out / "val_costs.json"), str(out / "val_invalid.json"))
            mism = [r for r in val if abs(float(sres["route_scores"][r]) - float(per[r]["score"])) > 1e-12]
            if mism:
                raise SystemExit(f"re-scored shortest tours differ from v0_shortest_val.json on {len(mism)} routes")
            shortest = {r: float(sres["route_scores"][r]) for r in val}
            shortest_source = ("results/v0_shortest_val.json (run_plain_baseline settings: OR-Tools PATH_CHEAPEST_ARC + "
                               "GUIDED_LOCAL_SEARCH, 10 s per route, 3 workers); saved tours re-scored here with the "
                               "official scorer, all 300 per-route scores matched exactly")
    if shortest is None:
        jobs = [(r, stops[r][0], costs[r], args.time_limit) for r in val]
        t0 = time.time()
        with mp.Pool(args.workers) as pool:
            solved = pool.map(baseline.solve_shortest, jobs, chunksize=1)
        print(f"shortest solved in {time.time() - t0:.0f}s", flush=True)
        tours = {r: o for r, o, _ in solved}
        assert all(tours[r] is not None for r in val)
        (out / "shortest_proposal.json").write_text(json.dumps({r: as_proposal(tours[r]) for r in val}))
        sres = official.evaluate(str(out / "val_actual.json"), str(out / "shortest_proposal.json"),
                                 str(out / "val_costs.json"), str(out / "val_invalid.json"))
        shortest = {r: float(sres["route_scores"][r]) for r in val}
        shortest_source = (f"computed here with run_plain_baseline.solve_shortest (OR-Tools PATH_CHEAPEST_ARC + "
                           f"GUIDED_LOCAL_SEARCH, {args.time_limit} s per route, {args.workers} workers)")

    o = [opus[r] for r in val]
    s = [shortest[r] for r in val]
    valid_ids = [r for r in val if validity[r] is None]
    median_stops = float(np.median([meta[r]["n_stops"] for r in val]))
    by_size = {}
    for label, keep in (("le_median", lambda n: n <= median_stops), ("gt_median", lambda n: n > median_stops)):
        g = [r for r in val if keep(meta[r]["n_stops"])]
        by_size[label] = {"n_routes": len(g), "opus": summarize([opus[r] for r in g], args.seed),
                          "shortest": summarize([shortest[r] for r in g], args.seed),
                          "opus_minus_shortest": paired([opus[r] for r in g], [shortest[r] for r in g], args.seed)}
    result = {
        "description": "Opus 5.5 fair fight: Opus built its own predictor in a sandbox; scored on the 300 val routes",
        "n_routes": len(val),
        "metric": "official rc-cli score, lower is better, 0 = driver's exact order; invalid routes get the "
                  "challenge's invalid-sequence score",
        "predictions_file": str(args.predictions.resolve().relative_to(ROOT)),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "predictions_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(args.predictions.stat().st_mtime)),
        "opus": summarize(o, args.seed),
        "valid_predictions": len(valid_ids),
        "invalid_reasons": {r: validity[r] for r in val if validity[r] is not None},
        "shortest": summarize(s, args.seed),
        "shortest_source": shortest_source,
        "opus_minus_shortest": paired(o, s, args.seed),
        "by_stop_count_descriptive": {"median_stops_including_station": median_stops, **by_size},
        "bootstrap": "10000 percentile resamples of routes, seed " + str(args.seed),
        "per_route": [{"route_id": r, "station": meta[r]["station"], "n_stops": meta[r]["n_stops"],
                       "opus_score": opus[r], "shortest_score": shortest[r], "opus_valid": validity[r] is None}
                      for r in val],
    }
    path = ROOT / "results" / f"{args.tag}.json"
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, path)
    print(json.dumps({k: result[k] for k in ("opus", "valid_predictions", "shortest", "opus_minus_shortest")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
