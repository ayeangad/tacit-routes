#!/usr/bin/env python3
"""Score jpt-amz's proposals on the 300 val routes with the unmodified official scorer.

Run from tacit-routes/ with -I:
  .venv/bin/python -I scripts/score_jpt_baseline.py --run-dir runs/jpt --tags t1_1 \
      --primary t1_1 --results-dir results

Writes results/jpt_baseline.json. Paired comparisons use the same 300 routes:
  - shortest route: per-route scores from results/rules_baseline.json if it has them, else the
    orders stored in results/v0_shortest_val.json (OR-Tools, GLS, 10 s, as run_plain_baseline.py),
    re-scored here with the official scorer;
  - best v0 variant / best rules arm, if results/v0.json / results/rules_baseline.json exist.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORER = HERE / "official_rc_cli" / "score.py"
SEED = 20261111


def load_scorer():
    spec = importlib.util.spec_from_file_location("official_score", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_ci(values, seed=SEED, resamples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(scores):
    return {"n": len(scores), "mean": float(np.mean(scores)), "median": float(np.median(scores)),
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "bootstrap_95_ci": bootstrap_ci(scores)}


def sign_test(wins, losses):
    """Exact two-sided binomial sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a, b, ids, name_a, name_b):
    """a minus b per route; negative means a is closer to the driver (lower is better)."""
    d = [a[r] - b[r] for r in ids]
    wins = sum(1 for x in d if x < 0)
    losses = sum(1 for x in d if x > 0)
    return {"difference": f"{name_a} minus {name_b} (negative = {name_a} better)", "n": len(ids),
            "mean_difference": float(np.mean(d)), "median_difference": float(np.median(d)),
            "bootstrap_95_ci": bootstrap_ci(d),
            f"{name_a}_better": wins, f"{name_a}_worse": losses, "tied": len(d) - wins - losses,
            "sign_test_p_two_sided": sign_test(wins, losses)}


def write_json(obj, path):
    tmp = path.with_suffix(".part")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def per_route_from(obj):
    """Find a {route_id: score} mapping inside a results file, whatever its exact layout."""
    pr = obj.get("per_route")
    if isinstance(pr, dict):
        return {r: (v["score"] if isinstance(v, dict) else v) for r, v in pr.items()}
    if isinstance(pr, list):
        key = next(k for k in pr[0] if k.endswith("score") and k != "official_invalid_score")
        return {p["route_id"]: p[key] for p in pr}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    a = ap.parse_args()
    official = load_scorer()

    data = a.run_dir / "data"
    actual_p = data / "model_score_inputs" / "new_actual_sequences.json"
    invalid_p = data / "model_score_inputs" / "new_invalid_sequence_scores.json"
    tt_p = data / "model_apply_inputs" / "new_travel_times.json"
    actual = json.load(open(actual_p))
    travel = json.load(open(tt_p))
    route_meta = json.load(open(data / "model_apply_inputs" / "new_route_data.json"))
    splits = json.load(open(a.splits))
    val_ids = sorted(r["route_id"] for r in splits["val"])
    assert sorted(actual) == val_ids and sorted(travel) == val_ids
    ids = val_ids

    # Wiring check: the driver's own route must score exactly 0.
    for r in ids[:3]:
        order = sorted(actual[r]["actual"], key=actual[r]["actual"].get)
        check = official.score(order + [order[0]], order + [order[0]], copy.deepcopy(travel[r]))
        if check != 0:
            raise SystemExit(f"{r}: driver route scored {check}, expected 0")

    runs = {}
    for tag in a.tags:
        prop_p = a.run_dir / f"out_{tag}" / "proposed_sequences.json"
        out = official.evaluate(str(actual_p), str(prop_p), str(tt_p), str(invalid_p))
        scores = {r: float(out["route_scores"][r]) for r in ids}
        feas = {r: bool(out["route_feasibility"][r]) for r in ids}
        timing = json.load(open(a.run_dir / "logs" / f"apply_{tag}.json"))
        # LKH per-instance CPU: two passes per route (two constraint sets).
        lkh = timing["lkh_cpu_seconds_per_instance"]
        runs[tag] = {"scores": scores, "feasible": feas, "timing": timing, "proposals": str(prop_p),
                     "lkh_instances": len(lkh), "lkh_cpu_total": float(sum(lkh))}
        print(f"{tag}: mean {np.mean(list(scores.values())):.5f}, valid {sum(feas.values())}/{len(ids)}",
              flush=True)

    # Shortest route on the same 300 routes.
    rules_p, v0_p = a.results_dir / "rules_baseline.json", a.results_dir / "v0.json"
    shortest, shortest_source = None, None
    if rules_p.exists():
        rb = json.load(open(rules_p))
        for key in ("S", "shortest", "S_shortest"):
            arm = rb.get("arms", {}).get(key) if isinstance(rb.get("arms"), dict) else None
            if arm and per_route_from(arm):
                shortest, shortest_source = per_route_from(arm), f"results/rules_baseline.json arm {key}"
                break
    if shortest is None:
        sv = json.load(open(a.results_dir / "v0_shortest_val.json"))
        tmp = a.run_dir / "score_tmp"
        tmp.mkdir(exist_ok=True)
        prop = {r: {"proposed": {s: i for i, s in enumerate(sv["per_route"][r]["order"])}} for r in ids}
        write_json(prop, tmp / "shortest_val_proposed.json")
        out = official.evaluate(str(actual_p), str(tmp / "shortest_val_proposed.json"), str(tt_p), str(invalid_p))
        shortest = {r: float(out["route_scores"][r]) for r in ids}
        stored = {r: sv["per_route"][r]["score"] for r in ids}
        max_abs = max(abs(shortest[r] - stored[r]) for r in ids)
        shortest_source = ("orders from results/v0_shortest_val.json (OR-Tools GLS, 10 s/route, as "
                           f"run_plain_baseline.py), re-scored here; max |diff| vs stored scores = {max_abs:.2e}")

    primary = runs[a.primary]["scores"]
    comparisons = {"jpt_minus_shortest": paired(primary, shortest, ids, "jpt", "shortest")}
    others = {}
    for path, label in ((v0_p, "best_v0_variant"), (rules_p, "best_rules_arm")):
        if not path.exists():
            comparisons[f"jpt_minus_{label}"] = f"not run: {path.name} did not exist when this was scored"
            continue
        obj = json.load(open(path))
        cands = {}
        for name, arm in (obj.get("arms") or obj.get("variants") or {}).items():
            pr = per_route_from(arm) if isinstance(arm, dict) else None
            if pr and set(ids) <= set(pr):
                cands[name] = pr
        if not cands:
            comparisons[f"jpt_minus_{label}"] = f"not run: no per-route val scores found in {path.name}"
            continue
        best = min(cands, key=lambda n: np.mean([cands[n][r] for r in ids]))
        others[label] = best
        comparisons[f"jpt_minus_{label}"] = {"other": f"{path.name}: {best}",
                                             **paired(primary, cands[best], ids, "jpt", label)}
    for tag in a.tags:
        if tag != a.primary:
            comparisons[f"{tag}_minus_{a.primary}"] = paired(runs[tag]["scores"], primary, ids, tag, a.primary)

    # Per-route CPU: sum of LKH's own CPU time over both passes, matched by instance order is not
    # needed; we report the distribution per instance and the per-route mean.
    run_summ = {}
    for tag, rr in runs.items():
        t = rr["timing"]
        lkh = t["lkh_cpu_seconds_per_instance"]
        run_summ[tag] = {
            "time_limits_s": t["time_limits_s"],
            "score": summarize(list(rr["scores"].values())),
            "valid_routes": sum(rr["feasible"].values()),
            "apply_wall_seconds": t["wall_seconds"],
            "apply_cpu_seconds_total": t["cpu_seconds_user"] + t["cpu_seconds_sys"],
            "apply_cpu_seconds_per_route": (t["cpu_seconds_user"] + t["cpu_seconds_sys"]) / len(ids),
            "lkh_instances": len(lkh),
            "lkh_cpu_seconds_per_instance": {"mean": float(np.mean(lkh)), "median": float(np.median(lkh)),
                                             "max": float(np.max(lkh))} if lkh else None,
            "lkh_cpu_seconds_per_route_both_passes_mean": float(np.sum(lkh)) / len(ids) if lkh else None,
            "peak_tree_rss_mb_sampled": t["peak_tree_rss_mb_sampled"],
        }
    build = json.load(open(a.run_dir / "logs" / "build.json"))

    stops = {r: len(travel[r]) for r in ids}
    res = {
        "description": "2021 Amazon Last Mile winner (jpt-amz: Cook, Held, Helsgaun) run locally on the 300 val routes",
        "code": {"repo": "https://github.com/heldstephan/jpt-amz",
                 "commit": (a.run_dir / "JPT_COMMIT.txt").read_text().strip(),
                 "licence": "MIT",
                 "local_changes": (a.run_dir / "LOCAL_CHANGES.txt").read_text()},
        "data": {"build": "5,762 train routes (route_data, actual_sequences, package_data, travel_times)",
                 "apply": "300 val routes (route_data without route_score, package_data, travel_times); no sequences",
                 "pilot_routes_used": 0, "eval_frozen_touched": False,
                 "stops_min_mean_max": [min(stops.values()), float(np.mean(list(stops.values()))), max(stops.values())]},
        "metric": {"scorer": "official rc-cli score.py, unmodified (scripts/official_rc_cli)",
                   "direction": "lower is better; 0 = driver's exact order"},
        "primary_tag": a.primary,
        "runs": run_summ,
        "build": {k: build[k] for k in ("wall_seconds", "cpu_seconds_user", "cpu_seconds_sys",
                                         "peak_tree_rss_mb_sampled", "exit_code")},
        "shortest_on_val": {"source": shortest_source, "score": summarize([shortest[r] for r in ids])},
        "comparisons": comparisons,
        "published_reference": {
            "evaluation_side": {"jpt_official_hidden_evaluation_score": 0.0248, "second_place": 0.0353,
                                "third_place": 0.0391,
                                "source": "team page https://www.math.uwaterloo.ca/tsp/amz/index.html (confirmed)",
                                "note": "hidden evaluation set, ~3,000 routes, contest time limits; not our val set"},
            "training_side": {"best_mean": 0.01978, "best_median": 0.00732, "routes": "1,107 High, fully delivered training routes",
                              "time_limits": "1 s + 1 s per route",
                              "source": "Cook, Held, Helsgaun arXiv 2112.15192 "
                                        "(from a summary, not read verbatim here)",
                              "note": "their training routes are also their history pool's neighbours; our val routes "
                                      "are held out from the build data"},
        },
        "bootstrap": "10000 percentile resamples of routes, seed 20261111",
        "per_route": [{"route_id": r, "station_code": route_meta[r]["station_code"], "n_stops": stops[r],
                       "shortest_score": shortest[r],
                       **{f"jpt_{t}_score": runs[t]["scores"][r] for t in a.tags},
                       **{f"jpt_{t}_feasible": runs[t]["feasible"][r] for t in a.tags}} for r in ids],
    }
    with open(a.results_dir / "jpt_baseline.json.part", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(a.results_dir / "jpt_baseline.json.part", a.results_dir / "jpt_baseline.json")
    print(json.dumps({"runs": {t: {k: v for k, v in s.items()} for t, s in run_summ.items()},
                      "shortest": res["shortest_on_val"], "comparisons": comparisons}, indent=1))


if __name__ == "__main__":
    main()
