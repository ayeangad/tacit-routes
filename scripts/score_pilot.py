#!/usr/bin/env python3
"""Score the direct-ordering pilot with the official, unmodified rc-cli scorer.

  .venv/bin/python -I scripts/score_pilot.py --sample-dir data/raw/sample \
      --pilot-dir runs/pilot --results-dir results

Scoring follows run_plain_baseline.py: proposals are written in the challenge format and
passed to the official evaluate(). An invalid answer (missing, extra or duplicate stops,
or not starting at the depot) is left out of the proposal file, so evaluate() gives it the
challenge's invalid-sequence score, exactly as it does for any route without a valid
submission. Only routes the arm was actually run on are reported.
"""
import argparse
import copy
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


official = load_module("official_score", HERE / "official_rc_cli" / "score.py")
baseline = load_module("run_plain_baseline", HERE / "run_plain_baseline.py")
bootstrap_ci, summarize, as_proposal = baseline.bootstrap_ci, baseline.summarize, baseline.as_proposal


def read_jsonl(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("status") == "ok":
                rows[rec["route_id"]] = rec
    return rows


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


def diagnose(text, driver_order, matrix):
    """Descriptive only, never used for scoring. Stop IDs as they appear in the raw output,
    where the first repeat occurs, and the rank correlation of the answer's order with the
    driver's (the official metric scores an exact reverse as 0, so direction matters)."""
    ids = [t for t in re.findall(r'"([A-Z]{2})"', text) if t in matrix]
    seen, first_repeat = set(), None
    for i, t in enumerate(ids):
        if t in seen:
            first_repeat = i
            break
        seen.add(t)
    uniq = list(dict.fromkeys(ids))
    pos = {s: i for i, s in enumerate(driver_order)}
    rho = None
    if len(uniq) > 2:
        a = np.argsort(np.argsort([pos[t] for t in uniq]))
        rho = float(np.corrcoef(a, np.arange(len(uniq)))[0, 1])
    return {"ids_in_output": len(ids), "distinct_ids_in_output": len(uniq),
            "first_repeat_at_index": first_repeat,
            "spearman_vs_driver_order_distinct_ids": rho}


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, type=Path)
    ap.add_argument("--pilot-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    sd = args.sample_dir

    manifest = json.loads((sd / "sample_manifest.json").read_text())
    routes = json.loads((sd / "sample_route_data.json").read_text())
    actual = json.loads((sd / "sample_actual.json").read_text())
    travel = json.loads((sd / "sample_travel_times.json").read_text())
    shortest_props = json.loads((args.results_dir / "proposed_shortest.json").read_text())
    plain = json.loads((args.results_dir / "plain_baseline.json").read_text())
    prompts = json.loads((args.pilot_dir / "prompts_index.json").read_text())
    order_of = lambda d: sorted(d, key=d.get)
    driver = {r: order_of(actual[r]["actual"]) for r in actual}
    shortest = {r: order_of(shortest_props[r]["proposed"]) for r in shortest_props}
    zone = {r: {s: (v["zone_id"] or "none") for s, v in routes[r]["stops"].items()} for r in routes}

    def official_eval(name, orders):
        path = args.pilot_dir / f"proposed_{name}.json"
        path.write_text(json.dumps({r: as_proposal(o) for r, o in orders.items()}))
        out = official.evaluate(str(sd / "sample_actual.json"), str(path),
                                str(sd / "sample_travel_times.json"), str(sd / "sample_invalid.json"))
        return out

    # Re-score the saved shortest tours with the same call, so baseline numbers match exactly.
    short_eval = official_eval("shortest_rescored", shortest)
    old = {p["route_id"]: p["shortest_score"] for p in plain["per_route"]}
    assert all(abs(float(short_eval["route_scores"][r]) - old[r]) < 1e-12 for r in old), "baseline drift"

    arms = {}
    for arm in ("deepseek", "opus"):
        recs = read_jsonl(args.pilot_dir / f"{arm}.jsonl")
        ran = [r for r in manifest["route_ids"] if r in recs]
        valid_orders = {r: recs[r]["parsed_sequence"] for r in ran if recs[r]["valid"]}
        ev = official_eval(arm, valid_orders)
        per = []
        for r in ran:
            rec = recs[r]
            feasible = bool(ev["route_feasibility"][r])
            if feasible != rec["valid"]:
                raise SystemExit(f"{arm} {r}: our validity {rec['valid']} != official {feasible}")
            row = {"route_id": r, "n_stops": len(travel[r]), "valid": rec["valid"],
                   "invalid_reason": rec["invalid_reason"],
                   "score": float(ev["route_scores"][r]),
                   "shortest_score": float(short_eval["route_scores"][r]),
                   "prompt_chars": rec["prompt_chars"], "output_chars": rec["output_chars"],
                   "wall_seconds": rec["wall_seconds"]}
            if arm == "deepseek":
                u = rec["usage"]
                row.update(input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"),
                           cost_usd_list_price=rec["cost_usd_list_price"],
                           model_returned=rec["model_returned"], finish_reason=rec["finish_reason"])
            else:
                u = rec["usage"] or {}
                row.update(input_tokens=sum(u.get(k) or 0 for k in (
                               "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")),
                           uncached_input_tokens=u.get("input_tokens"),
                           thinking_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens"),
                           cache_creation_input_tokens=u.get("cache_creation_input_tokens"),
                           cache_read_input_tokens=u.get("cache_read_input_tokens"),
                           output_tokens=u.get("output_tokens"),
                           total_cost_usd=rec["total_cost_usd"], duration_ms=rec["duration_ms"],
                           duration_api_ms=rec["duration_api_ms"], model_returned=rec["model_returned"],
                           stop_reason=rec.get("stop_reason"))
            row.update(diagnose(rec["raw_output"], driver[r], travel[r]))
            row["shortest_spearman_vs_driver_order"] = diagnose(
                json.dumps(shortest[r]), driver[r], travel[r])["spearman_vs_driver_order_distinct_ids"]
            per.append(row)
        if not per:
            continue
        scores = [p["score"] for p in per]
        base = [p["shortest_score"] for p in per]
        reasons = {}
        for p in per:
            if not p["valid"]:
                reasons[p["invalid_reason"]] = reasons.get(p["invalid_reason"], 0) + 1
        summary = {
            "n_routes": len(per),
            "score": summarize(scores, args.seed),
            "valid_only_mean_score": mean_or_none([p["score"] for p in per if p["valid"]]),
            "shortest_on_same_routes": summarize(base, args.seed),
            "arm_minus_shortest_paired": paired(scores, base, args.seed),
            "valid_count": sum(p["valid"] for p in per),
            "invalid_count": sum(not p["valid"] for p in per),
            "invalid_reasons": reasons,
            "mean_prompt_chars": float(np.mean([p["prompt_chars"] for p in per])),
            "mean_input_tokens": mean_or_none([p["input_tokens"] for p in per]),
            "mean_output_chars": float(np.mean([p["output_chars"] for p in per])),
            "mean_output_tokens": mean_or_none([p["output_tokens"] for p in per]),
            "mean_wall_seconds": float(np.mean([p["wall_seconds"] for p in per])),
            "mean_spearman_vs_driver_order": mean_or_none(
                [p["spearman_vs_driver_order_distinct_ids"] for p in per]),
            "model_returned": sorted({json.dumps(p["model_returned"]) for p in per}),
        }
        if arm == "deepseek":
            summary["spend_usd_list_price"] = float(sum(p["cost_usd_list_price"] for p in per))
        else:
            if len(per) < 20:
                summary["warning"] = f"n={len(per)}: far too small for conclusions"
            summary["total_cost_usd_field_sum"] = float(sum(p["total_cost_usd"] or 0 for p in per))
            summary["totals"] = {k: float(sum(p[k] or 0 for p in per)) for k in (
                "input_tokens", "uncached_input_tokens", "cache_creation_input_tokens",
                "cache_read_input_tokens", "output_tokens", "thinking_tokens", "duration_ms",
                "duration_api_ms", "wall_seconds")}
            summary["mean_duration_ms"] = float(np.mean([p["duration_ms"] for p in per]))
            summary["mean_thinking_tokens"] = mean_or_none([p["thinking_tokens"] for p in per])
            # Earlier pilot: the first 5 routes in manifest order, kept for the record.
            first5 = [p for p in per if p["route_id"] in manifest["route_ids"][:5]]
            if first5:
                summary["first_5_routes_earlier_pilot"] = {
                    "route_ids": [p["route_id"] for p in first5],
                    "score": summarize([p["score"] for p in first5], args.seed),
                    "shortest_on_same_routes": summarize([p["shortest_score"] for p in first5], args.seed),
                    "arm_minus_shortest_paired": paired([p["score"] for p in first5],
                                                        [p["shortest_score"] for p in first5], args.seed)}
            # Descriptive split by stop count at the median over the routes Opus ran.
            med = float(np.median([p["n_stops"] for p in per]))
            groups = {}
            for name, keep in ((f"n_stops <= {med:g}", lambda n: n <= med),
                               (f"n_stops > {med:g}", lambda n: n > med)):
                g = [p for p in per if keep(p["n_stops"])]
                if g:
                    groups[name] = {
                        "n_routes": len(g),
                        "stops_min_max": [min(p["n_stops"] for p in g), max(p["n_stops"] for p in g)],
                        "mean_score": float(np.mean([p["score"] for p in g])),
                        "mean_shortest": float(np.mean([p["shortest_score"] for p in g])),
                        "valid_count": sum(p["valid"] for p in g),
                        "arm_minus_shortest_paired": paired([p["score"] for p in g],
                                                            [p["shortest_score"] for p in g], args.seed)}
            summary["by_stop_count_descriptive"] = {"median_n_stops": med, "groups": groups}
        arms[arm] = {"summary": summary, "per_route": per,
                     "orders": {r: recs[r]["parsed_sequence"] for r in ran}}

    comparisons = {}
    if "deepseek" in arms and "opus" in arms:
        both = [p["route_id"] for p in arms["opus"]["per_route"]]
        ds = {p["route_id"]: p["score"] for p in arms["deepseek"]["per_route"]}
        op = {p["route_id"]: p["score"] for p in arms["opus"]["per_route"]}
        both = [r for r in both if r in ds]
        comparisons["opus_minus_deepseek_on_opus_routes"] = {
            "n_routes": len(both), **paired([op[r] for r in both], [ds[r] for r in both], args.seed)}
        comparisons["deepseek_on_opus_routes"] = summarize([ds[r] for r in both], args.seed)

    # Worked route: the plain_baseline.md median route if Opus ran it, else the Opus route
    # with the median shortest score among Opus's routes.
    worked = None
    if "opus" in arms:
        opus_routes = [p["route_id"] for p in arms["opus"]["per_route"]]
        target = plain["worked_example"]["route_id"]
        if target in opus_routes:
            r, why = target, "same median route as plain_baseline.md"
        else:
            ranked = sorted(arms["opus"]["per_route"], key=lambda p: p["shortest_score"])
            r = ranked[len(ranked) // 2]["route_id"]
            why = (f"plain_baseline.md's route {target} is not among Opus's routes; this is the Opus "
                   f"route with the median shortest score among them")
        rows = {"driver": driver[r], "shortest": shortest[r]}
        for arm in ("deepseek", "opus"):
            rows[arm] = arms[arm]["orders"].get(r) if arm in arms else None
        score_of = {"driver": 0.0, "shortest": float(short_eval["route_scores"][r])}
        for arm in ("deepseek", "opus"):
            p = next((p for p in arms.get(arm, {}).get("per_route", []) if p["route_id"] == r), None)
            score_of[arm] = p["score"] if p else None
        valid_of = {arm: next((p["valid"] for p in arms[arm]["per_route"] if p["route_id"] == r), None)
                    for arm in arms}
        first10 = {}
        for name, seq in rows.items():
            if seq:
                ids = [s for s in seq[1:11]]
                first10[name] = [{"stop": s, "zone": zone[r].get(s, "?") if isinstance(s, str) else "?"}
                                 for s in ids]
        worked = {"route_id": r, "selection": why, "station_code": routes[r]["station_code"],
                  "depot": driver[r][0], "n_stops_including_depot": len(travel[r]),
                  "scores": score_of, "valid": valid_of, "first_10_after_depot": first10}

    results = {
        "description": "Direct stop ordering by DeepSeek and Opus, scored with the official rc-cli scorer",
        "metric": plain["metric"],
        "invalid_handling": "invalid answers are omitted from the proposal file, so official evaluate() "
                            "assigns the challenge's invalid-sequence score",
        "prompts": {k: prompts[k] for k in ("seed", "listing_order", "neighbours", "system_prompt_chars",
                                            "total_prompt_chars", "mean_prompt_chars")},
        "arms": {a: {"summary": v["summary"], "per_route": v["per_route"]} for a, v in arms.items()},
        "comparisons": comparisons,
        "worked_example": worked,
        "baseline_rescore_check": "saved shortest tours re-scored identically to plain_baseline.json",
    }
    (args.results_dir / "pilot.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({a: v["summary"] for a, v in arms.items()}, indent=1))
    print(json.dumps(comparisons, indent=1))
    print(json.dumps(worked, indent=1))


if __name__ == "__main__":
    sys.exit(main())
