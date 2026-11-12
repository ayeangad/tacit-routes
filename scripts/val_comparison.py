#!/usr/bin/env python3
"""Paired, per-route comparison on the 300 val routes: v0 (tiny model) against Opus (fair-fight
runs 1 and 2), jpt-amz (2021 winner), the plain shortest route and, if it exists, the rules arm.

  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/val_comparison.py --workers 3

Every arm is re-scored here from its saved stop orders with v0_decode.official_scores (the
official, unmodified rc-cli evaluate(), travel times rounded to 0.1 s, chunks of 20 routes),
so all arms share one scoring path. The re-scored per-route values are checked against each
arm's own results file. Rules (results/rules_baseline.json) stores no orders, so its stored
per-route scores are used (same 0.1 s rounding, official score()).

Selection rule (fixed before looking at val): per family, the training-data choice with the
lower held-out-train loss (v0.json best_model_per_family_by_heldout_loss) and decoding beam 5 +
travel at the lambda tuned on held-out train routes. Best-on-val variants are reported too,
labelled as optimistically selected.

Writes results/val_comparison.json; per-arm score caches in runs/val_comparison/.
Never reads pilot routes' answers or data/eval_frozen/.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v0_decode as vd                                   # noqa: E402
from v0_model import zone_membership                     # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "runs" / "val_comparison"
SEED = 20261111
FAMILY_OF = {"high": "v0", "high_medium": "v0", "hist_high": "v0-hist", "hist_high_medium": "v0-hist",
             "past_high": "v0-hist-past", "past_high_medium": "v0-hist-past"}


def load(p):
    with open(p) as f:
        return json.load(f)


def check_seq(seq, stop_ids):
    if not isinstance(seq, list) or not all(isinstance(x, str) for x in seq):
        return "not a list of strings"
    bad = []
    if set(seq) != set(stop_ids):
        bad.append("wrong stop set")
    if len(seq) != len(set(seq)):
        bad.append("duplicates")
    if not seq or seq[0] != stop_ids[0]:
        bad.append("does not start at station")
    return "; ".join(bad) or None


def holm(pvals):
    """Holm step-down adjusted p-values, same order as the input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired(a, b, ids, name_a, name_b):
    d = np.array([a[r] - b[r] for r in ids])
    first, second = int((d < 0).sum()), int((d > 0).sum())
    return {"first": name_a, "second": name_b, "n_routes": len(ids),
            "mean_first": float(np.mean([a[r] for r in ids])), "mean_second": float(np.mean([b[r] for r in ids])),
            "mean_difference": float(d.mean()), "median_difference": float(np.median(d)),
            "bootstrap_95_ci": vd.bootstrap_ci(d, SEED),
            "first_better": first, "second_better": second, "ties": int((d == 0).sum()),
            "sign_test_p_two_sided": vd.sign_test(first, second)}


def zone_order_from_stops(route, order):
    """First visit to each zone along a stop order (station dropped), as zone names; stops with
    no zone borrow the nearest stop's zone, exactly as v0's driver_zone_order does."""
    names, zone_of = zone_membership(route)
    pos = {str(s): i for i, s in enumerate(route["stop_ids"])}
    seen, out = set(), []
    for s in order[1:]:
        k = int(zone_of[pos[s]])
        if k not in seen:
            seen.add(k)
            out.append(names[k])
    return out


def direction(zone_orders, driver, ids):
    t = np.array([vd.kendall_tau(zone_orders[r], driver[r]) for r in ids])
    return {"kendall_tau_mean": float(t.mean()), "kendall_tau_median": float(np.median(t)),
            "routes_mostly_backwards_tau_below_0": int((t < 0).sum()),
            "exact_zone_order": int(sum(zone_orders[r] == driver[r] for r in ids)),
            "exact_reverse_zone_order": int(sum(zone_orders[r] == driver[r][::-1] for r in ids))}, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = load(ROOT / "data" / "processed" / "splits.json")
    val_ids = sorted(r["route_id"] for r in splits["val"])
    pilot = {r["route_id"] for r in splits["pilot"]}
    assert len(val_ids) == 300 and not pilot & set(val_ids)
    invalid = load(vd.INVALID)
    invalid = {r: invalid[r] for r in val_ids}

    v0 = load(RESULTS / "v0.json")
    val_orders = load(ROOT / "runs" / "v0" / "val_orders.json")

    # ---- 1. selection -------------------------------------------------------------------------
    chosen_tag = v0["best_model_per_family_by_heldout_loss"]
    prereg, best_on_val = {}, {}
    for fam, tag in chosen_tag.items():
        lam = v0["tuning_on_heldout_train"][tag]["chosen_lambda"]
        prereg[fam] = f"{tag}/beam5+travel(lambda={lam:g})"
        assert prereg[fam] in v0["variants"][tag], prereg[fam]
        cands = {n: e["mean"] for t, vs in v0["variants"].items() if FAMILY_OF[t] == fam for n, e in vs.items()}
        best_on_val[fam] = min(cands, key=cands.get)
    selection = {
        "rule": "fixed before val: per family, the training-data choice with the lower held-out-train loss "
                "(v0.json best_model_per_family_by_heldout_loss) and decoding beam 5 + travel at the lambda "
                "tuned on 234 held-out High train routes (v0.json tuning_on_heldout_train)",
        "pre_registered": prereg,
        "heldout_train_loss": {fam: v0["models"][t]["best_heldout_loss"] for fam, t in chosen_tag.items()},
        "tuned_lambda": {fam: v0["tuning_on_heldout_train"][t]["chosen_lambda"] for fam, t in chosen_tag.items()},
        "best_on_val_OPTIMISTIC": best_on_val,
        "best_on_val_note": "lowest val mean among the 6 variants of each family (2 training sets x greedy/beam5/"
                            "beam5+travel); chosen by looking at val, so optimistically biased",
        "variants_scored_on_val_in_v0_json": sum(len(v) for v in v0["variants"].values()),
    }

    # ---- 2. stop orders of every arm ----------------------------------------------------------
    routes = {r: vd.load_npz(r) for r in val_ids}
    stop_ids = {r: [str(s) for s in routes[r]["stop_ids"]] for r in val_ids}
    arms = {}
    for name in sorted(set(prereg.values()) | set(best_on_val.values())):
        arms[f"v0:{name}"] = {r: val_orders[name]["stop_orders"][r] for r in val_ids}
    op1 = load(ROOT / "runs" / "opus_fight" / "val_predictions_opus_copy.json")
    op2 = load(ROOT / "runs" / "opus_fight_run2" / "val_predictions_opus_copy.json")
    arms["opus_run1"] = {r: op1[r] for r in val_ids}
    arms["opus_run2"] = {r: op2[r] for r in val_ids}
    jp = load(ROOT / "runs" / "jpt" / "out_t1_1" / "proposed_sequences.json")
    arms["jpt"] = {r: sorted(jp[r]["proposed"], key=jp[r]["proposed"].get) for r in val_ids}
    sv = load(vd.SHORTEST_VAL)
    arms["shortest"] = {r: sv["per_route"][r]["order"] for r in val_ids}
    # Rules arms (results/rules_baseline.json stores each arm's stop order per route).
    rb_path = RESULTS / "rules_baseline.json"
    rules_arm_names = []
    if rb_path.exists():
        rb = load(rb_path)
        rb_per = {p["route_id"]: p for p in rb["per_route"]}
        assert sorted(rb_per) == val_ids
        rules_arm_names = list(rb_per[val_ids[0]]["arms"])           # S, R1, R2, R3, R4
        for arm in rules_arm_names:
            arms[f"rules:{arm}"] = {r: rb_per[r]["arms"][arm]["order"] for r in val_ids}
    validity = {a: {r: check_seq(o[r], stop_ids[r]) for r in val_ids} for a, o in arms.items()}
    bad = {a: {r: v for r, v in vs.items() if v} for a, vs in validity.items()}
    assert not any(bad.values()), {a: len(v) for a, v in bad.items()}   # every arm: 300/300 valid

    # ---- 3. score every arm the same way -------------------------------------------------------
    scores = {}
    with mp.Pool(args.workers) as pool:
        for a, orders in arms.items():
            cache = OUT_DIR / ("scores_" + a.replace("/", "__").replace(":", "_") + ".json")
            if cache.exists():
                scores[a] = load(cache)
            else:
                t0 = time.time()
                s, feas = vd.official_scores(orders, invalid, pool)
                assert all(feas.values())
                scores[a] = s
                json.dump(s, open(cache, "w"))
                print(f"scored {a} in {time.time() - t0:.0f}s: mean {np.mean(list(s.values())):.5f}", flush=True)

    # ---- 4. reproduction checks against each arm's own results file ---------------------------
    def maxdiff(a, b):
        return float(max(abs(a[r] - b[r]) for r in val_ids))
    checks = {}
    for a in arms:
        if a.startswith("v0:"):
            name = a[3:]
            tag = name.split("/")[0]
            stored_mean = v0["variants"][tag][name]["mean"]
            cache = load(ROOT / "runs" / "v0" / "cache" / f"val_{name.replace('/', '__')}_n300.json")
            checks[a] = {"recomputed_mean": float(np.mean([scores[a][r] for r in val_ids])),
                         "v0_json_mean": stored_mean,
                         "abs_mean_diff": abs(float(np.mean([scores[a][r] for r in val_ids])) - stored_mean),
                         "max_abs_per_route_diff_vs_runs_v0_cache": maxdiff(scores[a], cache["scores"])}
    for a, path, key in (("opus_run1", "opus_fight.json", "opus_score"),
                         ("opus_run2", "opus_fight_run2.json", "opus_score"),
                         ("jpt", "jpt_baseline.json", "jpt_t1_1_score")):
        per = {p["route_id"]: p[key] for p in load(RESULTS / path)["per_route"]}
        assert sorted(per) == val_ids
        checks[a] = {"recomputed_mean": float(np.mean([scores[a][r] for r in val_ids])),
                     "stored_mean": float(np.mean([per[r] for r in val_ids])),
                     "max_abs_per_route_diff": maxdiff(scores[a], per), "source": f"results/{path}"}
    stored_short = {r: sv["per_route"][r]["score"] for r in val_ids}
    checks["shortest"] = {"recomputed_mean": float(np.mean(list(scores["shortest"].values()))),
                          "stored_mean": float(np.mean(list(stored_short.values()))),
                          "max_abs_per_route_diff": maxdiff(scores["shortest"], stored_short),
                          "source": "results/v0_shortest_val.json"}
    for arm in rules_arm_names:
        a = f"rules:{arm}"
        stored = {r: float(rb_per[r]["arms"][arm]["score"]) for r in val_ids}
        checks[a] = {"recomputed_mean": float(np.mean([scores[a][r] for r in val_ids])),
                     "stored_mean": float(np.mean(list(stored.values()))),
                     "max_abs_per_route_diff": maxdiff(scores[a], stored), "source": "results/rules_baseline.json"}
    for a, c in checks.items():
        worst = max(v for k, v in c.items() if k.startswith("max_abs") or k == "abs_mean_diff")
        c["reproduced_within_1e-9"] = bool(worst < 1e-9)
    print(json.dumps(checks, indent=1), flush=True)

    rules_status, rules_arms = "pending: results/rules_baseline.json does not exist yet", {}
    s_vs_shortest = None
    if rules_arm_names:
        rules_arms = {f"rules:{a}": True for a in rules_arm_names}
        rules_status = (f"included: arms {rules_arm_names} re-scored here from the stop orders stored in "
                        f"results/rules_baseline.json, and checked against its stored per-route scores")
        # rules S is its own OR-Tools run (same settings, 10 s time limit), not the saved v0 shortest tours.
        same_tour = sum(arms["rules:S"][r] == arms["shortest"][r] for r in val_ids)
        d = [scores["rules:S"][r] - scores["shortest"][r] for r in val_ids]
        s_vs_shortest = {
            "identical_tours": int(same_tour),
            "per_route_score_equal_where_tour_identical": bool(all(
                scores["rules:S"][r] == scores["shortest"][r] for r in val_ids
                if arms["rules:S"][r] == arms["shortest"][r])),
            "mean_rules_S": float(np.mean([scores["rules:S"][r] for r in val_ids])),
            "mean_v0_shortest": float(np.mean([scores["shortest"][r] for r in val_ids])),
            "max_abs_per_route_diff": float(max(abs(x) for x in d)),
            "note": "two separate time-limited (10 s) OR-Tools GLS runs, so tours differ on some routes; the "
                    "scoring convention is the same (rules' stored scores reproduce exactly when re-scored here)"}

    # ---- 5. paired comparisons ----------------------------------------------------------------
    P = {fam: f"v0:{n}" for fam, n in prereg.items()}
    B = {fam: f"v0:{n}" for fam, n in best_on_val.items()}
    main_family = [(P["v0-hist"], "opus_run2"), (P["v0-hist-past"], "opus_run2"), (P["v0"], "opus_run2")]
    secondary = [(P[f], "opus_run1") for f in ("v0-hist", "v0-hist-past", "v0")]
    secondary += [(P["v0-hist"], "jpt"), (P["v0-hist-past"], "jpt"), ("opus_run2", "jpt"),
                  ("opus_run2", "opus_run1")]
    secondary += [(P[f], "shortest") for f in ("v0-hist", "v0-hist-past", "v0")]
    secondary += [("opus_run2", "shortest"), ("jpt", "shortest")]
    optimistic = [(B[f], "opus_run2") for f in ("v0-hist", "v0-hist-past", "v0") if B[f] != P[f]]
    rules_pairs = []
    if rules_arms:
        for ra in ("rules:R2", "rules:R4", "rules:R3"):
            rules_pairs += [(P["v0-hist"], ra), (P["v0-hist-past"], ra), (P["v0"], ra), ("opus_run2", ra)]
        rules_pairs += [("jpt", "rules:R2")]

    def run(pairs):
        return [paired(scores[a], scores[b], val_ids, a, b) for a, b in pairs]
    main_res = run(main_family)
    for res, adj in zip(main_res, holm([x["sign_test_p_two_sided"] for x in main_res])):
        res["holm_adjusted_sign_p"] = adj
    comparisons = {"main_family_vs_opus_run2_holm": main_res, "secondary_unadjusted": run(secondary),
                   "best_on_val_OPTIMISTIC_vs_opus_run2": run(optimistic), "vs_rules": run(rules_pairs)}

    arm_summary = {}
    for a, s in scores.items():
        v = [s[r] for r in val_ids]
        arm_summary[a] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                          "bootstrap_95_ci": vd.bootstrap_ci(v, SEED)}

    # ---- 6. direction: zone order derived from each arm's stop order ---------------------------
    from v0_model import prepare_route, driver_zone_order
    driver = {}
    for r in val_ids:
        Pr = prepare_route(routes[r])
        driver[r] = [Pr["names"][z] for z in driver_zone_order(Pr, routes[r])]
    derived = {a: {r: zone_order_from_stops(routes[r], arms[a][r]) for r in val_ids} for a in arms}
    sanity = {a: int(sum(derived[a][r] == val_orders[a[3:]]["zone_orders"][r] for r in val_ids))
              for a in arms if a.startswith("v0:")}
    dir_out, taus = {}, {}
    for a in arms:
        dir_out[a], taus[a] = direction(derived[a], driver, val_ids)
    tau_paired = {}
    for a in (P["v0-hist"], P["v0-hist-past"], P["v0"]):
        d = taus[a] - taus["opus_run2"]
        tau_paired[f"{a} minus opus_run2"] = {
            "mean_tau_difference": float(d.mean()), "bootstrap_95_ci": vd.bootstrap_ci(d, SEED),
            "first_higher": int((d > 0).sum()), "second_higher": int((d < 0).sum()), "ties": int((d == 0).sum()),
            "sign_test_p_two_sided": vd.sign_test(int((d > 0).sum()), int((d < 0).sum()))}
    direction_out = {
        "method": "zone order = first visit to each zone along the arm's stop order (zone-less stops take the "
                  "nearest stop's zone, as v0); Kendall tau vs the driver's zone order (v0_decode.kendall_tau)",
        "sanity_check_v0_derived_equals_stored_zone_order_routes": sanity,
        "per_arm": dir_out, "paired_tau_vs_opus_run2": tau_paired}

    # ---- 7. cost and latency ------------------------------------------------------------------
    jb = load(RESULTS / "jpt_baseline.json")["runs"]["t1_1"]
    cost = {
        "v0_per_route_cpu_seconds_one_worker": {
            P[f]: v0["variants"][prereg[f].split("/")[0]][prereg[f]]["per_route_seconds"] for f in prereg},
        "v0_training_one_off": {chosen_tag[f]: v0["training_cost"][chosen_tag[f]] for f in chosen_tag},
        "v0_money": "$0 (local CPU)",
        "opus_run2": {"cost_usd_cli_list_price": 1.37, "session_wall_seconds": 4298,
                      "session_wall": "71 min 38 s (22:48:40 -> 00:00:18 UTC)", "cli_duration_ms": 4296544,
                      "api_seconds": 328, "session_cpu": "2 h 31 min total",
                      "output_tokens": 34580, "thinking_tokens": 10653,
                      "predict_py_runtime": "about 2.5 min wall on 3 processes for 300 routes (Opus's own "
                                            "APPROACH.md / final message; not timed by us) = about 0.5 s wall, "
                                            "roughly 1.5 CPU-s per route",
                      "source": "results/opus_fight_run2.md"},
        "opus_run1": {"cost_usd": None, "note": "no cost field (run killed before summary)",
                      "session_wall": "66 min 57 s", "predict_py_runtime": "12.6 min wall on 3 processes",
                      "source": "results/opus_fight.md"},
        "jpt": {"apply_cpu_seconds_per_route": jb["apply_cpu_seconds_per_route"],
                "apply_wall_seconds_300_routes_3_threads": jb["apply_wall_seconds"],
                "build_wall_seconds": load(RESULTS / "jpt_baseline.json")["build"]["wall_seconds"],
                "money": "$0 (local CPU)", "source": "results/jpt_baseline.json"},
        "shortest": {"per_route": "10 s OR-Tools search per route (3 workers)"},
    }

    hist = load(ROOT / "runs" / "v0" / "history_train.json")
    out = {
        "description": "Paired per-route comparison on the 300 val routes: v0 vs Opus fair-fight runs, jpt-amz, "
                       "shortest route (and rules if available). All arms re-scored with one scoring path.",
        "split": "val, 300 High routes; pilot never used; data/eval_frozen never touched",
        "metric": "official rc-cli score, lower is better; differences are first minus second (negative = first "
                  "better)",
        "scoring": "v0_decode.official_scores: official unmodified evaluate(), travel times rounded to 0.1 s "
                   "(raw data precision), chunks of 20 routes; same for every re-scored arm",
        "selection": selection,
        "arms": arm_summary,
        "all_arms_300_of_300_valid": True,
        "reproduction_checks": checks,
        "rules_status": rules_status,
        "rules_S_vs_v0_shortest": s_vs_shortest,
        "comparisons": comparisons,
        "direction": direction_out,
        "cost_latency": cost,
        "size": {"v0_hist_parameters": 28481, "v0_parameters": 27329,
                 "history_table": {"train_routes": len(hist["routes"]),
                                   "route_zone_entries": sum(len(r["zones"]) for r in hist["routes"]),
                                   "json_bytes": os.path.getsize(ROOT / "runs" / "v0" / "history_train.json"),
                                   "note": "v0-hist and v0-hist-past look up counts from this table at predict "
                                           "time; it is data the model carries, beyond its 28,481 weights"}},
        "settings": {"bootstrap": f"10000 percentile resamples of routes on paired differences, seed {SEED}",
                     "sign_test": "exact two-sided binomial, ties dropped",
                     "holm": "Holm step-down over the 3 main comparisons vs Opus run 2 (sign-test p)"},
        "per_route": [{"route_id": r, **{a: scores[a][r] for a in scores},
                       **{f"tau:{a}": float(taus[a][i]) for a in taus}} for i, r in enumerate(val_ids)],
        "run_seconds": time.time() - t_start,
    }
    tmp = RESULTS / "val_comparison.json.part"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, RESULTS / "val_comparison.json")
    print(json.dumps({k: out[k] for k in ("selection", "arms", "comparisons", "direction", "rules_status")},
                     indent=1))


if __name__ == "__main__":
    sys.exit(main())
