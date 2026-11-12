#!/usr/bin/env python3
"""Score the zone-level "swap the brain" arm with the official, unmodified rc-cli scorer.

  .venv/bin/python -I scripts/score_pilot_zone.py

Writes ONLY results/pilot_zone.json and results/pilot_zone.md (refuses any other results
stem). Proposal files for official evaluate() go to runs/pilot_zone/proposed_*.json.

Pipeline per route:
  1. Opus's zone list (runs/pilot_zone/opus.jsonl, re-parsed from raw_output and checked
     against the logged parse).
  2. Zone order -> full stop order with v0's own inside-zone solver, imported unchanged:
     v0_decode.stops_from_zone_order(route, zone_id_strings), called exactly as v0 calls it
     (time_limit_s left at its default None = OR-Tools PATH_CHEAPEST_ARC + GREEDY_DESCENT in
     each zone, entering from the previous zone's last stop, ending near the next zone).
  3. Two versions:
       strict:   an invalid zone list (missing, duplicate or unknown zones, or no list) is
                 left out of the proposal file, so evaluate() gives the challenge's invalid
                 score, exactly as score_pilot.py treats an invalid stop list;
       fallback: keep the valid zones in Opus's order (first occurrence; unknown IDs dropped),
                 then insert each missing zone by cheapest insertion on v0's zone travel
                 times (station -> zones -> station), repeatedly taking the cheapest
                 (zone, position) pair. Valid answers are unchanged.
  4. Official evaluate() on the sample files, like score_pilot_history.py. Opus direct and
     shortest are re-scored and must reproduce results/pilot.json exactly. Opus + history
     (runs/pilot_history/opus.jsonl) is included on whatever routes it has finished.

No v0 model is scored here. The pilot npz files are opened for the stop/zone geometry, and
their actual_order is used ONLY after Opus has answered, to measure zone-order agreement with
the driver.

--arm-jsonl / --out-stem / --work-dir exist only to test this script with a stand-in log.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from v0_model import prepare_route, driver_zone_order  # noqa: E402
from v0_decode import load_npz, stops_from_zone_order   # noqa: E402  (v0's reusable zone -> stops)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp = load_module("score_pilot", HERE / "score_pilot.py")
zr = load_module("zone_revisits", HERE / "zone_revisits.py")
rz = load_module("run_pilot_zone", HERE / "run_pilot_zone.py")
official, as_proposal, summarize, paired, read_jsonl = (sp.official, sp.as_proposal, sp.summarize,
                                                        sp.paired, sp.read_jsonl)


def cheapest_insertion(P, partial):
    """Insert every zone missing from `partial` (zone indices) by cheapest insertion.
    Cost of a path station -> z1 -> ... -> zk -> station uses v0's zone travel times."""
    Z, ztt, back = P["Z"], P["ztt"], P["to_station"]
    order = list(partial)
    missing = [z for z in range(Z) if z not in set(order)]
    inserted = []

    def leg(a, b):                     # a, b: zone index, or -1 for the station
        if a == -1:
            return ztt[Z, b]
        if b == -1:
            return back[a]
        return ztt[a, b]

    while missing:
        best = None
        for z in missing:
            for pos in range(len(order) + 1):
                a = order[pos - 1] if pos > 0 else -1
                b = order[pos] if pos < len(order) else -1
                c = leg(a, z) + leg(z, b) - leg(a, b)
                if best is None or c < best[0]:
                    best = (c, z, pos)
        _, z, pos = best
        order.insert(pos, z)
        missing.remove(z)
        inserted.append(z)
    return order, inserted


def kendall_tau(a, b):
    """Rank agreement of two orders of the same items: +1 identical, -1 reversed."""
    pos = {x: i for i, x in enumerate(b)}
    r = [pos[x] for x in a]
    n, s = len(r), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 if r[j] > r[i] else -1
    return s / (n * (n - 1) / 2) if n > 1 else 1.0


def agreement(order, driver):
    """How alike a zone order is to the driver's first-visit zone order."""
    pairs = set(zip(driver, driver[1:]))
    nxt = sum((a, b) in pairs for a, b in zip(order, order[1:]))
    return {"kendall_tau": kendall_tau(order, driver), "first_zone_match": bool(order[0] == driver[0]),
            "next_zone_pairs_shared": nxt / max(1, len(driver) - 1), "exact_match": order == driver}


def first_visit(stop_order, ids, P):
    pos = {str(s): i for i, s in enumerate(ids)}
    seen, out = set(), []
    for s in stop_order:
        k = int(P["zone_of"][pos[s]])
        if k >= 0 and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def fmt_ci(s):
    return f"{s['mean']:.4f} [{s['bootstrap_95_ci'][0]:.4f}, {s['bootstrap_95_ci'][1]:.4f}]"


def fmt_pair(p):
    return (f"{p['mean_difference']:+.4f} [{p['bootstrap_95_ci'][0]:+.4f}, {p['bootstrap_95_ci'][1]:+.4f}]; "
            f"zone arm better on {p['routes_first_better']}, worse on {p['routes_second_better']}, "
            f"tied {p['ties']}; sign test p = {p['exact_sign_test_two_sided_p']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=ROOT / "data/raw/sample")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--arm-jsonl", type=Path, default=ROOT / "runs/pilot_zone/opus.jsonl")
    ap.add_argument("--direct-jsonl", type=Path, default=ROOT / "runs/pilot/opus.jsonl")
    ap.add_argument("--history-jsonl", type=Path, default=ROOT / "runs/pilot_history/opus.jsonl")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "runs/pilot_zone")
    ap.add_argument("--out-stem", type=Path, default=None, help="default <results-dir>/pilot_zone")
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    stem = a.out_stem or a.results_dir / "pilot_zone"
    if stem.parent.resolve() == (ROOT / "results").resolve() and stem.name != "pilot_zone":
        raise SystemExit("refusing to write any results file other than results/pilot_zone.*")
    a.work_dir.mkdir(parents=True, exist_ok=True)

    sd = a.sample_dir
    manifest = json.loads((sd / "sample_manifest.json").read_text())["route_ids"]
    routes = json.loads((sd / "sample_route_data.json").read_text())
    actual = json.loads((sd / "sample_actual.json").read_text())
    shortest_props = json.loads((a.results_dir / "proposed_shortest.json").read_text())
    pilot_res = json.loads((a.results_dir / "pilot.json").read_text())
    zindex = {x["route_id"]: x for x in
              json.loads((ROOT / "runs/pilot_zone/prompts_index.json").read_text())["routes"]}
    order_of = lambda d: sorted(d, key=d.get)
    driver = {r: order_of(actual[r]["actual"]) for r in manifest}
    shortest = {r: order_of(shortest_props[r]["proposed"]) for r in manifest}

    def evaluate(name, orders):
        path = a.work_dir / f"proposed_{name}.json"
        path.write_text(json.dumps({r: as_proposal(o) for r, o in orders.items()}))
        return official.evaluate(str(sd / "sample_actual.json"), str(path),
                                 str(sd / "sample_travel_times.json"), str(sd / "sample_invalid.json"))

    arm = read_jsonl(a.arm_jsonl)
    direct = read_jsonl(a.direct_jsonl)
    hist = read_jsonl(a.history_jsonl)
    ran = [r for r in manifest if r in arm]
    if not ran:
        raise SystemExit(f"no finished routes in {a.arm_jsonl}")

    strict, fallback, info = {}, {}, {}
    for r in ran:
        rec = arm[r]
        names = zindex[r]["zone_ids"]
        zones, valid, reason, details = rz.parse_zone_order(rec.get("raw_output") or "", names)
        assert valid == rec["valid"] and zones == rec["parsed_zone_order"], f"{r}: re-parse differs from log"
        route = load_npz(r)
        P = prepare_route(route)
        assert P["names"] == names, f"{r}: zone list differs from the prompt"
        k = {z: i for i, z in enumerate(names)}
        kept = list(dict.fromkeys(z for z in (zones or []) if z in k))
        partial = [k[z] for z in kept]
        full, inserted = cheapest_insertion(P, partial)
        if valid:
            assert not inserted and len(full) == P["Z"]
        t0 = time.process_time()
        stops = stops_from_zone_order(route, [names[z] for z in full])
        cpu = time.process_time() - t0
        fallback[r] = stops
        if valid:
            strict[r] = stops
        drv = driver_zone_order(P, route)               # used only for agreement, after Opus answered
        assert stops[0] == driver[r][0] and sorted(stops) == sorted(driver[r])
        info[r] = {"route": route, "P": P, "zones": full, "valid": valid, "reason": reason,
                   "details": details, "inserted": [names[z] for z in inserted], "cpu": cpu,
                   "driver_zones": drv}

    ev = {"strict": evaluate("opus_zone_strict", strict),
          "fallback": evaluate("opus_zone_fallback", fallback),
          "direct": evaluate("opus_direct_rescored",
                             {r: direct[r]["parsed_sequence"] for r in manifest if direct[r]["valid"]}),
          "shortest": evaluate("shortest_rescored", shortest)}
    hist_ran = [r for r in ran if r in hist]
    if hist_ran:
        ev["history"] = evaluate("opus_history_rescored",
                                 {r: hist[r]["parsed_sequence"] for r in hist_ran if hist[r]["valid"]})
    old_direct = {p["route_id"]: p["score"] for p in pilot_res["arms"]["opus"]["per_route"]}
    old_short = {p["route_id"]: p["shortest_score"] for p in pilot_res["arms"]["opus"]["per_route"]}
    assert all(abs(float(ev["direct"]["route_scores"][r]) - old_direct[r]) < 1e-12 for r in manifest), "direct drift"
    assert all(abs(float(ev["shortest"]["route_scores"][r]) - old_short[r]) < 1e-12 for r in manifest), "shortest drift"

    per = []
    for r in ran:
        rec, x, stops_raw = arm[r], info[r], routes[r]["stops"]
        P, ids = x["P"], x["route"]["stop_ids"]
        if bool(ev["strict"]["route_feasibility"][r]) != x["valid"]:
            raise SystemExit(f"{r}: zone validity {x['valid']} != official feasibility")
        assert ev["fallback"]["route_feasibility"][r]
        u = rec.get("usage") or {}
        row = {"route_id": r, "n_stops": len(stops_raw), "n_zones": P["Z"],
               "n_references": len(zindex[r]["references"]),
               "valid": x["valid"], "invalid_reason": None if x["valid"] else x["reason"],
               "invalid_details": None if x["valid"] else x["details"],
               "fallback_inserted_zones": x["inserted"],
               "score_strict": float(ev["strict"]["route_scores"][r]),
               "score_fallback": float(ev["fallback"]["route_scores"][r]),
               "direct_score": float(ev["direct"]["route_scores"][r]),
               "shortest_score": float(ev["shortest"]["route_scores"][r]),
               "history_score": float(ev["history"]["route_scores"][r]) if r in hist_ran else None,
               "opus_zone_order": [P["names"][z] for z in x["zones"]],
               "driver_zone_order": [P["names"][z] for z in x["driver_zones"]],
               "agreement_zone_arm": agreement(x["zones"], x["driver_zones"]),
               "agreement_direct": (agreement(first_visit(direct[r]["parsed_sequence"], ids, P), x["driver_zones"])
                                    if direct[r]["valid"] else None),
               "agreement_shortest": agreement(first_visit(shortest[r], ids, P), x["driver_zones"]),
               "revisits_zone_arm": zr.revisits(fallback[r], stops_raw)[0],
               "revisits_driver": zr.revisits(driver[r], stops_raw)[0],
               "revisits_direct": zr.revisits(direct[r]["parsed_sequence"], stops_raw)[0] if direct[r]["valid"] else None,
               "input_tokens": sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                          "cache_read_input_tokens")),
               "output_tokens": u.get("output_tokens"),
               "total_cost_usd": rec.get("total_cost_usd"), "duration_ms": rec.get("duration_ms"),
               "wall_seconds": rec.get("wall_seconds"), "solver_cpu_seconds": x["cpu"],
               "model_returned": rec.get("model_returned"), "prompt_chars": rec.get("prompt_chars")}
        per.append(row)

    col = lambda k, rows=per: [p[k] for p in rows]
    s, f, d, b = col("score_strict"), col("score_fallback"), col("direct_score"), col("shortest_score")
    hrows = [p for p in per if p["history_score"] is not None]
    num = lambda k: [p[k] for p in per if p[k] is not None]
    agree = {}
    for who in ("zone_arm", "direct", "shortest"):
        rows = [p[f"agreement_{who}"] for p in per if p[f"agreement_{who}"] is not None]
        agree[who] = {"n_routes": len(rows),
                      "mean_kendall_tau": float(np.mean([x["kendall_tau"] for x in rows])),
                      "first_zone_match": sum(x["first_zone_match"] for x in rows),
                      "mean_next_zone_pairs_shared": float(np.mean([x["next_zone_pairs_shared"] for x in rows])),
                      "exact_zone_order_match": sum(x["exact_match"] for x in rows)}
    both = [p for p in per if p["agreement_direct"] is not None]
    summary = {
        "n_routes": len(per), "n_planned": len(manifest),
        "zone_arm_strict": summarize(s, a.seed), "zone_arm_fallback": summarize(f, a.seed),
        "valid_count": sum(p["valid"] for p in per),
        "invalid": [{"route_id": p["route_id"], "reason": p["invalid_reason"],
                     "inserted_by_fallback": p["fallback_inserted_zones"]} for p in per if not p["valid"]],
        "opus_direct_same_routes": summarize(d, a.seed), "shortest_same_routes": summarize(b, a.seed),
        "strict_minus_direct": paired(s, d, a.seed), "fallback_minus_direct": paired(f, d, a.seed),
        "strict_minus_shortest": paired(s, b, a.seed), "fallback_minus_shortest": paired(f, b, a.seed),
        "history_arm": None,
        "zone_revisits": {"zone_arm_total": sum(col("revisits_zone_arm")),
                          "driver_mean": float(np.mean(col("revisits_driver"))),
                          "direct_mean_valid": float(np.mean(num("revisits_direct")))},
        "zone_order_agreement_with_driver": agree,
        "zone_arm_minus_direct_kendall_tau": paired([p["agreement_zone_arm"]["kendall_tau"] for p in both],
                                                    [p["agreement_direct"]["kendall_tau"] for p in both], a.seed),
        "cost_latency": {
            "opus_cost_usd_total": float(np.sum(num("total_cost_usd"))) if num("total_cost_usd") else None,
            "opus_cost_usd_mean": float(np.mean(num("total_cost_usd"))) if num("total_cost_usd") else None,
            "opus_duration_s_mean": float(np.mean(num("duration_ms")) / 1000) if num("duration_ms") else None,
            "opus_duration_s_median": float(np.median(num("duration_ms")) / 1000) if num("duration_ms") else None,
            "input_tokens_total": int(np.sum(num("input_tokens"))),
            "output_tokens_total": int(np.sum(num("output_tokens"))) if num("output_tokens") else None,
            "solver_cpu_s_mean": float(np.mean(col("solver_cpu_seconds"))),
            "direct_cost_usd_same_routes": float(np.sum([direct[r].get("total_cost_usd") or 0 for r in ran])),
            "direct_duration_s_mean_same_routes": float(np.mean([(direct[r].get("duration_ms") or 0) / 1000
                                                                 for r in ran]))},
    }
    if hrows:
        h = col("history_score", hrows)
        summary["history_arm"] = {
            "n_routes": len(hrows), "note": "Opus direct + history (stop level; its history rule allows "
                                            "references dated after the route)",
            "history_same_routes": summarize(h, a.seed),
            "strict_minus_history": paired(col("score_strict", hrows), h, a.seed),
            "fallback_minus_history": paired(col("score_fallback", hrows), h, a.seed)}

    out = {"description": ("Zone-level swap-the-brain arm: Opus 5.5 picks only the zone order; v0's own "
                           "inside-zone solver (v0_decode.stops_from_zone_order) fills in the stops."),
           "metric": "official rc-cli score, lower is better, 0 = driver's exact order",
           "invalid_handling": {"strict": "invalid zone lists omitted -> challenge invalid score",
                                "fallback": "missing zones added by cheapest insertion on v0 zone travel times"},
           "prompts": "runs/pilot_zone/prompts (scripts/build_prompts_zone.py), time-respecting history",
           "summary": summary, "per_route": per}
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1))
    stem.with_suffix(".md").write_text(report(out, info))
    print(f"zone arm strict {fmt_ci(summary['zone_arm_strict'])}, fallback {fmt_ci(summary['zone_arm_fallback'])}, "
          f"valid {summary['valid_count']}/{len(per)}; minus direct {fmt_pair(summary['strict_minus_direct'])}")
    print(f"wrote {stem.with_suffix('.json')} and .md")


def report(out, info):
    S, per = out["summary"], out["per_route"]
    cl, ag = S["cost_latency"], S["zone_order_agreement_with_driver"]
    # Worked example: the route whose strict score is closest to the median.
    med = float(np.median([p["score_fallback"] for p in per]))
    ex = min(per, key=lambda p: (abs(p["score_fallback"] - med), p["route_id"]))
    hist = S["history_arm"]
    money = lambda v: "n/a" if v is None else f"${v:.3f}"
    secs = lambda v: "n/a" if v is None else f"{v:.1f} s"
    L = []
    L.append("# Zone-level \"swap the brain\" arm: Opus picks the zone order, our solver does the rest\n")
    L.append(f"Routes scored: {S['n_routes']} of {S['n_planned']} pilot routes. Produced by "
             "`scripts/score_pilot_zone.py` from `runs/pilot_zone/opus.jsonl`. No v0 model is scored here.\n")
    L.append("## The idea, from zero\n")
    L.append("A delivery route has about 150 stops. Amazon groups nearby stops into **zones** (IDs like "
             "`J-16.2C`), about 20 per route. Drivers almost always finish one zone before moving to the next. "
             "So you can build a route in two steps:\n")
    L.append("1. **Choose the order of the zones** (the big decision: which neighbourhood first, which next).")
    L.append("2. **Inside each zone, order its few stops** (a small puzzle a solver handles well: shortest path "
             "from where you came in to where you leave towards the next zone).\n")
    L.append("Our tiny model does step 1, and a fixed solver (OR-Tools, the function "
             "`stops_from_zone_order` in `scripts/v0_decode.py`) does step 2. In this arm we **swap the brain**: "
             "Opus does step 1 and the *same* solver does step 2. If our model later beats this arm, the "
             "difference can only come from who picked the zone order, because everything else is identical.\n")
    L.append("What Opus saw for each route: the station; each zone's ID, centre point and stop count; travel "
             "times between zones (the average over all pairs of stops, the same numbers our model sees), from "
             "the station to each zone and back; and the zone orders of up to 3 earlier routes from the same "
             "station that share the most zones, all driven **strictly before** this route's date and all from "
             "the training split. It answered with a list of zone IDs.\n")
    L.append("## Results\n")
    L.append("Score: the challenge's official metric. Lower is better; 0 means exactly the driver's order. "
             "The bracket is a 95% bootstrap confidence interval: resample the routes 10,000 times and see "
             "where the average lands.\n")
    L.append("| Arm (same routes) | Mean score [95% CI] |")
    L.append("|---|---|")
    L.append(f"| Opus zone order + our solver, strict (invalid answers get the challenge's invalid score) | "
             f"{fmt_ci(S['zone_arm_strict'])} |")
    L.append(f"| Opus zone order + our solver, with cheapest-insertion fallback | {fmt_ci(S['zone_arm_fallback'])} |")
    L.append(f"| Opus direct (orders all stops itself, no history) | {fmt_ci(S['opus_direct_same_routes'])} |")
    if hist:
        L.append(f"| Opus direct + history (on its {hist['n_routes']} finished routes) | "
                 f"{fmt_ci(hist['history_same_routes'])} |")
    L.append(f"| Shortest route (no learning) | {fmt_ci(S['shortest_same_routes'])} |\n")
    L.append(f"Valid zone lists: {S['valid_count']} of {S['n_routes']}.")
    for x in S["invalid"]:
        L.append(f"- `{x['route_id'][:16]}…`: {x['reason']}; fallback inserted {x['inserted_by_fallback']}")
    L.append("")
    L.append("### Paired comparisons (zone arm minus the other; negative = zone arm closer to the driver)\n")
    L.append("Paired means: compare the two arms on the same route, then average the differences. The sign "
             "test asks: if each arm were equally likely to win any route, how surprising is a win/loss "
             "split this lopsided? (ties dropped)\n")
    L.append(f"- vs Opus direct, strict: {fmt_pair(S['strict_minus_direct'])}")
    L.append(f"- vs Opus direct, fallback: {fmt_pair(S['fallback_minus_direct'])}")
    if hist:
        L.append(f"- vs Opus + history ({hist['n_routes']} routes), strict: {fmt_pair(hist['strict_minus_history'])}")
        L.append(f"- vs Opus + history, fallback: {fmt_pair(hist['fallback_minus_history'])}")
        L.append("  (Note: the history arm's rule allowed references dated after the route; this arm's did not.)")
    else:
        L.append("- vs Opus + history: not available (`runs/pilot_history/opus.jsonl` has no finished routes yet).")
    L.append(f"- vs shortest, strict: {fmt_pair(S['strict_minus_shortest'])}")
    L.append(f"- vs shortest, fallback: {fmt_pair(S['fallback_minus_shortest'])}\n")
    L.append("### Zone revisits and zone-order agreement\n")
    L.append(f"Zone revisits (leaving a zone and coming back later): zone arm {S['zone_revisits']['zone_arm_total']} "
             f"in total, which is zero **by construction** (the solver finishes each zone in one go); driver mean "
             f"{S['zone_revisits']['driver_mean']:.2f} per route; Opus direct mean "
             f"{S['zone_revisits']['direct_mean_valid']:.2f}. Since zero is guaranteed, the useful question is "
             "whether the zone *order* matches the driver's (the order in which the driver first entered each zone):\n")
    L.append("| Whose zone order | Kendall tau (1 = same order, 0 = unrelated, -1 = reversed) | First zone right | "
             "Next-zone pairs shared with driver | Exactly the driver's order |")
    L.append("|---|---|---|---|---|")
    for who, label in (("zone_arm", "Opus zone arm"), ("direct", "Opus direct (zones in first-visit order)"),
                       ("shortest", "Shortest route (zones in first-visit order)")):
        g = ag[who]
        L.append(f"| {label} | {g['mean_kendall_tau']:.3f} | {g['first_zone_match']}/{g['n_routes']} | "
                 f"{g['mean_next_zone_pairs_shared']:.1%} | {g['exact_zone_order_match']}/{g['n_routes']} |")
    k = S["zone_arm_minus_direct_kendall_tau"]
    L.append(f"\nKendall tau, zone arm minus Opus direct (paired, {k['n_routes']} routes): "
             f"{k['mean_difference']:+.3f} [{k['bootstrap_95_ci'][0]:+.3f}, {k['bootstrap_95_ci'][1]:+.3f}] "
             f"(positive = the zone arm's order is closer to the driver's). The zone arm's tau is higher on "
             f"{k['routes_second_better']} routes and lower on {k['routes_first_better']}. Note: tau punishes a "
             "reversed order, while the official score treats an exact reverse as perfect.\n")
    L.append("### Cost and latency per route\n")
    L.append(f"- Opus zone arm: cost field {money(cl['opus_cost_usd_mean'])} mean, {money(cl['opus_cost_usd_total'])} "
             f"total; {secs(cl['opus_duration_s_mean'])} mean, {secs(cl['opus_duration_s_median'])} median per call; "
             f"{cl['input_tokens_total']:,} input and {cl['output_tokens_total'] or 0:,} output tokens in total.")
    L.append(f"- Inside-zone solver (our laptop): {cl['solver_cpu_s_mean']:.2f} s CPU per route.")
    L.append(f"- Opus direct on the same routes: {money(cl['direct_cost_usd_same_routes'])} total, "
             f"{cl['direct_duration_s_mean_same_routes']:.1f} s mean per call.\n")
    L.append("## One worked route\n")
    L.append(f"Route `{ex['route_id']}` ({ex['n_stops']} locations, {ex['n_zones']} zones, "
             f"{ex['n_references']} earlier reference route(s)), chosen because its score is closest to the median.\n")
    L.append(f"- Opus's zone order: {' '.join(ex['opus_zone_order'])}")
    L.append(f"- Driver's zone order: {' '.join(ex['driver_zone_order'])}")
    a2 = ex["agreement_zone_arm"]
    L.append(f"- Agreement: Kendall tau {a2['kendall_tau']:.3f}; first zone "
             f"{'right' if a2['first_zone_match'] else 'wrong'}; {a2['next_zone_pairs_shared']:.0%} of the driver's "
             f"zone-to-zone steps also appear in Opus's order.")
    L.append(f"- The solver then ordered the stops inside each zone. Score: zone arm {ex['score_fallback']:.4f}, "
             f"Opus direct {ex['direct_score']:.4f}, shortest {ex['shortest_score']:.4f} (lower is better).\n")
    L.append("## How to read this honestly\n")
    L.append("- 50 routes is small; trust the confidence intervals and sign tests, not the point estimates.")
    L.append("- This arm is an opponent for our model, not a result for it. The model has not been scored on "
             "these routes; that happens once, later, by Angad's decision.")
    L.append("- Time-respecting history leaves some routes with fewer than 3 references (see "
             "`runs/pilot_zone/prompts_index.json`), because the split is random rather than by date.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
