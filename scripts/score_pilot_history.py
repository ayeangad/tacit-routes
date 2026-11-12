#!/usr/bin/env python3
"""Score the "Opus direct + history" arm with the official, unmodified rc-cli scorer.

  .venv/bin/python -I scripts/score_pilot_history.py

Writes ONLY results/pilot_history.json and results/pilot_history.md (never results/pilot.*).
Proposal files for the official evaluate() go to runs/pilot_history/proposed_*.json.

Scoring is exactly as in score_pilot.py: valid answers are written in the challenge format
and passed to official evaluate(); an invalid answer is left out, so evaluate() gives it the
challenge's invalid-sequence score. Opus direct (runs/pilot/opus.jsonl) and the saved
shortest tours (results/proposed_shortest.json) are re-scored in the same call style, and
must reproduce results/pilot.json and plain_baseline.json exactly.

Reported: Opus+history mean with bootstrap 95% CI and validity; paired differences against
Opus direct and against shortest (mean with CI, wins/losses, exact sign test); zone revisits
(method of scripts/zone_revisits.py) for both Opus arms and the driver; usage, cost field
and duration per call. Only routes the history arm actually finished are reported.

--arm-jsonl and --out-stem exist only to test this script (e.g. on the Opus direct log,
which must give a difference of exactly 0); defaults are the real paths.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sp = load_module("score_pilot", HERE / "score_pilot.py")      # official scorer, paired(), read_jsonl()
zr = load_module("zone_revisits", HERE / "zone_revisits.py")   # revisits()
official, as_proposal, summarize, paired, read_jsonl = (sp.official, sp.as_proposal, sp.summarize,
                                                        sp.paired, sp.read_jsonl)


def first_visit_zones(order, stops):
    seen, out = set(), []
    for s in order:
        z = stops[s].get("zone_id")
        if stops[s]["type"] != "Station" and z not in (None, "-") and z not in seen:
            seen.add(z)
            out.append(z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", type=Path, default=ROOT / "data/raw/sample")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--arm-jsonl", type=Path, default=ROOT / "runs/pilot_history/opus.jsonl")
    ap.add_argument("--direct-jsonl", type=Path, default=ROOT / "runs/pilot/opus.jsonl")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "runs/pilot_history")
    ap.add_argument("--out-stem", type=Path, default=None, help="default <results-dir>/pilot_history")
    ap.add_argument("--seed", type=int, default=20261110)
    a = ap.parse_args()
    stem = a.out_stem or a.results_dir / "pilot_history"
    if stem.name in ("pilot", "plain_baseline", "zone_revisits"):
        raise SystemExit("refusing to overwrite an existing results file")

    sd = a.sample_dir
    manifest = json.loads((sd / "sample_manifest.json").read_text())["route_ids"]
    routes = json.loads((sd / "sample_route_data.json").read_text())
    actual = json.loads((sd / "sample_actual.json").read_text())
    shortest_props = json.loads((a.results_dir / "proposed_shortest.json").read_text())
    pilot_res = json.loads((a.results_dir / "pilot.json").read_text())
    hist_index = json.loads((ROOT / "runs/pilot_history/prompts_index.json").read_text())
    refs_of = {x["route_id"]: x["references"] for x in hist_index["routes"]}
    order_of = lambda d: sorted(d, key=d.get)
    driver = {r: order_of(actual[r]["actual"]) for r in manifest}
    shortest = {r: order_of(shortest_props[r]["proposed"]) for r in manifest}

    def evaluate(name, orders):
        path = a.work_dir / f"proposed_{name}.json"
        path.write_text(json.dumps({r: as_proposal(o) for r, o in orders.items()}))
        return official.evaluate(str(sd / "sample_actual.json"), str(path),
                                 str(sd / "sample_travel_times.json"), str(sd / "sample_invalid.json"))

    hist = read_jsonl(a.arm_jsonl)
    direct = read_jsonl(a.direct_jsonl)
    ran = [r for r in manifest if r in hist]
    if not ran:
        raise SystemExit(f"no finished routes in {a.arm_jsonl}")

    ev = {"history": evaluate("opus_history", {r: hist[r]["parsed_sequence"] for r in ran if hist[r]["valid"]}),
          "direct": evaluate("opus_direct_rescored",
                             {r: direct[r]["parsed_sequence"] for r in manifest if direct[r]["valid"]}),
          "shortest": evaluate("shortest_rescored", shortest)}
    old_direct = {p["route_id"]: p["score"] for p in pilot_res["arms"]["opus"]["per_route"]}
    old_short = {p["route_id"]: p["shortest_score"] for p in pilot_res["arms"]["opus"]["per_route"]}
    assert all(abs(float(ev["direct"]["route_scores"][r]) - old_direct[r]) < 1e-12 for r in manifest), "direct drift"
    assert all(abs(float(ev["shortest"]["route_scores"][r]) - old_short[r]) < 1e-12 for r in manifest), "shortest drift"

    per = []
    for r in ran:
        rec, stops = hist[r], routes[r]["stops"]
        feasible = bool(ev["history"]["route_feasibility"][r])
        if feasible != rec["valid"]:
            raise SystemExit(f"{r}: our validity {rec['valid']} != official {feasible}")
        u = rec.get("usage") or {}
        row = {"route_id": r, "n_stops": len(stops), "valid": rec["valid"], "invalid_reason": rec["invalid_reason"],
               "n_references": len(refs_of[r]),
               "score": float(ev["history"]["route_scores"][r]),
               "direct_score": float(ev["direct"]["route_scores"][r]),
               "shortest_score": float(ev["shortest"]["route_scores"][r]),
               "input_tokens": sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                          "cache_read_input_tokens")),
               "output_tokens": u.get("output_tokens"),
               "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens"),
               "total_cost_usd": rec.get("total_cost_usd"), "duration_ms": rec.get("duration_ms"),
               "wall_seconds": rec.get("wall_seconds"), "model_returned": rec.get("model_returned"),
               "stop_reason": rec.get("stop_reason"), "prompt_chars": rec.get("prompt_chars")}
        for who, order in (("driver", driver[r]), ("history", rec["parsed_sequence"] if rec["valid"] else None),
                           ("direct", direct[r]["parsed_sequence"] if direct[r]["valid"] else None)):
            row[f"{who}_revisits"] = zr.revisits(order, stops)[0] if order else None
        per.append(row)

    s = [p["score"] for p in per]
    d = [p["direct_score"] for p in per]
    b = [p["shortest_score"] for p in per]
    both_valid = [p for p in per if p["history_revisits"] is not None and p["direct_revisits"] is not None]
    rv = lambda k, rows: [p[k] for p in rows]
    summary = {
        "n_routes": len(per), "n_planned": len(manifest),
        "opus_history": summarize(s, a.seed),
        "valid_count": sum(p["valid"] for p in per),
        "invalid_reasons": [p["invalid_reason"] for p in per if not p["valid"]],
        "opus_direct_same_routes": summarize(d, a.seed),
        "shortest_same_routes": summarize(b, a.seed),
        "history_minus_direct": paired(s, d, a.seed),
        "history_minus_shortest": paired(s, b, a.seed),
        "direct_minus_shortest_same_routes": paired(d, b, a.seed),
        "zone_revisits": {
            "n_routes_both_valid": len(both_valid),
            "mean": {w: float(np.mean(rv(f"{w}_revisits", both_valid))) for w in ("driver", "direct", "history")},
            "median": {w: float(np.median(rv(f"{w}_revisits", both_valid))) for w in ("driver", "direct", "history")},
            "routes_with_zero": {w: sum(p[f"{w}_revisits"] == 0 for p in both_valid)
                                 for w in ("driver", "direct", "history")},
            "history_minus_direct": paired(rv("history_revisits", both_valid), rv("direct_revisits", both_valid), a.seed),
        },
        "usage": {
            "total_cost_usd_field_sum": float(sum(p["total_cost_usd"] or 0 for p in per)),
            "mean_cost_usd_field": float(np.mean([p["total_cost_usd"] or 0 for p in per])),
            "mean_input_tokens": float(np.mean([p["input_tokens"] for p in per])),
            "mean_output_tokens": float(np.mean([p["output_tokens"] or 0 for p in per])),
            "mean_thinking_tokens": float(np.mean([p["thinking_tokens"] or 0 for p in per])),
            "mean_duration_ms": float(np.mean([p["duration_ms"] or 0 for p in per])),
            "mean_wall_seconds": float(np.mean([p["wall_seconds"] or 0 for p in per])),
            "models_returned": sorted({json.dumps(p["model_returned"]) for p in per}),
            "stop_reasons": sorted({str(p["stop_reason"]) for p in per}),
        },
    }

    # Worked route: the route whose history-minus-direct difference is the median.
    ranked = sorted(per, key=lambda p: (p["score"] - p["direct_score"], p["route_id"]))
    w = ranked[len(ranked) // 2]
    r, stops = w["route_id"], routes[w["route_id"]]["stops"]
    worked = {"route_id": r, "station": routes[r]["station_code"], "n_stops": w["n_stops"],
              "selection": "route with the median (Opus+history − Opus direct) score difference",
              "scores": {"driver": 0.0, "shortest": w["shortest_score"], "opus_direct": w["direct_score"],
                         "opus_history": w["score"]},
              "revisits": {k: w[f"{k}_revisits"] for k in ("driver", "direct", "history")},
              "zone_order_first_visit": {
                  "driver": first_visit_zones(driver[r], stops),
                  "opus_direct": first_visit_zones(direct[r]["parsed_sequence"], stops) if direct[r]["valid"] else None,
                  "opus_history": first_visit_zones(hist[r]["parsed_sequence"], stops) if hist[r]["valid"] else None,
                  "reference_1_shown_in_prompt": refs_of[r][0]["zone_order"] if refs_of[r] else None},
              "reference_1": ({k: refs_of[r][0][k] for k in ("date", "shared_zones")} if refs_of[r] else None)}

    out = {"description": "Opus direct + history: pilot prompt plus up to 3 similar TRAIN routes' zone orders",
           "metric": pilot_res["metric"], "invalid_handling": pilot_res["invalid_handling"],
           "rescore_checks": "Opus direct and shortest re-scored identically to results/pilot.json",
           "summary": summary, "worked_example": worked, "per_route": per}
    stem.with_suffix(".json").write_text(json.dumps(out, indent=1))
    stem.with_suffix(".md").write_text(markdown(out, a.arm_jsonl))
    print(json.dumps({k: summary[k] for k in ("n_routes", "opus_history", "valid_count",
                                                "history_minus_direct", "history_minus_shortest")}, indent=1))
    return 0


def ci(x):
    return f"[{x[0]:+.4f}, {x[1]:+.4f}]"


def markdown(out, arm_path):
    s, w = out["summary"], out["worked_example"]
    hd, hs, ds = s["history_minus_direct"], s["history_minus_shortest"], s["direct_minus_shortest_same_routes"]
    z, u = s["zone_revisits"], s["usage"]
    n = s["n_routes"]
    L = [f"# Opus direct + history: results on {n} of {s['n_planned']} pilot routes", "",
         f"Produced by `scripts/score_pilot_history.py` from `{arm_path.relative_to(ROOT) if arm_path.is_relative_to(ROOT) else arm_path}`; "
         "every number here is in `results/pilot_history.json`.", ""]
    if n < s["n_planned"]:
        L += [f"**Warning: only {n} routes finished. Treat everything below as incomplete.**", ""]
    L += ["## What was tested, from zero", "",
          "Each pilot route is one real van trip: a depot and 60–190 stops. The driver visited the stops in some order, "
          "and we know that order. Each competitor guesses the order, and the official challenge scorer compares the "
          "guess with the driver's real order. **0 means a perfect match, and lower is better.**", "",
          "- **Opus direct** (already run, `results/pilot.md`): Opus sees only this route's stops: their IDs, map "
          "positions, zone labels and nearest neighbours.",
          "- **Opus + history** (this run): the same prompt, word for word, plus a short list of up to 3 other routes "
          "from the same depot that share the most zone labels. Each one shows the order its driver visited those "
          "zones in. This is the same kind of history our own model learns from.",
          "- **Shortest route:** a solver that just finds a short tour and knows nothing about drivers.", "",
          "The question: does showing Opus how drivers at this depot usually move through the zones make its guess "
          "closer to the real driver?", "",
          "## Main results (lower is better)", "",
          "| Arm | Routes | Mean score | Bootstrap 95% CI | Median | Valid answers |", "|---|---|---|---|---|---|"]
    for name, k in (("Opus + history", "opus_history"), ("Opus direct", "opus_direct_same_routes"),
                    ("Shortest route", "shortest_same_routes")):
        x = s[k]
        valid = f"{s['valid_count']} / {n}" if k == "opus_history" else "—"
        L.append(f"| {name} | {n} | {x['mean']:.4f} | {x['bootstrap_95_ci'][0]:.4f} – {x['bootstrap_95_ci'][1]:.4f} "
                 f"| {x['median']:.4f} | {valid} |")
    L += ["", "**How to read the CI.** We resample the routes 10,000 times. The range holds the middle 95% of the "
          "resampled means. A wide range means the mean could easily have come out differently with other routes.", "",
          "## Paired comparisons (first minus second; negative = first is better)", "",
          "Every route is scored under both arms, so we subtract the two scores route by route and average the "
          "differences. This removes the fact that some routes are simply harder than others.", "",
          "The **sign test** asks: if neither arm were really better, how surprising is this win/loss split? It is "
          "like flipping a fair coin once per route. A small p (below about 0.05) means the split would be rare by "
          "luck alone.", "",
          "| Comparison | Mean difference | 95% CI | Median | First better / worse / tied | Sign test p |",
          "|---|---|---|---|---|---|"]
    for name, x in (("Opus+history − Opus direct", hd), ("Opus+history − shortest", hs),
                    ("Opus direct − shortest (same routes)", ds)):
        p = x["exact_sign_test_two_sided_p"]
        L.append(f"| {name} | {x['mean_difference']:+.4f} | {ci(x['bootstrap_95_ci'])} | {x['median_difference']:+.4f} | "
                 f"{x['routes_first_better']} / {x['routes_second_better']} / {x['ties']} | "
                 f"{'—' if p is None else f'{p:.3g}'} |")
    zd = z["history_minus_direct"]
    L += ["", "## Zone revisits", "",
          "A zone is a small area with a label such as `D-16.2C`. A **revisit** means the route leaves a zone and later "
          "comes back to it. Real drivers almost never do this. The method is `scripts/zone_revisits.py`'s: drop "
          "the depot and stops without a zone, merge back-to-back stops in the same zone into one visit, and count "
          "visits minus distinct zones.", "",
          f"Routes where both Opus answers are valid: {z['n_routes_both_valid']}.", "",
          "| Order | Mean revisits | Median | Routes with 0 |", "|---|---|---|---|"]
    for name, k in (("Driver", "driver"), ("Opus direct", "direct"), ("Opus + history", "history")):
        L.append(f"| {name} | {z['mean'][k]:.2f} | {z['median'][k]:g} | {z['routes_with_zero'][k]} |")
    L += ["", f"Opus+history − Opus direct revisits: mean {zd['mean_difference']:+.2f}, 95% CI "
          f"[{zd['bootstrap_95_ci'][0]:+.2f}, {zd['bootstrap_95_ci'][1]:+.2f}], fewer on {zd['routes_first_better']}, "
          f"more on {zd['routes_second_better']}, tied on {zd['ties']} (sign test p = "
          f"{zd['exact_sign_test_two_sided_p'] if zd['exact_sign_test_two_sided_p'] is None else format(zd['exact_sign_test_two_sided_p'], '.3g')}).", "",
          "## Worked route", "",
          f"`{w['route_id']}` (station {w['station']}, {w['n_stops']} locations). Chosen as the {w['selection']}, "
          "so it is a typical route for this comparison, not the best or the worst.", "",
          "| | Score | Zone revisits |", "|---|---|---|",
          f"| Driver (the answer) | 0 | {w['revisits']['driver']} |",
          f"| Shortest route | {w['scores']['shortest']:.4f} | — |",
          f"| Opus direct | {w['scores']['opus_direct']:.4f} | {w['revisits']['direct']} |",
          f"| Opus + history | {w['scores']['opus_history']:.4f} | {w['revisits']['history']} |", "",
          "Zones in the order each one first enters them:", ""]
    for k, label in (("reference_1_shown_in_prompt", "Past route #1 shown to Opus" +
                      (f" ({w['reference_1']['date']}, shares {w['reference_1']['shared_zones']} zones)"
                       if w["reference_1"] else "")),
                     ("driver", "Real driver"), ("opus_direct", "Opus direct"), ("opus_history", "Opus + history")):
        v = w["zone_order_first_visit"][k]
        L.append(f"- **{label}:** {' '.join(v) if v else '(none)'}")
    L += ["", "Things to compare: does Opus + history follow the past route's zone order more closely than Opus "
          "direct does? Is the past route itself close to what the real driver did?", "",
          "## Cost and time per call", "",
          f"- Cost field from `claude -p` (`total_cost_usd`): total ${u['total_cost_usd_field_sum']:.2f}, "
          f"mean ${u['mean_cost_usd_field']:.3f} per call.",
          f"- Mean tokens per call: input {u['mean_input_tokens']:.0f}, output {u['mean_output_tokens']:.0f} "
          f"(of which thinking {u['mean_thinking_tokens']:.0f}).",
          f"- Mean duration: {u['mean_duration_ms'] / 1000:.1f} s reported by the CLI, {u['mean_wall_seconds']:.1f} s wall clock.",
          f"- Models returned: {', '.join(u['models_returned'])}; stop reasons: {', '.join(u['stop_reasons'])}.", "",
          "## Caveats", "",
          "- 50 routes. The comparisons were not pre-registered as a final test, and the pilot routes are the "
          "frontier-comparison sample.",
          "- Opus uses extended thinking. Each arm is one sample per route, so run-to-run randomness in Opus is part "
          "of the difference.",
          "- The past routes come from the TRAIN split, picked by shared zones. Some are dated after the pilot route "
          "(the split is random, not by date). Our v0-hist model has the same information.",
          "- One DLA4 route had no past route sharing any zone, so its prompt contained no history.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
