#!/usr/bin/env python3
"""Score the DeepSeek thinking-on rerun exactly as score_pilot.py scores its arms.

  .venv/bin/python -I scripts/score_deepseek_thinking.py --sample-dir data/raw/sample \
      --pilot-dir runs/pilot --results-dir results

Uses score_pilot.py's own helpers (official evaluate(), as_proposal, summarize, bootstrap_ci,
paired, diagnose) unchanged. Invalid answers are left out of the proposal file, so the
official evaluate() assigns the challenge's invalid-sequence score.

Reads (never writes) runs/pilot/deepseek.jsonl and runs/pilot/opus.jsonl. Writes only
results/pilot_deepseek_thinking.{json,md} and runs/pilot/proposed_*_dsthink.json, so it does
not touch the files score_pilot.py writes.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("score_pilot", HERE / "score_pilot.py")
sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sp)


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def sf4(x):
    return f"{x:+.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, type=Path)
    ap.add_argument("--pilot-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    sd, seed = args.sample_dir, args.seed

    manifest = json.loads((sd / "sample_manifest.json").read_text())
    actual = json.loads((sd / "sample_actual.json").read_text())
    travel = json.loads((sd / "sample_travel_times.json").read_text())
    shortest_props = json.loads((args.results_dir / "proposed_shortest.json").read_text())
    plain = json.loads((args.results_dir / "plain_baseline.json").read_text())
    order_of = lambda d: sorted(d, key=d.get)
    driver = {r: order_of(actual[r]["actual"]) for r in actual}
    shortest = {r: order_of(shortest_props[r]["proposed"]) for r in shortest_props}

    def official_eval(name, orders):
        path = args.pilot_dir / f"proposed_{name}_dsthink.json"
        path.write_text(json.dumps({r: sp.as_proposal(o) for r, o in orders.items()}))
        return sp.official.evaluate(str(sd / "sample_actual.json"), str(path),
                                    str(sd / "sample_travel_times.json"), str(sd / "sample_invalid.json"))

    short_eval = official_eval("shortest_rescored", shortest)
    old = {p["route_id"]: p["shortest_score"] for p in plain["per_route"]}
    assert all(abs(float(short_eval["route_scores"][r]) - old[r]) < 1e-12 for r in old), "baseline drift"

    def score_arm(arm, fname):
        recs = sp.read_jsonl(args.pilot_dir / fname)
        ran = [r for r in manifest["route_ids"] if r in recs]
        ev = official_eval(arm, {r: recs[r]["parsed_sequence"] for r in ran if recs[r]["valid"]})
        out = {}
        for r in ran:
            feasible = bool(ev["route_feasibility"][r])
            if feasible != recs[r]["valid"]:
                raise SystemExit(f"{arm} {r}: our validity {recs[r]['valid']} != official {feasible}")
            out[r] = float(ev["route_scores"][r])
        return recs, ran, out

    recs, ran, scores = score_arm("deepseek_thinking", "deepseek_thinking.jsonl")
    off_recs, off_ran, off_scores = score_arm("deepseek_off", "deepseek.jsonl")
    op_recs, op_ran, op_scores = score_arm("opus_snapshot", "opus.jsonl")

    per = []
    for r in ran:
        rec, u = recs[r], recs[r]["usage"]
        row = {"route_id": r, "n_stops": len(travel[r]), "valid": rec["valid"],
               "invalid_reason": rec["invalid_reason"], "score": scores[r],
               "shortest_score": float(short_eval["route_scores"][r]),
               "thinking_off_score": off_scores.get(r), "thinking_off_valid":
                   off_recs[r]["valid"] if r in off_recs else None,
               "opus_score": op_scores.get(r),
               "prompt_chars": rec["prompt_chars"], "output_chars": rec["output_chars"],
               "wall_seconds": rec["wall_seconds"], "attempts": rec["attempts"],
               "input_tokens": u.get("prompt_tokens"),
               "cache_hit_tokens": u.get("prompt_cache_hit_tokens"),
               "output_tokens": u.get("completion_tokens"),
               "reasoning_tokens": rec.get("reasoning_tokens"), "reasoning_chars": rec.get("reasoning_chars"),
               "cost_usd_list_price": rec["cost_usd_list_price"],
               "model_returned": rec["model_returned"], "finish_reason": rec["finish_reason"]}
        row.update(sp.diagnose(rec["raw_output"], driver[r], travel[r]))
        per.append(row)
    if not per:
        raise SystemExit("no scored routes")

    sc = [p["score"] for p in per]
    base = [p["shortest_score"] for p in per]
    reasons = {}
    for p in per:
        if not p["valid"]:
            reasons[p["invalid_reason"]] = reasons.get(p["invalid_reason"], 0) + 1
    finish = {}
    for p in per:
        finish[p["finish_reason"]] = finish.get(p["finish_reason"], 0) + 1
    tot = lambda k: int(sum(p[k] or 0 for p in per))
    walls = [p["wall_seconds"] for p in per]
    summary = {
        "n_routes": len(per),
        "score": sp.summarize(sc, seed),
        "valid_only_mean_score": sp.mean_or_none([p["score"] for p in per if p["valid"]]),
        "valid_count": sum(p["valid"] for p in per), "invalid_count": sum(not p["valid"] for p in per),
        "invalid_reasons": reasons, "finish_reasons": finish,
        "shortest_on_same_routes": sp.summarize(base, seed),
        "arm_minus_shortest_paired": sp.paired(sc, base, seed),
        "spend_usd_list_price": float(sum(p["cost_usd_list_price"] for p in per)),
        "totals": {k: tot(k) for k in ("input_tokens", "cache_hit_tokens", "output_tokens",
                                       "reasoning_tokens", "reasoning_chars", "output_chars")},
        "mean_input_tokens": float(np.mean([p["input_tokens"] for p in per])),
        "mean_output_tokens": float(np.mean([p["output_tokens"] for p in per])),
        "mean_reasoning_tokens": sp.mean_or_none([p["reasoning_tokens"] for p in per]),
        "min_max_output_tokens": [min(p["output_tokens"] for p in per), max(p["output_tokens"] for p in per)],
        "wall_seconds": {"mean": float(np.mean(walls)), "median": float(np.median(walls)),
                         "min": float(min(walls)), "max": float(max(walls)), "sum": float(sum(walls))},
        "calls_with_network_retries": sum(p["attempts"] > 1 for p in per),
        "mean_spearman_vs_driver_order": sp.mean_or_none(
            [p["spearman_vs_driver_order_distinct_ids"] for p in per]),
        "outputs_with_a_repeated_id": sum(p["first_repeat_at_index"] is not None for p in per),
        "model_returned": sorted({p["model_returned"] for p in per}),
    }

    comps = {}
    both = [r for r in ran if r in off_scores]
    comps["thinking_on_minus_off"] = {
        **sp.paired([scores[r] for r in both], [off_scores[r] for r in both], seed),
        "thinking_off": sp.summarize([off_scores[r] for r in both], seed),
        "thinking_off_valid_count": sum(off_recs[r]["valid"] for r in both),
        "both_valid_routes": [r for r in both if recs[r]["valid"] and off_recs[r]["valid"]]}
    op_both = [r for r in ran if r in op_scores]
    if op_both:
        comps["opus_minus_thinking_on_opus_routes"] = {
            **sp.paired([op_scores[r] for r in op_both], [scores[r] for r in op_both], seed),
            "opus_jsonl_ok_lines_at_scoring": len(op_ran),
            "opus": sp.summarize([op_scores[r] for r in op_both], seed),
            "opus_valid_count": sum(op_recs[r]["valid"] for r in op_both),
            "thinking_on": sp.summarize([scores[r] for r in op_both], seed),
            "thinking_on_valid_count": sum(recs[r]["valid"] for r in op_both),
            "thinking_off": sp.summarize([off_scores[r] for r in op_both if r in off_scores], seed),
            "shortest": sp.summarize([float(short_eval["route_scores"][r]) for r in op_both], seed)}
        both_valid = [r for r in op_both if recs[r]["valid"] and op_recs[r]["valid"]]
        if both_valid:
            comps["opus_minus_thinking_on_both_valid"] = sp.paired(
                [op_scores[r] for r in both_valid], [scores[r] for r in both_valid], seed)

    settings = recs[ran[0]]["settings"]
    results = {
        "description": "DeepSeek deepseek-flash with thinking ON, direct stop ordering, same prompts as "
                       "the pilot, scored with the official rc-cli scorer as in score_pilot.py",
        "metric": plain["metric"],
        "settings": settings, "model_requested": recs[ran[0]]["model_requested"],
        "price_basis": recs[ran[0]]["price_basis"],
        "invalid_handling": "invalid answers are omitted from the proposal file, so official evaluate() "
                            "assigns the challenge's invalid-sequence score",
        "summary": summary, "comparisons": comps, "per_route": per,
        "baseline_rescore_check": "saved shortest tours re-scored identically to plain_baseline.json",
    }
    (args.results_dir / "pilot_deepseek_thinking.json").write_text(json.dumps(results, indent=1))

    # Markdown report
    s, ci = summary["score"], summary["score"]
    pd_ = summary["arm_minus_shortest_paired"]
    L = ["# Pilot rerun: DeepSeek with thinking on", "",
         "All numbers come from `results/pilot_deepseek_thinking.json`, built by "
         "`scripts/score_deepseek_thinking.py` from `runs/pilot/deepseek_thinking.jsonl` "
         "(calls made by `scripts/run_deepseek_thinking.py`).", "",
         "## What ran", "",
         f"- **Routes:** {len(per)} of the 50 pilot routes, the same prompts (`runs/pilot/prompts/`) and "
         "system prompt (`runs/pilot/system_prompt.txt`), unchanged.",
         f"- **Model:** `{results['model_requested']}` (returned: {', '.join(summary['model_returned'])}), "
         "the same model as the thinking-off run.",
         f"- **Settings:** `{json.dumps(settings)}`. Thinking enabled with effort `high` (DeepSeek's documented "
         "default). Temperature not sent: DeepSeek's docs say it has no effect in thinking mode. top_p left at "
         "its default. max_tokens 393,216 is the documented maximum (384K). The thinking-off run used "
         "temperature 0, max_tokens 4,096, thinking disabled.",
         "- **Parsing and validity:** `run_pilot.py`'s `parse_sequence`, unchanged, applied to the answer "
         "(`content`), not the reasoning. Our validity flag matched the official feasibility flag on every route.",
         "- **One call per route**, up to 4 in parallel, no retries for wrong answers, at most 2 retries for "
         f"network errors (calls that needed a retry: {summary['calls_with_network_retries']}).",
         "", "## Results (lower is better)", "",
         "| Arm | Routes | Mean score | Bootstrap 95% CI | Median | Valid |", "|---|---|---|---|---|---|",
         f"| DeepSeek, thinking on | {len(per)} | **{f4(s['mean'])}** | {f4(s['bootstrap_95_ci'][0])} – "
         f"{f4(s['bootstrap_95_ci'][1])} | {f4(s['median'])} | {summary['valid_count']} / {len(per)} |"]
    off = comps["thinking_on_minus_off"]["thinking_off"]
    L.append(f"| DeepSeek, thinking off, same routes | {len(both)} | {f4(off['mean'])} | "
             f"{f4(off['bootstrap_95_ci'][0])} – {f4(off['bootstrap_95_ci'][1])} | {f4(off['median'])} | "
             f"{comps['thinking_on_minus_off']['thinking_off_valid_count']} / {len(both)} |")
    sh = summary["shortest_on_same_routes"]
    L.append(f"| Shortest route, same routes | {len(per)} | {f4(sh['mean'])} | {f4(sh['bootstrap_95_ci'][0])} – "
             f"{f4(sh['bootstrap_95_ci'][1])} | {f4(sh['median'])} | {len(per)} / {len(per)} |")
    if "opus_minus_thinking_on_opus_routes" in comps:
        c = comps["opus_minus_thinking_on_opus_routes"]
        for name, key, vk in (("Opus", "opus", "opus_valid_count"),
                              ("DeepSeek thinking on, Opus's routes", "thinking_on", "thinking_on_valid_count")):
            x = c[key]
            L.append(f"| {name} | {c['n_routes']} | {f4(x['mean'])} | {f4(x['bootstrap_95_ci'][0])} – "
                     f"{f4(x['bootstrap_95_ci'][1])} | {f4(x['median'])} | {c[vk]} / {c['n_routes']} |")
        x = c["shortest"]
        L.append(f"| Shortest route, Opus's routes | {c['n_routes']} | {f4(x['mean'])} | "
                 f"{f4(x['bootstrap_95_ci'][0])} – {f4(x['bootstrap_95_ci'][1])} | {f4(x['median'])} | "
                 f"{c['n_routes']} / {c['n_routes']} |")
    L.append(f"\nMean score over valid answers only: {f4(summary['valid_only_mean_score'])}.\n")
    L += ["Paired differences (first minus second, negative = first better):", "",
          "| Comparison | Routes | Mean difference | Bootstrap 95% CI | First better / worse / ties | Sign test p |",
          "|---|---|---|---|---|---|"]

    def prow(name, d):
        lo, hi = d["bootstrap_95_ci"]
        p = d["exact_sign_test_two_sided_p"]
        return (f"| {name} | {d['n_routes']} | {sf4(d['mean_difference'])} | {sf4(lo)} – {sf4(hi)} | "
                f"{d['routes_first_better']} / {d['routes_second_better']} / {d['ties']} | "
                f"{'n/a' if p is None else f'{p:.3g}'} |")
    L.append(prow("Thinking on − shortest", pd_))
    L.append(prow("Thinking on − thinking off", comps["thinking_on_minus_off"]))
    if "opus_minus_thinking_on_opus_routes" in comps:
        L.append(prow("Opus − thinking on (Opus's routes)", comps["opus_minus_thinking_on_opus_routes"]))
    if "opus_minus_thinking_on_both_valid" in comps:
        L.append(prow("Opus − thinking on (both valid)", comps["opus_minus_thinking_on_both_valid"]))
    if "opus_minus_thinking_on_opus_routes" in comps:
        c = comps["opus_minus_thinking_on_opus_routes"]
        L.append(f"\nOpus comparison: `runs/pilot/opus.jsonl` had {c['opus_jsonl_ok_lines_at_scoring']} "
                 f"completed routes when this was scored; n = {c['n_routes']}. Opus scores here are computed by this script "
                 "with the same official scorer, not read from `results/pilot.json`.")
    L += ["", "## Validity", "",
          f"{summary['valid_count']} of {len(per)} valid. Finish reasons: "
          + ", ".join(f"`{k}` {v}" for k, v in finish.items()) + ".", ""]
    if reasons:
        L += ["| Invalid reason | Routes |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
    L.append(f"\n{summary['outputs_with_a_repeated_id']} of {len(per)} answers repeat at least one stop ID "
             "(thinking off: 46 of 50).")
    L.append(f"Thinking-off run, same routes: {comps['thinking_on_minus_off']['thinking_off_valid_count']} "
             f"valid. Routes valid in both runs: {len(comps['thinking_on_minus_off']['both_valid_routes'])}.")
    t = summary["totals"]
    w = summary["wall_seconds"]
    L += ["", "## Tokens, latency and spend", "",
          f"- **Spend:** ${summary['spend_usd_list_price']:.4f} at DeepSeek's peak list prices "
          "($0.006 / $0.30 / $1.20 per 1M cache-hit input / cache-miss input / output tokens; reasoning billed "
          "as output). Hard cap $5.00 (raised from $2.00 with Angad's approval after the 3-call projection of $3.81). Off-peak prices are half; failed network attempts, if any, are not counted.",
          f"- **Input tokens:** {t['input_tokens']:,} total (mean {summary['mean_input_tokens']:,.0f}), "
          f"{t['cache_hit_tokens']:,} of them cache hits.",
          f"- **Output tokens:** {t['output_tokens']:,} total, mean {summary['mean_output_tokens']:,.0f}, "
          f"range {summary['min_max_output_tokens'][0]:,} – {summary['min_max_output_tokens'][1]:,}. "
          f"Reasoning tokens {t['reasoning_tokens']:,} total, mean "
          f"{f4(summary['mean_reasoning_tokens']) if summary['mean_reasoning_tokens'] is None else format(summary['mean_reasoning_tokens'], ',.0f')}.",
          f"- **Answer length:** mean {t['output_chars'] / len(per):,.0f} characters "
          f"(reasoning text {t['reasoning_chars']:,} characters in total).",
          f"- **Wall time per call:** mean {w['mean']:.0f} s, median {w['median']:.0f} s, range "
          f"{w['min']:.0f} – {w['max']:.0f} s (up to 4 calls ran at once).",
          "", "## Per route", "",
          "| Route | Stops | Valid | Score | Shortest | Thinking off | Opus | Output tok | Reasoning tok | Wall s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in per:
        L.append(f"| {p['route_id'][8:16]} | {p['n_stops']} | {'yes' if p['valid'] else 'no'} | {f4(p['score'])} | "
                 f"{f4(p['shortest_score'])} | {f4(p['thinking_off_score'])} | {f4(p['opus_score'])} | "
                 f"{p['output_tokens']:,} | {p['reasoning_tokens'] if p['reasoning_tokens'] is not None else 'n/a'} | "
                 f"{p['wall_seconds']:.0f} |")
    L += ["", "## Caveats", "",
          "- One call per route; thinking mode samples (temperature has no effect), so a rerun could differ.",
          "- Reasoning effort `high` is DeepSeek's default, not `max`. Opus's effort may have been medium "
          "(see `pilot.md`), so the arms are closer on reasoning but still not identical.",
          "- The Opus comparison covers only the routes Opus had finished when this was scored.",
          "", "## How to rerun", "", "```",
          ".venv/bin/python -I scripts/run_deepseek_thinking.py --pilot-dir runs/pilot --sample-dir data/raw/sample --cap-usd 5.00 --workers 4",
          ".venv/bin/python -I scripts/score_deepseek_thinking.py --sample-dir data/raw/sample --pilot-dir runs/pilot --results-dir results",
          "```", ""]
    (args.results_dir / "pilot_deepseek_thinking.md").write_text("\n".join(L))
    print(json.dumps({"summary": summary, "comparisons": comps}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
