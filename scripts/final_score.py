#!/usr/bin/env python3
"""FINAL TEST, STAGE 2: the ONLY code that opens data/eval_frozen/raw/score/. Run it ONCE.

Refuses to run unless runs/final/PREREG_CONFIG.json has "locked": true AND its
"preregistration_sha256" equals the sha256 of PREREGISTRATION.md on disk. Before opening the sealed
files it checks their sha256 against data/eval_frozen/MANIFEST.json (recorded when they were
downloaded and never parsed).

Scoring: the official, unmodified rc-cli evaluate() (scripts/official_rc_cli/score.py), called in
chunks of 20 routes exactly as scripts/v0_decode.official_scores does for every val number (travel
times from the npz rounded to 0.1 s; missing or malformed proposals get the challenge's invalid
score through evaluate()'s own isinvalid check). Statistics as on val: paired differences, 10,000
bootstrap resamples of routes (seed 20261111), exact sign test, Holm within the configured family.

Every comparison is reported twice: on all routes, and excluding routes at a station with no train
history (DBO6). Each Opus fair-fight system's two sandbox repeats are reported separately with their
spread; the arm score is their per-route mean ("mean:a+b" in the config). The OPUS-MODEL arm has a
REPAIRED (primary) and a STRICT view, both listed as arms in the config.

Everything that decides the outcome (arms, primary comparison, secondary family, win rules) is read
from PREREG_CONFIG.json; nothing is chosen in this script.

BOTH VIEWS (fix 2026-11-12, before the lock; PREREGISTRATION.md section 6): a comparison is a win only
if it is a win on all routes AND excluding the no-history station, each view with its own Holm
adjustment over the secondary family. Claims use that combined flag; each view's flag is also saved.
NON-INFERIORITY (PREREGISTRATION.md section 6), if configured: upper CI end of the primary comparison
below the margin in BOTH views AND the cost ratio (b's mean CLI USD per route / a's CPU time priced
at the configured USD per CPU-hour) at least the configured minimum. Reported always; it is the
fallback claim only when the primary win fails.
OPTIONAL ARMS (descriptive only, e.g. OPUS-MODEL at default effort): scored if their predictions
exist, skipped with a note otherwise. They are in no comparison that decides anything.

  .venv/bin/python -I scripts/final_score.py                      # the real, single run
  .venv/bin/python -I scripts/final_score.py --test-fake          # self-test on 3 VAL routes only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_common as fc  # noqa: E402
import v0_decode  # noqa: E402

RESULTS = fc.RESULTS


# -------------------------------------------------------------------------------- gatekeeping
def check_config(cfg_path, doc_path):
    cfg = json.load(open(cfg_path))
    if cfg.get("locked") is not True:
        raise SystemExit(f"REFUSING: {cfg_path} is not locked.")
    actual = fc.sha256_file(doc_path)
    if cfg.get("preregistration_sha256") != actual:
        raise SystemExit(f"REFUSING: {doc_path} sha256 {actual} != config {cfg.get('preregistration_sha256')}")
    return cfg


def open_sealed(answers_dir, real):
    files = {"actual": answers_dir / "eval_actual_sequences.json",
             "invalid": answers_dir / "eval_invalid_sequence_scores.json"}
    if real:
        man = json.load(open(fc.EVAL_DIR / "MANIFEST.json"))
        expected = json.dumps(man)
        for key, p in files.items():
            h = fc.sha256_file(p)
            if h not in expected:
                raise SystemExit(f"REFUSING: sha256 of {p.name} ({h}) not in data/eval_frozen/MANIFEST.json")
    return json.load(open(files["actual"])), json.load(open(files["invalid"]))


# ------------------------------------------------------------------------------------ scoring
_CTX = {}


def _init(answers, invalid, routes_dir):
    _CTX.update(answers=answers, invalid=invalid, routes_dir=Path(routes_dir))


def _score_chunk(job):
    """v0_decode.score_chunk with the driver order taken from the sealed file and the route from
    the eval npz (same four JSON files, same official evaluate())."""
    if v0_decode._SCORER is None:
        v0_decode._SCORER = v0_decode.load_scorer()
    orders = job
    tmp = fc.FINAL / "score_tmp" / str(os.getpid())
    tmp.mkdir(parents=True, exist_ok=True)
    part = sorted(orders)
    routes = {}
    for r in part:
        with np.load(_CTX["routes_dir"] / f"{r}.npz", allow_pickle=False) as d:
            routes[r] = {k: d[k] for k in d.files}
    files = {
        "actual": {r: _CTX["answers"][r] for r in part},
        "proposed": {r: v0_decode.as_proposal(orders[r]) for r in part},
        "costs": {r: v0_decode.cost_dict(routes[r]) for r in part},
        "invalid": {r: _CTX["invalid"][r] for r in part},
    }
    for name, data in files.items():
        with open(tmp / f"{name}.json", "w") as f:
            json.dump(data, f)
    out = v0_decode._SCORER.evaluate(str(tmp / "actual.json"), str(tmp / "proposed.json"),
                                     str(tmp / "costs.json"), str(tmp / "invalid.json"))
    for name in files:
        (tmp / f"{name}.json").unlink(missing_ok=True)
    return {r: (float(out["route_scores"][r]), bool(out["route_feasibility"][r])) for r in part}


def score_arm(orders, ids, pool):
    """Every route in ids is scored; a route the arm did not predict gets an empty proposal, which
    evaluate()'s isinvalid() turns into the challenge's invalid score."""
    full = {r: (orders.get(r) or []) for r in ids}
    jobs = [{r: full[r] for r in ids[k:k + 20]} for k in range(0, len(ids), 20)]
    parts = pool.map(_score_chunk, jobs, chunksize=1)
    merged = {r: v for p in parts for r, v in p.items()}
    return {r: v[0] for r, v in merged.items()}, {r: v[1] for r, v in merged.items()}


# -------------------------------------------------------------------------------------- stats
def resolve(name, scores):
    """'arm' or 'mean:armA+armB' (per-route mean, e.g. the two repeats of one Opus system)."""
    if name.startswith("mean:"):
        parts = name[5:].split("+")
        return {r: float(np.mean([scores[p][r] for p in parts])) for r in scores[parts[0]]}
    return scores[name]


def paired(a, b, ids):
    d = np.array([a[r] - b[r] for r in ids])
    lo, hi = v0_decode.bootstrap_ci(d, fc.BOOT_SEED)
    w, l_ = int((d < 0).sum()), int((d > 0).sum())
    return {"n": len(ids), "mean_a": float(np.mean([a[r] for r in ids])), "mean_b": float(np.mean([b[r] for r in ids])),
            "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "a_better": w, "a_worse": l_,
            "tied": len(ids) - w - l_, "sign_p": float(v0_decode.sign_test(w, l_))}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def win(c, holm_p=None, alpha=0.05):
    """A comparison is a WIN for 'a' iff mean_diff < 0 and the bootstrap CI upper bound < 0
    (and, where a Holm-adjusted sign-test p exists, it is < alpha)."""
    ok = c["mean_diff"] < 0 and c["ci95"][1] < 0
    return ok and (holm_p is None or holm_p < alpha)


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-fake", action="store_true", help="self-test on 3 val routes; never touches raw/score")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    import multiprocessing as mp

    if a.test_fake:
        T = fc.FINAL / "score_selftest"
        cfg = check_config(T / "PREREG_CONFIG.json", T / "PREREGISTRATION.md")
        answers, invalid = open_sealed(T / "answers", real=False)
        routes_dir, pred_dir, out_json, out_md = fc.ROOT / "data" / "processed" / "routes", T / "predictions", T / "result.json", T / "result.md"
        all_ids = sorted(answers)
        assert "eval_frozen" not in str(T)
    else:
        cfg = check_config(fc.PREREG_CONFIG, fc.PREREG_DOC)
        answers, invalid = open_sealed(fc.SEALED, real=True)
        routes_dir, pred_dir, out_json, out_md = fc.EVAL_ROUTES, fc.PRED, RESULTS / "final_test.json", RESULTS / "final_test.md"
        RESULTS.mkdir(parents=True, exist_ok=True)
        all_ids = fc.eval_ids()
        assert sorted(answers) == all_ids, "sealed answers and eval index disagree"
        if out_json.exists():
            raise SystemExit(f"REFUSING: {out_json} exists; the final test runs once.")

    sample = json.load(open(fc.ROOT / cfg["primary_sample_file"]))["route_ids"] if not a.test_fake else all_ids
    man = json.load(open(pred_dir / "MANIFEST.json"))
    t0 = time.time()
    scores, feasible, missing = {}, {}, {}
    with mp.Pool(a.workers, initializer=_init, initargs=(answers, invalid, str(routes_dir))) as pool:
        optional = [x for x in cfg.get("optional_arms", []) if x in man]
        skipped_optional = [x for x in cfg.get("optional_arms", []) if x not in man]
        for arm in cfg["arms"] + optional:
            p = pred_dir / f"{arm}.json"
            if fc.sha256_file(p) != man[arm]["sha256"]:
                raise SystemExit(f"REFUSING: predictions for {arm} changed after stage 1")
            orders = json.load(open(p))["orders"]
            ids = sample if (arm in cfg.get("sample_only_arms", []) or arm in optional) else all_ids
            missing[arm] = sorted(set(ids) - set(orders))
            scores[arm], feasible[arm] = score_arm(orders, ids, pool)
            print(f"{arm}: mean {np.mean(list(scores[arm].values())):.4f} on {len(ids)} routes "
                  f"(missing {len(missing[arm])}, infeasible {sum(not v for v in feasible[arm].values())})", flush=True)

    nohist = set(fc.no_history_ids()) if not a.test_fake else set()
    comps = {}
    for c in cfg["comparisons"]:
        if c.get("optional") and not all(x in scores for x in (c["a"], c["b"])):
            continue
        ids = sample if c["routes"] == "sample" else all_ids
        A, B = resolve(c["a"], scores), resolve(c["b"], scores)
        comps[c["name"]] = {**c, **paired(A, B, ids),
                            "excluding_no_history_station": paired(A, B, [r for r in ids if r not in nohist])}
    reps = {}
    for arm in cfg["arms"]:
        if arm.endswith("_rep1") and arm[:-1] + "2" in scores:
            r1, r2 = arm, arm[:-1] + "2"
            ids_r = sorted(set(scores[r1]) & set(scores[r2]))  # option B: repeat 2 covers the 500 sample only
            d = paired(scores[r1], scores[r2], ids_r)
            reps[arm[:-5]] = {"rep1_mean": d["mean_a"], "rep2_mean": d["mean_b"], "rep1_minus_rep2": d,
                              "mean_abs_route_diff": float(np.mean([abs(scores[r1][r] - scores[r2][r]) for r in ids_r])),
                              "identical_routes": int(sum(scores[r1][r] == scores[r2][r] for r in ids_r))}
    fam = [n for n, c in comps.items() if c["role"] == "secondary"]
    for n, p in zip(fam, holm([comps[n]["sign_p"] for n in fam])):
        comps[n]["holm_p"] = p
    for n, p in zip(fam, holm([comps[n]["excluding_no_history_station"]["sign_p"] for n in fam])):
        comps[n]["excluding_no_history_station"]["holm_p"] = p
    for n, c in comps.items():
        e = c["excluding_no_history_station"]
        c["win_for_a_all_routes"] = win(c, c.get("holm_p"))
        e["win_for_a"] = win(e, e.get("holm_p"))
        c["views_agree"] = c["win_for_a_all_routes"] == e["win_for_a"]
        c["win_for_a"] = c["win_for_a_all_routes"] and e["win_for_a"]  # both views must hold
    ni = None
    nic = cfg.get("non_inferiority")
    if nic and nic["comparison"] in comps:
        c = comps[nic["comparison"]]
        e = c["excluding_no_history_station"]
        ids_s = sample if c["routes"] == "sample" else all_ids
        b_per = man[nic["b_cost_arm"]]["meta"].get("per_route", {})
        # Deciding cost (check-in 1): answered routes only, the answering call only, so refused attempts are
        # never charged to Opus (the kinder direction). The all-attempts view is reported beside it.
        b_usd = [b_per[r]["cost_usd"] for r in ids_s if r in b_per and not b_per[r].get("refused") and b_per[r].get("cost_usd") is not None]
        b_usd_all = [b_per[r].get("cost_usd_all_attempts", b_per[r].get("cost_usd")) for r in ids_s
                     if r in b_per and b_per[r].get("cost_usd_all_attempts", b_per[r].get("cost_usd")) is not None]
        b_wall = [b_per[r]["wall_s"] for r in ids_s if r in b_per and b_per[r].get("wall_s") is not None]
        a_cpu = man[nic["a_cost_arm"]]["meta"].get("cpu_s_per_route_mean")
        b_usd_mean = float(np.mean(b_usd)) if b_usd else None
        a_usd = a_cpu / 3600.0 * nic["cpu_usd_per_hour"] if a_cpu is not None else None
        ratio = (b_usd_mean / a_usd) if (b_usd_mean and a_usd) else None
        acc_ok = c["ci95"][1] < nic["margin"] and e["ci95"][1] < nic["margin"]
        cost_ok = ratio is not None and ratio >= nic["min_cost_ratio"]
        ni = {**nic, "upper_all": c["ci95"][1], "upper_excluding_no_history": e["ci95"][1], "accuracy_within_margin_both_views": acc_ok,
              "b_usd_per_route_mean": b_usd_mean, "b_calls_with_cost": len(b_usd),
              "b_usd_per_route_mean_incl_refused_attempts": float(np.mean(b_usd_all)) if b_usd_all else None,
              "b_routes_refused": sum(1 for r in ids_s if r in b_per and b_per[r].get("refused")), "b_wall_s_per_route_mean": float(np.mean(b_wall)) if b_wall else None,
              "a_cpu_s_per_route_mean": a_cpu, "a_usd_per_route_imputed": a_usd, "cost_ratio": ratio, "cost_ok": cost_ok,
              "holds": acc_ok and cost_ok, "applies_as_claim": not c["win_for_a"]}
    groups = {g: {"members": m["members"], "rule": m["rule"],
                  "claim_holds": all(comps[x]["win_for_a"] for x in m["members"]) if m["rule"] == "all"
                  else any(comps[x]["win_for_a"] for x in m["members"])}
              for g, m in cfg.get("claims", {}).items()}
    out = {"config": cfg, "seconds": time.time() - t0, "comparisons": comps, "claims": groups,
           "opus_repeat_spread": reps, "no_history_routes": sorted(nohist), "non_inferiority": ni,
           "optional_arms_skipped": skipped_optional,
           "arm_means": {k: float(np.mean(list(v.values()))) for k, v in scores.items()},
           "missing_routes": {k: len(v) for k, v in missing.items()},
           "infeasible_routes": {k: int(sum(not x for x in v.values())) for k, v in feasible.items()},
           "cost_latency": {arm: {k: v for k, v in man[arm]["meta"].items() if k not in ("zone_orders",)}
                            for arm in cfg["arms"] + optional},
           "per_route": scores}
    json.dump(out, open(out_json, "w"), indent=1)
    lines = ["# Final test result" + (" (SELF-TEST on 3 val routes, not the real test)" if a.test_fake else ""), "",
             f"Config: `{'score_selftest/' if a.test_fake else 'runs/final/'}PREREG_CONFIG.json` (locked, preregistration sha256 "
             f"{cfg['preregistration_sha256'][:12]}…). Lower is better.", "", "| Arm | Mean | Routes | Missing |", "|---|---|---|---|"]
    lines += [f"| {k} | {out['arm_means'][k]:.4f} | {len(scores[k])} | {out['missing_routes'][k]} |" for k in cfg["arms"] + optional]
    if skipped_optional:
        lines += ["", f"Optional (descriptive) arms with no predictions, skipped: {', '.join(skipped_optional)}."]
    lines += ["", "| Comparison (a − b) | Role | Routes | Mean diff [95% CI] | a better / worse | Sign p | Holm p | Win, all routes |",
              "|---|---|---|---|---|---|---|---|"]
    for n, c in comps.items():
        lines.append(f"| {n}: {c['a']} − {c['b']} | {c['role']} | {c['n']} | {c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | "
                     f"{c['a_better']} / {c['a_worse']} | {c['sign_p']:.2g} | {c.get('holm_p', float('nan')):.2g} | {'YES' if c['win_for_a_all_routes'] else 'no'} |")
    lines += ["", "Same comparisons excluding routes at a station with no train history (DBO6), and the combined verdict "
              "(a win needs BOTH views):", "",
              "| Comparison | Routes | Mean diff [95% CI] | a better / worse | Holm p | Win, excl. | Views agree | WIN (both) |",
              "|---|---|---|---|---|---|---|---|"]
    for n, c in comps.items():
        e = c["excluding_no_history_station"]
        lines.append(f"| {n} | {e['n']} | {e['mean_diff']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {e['a_better']} / {e['a_worse']} | "
                     f"{e.get('holm_p', float('nan')):.2g} | {'YES' if e['win_for_a'] else 'no'} | {'yes' if c['views_agree'] else 'NO (not established)'} | "
                     f"{'YES' if c['win_for_a'] else 'no'} |")
    if ni:
        lines += ["", f"Non-inferiority ({ni['comparison']}, margin +{ni['margin']}, cost ratio ≥ {ni['min_cost_ratio']}×; "
                  f"{'the fallback claim, since the primary win failed' if ni['applies_as_claim'] else 'reported only, since the primary win holds'}):",
                  f"- CI upper end: all routes {ni['upper_all']:+.4f}, excluding no-history station {ni['upper_excluding_no_history']:+.4f} "
                  f"-> {'within' if ni['accuracy_within_margin_both_views'] else 'NOT within'} the margin in both views",
                  f"- Cost per route: b {ni['b_usd_per_route_mean']} USD (CLI cost field, answering call only, {ni['b_calls_with_cost']} calls; "
                  f"{ni['b_usd_per_route_mean_incl_refused_attempts']} USD including retried and refused attempts; {ni['b_routes_refused']} routes refused; "
                  f"wall {ni['b_wall_s_per_route_mean']} s); "
                  f"a {ni['a_cpu_s_per_route_mean']} CPU-s = {ni['a_usd_per_route_imputed']} USD at {ni['cpu_usd_per_hour']} USD per CPU-hour; "
                  f"ratio {ni['cost_ratio']} -> {'ok' if ni['cost_ok'] else 'NOT ok'}",
                  f"- **Non-inferiority {'HOLDS' if ni['holds'] else 'does not hold'}**"]
    if reps:
        lines += ["", "Opus fair-fight systems, the two sandbox repeats (solver noise):", "",
                  "| System | Repeat 1 | Repeat 2 | rep1 − rep2 [95% CI] | Mean abs route diff | Identical routes |", "|---|---|---|---|---|---|"]
        for k, v in reps.items():
            d = v["rep1_minus_rep2"]
            lines.append(f"| {k} | {v['rep1_mean']:.4f} | {v['rep2_mean']:.4f} | {d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                         f"{v['mean_abs_route_diff']:.4f} | {v['identical_routes']} |")
    lines += ["", "Claims:"] + [f"- **{g}** ({v['rule']} of {', '.join(v['members'])}): {'HOLDS' if v['claim_holds'] else 'does not hold'}"
                                for g, v in groups.items()]
    Path(out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
