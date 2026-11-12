#!/usr/bin/env python3
"""PREREGISTRATION.md section 7 (reported regardless of outcome) and section 8 ("after the answers are opened"):
the DESCRIPTIVE analyses of the final test. Written and tested 2026-11-12 ~11:50 IST, BEFORE the lock, after
a pre-lock review found that no code produced the zone-order direction (Kendall tau) for every arm.
Nothing here can change a claim: claims come only from scripts/final_score.py's single run.

Gate (real run): PREREG_CONFIG locked with the matching sha256 (same check as final_score.py), AND
results/final_test.json exists with the sha256 logged in runs/final/LOCK_RECORD.json ("final_test_json_sha256"),
i.e. the answers were already opened once by the scorer. Runs once (refuses if results/final_descriptive.json exists).

What it computes (all labelled descriptive):
  1. Every arm: mean, median, bootstrap 95% CI of the mean (seed 20261111), on the routes it was scored on.
  2. Direction: Kendall tau between the arm's zone order and the driver's (both = first visit to each zone, the
     v0_model.zone_membership zones), mean / median / share below 0 / exact zone order / first zone right.
  3. Period: mean score before / inside / after the training dates, and by gap in days to the training window.
  4. Error analysis, FINAL vs OPUS-MODEL (500 sample) and FINAL vs each same-solver Opus run (all routes):
     per-route score difference against the tau difference; FINAL's 20 largest losses to OPUS-MODEL with
     station, stops, zones, gap and both taus; wins/losses by route size tercile.

  .venv/bin/python -I scripts/final_descriptive.py              # real, once, after final_score.py
  .venv/bin/python -I scripts/final_descriptive.py --test-fake  # on runs/final/score_selftest (3 val routes)
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_common as fc  # noqa: E402
import v0_decode  # noqa: E402
from final_score import check_config, open_sealed  # noqa: E402
from v0_model import zone_membership  # noqa: E402

SEED = 20261111
TRAIN_FIRST, TRAIN_LAST = dt.date(2018, 7, 19), dt.date(2018, 8, 26)
LOCK_RECORD = fc.FINAL / "LOCK_RECORD.json"


def first_visit_zones(route, stop_order):
    names, zone_of = zone_membership(route)
    pos = {str(s): i for i, s in enumerate(route["stop_ids"])}
    seen = []
    for sid in stop_order:
        k = int(zone_of[pos[str(sid)]])
        if k >= 0 and names[k] not in seen:
            seen.append(names[k])
    return seen


def driver_stops(answer):
    a = answer["actual"]
    return [s for s, _ in sorted(a.items(), key=lambda kv: kv[1])]


def summ(xs):
    xs = np.asarray(xs, float)
    lo, hi = v0_decode.bootstrap_ci(xs, SEED)
    return {"n": len(xs), "mean": float(xs.mean()), "median": float(np.median(xs)), "ci95": [lo, hi]}


def gap_days(d):
    if d < TRAIN_FIRST:
        return "before", (TRAIN_FIRST - d).days
    if d > TRAIN_LAST:
        return "after", (d - TRAIN_LAST).days
    return "inside", 0


GAP_BINS = [(0, 0, "0 (inside)"), (1, 14, "1-14"), (15, 30, "15-30"), (31, 60, "31-60"), (61, 10_000, "61+")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-fake", action="store_true")
    a = ap.parse_args()
    if a.test_fake:
        T = fc.FINAL / "score_selftest"
        cfg = check_config(T / "PREREG_CONFIG.json", T / "PREREGISTRATION.md")
        result_path, pred_dir, routes_dir = T / "result.json", T / "predictions", fc.ROOT / "data" / "processed" / "routes"
        answers, _ = open_sealed(T / "answers", real=False)
        out_json, out_md = T / "descriptive.json", T / "descriptive.md"
        index = {}
        final_arm, model_arm = "ours", "opus_a"
    else:
        cfg = check_config(fc.PREREG_CONFIG, fc.PREREG_DOC)
        result_path, pred_dir, routes_dir = fc.RESULTS / "final_test.json", fc.PRED, fc.EVAL_ROUTES
        if not result_path.exists():
            raise SystemExit("REFUSING: results/final_test.json does not exist; final_score.py runs first.")
        rec = json.load(open(LOCK_RECORD))
        if rec.get("final_test_json_sha256") != fc.sha256_file(result_path):
            raise SystemExit("REFUSING: final_test.json sha256 not logged in LOCK_RECORD.json (or changed).")
        out_json, out_md = fc.RESULTS / "final_descriptive.json", fc.RESULTS / "final_descriptive.md"
        if out_json.exists():
            raise SystemExit(f"REFUSING: {out_json} exists; this runs once.")
        answers, _ = open_sealed(fc.SEALED, real=True)
        index = {r["route_id"]: r for r in fc.eval_index()}
        final_arm, model_arm = "v1_exact", "opus_model_repaired_exact"
    res = json.load(open(result_path))
    per = res["per_route"]
    man = json.load(open(pred_dir / "MANIFEST.json"))
    arms = [x for x in per]

    # 1. means, medians, CIs
    summary = {arm: summ(list(per[arm].values())) for arm in arms}

    # 2. direction
    route_cache, driver_z, tau = {}, {}, {}
    def route(r):
        if r not in route_cache:
            with np.load(routes_dir / f"{r}.npz", allow_pickle=False) as d:
                route_cache[r] = {k: d[k] for k in d.files if k != "actual_order"}
        return route_cache[r]
    direction = {}
    for arm in arms:
        p = pred_dir / f"{arm}.json"
        if fc.sha256_file(p) != man[arm]["sha256"]:
            raise SystemExit(f"REFUSING: predictions for {arm} changed")
        orders = json.load(open(p))["orders"]
        t = {}
        for r in per[arm]:
            if r not in orders:
                continue
            rt = route(r)
            if r not in driver_z:
                driver_z[r] = first_visit_zones(rt, driver_stops(answers[r]))
            try:
                z = first_visit_zones(rt, orders[r])
            except (KeyError, IndexError):
                continue                                   # malformed prediction: no direction
            if sorted(z) != sorted(driver_z[r]):
                continue
            t[r] = (v0_decode.kendall_tau(z, driver_z[r]), z == driver_z[r], z[0] == driver_z[r][0])
        tau[arm] = {r: v[0] for r, v in t.items()}
        if t:
            xs = np.array([v[0] for v in t.values()])
            direction[arm] = {"routes": len(t), "tau_mean": float(xs.mean()), "tau_median": float(np.median(xs)),
                              "share_tau_below_0": float((xs < 0).mean()), "exact_zone_order": int(sum(v[1] for v in t.values())),
                              "first_zone_right": float(np.mean([v[2] for v in t.values()]))}

    # 3. period and gap (real run only: the eval index has the dates)
    period = {}
    if index:
        for arm in arms:
            rows = {"before": [], "inside": [], "after": []}
            gaps = {lab: [] for *_, lab in GAP_BINS}
            for r, s in per[arm].items():
                which, g = gap_days(dt.date.fromisoformat(index[r]["date"]))
                rows[which].append(s)
                gaps[next(lab for lo, hi, lab in GAP_BINS if lo <= g <= hi)].append(s)
            period[arm] = {"period": {k: (summ(v) if v else None) for k, v in rows.items()},
                           "gap_days": {k: ({"n": len(v), "mean": float(np.mean(v))} if v else None) for k, v in gaps.items()}}

    # 4. error analysis
    def versus(b):
        ids = sorted(set(per[final_arm]) & set(per[b]))
        d = {r: per[final_arm][r] - per[b][r] for r in ids}
        both = [r for r in ids if r in tau.get(final_arm, {}) and r in tau.get(b, {})]
        dt_ = [tau[final_arm][r] - tau[b][r] for r in both]
        corr = float(np.corrcoef([d[r] for r in both], dt_)[0, 1]) if len(both) > 2 and np.std(dt_) > 0 else None
        nz = {r: len(driver_z.get(r, [])) for r in ids}
        terc = np.percentile(list(nz.values()), [33.3, 66.7]) if ids else [0, 0]
        size = {}
        for lab, f in (("small", lambda n: n <= terc[0]), ("mid", lambda n: terc[0] < n <= terc[1]), ("large", lambda n: n > terc[1])):
            rr = [r for r in ids if f(nz[r])]
            size[lab] = {"n": len(rr), "final_better": sum(d[r] < 0 for r in rr), "final_worse": sum(d[r] > 0 for r in rr),
                         "mean_diff": float(np.mean([d[r] for r in rr])) if rr else None}
        worst = sorted(ids, key=lambda r: -d[r])[:20]
        return {"n": len(ids), "corr_score_diff_vs_tau_diff": corr,
                "tau_final_minus_b_mean": float(np.mean(dt_)) if dt_ else None, "by_zone_count_tercile": size,
                "final_largest_losses": [{"route_id": r, "diff": d[r], "final": per[final_arm][r], "b": per[b][r],
                                          "tau_final": tau.get(final_arm, {}).get(r), "tau_b": tau.get(b, {}).get(r),
                                          "zones": nz[r], **({"station": index[r]["station"], "n_stops": index[r]["n_stops"],
                                                              "period_gap": gap_days(dt.date.fromisoformat(index[r]["date"]))} if index else {})}
                                         for r in worst]}
    errors = {}
    for b in [model_arm] + [x for x in arms if x.startswith("samesolver_")]:
        if b in per and final_arm in per:
            errors[f"{final_arm} vs {b}"] = versus(b)

    out = {"label": "DESCRIPTIVE (PREREGISTRATION.md sections 7-8). Nothing here changes a claim.",
           "source_result_sha256": fc.sha256_file(result_path), "summary": summary, "direction": direction,
           "period": period, "error_analysis": errors}
    json.dump(out, open(out_json, "w"), indent=1)
    L = ["# Final test, descriptive analyses" + (" (SELF-TEST)" if a.test_fake else ""), "",
         "Made once by `scripts/final_descriptive.py` after `final_score.py`. **Descriptive: nothing here changes a claim.** Lower score is better; tau +1 = driver's zone order, -1 = reversed.", "",
         "| Arm | Routes | Mean [95% CI] | Median | Tau mean | Tau < 0 | Exact zone order | First zone right |", "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        s, dr = summary[arm], direction.get(arm)
        L.append(f"| {arm} | {s['n']} | {s['mean']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] | {s['median']:.4f} | "
                 + (f"{dr['tau_mean']:+.3f} | {dr['share_tau_below_0']:.0%} | {dr['exact_zone_order']} | {dr['first_zone_right']:.0%} |" if dr else "— | — | — | — |"))
    if period:
        L += ["", "| Arm | Before | Inside | After | " + " | ".join(f"gap {lab}" for *_, lab in GAP_BINS) + " |", "|---|---|---|---|" + "---|" * len(GAP_BINS)]
        for arm, p in period.items():
            cell = lambda v: f"{v['mean']:.4f} (n {v['n']})" if v else "—"
            L.append(f"| {arm} | {cell(p['period']['before'])} | {cell(p['period']['inside'])} | {cell(p['period']['after'])} | "
                     + " | ".join(cell(p["gap_days"][lab]) for *_, lab in GAP_BINS) + " |")
    for k, e in errors.items():
        L += ["", f"## Error analysis: {k} ({e['n']} routes)", "",
              f"- Correlation of per-route score difference with tau difference: {e['corr_score_diff_vs_tau_diff']}",
              f"- Mean tau difference (FINAL − other): {e['tau_final_minus_b_mean']}", "",
              "| Zone-count tercile | Routes | FINAL better / worse | Mean diff |", "|---|---|---|---|"]
        md = lambda x: "—" if x is None else f"{x:+.4f}"
        L += [f"| {lab} | {v['n']} | {v['final_better']} / {v['final_worse']} | {md(v['mean_diff'])} |"
              for lab, v in e["by_zone_count_tercile"].items()]
        L += ["", "FINAL's largest losses:", "", "| Route | Diff | FINAL | Other | Tau FINAL | Tau other | Zones |", "|---|---|---|---|---|---|---|"]
        f3 = lambda x: "—" if x is None else f"{x:+.2f}"
        L += [f"| {w['route_id'][-12:]} | {w['diff']:+.4f} | {w['final']:.4f} | {w['b']:.4f} | {f3(w['tau_final'])} | {f3(w['tau_b'])} | {w['zones']} |"
              for w in e["final_largest_losses"][:10]]
    Path(out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
