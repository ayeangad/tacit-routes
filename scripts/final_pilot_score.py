#!/usr/bin/env python3
"""PREREGISTRATION.md section 7: "v0-hist on the 50 pilot routes, next to the stored pilot Opus arms
(n = 50, descriptive)". Added 2026-11-12 ~03:05 IST because nothing in the harness produced it.

Runs ONLY after the lock (same gate as scripts/final_score.py: PREREG_CONFIG locked and the sha256 of
PREREGISTRATION.md matching) and only once (refuses if results/final_pilot.json exists). Until then the 50
pilot routes stay unscored for our models, as they have been all project.

Predictions: runs/final/predictions/pilot_{v0hist,v1,v1_exact}.json (made overnight from inputs, sha256 checked
against MANIFEST.json). Answers: the pilot routes' own driver order in data/processed/routes (the pilot split was
never used to train, tune or select anything). Scoring: v0_decode.official_scores, the path every val number used.
Opus arms: the stored per-route scores in results/pilot_history.json (Opus + 3 past routes; Opus direct;
shortest) and results/pilot_zone.json (Opus zone-level, fallback view). Descriptive only: no claim rests on it.

  .venv/bin/python -I scripts/final_pilot_score.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_common as fc  # noqa: E402
import v0_decode  # noqa: E402
from final_score import check_config  # noqa: E402

ARMS = ("pilot_v0hist", "pilot_v1", "pilot_v1_exact")


def paired(a, b, ids):
    d = np.array([a[r] - b[r] for r in ids])
    lo, hi = v0_decode.bootstrap_ci(d, fc.BOOT_SEED)
    w, l_ = int((d < 0).sum()), int((d > 0).sum())
    return {"n": len(ids), "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "a_better": w, "a_worse": l_,
            "tied": len(ids) - w - l_, "sign_p": float(v0_decode.sign_test(w, l_))}


def main():
    check_config(fc.PREREG_CONFIG, fc.PREREG_DOC)
    out_json, out_md = fc.RESULTS / "final_pilot.json", fc.RESULTS / "final_pilot.md"
    if out_json.exists():
        raise SystemExit(f"REFUSING: {out_json} exists; this runs once.")
    splits = json.load(open(fc.ROOT / "data" / "processed" / "splits.json"))
    ids = sorted(r["route_id"] for r in splits["pilot"])
    assert len(ids) == 50
    man = json.load(open(fc.PRED / "MANIFEST.json"))
    invalid = json.load(open(v0_decode.INVALID))
    ours = {}
    for arm in ARMS:
        p = fc.PRED / f"{arm}.json"
        if fc.sha256_file(p) != man[arm]["sha256"]:
            raise SystemExit(f"REFUSING: {arm} changed after it was written")
        orders = json.load(open(p))["orders"]
        assert sorted(orders) == ids, f"{arm}: route set differs from the 50 pilot routes"
        s, feas = v0_decode.official_scores(orders, invalid)
        assert all(feas.values()), f"{arm}: infeasible route"
        ours[arm] = s
    hist = {x["route_id"]: x for x in json.load(open(fc.ROOT / "results" / "pilot_history.json"))["per_route"]}
    zone = {x["route_id"]: x for x in json.load(open(fc.ROOT / "results" / "pilot_zone.json"))["per_route"]}
    assert sorted(hist) == ids and sorted(zone) == ids
    theirs = {"Opus direct (pilot.json arm)": {r: hist[r]["direct_score"] for r in ids},
              "Opus + 3 past routes": {r: hist[r]["score"] for r in ids},
              "Opus zone-level (fallback view)": {r: zone[r]["score_fallback"] for r in ids},
              "Shortest route (OR-Tools)": {r: hist[r]["shortest_score"] for r in ids}}
    means = {k: float(np.mean(list(v.values()))) for k, v in {**ours, **theirs}.items()}
    comps = {f"{a} − {b}": paired(ours[a], theirs[b], ids) for a in ARMS for b in theirs}
    out = {"rule": "PREREGISTRATION.md section 7 (descriptive, n = 50)", "means": means, "comparisons": comps,
           "per_route": {**ours, **theirs}, "sources": ["runs/final/predictions/pilot_*.json", "results/pilot_history.json",
                                                      "results/pilot_zone.json", "data/processed/routes (pilot answers)"]}
    json.dump(out, open(out_json, "w"), indent=1)
    lines = ["# Our models on the 50 pilot routes (descriptive; written after the lock)", "",
             "Made once by `scripts/final_pilot_score.py`. n = 50, so intervals are wide. Opus numbers are the stored pilot runs. Lower is better.", "",
             "| Arm | Mean |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in means.items()] + [
             "", "| Comparison (a − b) | Mean diff [95% CI] | a better / worse / tied | Sign p |", "|---|---|---|---|"] + [
             f"| {k} | {c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | {c['a_better']} / {c['a_worse']} / {c['tied']} | {c['sign_p']:.2g} |"
             for k, c in comps.items()]
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
