#!/usr/bin/env python3
"""Score the adopted v1 on all 300 validation routes (competition history rule = all train routes).

v1 = 5-seed "main" ensemble, beam 5 + travel lambda 0.5 (fixed; results/v1_adoption.md). Two in-zone
solvers from the SAME zone orders: greedy descent (as every earlier v0/v1 number) and exact
(scripts/exact_zone.py; its adoption is decided separately by results/exact_solver_gate.*).
Paired comparisons as in results/val_comparison.md: mean difference with bootstrap 95% CI
(10,000, seed 20261111), wins / losses / ties, exact sign test.

Opponents' per-route scores come from stored files:
  v0-hist, Opus run 2, jpt: results/val_comparison.json (per_route)
  Opus runs 3, 4:           results/opus_fight_run{3,4}.json (per_route, opus_score)
  Opus runs 5, 6:           results/opus_fight_run{5,6}.json if present (informed + literature)

  .venv/bin/python -I scripts/v1_val.py --workers 3            (writes results/v1_val.json and .md)
  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/v1_val.py --limit 3 --workers 1 --no-write
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v1_model  # noqa: E402
from exact_zone import stops_from_zone_order_exact  # noqa: E402
from v0_decode import INVALID, bootstrap_ci, official_scores, sign_test  # noqa: E402
from v0_model import SPLITS, load_npz  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
SEED = 20261111
H = "v0:hist_high_medium/beam5+travel(lambda=0.5)"


def job(rid):
    j, greedy, names = v1_model.decode_job((rid, "v1", (), v1_model.LAMBDA, v1_model.BEAM))
    exact = stops_from_zone_order_exact(load_npz(rid), names)
    return rid, greedy, exact, names


def paired(a, b, ids):
    d = np.array([a[r] - b[r] for r in ids])
    lo, hi = bootstrap_ci(d, SEED)
    w, l = int((d < 0).sum()), int((d > 0).sum())
    return {"first_mean": float(np.mean([a[r] for r in ids])), "second_mean": float(np.mean([b[r] for r in ids])),
            "mean_diff": float(d.mean()), "ci95": [lo, hi], "first_better": w, "first_worse": l,
            "tied": len(ids) - w - l, "sign_p": sign_test(w, l)}


def opponents(ids):
    pr = {r["route_id"]: r for r in json.load(open(RES / "val_comparison.json"))["per_route"]}
    opp = {"v0-hist (pre-registered)": {r: pr[r][H] for r in ids},
           "Opus run 2 (blind)": {r: pr[r]["opus_run2"] for r in ids},
           "jpt (2021 winner)": {r: pr[r]["jpt"] for r in ids}}
    for n, label in ((3, "blind"), (4, "blind"), (5, "informed + literature"), (6, "informed + literature")):
        f = RES / f"opus_fight_run{n}.json"
        if f.exists():
            s = {x["route_id"]: x["opus_score"] for x in json.load(open(f))["per_route"]}
            opp[f"Opus run {n} ({label})"] = {r: s[r] for r in ids}
    return opp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    splits = json.load(open(SPLITS))
    ids = sorted(r["route_id"] for r in splits["val"])
    assert len(ids) == 300
    if a.limit:
        ids = ids[:a.limit]
    invalid = json.load(open(INVALID))
    t0 = time.time()
    with mp.Pool(a.workers, initializer=v1_model.set_ensemble, initargs=("main", v1_model.SEEDS)) as pool:
        done = pool.map(job, ids, chunksize=2)
        greedy = {r: g for r, g, _, _ in done}
        exact = {r: e for r, _, e, _ in done}
        s_g, f_g = official_scores(greedy, invalid, pool)
        s_e, f_e = official_scores(exact, invalid, pool)
    assert all(f_g.values()) and all(f_e.values())
    opp = opponents(ids)
    comps = {}
    for mine, sc in (("v1 greedy", s_g), ("v1 exact", s_e)):
        for name, o in opp.items():
            comps[f"{mine} − {name}"] = paired(sc, o, ids)
    comps["v1 exact − v1 greedy"] = paired(s_e, s_g, ids)
    out = {"description": "adopted v1 on the 300 val routes; competition history rule", "n_routes": len(ids),
           "means": {"v1 greedy": float(np.mean(list(s_g.values()))), "v1 exact": float(np.mean(list(s_e.values()))),
                     **{k: float(np.mean(list(v.values()))) for k, v in opp.items()}},
           "comparisons": comps, "seconds": time.time() - t0,
           "per_route": [{"route_id": r, "v1_greedy": s_g[r], "v1_exact": s_e[r]} for r in ids],
           "orders": {"zone_orders": {r: n for r, _, _, n in done}, "greedy": greedy, "exact": exact}}
    for k, v in out["means"].items():
        print(f"{k:40s} {v:.4f}")
    for k, v in comps.items():
        print(f"{k:60s} {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {v['first_better']}/{v['first_worse']} p={v['sign_p']:.2g}")
    if a.no_write:
        return
    json.dump(out, open(RES / "v1_val.json", "w"), indent=1)
    rows = "\n".join(f"| {k} | {v['first_mean']:.4f} / {v['second_mean']:.4f} | {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                     f"| {v['first_better']} / {v['first_worse']} / {v['tied']} | {v['sign_p']:.2g} |" for k, v in comps.items())
    means = "\n".join(f"| {k} | {v:.4f} |" for k, v in out["means"].items())
    (RES / "v1_val.md").write_text(f"""# Adopted v1 on the 300 validation routes

Made by `scripts/v1_val.py`; numbers in `results/v1_val.json`. Validation only; it has been looked at many times (see `results/val_comparison.md` caveats), so treat every gap as optimistic. The verdict is the frozen final test. v1 was adopted on held-out TRAIN routes (`results/v1_adoption.md`), not on these numbers. "v1 exact" uses the exact in-zone solver, whose adoption is decided by `results/exact_solver_gate.md`, not here.

| Arm | Mean |
|---|---|
{means}

Paired, "first minus second" on the same routes (negative = first closer to the drivers). Bootstrap 10,000, seed {SEED}; exact sign test. Unadjusted; descriptive.

| Comparison | Means | Mean difference [95% CI] | W / L / T | Sign p |
|---|---|---|---|---|
{rows}

Runtime {out['seconds']:.0f} s.
""")


if __name__ == "__main__":
    main()
