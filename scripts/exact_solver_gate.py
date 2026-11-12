#!/usr/bin/env python3
"""Gate for the exact in-zone solver (docs/plan.md, Amendment after
Opus run 3, P2), fixed before any number existed:

  Targets: the 234 held-out High TRAIN routes (same set as scripts/v1_adoption_test.py).
  Zone order: the DRIVER's own (first-visit) zone order, so only the in-zone step varies.
  Old: v0_decode.stops_from_zone_order (OR-Tools greedy descent). New: exact_zone (Held–Karp <= 12
  stops, OR-Tools guided local search 1 s above).
  Score: official scorer (v0_decode.official_scores). Paired bootstrap 10,000, seed 20261111.
  ADOPT iff the upper end of the paired 95% CI of (new − old) is below 0.

  .venv/bin/python -I scripts/exact_solver_gate.py --workers 3        (writes results/exact_solver_gate.*)
  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/exact_solver_gate.py --limit 3 --workers 1 --no-write
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
from exact_zone import stops_from_zone_order_exact  # noqa: E402
from v0_decode import INVALID, bootstrap_ci, official_scores, sign_test, stops_from_zone_order  # noqa: E402
from v0_model import SPLITS, driver_zone_order, load_npz, prepare_route, split_train  # noqa: E402

ROOT = HERE.parent
SEED = 20261111


def job(rid):
    route = load_npz(rid)
    P = prepare_route(route)
    names = [P["names"][z] for z in driver_zone_order(P, route)]
    t0 = time.process_time()
    old = stops_from_zone_order(route, names)
    t1 = time.process_time()
    new = stops_from_zone_order_exact(route, names)
    t2 = time.process_time()
    return rid, old, new, t1 - t0, t2 - t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    splits = json.load(open(SPLITS))
    _, held, quality = split_train(splits)
    ids = sorted(r for r in held if quality[r] == "High")
    assert len(ids) == 234
    forbidden = {r["route_id"] for r in splits["val"]} | {r["route_id"] for r in splits["pilot"]}
    assert not set(ids) & forbidden
    if a.limit:
        ids = ids[:a.limit]
    invalid = json.load(open(INVALID))
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        done = pool.map(job, ids, chunksize=2)
        old = {r: o for r, o, _, _, _ in done}
        new = {r: n for r, _, n, _, _ in done}
        s_old, f_old = official_scores(old, invalid, pool)
        s_new, f_new = official_scores(new, invalid, pool)
    assert all(f_old.values()) and all(f_new.values())
    d = np.array([s_new[r] - s_old[r] for r in ids])
    lo, hi = bootstrap_ci(d, SEED)
    w, l = int((d < 0).sum()), int((d > 0).sum())
    verdict = "ADOPT" if hi < 0 else "KEEP greedy descent"
    cpu_old = float(np.mean([x[3] for x in done]))
    cpu_new = float(np.mean([x[4] for x in done]))
    out = {"rule": "joint decisions, Amendment after Opus run 3, P2", "targets": len(ids),
           "mean_old": float(np.mean([s_old[r] for r in ids])), "mean_new": float(np.mean([s_new[r] for r in ids])),
           "new_minus_old": {"mean": float(d.mean()), "ci95": [lo, hi], "new_better": w, "new_worse": l,
                             "tied": len(ids) - w - l, "sign_p": sign_test(w, l)},
           "verdict": verdict, "cpu_s_per_route": {"old": cpu_old, "new": cpu_new},
           "per_route": {r: {"old": s_old[r], "new": s_new[r]} for r in ids}, "seconds": time.time() - t0}
    print(json.dumps({k: v for k, v in out.items() if k != "per_route"}, indent=1))
    if a.no_write:
        return
    json.dump(out, open(ROOT / "results" / "exact_solver_gate.json", "w"), indent=1)
    x = out["new_minus_old"]
    (ROOT / "results" / "exact_solver_gate.md").write_text(f"""# Exact in-zone solver gate (held-out TRAIN routes; val and the final test not touched)

Rule fixed before any number: `docs/plan.md`, "Amendment after Opus run 3", P2. Made by `scripts/exact_solver_gate.py`; numbers in `results/exact_solver_gate.json`. Prompted by Opus fair-fight run 3, which solves each zone path exactly.

**Verdict: {verdict}.**

Targets: the {len(ids)} held-out High training routes. Zone order = the driver's own, so only the order of stops inside zones differs. Old = OR-Tools greedy descent (every v0/v1 number so far). New = Held–Karp dynamic programming for zones of up to 12 stops, OR-Tools guided local search for 1 s above. Lower is better.

| | Score |
|---|---|
| Old (greedy descent) | {out['mean_old']:.4f} |
| New (exact) | {out['mean_new']:.4f} |
| **New − old [95% CI]** | **{x['mean']:+.5f} [{x['ci95'][0]:+.5f}, {x['ci95'][1]:+.5f}]** |
| New better / worse / tied | {w} / {l} / {len(ids) - w - l} (sign p {x['sign_p']:.2g}) |
| CPU per route (fill only) | old {cpu_old:.3f} s, new {cpu_new:.3f} s |

Rule: ADOPT iff the CI's upper end is below 0. Bootstrap: 10,000 route resamples, seed {SEED}. Runtime {out['seconds']:.0f} s.
""")


if __name__ == "__main__":
    main()
