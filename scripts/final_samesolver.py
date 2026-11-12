#!/usr/bin/env python3
"""PREREGISTRATION.md section 6.3, the same-solver engineer comparison (descriptive, not a claim).
Added 2026-11-12 ~11:45 IST after a pre-lock review found that no code produced it. Built BEFORE the lock,
from inputs only: no answer is read.

For each Opus-built system N (runs 2-6), take its repeat-1 prediction on the eval routes
(runs/final/predictions/opus_runN_rep1.json), read off its ZONE ORDER as the order of first visit to each zone
(zones as v0_model.zone_membership defines them, the same definition FINAL uses), then fill the stops inside
each zone with FINAL's in-zone solver (scripts/exact_zone.py, exactly as for v1_exact and OPUS-MODEL).
Written as runs/final/predictions/samesolver_runN.json. Compared with FINAL (v1_exact) by final_score.py as an
optional reference comparison. Same worker count as v1_exact (4), so the solver's 1 s searches get the
same machine share.

  .venv/bin/python -I scripts/final_samesolver.py --runs 5,6,2,3,4 [--limit K]
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_common as fc  # noqa: E402

WORKERS = 4


def zone_order_first_visit(route, order):
    from v0_model import zone_membership
    names, zone_of = zone_membership(route)
    pos = {str(s): i for i, s in enumerate(route["stop_ids"])}
    seen = []
    for sid in order:
        k = int(zone_of[pos[sid]])
        if k >= 0 and names[k] not in seen:
            seen.append(names[k])
    assert sorted(seen) == names, "prediction does not visit every zone"
    return seen


def _job(job):
    import exact_zone
    rid, order = job
    c0 = time.process_time()
    route = fc.load_eval_npz(rid)
    zones = zone_order_first_visit(route, order)
    stops = exact_zone.stops_from_zone_order_exact(route, zones)
    return rid, stops, zones, time.process_time() - c0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="5,6,2,3,4")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    man = json.load(open(fc.PRED / "MANIFEST.json"))
    for n in [int(x) for x in a.runs.split(",")]:
        src = f"opus_run{n}_rep1"
        p = fc.PRED / f"{src}.json"
        if fc.sha256_file(p) != man[src]["sha256"]:
            raise SystemExit(f"REFUSING: {src} changed after it was written")
        orders = json.load(open(p))["orders"]
        ids = sorted(orders)[: a.limit] if a.limit else sorted(orders)
        t0 = time.time()
        with mp.Pool(WORKERS) as pool:
            done = pool.map(_job, [(r, orders[r]) for r in ids], chunksize=4)
        out = {r: s for r, s, _, _ in done}
        bad = [r for r in ids if not fc.check_permutation(fc.load_eval_npz(r), out[r])]
        arm = ("smoke_" if a.limit else "") + f"samesolver_run{n}"
        if bad:
            fc.incident(arm, f"{len(bad)} invalid orders", " ".join(bad[:10]))
        meta = {"source_arm": src, "source_sha256": man[src]["sha256"],
                "zone_order": "first visit to each zone in the source prediction (v0_model.zone_membership zones)",
                "in_zone_solver": "exact (scripts/exact_zone.py), same as v1_exact", "routes": len(ids),
                "wall_s": time.time() - t0, "cpu_s_per_route_mean": sum(c for *_, c in done) / len(done),
                "workers": WORKERS, "zone_orders": {r: z for r, _, z, _ in done}}
        fc.write_predictions(arm, out, meta)
        print(f"{arm}: {len(ids)} routes, {time.time() - t0:.0f}s, invalid {len(bad)}", flush=True)


if __name__ == "__main__":
    main()
