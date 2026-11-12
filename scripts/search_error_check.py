#!/usr/bin/env python3
"""Search error or model error? (docs/plan.md item 3: a decoding
change is allowed only if this check, on held-out TRAIN routes, shows the search losing driver
orders the model rates higher.)

For each of the 234 held-out High train routes, with frozen v0-hist and full (leave-one-out)
history, compute the decoder's own objective
    J(order) = sum_t log p(zone_t | history) - lambda * travel_t / tt_scale
for (1) the zone order beam search returned and (2) the driver's real zone order.
  J(driver) > J(beam): SEARCH ERROR. The model prefers the driver's order; the search missed it.
  J(driver) < J(beam): MODEL ERROR. The search found something the model likes better.
  (equal: the beam found the driver's order.)
Done for the pre-registered decoder (beam 5, lambda 0.5) and the pure model (beam 5, lambda 0).
The official score per route is taken from v0's cached tuning runs (same routes, same decoder),
so we can also say how much of the score sits on search-error routes.

Heavy-ish (234 routes, about 0.3 s each per decoder):  .venv/bin/python -I scripts/search_error_check.py --workers 3
Smoke:  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/search_error_check.py --limit 3 --workers 1 --no-write
Writes results/search_error.json and .md.
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
import v0_decode  # noqa: E402
from v0_model import SPLITS, driver_zone_order, features, load_npz, prepare_route, split_train  # noqa: E402
from v1_model import sequence_objective  # noqa: E402

ROOT = HERE.parent
TAG = "hist_high_medium"
CACHE = ROOT / "runs" / "v0" / "cache"


def job(args):
    route_id, lam = args
    model = v0_decode.load_model(TAG)[0]
    route = load_npz(route_id)
    P = v0_decode.prepare_for(route, TAG)                 # v0-hist, full leave-one-out history
    beam = v0_decode.beam_decode(model, P, 5, lam)
    drv = driver_zone_order(P, route)
    jb = sequence_objective(model, P, beam, lam, features)
    jd = sequence_objective(model, P, drv, lam, features)
    return route_id, lam, {"J_beam": jb, "J_driver": jd, "beam_is_driver": beam == drv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    splits = json.load(open(SPLITS))
    _, held, quality = split_train(splits)
    ids = sorted(r for r in held if quality[r] == "High")
    if a.limit:
        ids = ids[:a.limit]
    forbidden = {r["route_id"] for r in splits["val"]} | {r["route_id"] for r in splits["pilot"]}
    assert not set(ids) & forbidden
    t0 = time.time()
    jobs = [(r, lam) for lam in (0.5, 0.0) for r in ids]
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            done = pool.map(job, jobs, chunksize=2)
    else:
        done = [job(j) for j in jobs]
    out = {"targets": len(ids), "model": TAG, "history": "full (leave-one-out)", "decoders": {}}
    lines = []
    for lam in (0.5, 0.0):
        rows = {r: d for r, l, d in done if l == lam}
        scores = json.load(open(CACHE / f"tune_{TAG}_beam5_lam{lam:g}_n234.json"))["scores"]
        exact = [r for r in ids if rows[r]["beam_is_driver"]]
        search = [r for r in ids if not rows[r]["beam_is_driver"] and rows[r]["J_driver"] > rows[r]["J_beam"] + 1e-9]
        model_err = [r for r in ids if r not in exact and r not in search]
        tot = sum(scores[r] for r in ids)
        summ = {
            "beam_found_driver_order": len(exact), "search_error": len(search), "model_error": len(model_err),
            "share_search_error_of_wrong": len(search) / max(1, len(search) + len(model_err)),
            "share_of_total_score_on_search_error_routes": sum(scores[r] for r in search) / tot,
            "share_of_total_score_on_model_error_routes": sum(scores[r] for r in model_err) / tot,
            "mean_score_search_error_routes": float(np.mean([scores[r] for r in search])) if search else None,
            "mean_score_model_error_routes": float(np.mean([scores[r] for r in model_err])) if model_err else None,
            "median_J_gap_search_error": float(np.median([rows[r]["J_driver"] - rows[r]["J_beam"] for r in search])) if search else None,
            "per_route": rows,
        }
        out["decoders"][f"beam5_lambda{lam:g}"] = summ
        lines.append(f"| beam 5, λ = {lam:g}{' (pre-registered)' if lam else ' (pure model)'} | {len(exact)} | "
                     f"{len(search)} | {len(model_err)} | {summ['share_of_total_score_on_search_error_routes']:.0%} | "
                     f"{summ['share_of_total_score_on_model_error_routes']:.0%} |")
        print(lines[-1], flush=True)
    out["seconds"] = time.time() - t0
    if a.no_write:
        return
    json.dump(out, open(ROOT / "results" / "search_error.json", "w"), indent=1)
    (ROOT / "results" / "search_error.md").write_text(f"""# Search error or model error? (v0-hist, held-out TRAIN routes)

Made by `scripts/search_error_check.py`; numbers in `results/search_error.json`. {len(ids)} held-out High training routes, full (leave-one-out) history. Validation and the final test were not touched.

The decoder maximises J = sum of log-probabilities − λ·travel/scale. For each route we compare J for the zone order the beam returned with J for the driver's real zone order.
- **Search error:** the model scores the driver's order HIGHER than what the beam returned. A better search could find it.
- **Model error:** the beam's order scores higher than the driver's. Only a better model can help.

| Decoder | Beam found driver's order | Search error | Model error | Score share on search-error routes | Score share on model-error routes |
|---|---|---|---|---|---|
{chr(10).join(lines)}

**Rule (agreed in advance):** a decoding change (a wider or stratified beam) is considered only if search errors carry a meaningful share of the score. Otherwise decoding stays frozen.
""")


if __name__ == "__main__":
    main()
