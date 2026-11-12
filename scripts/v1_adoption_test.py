#!/usr/bin/env python3
"""The v1 adoption test, exactly as written in docs/plan.md item 3
BEFORE any rehearsal or v1 number existed:

  Targets: the 234 held-out High TRAIN routes (v0_model.split_train; neither model trains on them;
           both used them for early stopping, and v0's lambda was tuned on them: disclosed).
  (a) Thinned history: for each target, every train route within ±14 days of its date is removed
      from its history. Adopt only if mean(v1 − v0-hist) < 0 AND the paired 95% bootstrap CI
      excludes 0 (upper bound < 0).
  (b) Full history (leave-one-out): the upper end of the paired 95% CI of v1 − v0-hist must be
      below +0.002 (non-inferiority margin).
  ADOPT only if (a) and (b) both hold; otherwise keep v0-hist.

AMENDED 2026-11-11 ~21:10 IST, BEFORE any adoption number existed (review 4; joint file
"Amendment 2"): the ±14-day window was found far harsher than the real final test (calibrated against
results/eval_coverage.md: the test looks like hiding ±5–7 days; before-half ±2–3, after-half ±8–9).
  (a) now decides on the MATCHED condition: per route, the mean of (v1 − v0-hist) at ±3 days and at
      ±9 days (the test's before/after halves). Adopt only if its paired 95% CI upper bound < 0.
  (b) unchanged (full history, upper < +0.002).
  ±6 and ±14 days are reported, not deciding (±14 = labelled stress test).
The original script is kept at runs/evening/v1_adoption_test_ORIGINAL.py.

Both arms: beam 5 + travel lambda = 0.5 (v0-hist's pre-registered decoder; v1 keeps it fixed).
Scoring: the official scorer through v0_decode.official_scores. Bootstrap: 10,000 resamples,
seed 20261111 (v0_decode.bootstrap_ci).

Heavy (run after Opus runs 3-4 finish):  .venv/bin/python -I scripts/v1_adoption_test.py --workers 3
Smoke:  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/v1_adoption_test.py --limit 3 --workers 1 --tag smoke --seeds 0 --no-write
Writes results/v1_adoption.json and .md (not with --no-write).
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
from v0_decode import INVALID, bootstrap_ci, official_scores, sign_test  # noqa: E402
from v0_model import SPLITS, split_train  # noqa: E402

ROOT = HERE.parent
RESULTS = ROOT / "results"
WINDOW_DAYS = 14  # stress test only (amended)
MATCHED = (3, 9)  # deciding pair (amended)
REPORT_ONLY = (6, 14)
MARGIN = 0.002
BOOT_SEED = 20261111


def paired(a, b, ids):
    d = np.array([a[r] - b[r] for r in ids])
    lo, hi = bootstrap_ci(d, BOOT_SEED)
    w, l = int((d < 0).sum()), int((d > 0).sum())
    return {"mean_diff": float(d.mean()), "ci95": [lo, hi], "v1_better": w, "v1_worse": l,
            "tied": len(d) - w - l, "sign_p": sign_test(w, l)}


def run(pool, ids, arm, spec, invalid):
    jobs = [(r, arm, spec, v1_model.LAMBDA, v1_model.BEAM) for r in ids]
    done = pool.map(v1_model.decode_job, jobs, chunksize=2) if pool else [v1_model.decode_job(j) for j in jobs]
    orders = {job[0]: order for job, order, _ in done}
    scores, feasible = official_scores(orders, invalid, pool)
    assert all(feasible.values())
    return scores, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="main", help="v1 model folder under runs/v1/")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(","))
    if (a.tag, seeds) != ("main", v1_model.SEEDS):
        assert a.no_write, "only the pre-registered 5-seed 'main' ensemble may write results"
    splits = json.load(open(SPLITS))
    _, held, quality = split_train(splits)
    ids = sorted(r for r in held if quality[r] == "High")
    assert len(ids) == 234 or a.limit
    if a.limit:
        ids = ids[:a.limit]
    forbidden = {r["route_id"] for r in splits["val"]} | {r["route_id"] for r in splits["pilot"]}
    assert not set(ids) & forbidden
    invalid = json.load(open(INVALID))

    t0 = time.time()
    res = {}
    with mp.Pool(a.workers, initializer=v1_model.set_ensemble, initargs=(a.tag, seeds)) if a.workers > 1 else _null() as pool:
        if pool is None:
            v1_model.set_ensemble(a.tag, seeds)
        conds = [("full", ())] + [(f"thinned_{d}d", ("window", d)) for d in sorted(set(MATCHED + REPORT_ONLY))]
        for cond, spec in conds:
            for arm in ("v0hist", "v1"):
                s, _ = run(pool, ids, arm, spec, invalid)
                res[(cond, arm)] = s
                print(f"[{cond}] {arm}: mean {np.mean(list(s.values())):.4f}  ({time.time() - t0:.0f}s)", flush=True)

    # Check: v0-hist with full history must reproduce v0's own cached tuning run exactly.
    cache = ROOT / "runs" / "v0" / "cache" / "tune_hist_high_medium_beam5_lam0.5_n234.json"
    ref = json.load(open(cache))["scores"]
    repro = max(abs(res[("full", "v0hist")][r] - ref[r]) for r in ids)
    print(f"v0-hist full-history reproduction vs v0 cache: max abs diff {repro}")

    by_d = {d: paired(res[(f"thinned_{d}d", "v1")], res[(f"thinned_{d}d", "v0hist")], ids) for d in sorted(set(MATCHED + REPORT_ONLY))}
    mv1 = {r: np.mean([res[(f"thinned_{d}d", "v1")][r] for d in MATCHED]) for r in ids}
    mv0 = {r: np.mean([res[(f"thinned_{d}d", "v0hist")][r] for d in MATCHED]) for r in ids}
    thin = paired(mv1, mv0, ids)
    full = paired(res[("full", "v1")], res[("full", "v0hist")], ids)
    cond_a = thin["mean_diff"] < 0 and thin["ci95"][1] < 0
    cond_b = full["ci95"][1] < MARGIN
    verdict = "ADOPT" if (cond_a and cond_b) else "KEEP v0-hist"
    for d, x in by_d.items():
        print(f"    ±{d} days (report): v1 - v0-hist {x['mean_diff']:+.4f} {x['ci95']}")
    print(f"(a) matched ±3/±9: v1 - v0-hist {thin['mean_diff']:+.4f} {thin['ci95']} -> {'pass' if cond_a else 'FAIL'}")
    print(f"(b) full:    v1 - v0-hist {full['mean_diff']:+.4f} {full['ci95']} (upper < +{MARGIN}) -> {'pass' if cond_b else 'FAIL'}")
    print(verdict)

    out = {
        "rule": "docs/plan.md item 3 (written before any rehearsal number)",
        "targets": len(ids), "matched_days": list(MATCHED), "report_only_days": list(REPORT_ONLY), "amended": "see docstring", "margin": MARGIN, "bootstrap_seed": BOOT_SEED,
        "decoder": {"beam": v1_model.BEAM, "lambda": v1_model.LAMBDA},
        "v1": {"tag": a.tag, "seeds": list(seeds)},
        "means": {f"{c}/{arm}": float(np.mean(list(s.values()))) for (c, arm), s in res.items()},
        "matched_v1_minus_v0hist": thin, "by_window_v1_minus_v0hist": {str(d): x for d, x in by_d.items()}, "full_v1_minus_v0hist": full,
        "condition_a_thinned_pass": cond_a, "condition_b_full_pass": cond_b, "verdict": verdict,
        "per_route": {f"{c}/{arm}": s for (c, arm), s in res.items()},
        "seconds": time.time() - t0, "v0hist_full_reproduces_v0_cache_max_abs_diff": repro,
    }
    if a.no_write:
        return
    json.dump(out, open(RESULTS / "v1_adoption.json", "w"), indent=1)
    m = out["means"]
    rows = "\n".join(f"| ±{d} days removed (report) | {m[f'thinned_{d}d/v0hist']:.4f} | {m[f'thinned_{d}d/v1']:.4f} | {x['mean_diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {x['v1_better']} / {x['v1_worse']} | — | — |" for d, x in by_d.items())
    (RESULTS / "v1_adoption.md").write_text(f"""# v1 adoption test (held-out TRAIN routes; validation and the final test not touched)

Rule (written before any number): `docs/plan.md`, item 3. Numbers: `results/v1_adoption.json`, made by `scripts/v1_adoption_test.py`.

**Verdict: {verdict}.**

Targets: the {len(ids)} held-out High training routes. Neither model trained on them. Both used them for early stopping, and v0's travel weight λ was tuned on them; the decoder is fixed at beam {v1_model.BEAM} + λ = {v1_model.LAMBDA} for both. Lower is better.

| History | v0-hist | v1 (5-seed) | v1 − v0-hist [95% CI] | v1 better / worse | Needed | Pass? |
|---|---|---|---|---|---|---|
| **Matched (mean of ±3 and ±9 days removed; DECIDES)** | {np.mean(list(mv0.values())):.4f} | {np.mean(list(mv1.values())):.4f} | {thin['mean_diff']:+.4f} [{thin['ci95'][0]:+.4f}, {thin['ci95'][1]:+.4f}] | {thin['v1_better']} / {thin['v1_worse']} | upper < 0 | {'yes' if cond_a else 'no'} |
{rows}
| Full (leave-one-out) | {m['full/v0hist']:.4f} | {m['full/v1']:.4f} | {full['mean_diff']:+.4f} [{full['ci95'][0]:+.4f}, {full['ci95'][1]:+.4f}] | {full['v1_better']} / {full['v1_worse']} | upper < +{MARGIN} | {'yes' if cond_b else 'no'} |

Rows marked "report" do not decide. ±14 days is a stress test, far thinner than the final test (see the amendment in the script's docstring and the joint decisions file).

Bootstrap: 10,000 route resamples, seed {BOOT_SEED}. Runtime {out['seconds']:.0f} s.
""")


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
