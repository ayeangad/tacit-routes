#!/usr/bin/env python3
"""Out-of-time rehearsal on VALIDATION routes. DIAGNOSTIC ONLY: not a decision tool and not a
test between methods (about 100 targets; paired intervals about ±0.004). Read each method's
degradation slope. Agreed in docs/plan.md (item 3, review 2b point 3).

Targets: val routes dated in the first 7 days (2018-07-19..07-25, "early") or the last 7 days
(2018-08-20..08-26, "late") of the training window.

History conditions (built per GROUP, so Opus's own code can later be run on identical thinned
train folders, which a per-route window would make impossible):
  full      all train routes (v0-hist: leave-one-out is irrelevant for val routes)
  thinned   early group: every train route dated <= (last early date + GAP) removed;
            late group:  every train route dated >= (first late date - GAP) removed.
            With GAP = 11, every target is at least 12 days from its nearest remaining history
            (the agreed "±11 days", applied group-wise; stricter for some targets).
  control   the same NUMBER of train routes removed, chosen at random (seed fixed per group):
            same history volume as "thinned", but no date gap. thinned − control isolates the gap.
Arms: frozen v0-hist; frozen v0-hist-past (its past-only rule PLUS the same removals);
v1 (5-seed ensemble) if runs/v1/main/seed{0..4}/model.pt exist. Decoder: beam 5 + lambda 0.5
for v0-hist and v1, lambda 1 for v0-hist-past (their pre-registered decoders).

Opus-code arms are NOT run here: Opus-written code must run inside its sandbox. See
write_thinned_arena() and the TODO at the bottom.

Heavy:  .venv/bin/python -I scripts/rehearsal_oot.py --workers 3
Smoke:  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/rehearsal_oot.py --limit 2 --workers 1 --no-write
Writes runs/rehearsal/exclusions.json and results/rehearsal_oot.json/.md.
"""
import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v1_model  # noqa: E402
from v0_decode import INVALID, bootstrap_ci, official_scores, sign_test  # noqa: E402
from v0_model import SPLITS  # noqa: E402

ROOT = HERE.parent
OUT_DIR = ROOT / "runs" / "rehearsal"
GAP = int(os.environ.get("REHEARSAL_GAP", "11"))  # 11 = stress read; 5 = matched to the final test (review 4)
SFX = "" if GAP == 11 else f"_gap{GAP}"
EARLY = (dt.date(2018, 7, 19), dt.date(2018, 7, 25))
LATE = (dt.date(2018, 8, 20), dt.date(2018, 8, 26))
SEED = 20261111
ARMS = {"v0hist": 0.5, "v0past": 1.0, "v1": 0.5}


def build_exclusions(splits):
    import csv
    date = {r["route_id"]: dt.date.fromisoformat(r["date"][:10])
            for r in csv.DictReader(open(ROOT / "data" / "processed" / "index.csv"))}
    val = {r["route_id"]: date[r["route_id"]] for r in splits["val"]}
    train = {r["route_id"]: date[r["route_id"]] for r in splits["train"]}
    groups = {}
    for name, (lo, hi) in (("early", EARLY), ("late", LATE)):
        targets = sorted(r for r, d in val.items() if lo <= d <= hi)
        if not targets:
            continue
        if name == "early":
            cut = max(val[r] for r in targets) + dt.timedelta(days=GAP)
            thinned = sorted(r for r, d in train.items() if d <= cut)
        else:
            cut = min(val[r] for r in targets) - dt.timedelta(days=GAP)
            thinned = sorted(r for r, d in train.items() if d >= cut)
        control = sorted(random.Random(f"{SEED}:{name}:control").sample(sorted(train), len(thinned)))
        groups[name] = {"targets": targets, "cut_date": str(cut), "thinned": thinned, "control": control}
    return groups


def paired(a, b, ids):
    d = np.array([a[r] - b[r] for r in ids])
    w, l = int((d < 0).sum()), int((d > 0).sum())
    return {"mean_diff": float(d.mean()), "ci95": bootstrap_ci(d, SEED), "first_better": w,
            "first_worse": l, "sign_p": sign_test(w, l)}


def spec_for(arm, cond, g):
    ids = () if cond == "full" else tuple(g[cond])
    if arm == "v0past":
        return ("past",) if cond == "full" else ("past+ids", ids)
    return () if cond == "full" else ("ids", ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="targets per group (testing)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    splits = json.load(open(SPLITS))
    pilot = {r["route_id"] for r in splits["pilot"]}
    groups = build_exclusions(splits)
    for g in groups.values():
        assert not set(g["targets"]) & pilot
        if a.limit:
            g["targets"] = g["targets"][:a.limit]
    arms = [x for x in ARMS if x != "v1" or all(
        (v1_model.V1_DIR / "main" / f"seed{s}" / "model.pt").exists() for s in v1_model.SEEDS)]
    invalid = json.load(open(INVALID))
    t0 = time.time()
    scores = {}
    pool = mp.Pool(a.workers) if a.workers > 1 else None
    try:
        for arm in arms:
            for cond in ("full", "thinned", "control"):
                jobs = [(r, arm, spec_for(arm, cond, g), ARMS[arm], v1_model.BEAM)
                        for g in groups.values() for r in g["targets"]]
                done = pool.map(v1_model.decode_job, jobs, chunksize=1) if pool else [v1_model.decode_job(j) for j in jobs]
                orders = {job[0]: order for job, order, _ in done}
                s, feas = official_scores(orders, invalid, pool)
                assert all(feas.values())
                scores[(arm, cond)] = s
                print(f"[{arm}/{cond}] mean {np.mean(list(s.values())):.4f} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        if pool:
            pool.close()
    ids = [r for g in groups.values() for r in g["targets"]]
    out = {"DIAGNOSTIC_ONLY": True, "gap_days": GAP, "targets": len(ids),
           "groups": {k: {"n_targets": len(g["targets"]), "cut_date": g["cut_date"],
                          "train_routes_removed": len(g["thinned"])} for k, g in groups.items()},
           "means": {f"{arm}/{c}": float(np.mean([scores[(arm, c)][r] for r in ids])) for arm, c in scores},
           "slopes": {}, "per_route": {f"{arm}/{c}": s for (arm, c), s in scores.items()}}
    for arm in arms:
        out["slopes"][arm] = {
            "thinned_minus_full": paired(scores[(arm, "thinned")], scores[(arm, "full")], ids),
            "control_minus_full": paired(scores[(arm, "control")], scores[(arm, "full")], ids),
            "thinned_minus_control (the date-gap effect)": paired(scores[(arm, "thinned")], scores[(arm, "control")], ids),
        }
    print(json.dumps(out["slopes"], indent=1))
    if a.no_write:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(groups, open(OUT_DIR / f"exclusions{SFX}.json", "w"))
    json.dump(out, open(ROOT / "results" / f"rehearsal_oot{SFX}.json", "w"), indent=1)
    rows = []
    for arm in arms:
        m = out["means"]
        sl = out["slopes"][arm]
        gap = sl["thinned_minus_control (the date-gap effect)"]
        rows.append(f"| {arm} | {m[f'{arm}/full']:.4f} | {m[f'{arm}/control']:.4f} | {m[f'{arm}/thinned']:.4f} | "
                    f"{sl['thinned_minus_full']['mean_diff']:+.4f} {fmt(sl['thinned_minus_full']['ci95'])} | "
                    f"{gap['mean_diff']:+.4f} {fmt(gap['ci95'])} |")
    (ROOT / "results" / f"rehearsal_oot{SFX}.md").write_text(f"""# Out-of-time rehearsal on validation routes (DIAGNOSTIC ONLY)

Made by `scripts/rehearsal_oot.py`; numbers in `results/rehearsal_oot.json`; exclusion sets in `runs/rehearsal/exclusions.json`. **Not a decision tool and not a test between methods.** {len(ids)} targets, so paired intervals are wide. Read how much each method degrades.

Targets: val routes from the first and last week of the training window. Thinned: train routes within the {GAP}-day gap (group-wise) removed, leaving every target at least {GAP + 1} days from its history. Control: the same number of train routes removed at random (same volume, no gap). Lower is better.

| Arm | Full | Control (volume only) | Thinned (gap) | Thinned − full [95% CI] | Thinned − control: the gap effect [95% CI] |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Groups: {json.dumps(out['groups'])}
""")


def fmt(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def write_thinned_arena(src_arena, dst_arena, exclude_ids, target_ids):
    """Copy an Opus fair-fight arena for a rehearsal condition (hard links, no data duplication):
    its code and trained model files, the train routes NOT excluded (train_index.csv filtered to
    match), and only the target val inputs. Derived history caches (*.pkl, *.npz feature dumps)
    and old predictions are NOT copied, so Opus's own history-building step must be re-run.

    TODO (must happen inside the Opus sandbox + memory scope, per the agreed rules, and twice
    for solver noise): for each fair-fight run, read its APPROACH.md to find which script builds
    its history tables (run 2: see opus_arena_run2/APPROACH.md, e.g. build_hist.py / prep.py),
    run that on the thinned arena WITHOUT retraining its networks (same as our frozen models),
    then run its predict.py; score with the same official scorer and add it to the table."""
    src, dst = Path(src_arena), Path(dst_arena)
    exclude = set(exclude_ids)
    (dst / "data" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "data" / "val_inputs").mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in (".py", ".pt", ".md", ".txt") and p.name != "val_predictions.json":
            os.link(p, dst / p.name)
    if (src / "scorer").exists():
        shutil.copytree(src / "scorer", dst / "scorer", copy_function=os.link, dirs_exist_ok=True)
    if (src / ".venv").exists():
        shutil.copytree(src / ".venv", dst / ".venv", symlinks=True, copy_function=os.link, dirs_exist_ok=True)
    kept = []
    for p in sorted((src / "data" / "train").glob("*.npz")):
        if p.stem not in exclude:
            os.link(p, dst / "data" / "train" / p.name)
            kept.append(p.stem)
    lines = (src / "data" / "train_index.csv").read_text().splitlines()
    keep = set(kept)
    (dst / "data" / "train_index.csv").write_text(
        "\n".join([lines[0]] + [l for l in lines[1:] if l.split(",")[0] in keep]) + "\n")
    for r in target_ids:
        os.link(src / "data" / "val_inputs" / f"{r}.npz", dst / "data" / "val_inputs" / f"{r}.npz")
    (dst / "data" / "val_ids.txt").write_text("\n".join(target_ids) + "\n")
    return len(kept)


if __name__ == "__main__":
    main()
