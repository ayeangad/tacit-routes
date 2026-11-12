"""Shared pieces of the final-test harness (stage 1 predictions and stage 2 scoring).

Plan of record: DECISION_LOG.md.

Stage 1 (scripts/final_predict.py) reads ONLY apply-side inputs: data/eval_frozen/routes/*.npz
(no driver order) and data/eval_frozen/raw/apply/ (for jpt-amz). Stage 2 (scripts/final_score.py)
is the only code allowed to open data/eval_frozen/raw/score/. Nothing here touches raw/score/.
"""
import csv
import datetime as dt
import os
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Env overrides exist ONLY for the dress rehearsal (scripts/final_dress_rehearsal.sh), which lays out
# the 300 VAL routes like data/eval_frozen and writes everything under runs/final/dress/.
EVAL_DIR = Path(os.environ.get("TR_EVAL_DIR", ROOT / "data" / "eval_frozen")).resolve()
EVAL_ROUTES = EVAL_DIR / "routes"
EVAL_INDEX = EVAL_DIR / "index.csv"
EVAL_APPLY = EVAL_DIR / "raw" / "apply"
SEALED = EVAL_DIR / "raw" / "score"          # stage 2 only
FINAL = Path(os.environ.get("TR_FINAL_DIR", ROOT / "runs" / "final")).resolve()
RESULTS = Path(os.environ.get("TR_RESULTS_DIR", ROOT / "results")).resolve()
INCIDENTS = FINAL / "incidents.log"
PRED = FINAL / "predictions"
SAMPLE_FILE = FINAL / "primary_sample_500.json"
PREREG_CONFIG = FINAL / "PREREG_CONFIG.json"
PREREG_DOC = Path(os.environ.get("TR_PREREG_DOC", ROOT / "PREREGISTRATION.md")).resolve()
TRAIN_START, TRAIN_END = dt.date(2018, 7, 19), dt.date(2018, 8, 26)
SAMPLE_SEED = 20261112
SAMPLE_N = 500
BOOT_SEED = 20261111


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def eval_index():
    """[{route_id, station, date, n_stops}] for the 3,052 eval routes (apply side)."""
    with open(EVAL_INDEX) as f:
        return [{"route_id": r["route_id"], "station": r["station"], "date": r["date"][:10],
                 "n_stops": int(r["n_stops"])} for r in csv.DictReader(f)]


def eval_ids():
    return sorted(r["route_id"] for r in eval_index())


def load_eval_npz(route_id):
    """An eval route as a dict of arrays (same layout as data/processed/routes, no actual_order)."""
    with np.load(EVAL_ROUTES / f"{route_id}.npz", allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    assert "actual_order" not in out, route_id
    return out


def period(date_str):
    d = dt.date.fromisoformat(date_str[:10])
    return "before" if d < TRAIN_START else "after" if d > TRAIN_END else "inside"


def select_primary_sample(write=True):
    """Fixed-seed 500-route sample of the eval set, stratified by station x period.

    Strata = (station, before/inside/after training dates). Proportional allocation by largest
    remainder (each non-empty stratum gets at least its floor; remainders broken by size, then
    name), then a seeded random draw inside each stratum. Uses only station and date (apply side).
    """
    idx = eval_index()
    strata = defaultdict(list)
    for r in idx:
        strata[(r["station"], period(r["date"]))].append(r["route_id"])
    total = len(idx)
    keys = sorted(strata)
    exact = {k: SAMPLE_N * len(strata[k]) / total for k in keys}
    alloc = {k: int(exact[k]) for k in keys}
    left = SAMPLE_N - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - alloc[k]), -len(strata[k]), k))[:left]:
        alloc[k] += 1
    rng = random.Random(SAMPLE_SEED)
    chosen = []
    for k in keys:
        ids = sorted(strata[k])
        chosen += rng.sample(ids, alloc[k])
    chosen = sorted(chosen)
    assert len(chosen) == SAMPLE_N == len(set(chosen))
    out = {"seed": SAMPLE_SEED, "n": SAMPLE_N, "rule": select_primary_sample.__doc__.strip(),
           "allocation": {f"{s}|{p}": n for (s, p), n in sorted(alloc.items())},
           "route_ids": chosen, "sha256_of_route_ids": sha256_json(chosen)}
    if write:
        FINAL.mkdir(parents=True, exist_ok=True)
        if SAMPLE_FILE.exists():
            old = json.load(open(SAMPLE_FILE))
            assert old["sha256_of_route_ids"] == out["sha256_of_route_ids"], "sample changed!"
        SAMPLE_FILE.write_text(json.dumps(out, indent=1))
    return out


def write_predictions(arm, orders, meta):
    """orders: {route_id: [stop ids, station first]}. Writes runs/final/predictions/<arm>.json
    and updates runs/final/predictions/MANIFEST.json with its sha256."""
    PRED.mkdir(parents=True, exist_ok=True)
    path = PRED / f"{arm}.json"
    path.write_text(json.dumps({"arm": arm, "meta": meta, "orders": orders}))
    man_path = PRED / "MANIFEST.json"
    man = json.load(open(man_path)) if man_path.exists() else {}
    man[arm] = {"file": path.name, "sha256": sha256_file(path), "n_routes": len(orders),
                "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "meta": meta}
    man_path.write_text(json.dumps(man, indent=1))
    return path


def check_permutation(route, order):
    ids = [str(s) for s in route["stop_ids"]]
    return order and order[0] == ids[0] and sorted(order) == sorted(ids)


def train_stations():
    rows = json.load(open(ROOT / "runs" / "v0" / "history_train.json"))["routes"]
    return {r["station"] for r in rows}


def no_history_ids():
    """Eval routes at a station with no train routes (DBO6 on the real eval set: 1 route)."""
    st = train_stations()
    return sorted(r["route_id"] for r in eval_index() if r["station"] not in st)


def incident(arm, what, detail=""):
    FINAL.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with open(INCIDENTS, "a") as f:
        f.write(f"{stamp} [{arm}] {what}" + (f"\n    {detail.strip()[-1500:]}".replace("\n", "\n    ") if detail else "") + "\n")
    print(f"INCIDENT [{arm}] {what}", flush=True)
