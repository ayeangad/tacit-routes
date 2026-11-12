#!/usr/bin/env python3
"""Write data/processed/splits.json from index.csv and the pilot manifest.

Run from tacit-routes/:
  .venv/bin/python -I scripts/make_splits.py --index data/processed/index.csv \
      --manifest data/raw/sample/sample_manifest.json --out data/processed/splits.json

  pilot: the 50 frontier-comparison routes. Never train or tune on them.
  val:   300 High routes from the rest, stratified by station (proportional allocation,
         largest remainder, ties broken by station code), drawn with seed 20261111.
  train: every other training route, all quality labels.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

SEED = 20261111
N_VAL = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    rows = {r["route_id"]: r for r in csv.DictReader(open(a.index))}
    pilot = sorted(json.load(open(a.manifest))["route_ids"])
    if not set(pilot) <= set(rows):
        raise SystemExit("pilot routes missing from index")

    pool = sorted(r for r, v in rows.items() if v["quality"] == "High" and r not in set(pilot))
    by_station = {}
    for r in pool:
        by_station.setdefault(rows[r]["station"], []).append(r)
    exact = {s: N_VAL * len(v) / len(pool) for s, v in by_station.items()}
    quota = {s: int(q) for s, q in exact.items()}
    for s in sorted(exact, key=lambda s: (-(exact[s] - quota[s]), s))[:N_VAL - sum(quota.values())]:
        quota[s] += 1
    rng = random.Random(SEED)
    val = sorted(r for s in sorted(by_station) for r in rng.sample(by_station[s], quota[s]))

    taken = set(pilot) | set(val)
    train = sorted(r for r in rows if r not in taken)
    assert len(set(pilot)) + len(set(val)) + len(set(train)) == len(rows)
    assert not (set(pilot) & set(val)) and not (set(pilot) & set(train)) and not (set(val) & set(train))

    def entry(r):
        return {"route_id": r, "quality": rows[r]["quality"], "station": rows[r]["station"]}

    def counts(ids):
        return dict(sorted(Counter(rows[r]["quality"] for r in ids).items()))

    out = {
        "seed": SEED,
        "rules": {
            "pilot": "the 50 routes in data/raw/sample/sample_manifest.json (High, seed 20261110); "
                     "frontier-comparison set, never used for training or tuning",
            "val": f"{N_VAL} High routes from the remaining training routes, stratified by station "
                   "(proportional allocation, largest remainder), random.Random(seed).sample per "
                   "station in sorted station order",
            "train": "every other training route, all quality labels; filter on quality as needed",
            "evaluation side": "not included; the bucket's almrrc2021-data-evaluation set is untouched",
        },
        "counts": {k: {"total": len(v), "by_quality": counts(v)}
                   for k, v in (("pilot", pilot), ("val", val), ("train", train))},
        "val_by_station": dict(sorted(Counter(rows[r]["station"] for r in val).items())),
        "pilot": [entry(r) for r in pilot],
        "val": [entry(r) for r in val],
        "train": [entry(r) for r in train],
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("counts", "val_by_station")}, indent=1))


if __name__ == "__main__":
    main()
