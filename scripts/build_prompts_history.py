#!/usr/bin/env python3
"""Build the "Opus direct + history" prompts: the pilot prompt, unchanged, plus past routes.

  .venv/bin/python -I scripts/build_prompts_history.py

For each of the 50 pilot routes (same routes, same order as runs/pilot, so results pair with
the existing Opus direct run) this writes runs/pilot_history/prompts/<route>.txt:

  1. the existing pilot prompt, runs/pilot/prompts/<route>.txt, copied byte for byte;
  2. an appended section with the 3 most similar routes from the same station in the TRAIN
     split, each as: date, zones in the order the driver first visited them, and how many of
     this route's zones it shares.

Similarity rule (documented in runs/pilot_history/README.md and prompts_index.json):
  - candidates: TRAIN routes (splits.json "train") with the same station_code. All quality
    labels, as in the v0-hist history table. Pilot and val routes are never candidates.
  - score: number of distinct zone IDs shared with this route (this route's zones are the
    zone IDs in its stop list, the same ones the original prompt already shows).
  - ties, in order: routes dated strictly before this route's date come first, nearest date
    first; then routes on or after its date, nearest date first; then route ID. The split is
    random, not by date, so some "past" routes can be dated after the pilot route; this is the
    same information v0-hist's history table holds.
  - a candidate must share at least 1 zone, so a route can get fewer than 3 (on the pilot:
    46 routes get 3, 3 get 2, and 1 DLA4 route gets none; its section says so).

Zone order of a past route = first visit to each zone in the driver's actual order, with the
same zone definition the model uses (v0_model.zone_membership: a stop with no zone ID takes
the zone of its nearest stop by travel time).

What is read:
  - runs/pilot/prompts/<route>.txt and runs/pilot/prompts_index.json (the original prompts)
  - data/raw/sample/sample_manifest.json (route order) and sample_route_data.json (this
    route's station, date and zone IDs; this file holds no sequences)
  - data/processed/splits.json, data/processed/index.csv (station and date per route)
  - data/processed/routes/<id>.npz for TRAIN routes only, including their actual_order
Never read: any pilot or val .npz, sample_actual.json, sample_invalid.json.
"""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from v0_model import zone_membership  # noqa: E402  (same zone definition as the model)

K_REF = 3
SYSTEM_SENTENCE = (" For reference, the prompt also lists the zone orders that drivers actually "
                   "drove on past routes from the same station.")

SECTION_HEAD = """

--- Past routes from the same station (for reference) ---
Below are {k} routes driven from this station on other occasions, chosen because they share the most zone IDs with this route. Each shows its date, how many of this route's {nz} zones it also visits, and its zones in the order its driver first visited them. These past routes may contain zones that are not on this route.

{lines}

Use these past routes only as a guide. The answer format above is unchanged: output the JSON list of all {n} location IDs and nothing else."""

SECTION_NONE = """

--- Past routes from the same station (for reference) ---
No past route from this station visits any of this route's zones, so none is listed.

The answer format above is unchanged: output the JSON list of all {n} location IDs and nothing else."""


def train_zone_order(route_id):
    """First-visit zone order of a TRAIN route, from its npz."""
    with np.load(ROOT / "data/processed/routes" / f"{route_id}.npz", allow_pickle=False) as d:
        route = {k: d[k] for k in ("stop_ids", "travel_time", "zone_ids", "actual_order")}
    names, zone_of = zone_membership(route)
    seen, order = set(), []
    for i in route["actual_order"][1:]:
        z = names[int(zone_of[int(i)])]
        if z not in seen:
            seen.add(z)
            order.append(z)
    return order


def rank_key(shared, cand_date, target_date, rid):
    before = cand_date < target_date
    gap = abs((cand_date - target_date).days)
    return (-shared, 0 if before else 1, gap, rid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", type=Path, default=ROOT / "runs/pilot")
    ap.add_argument("--sample-dir", type=Path, default=ROOT / "data/raw/sample")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs/pilot_history")
    args = ap.parse_args()

    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    train_ids = {x["route_id"] for x in splits["train"]}
    forbidden = {x["route_id"] for x in splits["pilot"]} | {x["route_id"] for x in splits["val"]}
    assert not train_ids & forbidden, "splits overlap"

    meta = {}
    with open(ROOT / "data/processed/index.csv") as f:
        for row in csv.DictReader(f):
            meta[row["route_id"]] = (row["station"], date.fromisoformat(row["date"]))

    manifest = json.loads((args.sample_dir / "sample_manifest.json").read_text())
    routes = json.loads((args.sample_dir / "sample_route_data.json").read_text())
    pilot_index = json.loads((args.pilot_dir / "prompts_index.json").read_text())
    pilot_info = {x["route_id"]: x for x in pilot_index["routes"]}
    assert set(manifest["route_ids"]) == {x["route_id"] for x in splits["pilot"]}

    # Train candidates per station, needed only for stations that pilot routes use.
    stations = {routes[r]["station_code"] for r in manifest["route_ids"]}
    by_station = {}
    for rid in sorted(train_ids):
        st, d = meta[rid]
        if st in stations:
            by_station.setdefault(st, []).append((rid, d))

    order_cache = {}

    def zones_of(rid):
        assert rid in train_ids and rid not in forbidden
        if rid not in order_cache:
            order_cache[rid] = train_zone_order(rid)
        return order_cache[rid]

    prompt_dir = args.out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    system = (args.pilot_dir / "system_prompt.txt").read_text() + SYSTEM_SENTENCE
    (args.out_dir / "system_prompt.txt").write_text(system)

    index = []
    for r in manifest["route_ids"]:
        route = routes[r]
        station, target_date = route["station_code"], date.fromisoformat(route["date_YYYY_MM_DD"])
        mine = {v["zone_id"] for v in route["stops"].values()
                if v["type"] != "Station" and isinstance(v["zone_id"], str) and v["zone_id"] not in ("-", "")}
        ranked = []
        for rid, d in by_station.get(station, []):
            shared = len(mine & set(zones_of(rid)))
            if shared > 0:
                ranked.append((rank_key(shared, d, target_date, rid), rid, d, shared))
        ranked.sort()
        refs = ranked[:K_REF]
        lines = []
        for _, rid, d, shared in refs:
            zs = zones_of(rid)
            lines.append(f"{d.isoformat()}: shares {shared} of {len(mine)} zones ({shared / len(mine):.0%}); "
                         f"{len(zs)} zones in driver order: {' '.join(zs)}")
        original = (args.pilot_dir / "prompts" / f"{r}.txt").read_text()
        n = pilot_info[r]["n_stops_including_depot"]
        if refs:
            section = SECTION_HEAD.format(k=len(refs), nz=len(mine), lines="\n".join(lines), n=n)
        else:
            section = SECTION_NONE.format(n=n)
        prompt = original + section
        (prompt_dir / f"{r}.txt").write_text(prompt)
        index.append({**pilot_info[r], "prompt_chars": len(prompt), "original_prompt_chars": len(original),
                      "size_ratio": len(prompt) / len(original), "station": station,
                      "date": target_date.isoformat(), "n_zones": len(mine),
                      "train_candidates_same_station": len(by_station.get(station, [])),
                      "references": [{"route_id": rid, "date": d.isoformat(), "shared_zones": shared,
                                      "zone_order": zones_of(rid)} for _, rid, d, shared in refs]})

    total = sum(x["prompt_chars"] for x in index)
    ratios = [x["size_ratio"] for x in index]
    info = {
        "built_from": "runs/pilot/prompts (copied byte for byte) + appended history section",
        "similarity_rule": ("TRAIN routes, same station_code, ranked by number of shared distinct zone IDs; "
                            "ties: dated strictly before the pilot date first (nearest first), then on/after "
                            "(nearest first), then route ID; must share >= 1 zone; top 3"),
        "zone_order": "first visit to each zone in the driver's actual order; v0_model.zone_membership",
        "never_read": ["pilot or val .npz files", "sample_actual.json", "sample_invalid.json"],
        "system_prompt_chars": len(system),
        "total_prompt_chars": total, "mean_prompt_chars": total / len(index),
        "original_total_prompt_chars": sum(x["original_prompt_chars"] for x in index),
        "size_ratio_min": min(ratios), "size_ratio_mean": sum(ratios) / len(ratios), "size_ratio_max": max(ratios),
        "routes": index,
    }
    (args.out_dir / "prompts_index.json").write_text(json.dumps(info, indent=1))
    print(f"{len(index)} prompts, {total} chars total (original {info['original_total_prompt_chars']}), "
          f"mean {total / len(index):.0f}; size ratio min {min(ratios):.3f} mean "
          f"{info['size_ratio_mean']:.3f} max {max(ratios):.3f}; "
          f"refs per route min {min(len(x['references']) for x in index)}")


if __name__ == "__main__":
    sys.exit(main())
