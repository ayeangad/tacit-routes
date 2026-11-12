#!/usr/bin/env python3
"""Build one direct-ordering prompt per sampled route, identical for every model.

  .venv/bin/python -I scripts/build_prompts.py --sample-dir data/raw/sample --out-dir runs/pilot

Each prompt gives the depot ID and, for every stop (depot included), its ID, latitude,
longitude, zone ID and its 5 nearest other stops by the supplied travel time.

Leakage rules:
  - Stops are listed in a seeded random order, random.Random(f"{seed}:prompt:{route_id}"),
    never the driver's order or the file's order.
  - Nothing from actual_sequences.json, the route quality label (route_score),
    invalid-sequence scores, or anything derived from them is read into a prompt.
    This script does not even open those files.
"""
import argparse
import json
import random
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert in last-mile parcel delivery. You know how experienced delivery "
    "drivers actually sequence their stops. Answer with a JSON list only, no other text."
)

TEMPLATE = """A delivery van leaves the depot and must visit every stop below exactly once.

Depot ID: {depot}

Each line below is one location: ID, latitude, longitude, zone ID, then its 5 nearest other locations by driving time from it (ID:seconds). The depot is marked [DEPOT]. Zone "none" means no zone was given. Lines are in random order.

{lines}

There are {n} locations in total, including the depot.

Give the order in which an experienced delivery driver would visit every stop, starting at the depot. Return a JSON list of location IDs that starts with the depot ID "{depot}" and then contains every other ID exactly once, {n} IDs in total. Do not return to the depot at the end. Output the JSON list and nothing else."""


def build(route_id, route, travel, seed, k=5):
    stops = route["stops"]
    depot = next(s for s, v in stops.items() if v["type"] == "Station")
    ids = sorted(stops)
    if set(ids) != set(travel):
        raise SystemExit(f"{route_id}: stop set differs between route data and travel times")
    order = ids[:]
    random.Random(f"{seed}:prompt:{route_id}").shuffle(order)
    lines = []
    for s in order:
        v = stops[s]
        zone = v["zone_id"] if isinstance(v["zone_id"], str) and v["zone_id"] != "-" else "none"
        near = sorted((t for t in ids if t != s), key=lambda t: (travel[s][t], t))[:k]
        near_txt = " ".join(f"{t}:{round(travel[s][t])}" for t in near)
        tag = " [DEPOT]" if s == depot else ""
        lines.append(f"{s}{tag} {v['lat']:.6f} {v['lng']:.6f} {zone} | {near_txt}")
    prompt = TEMPLATE.format(depot=depot, lines="\n".join(lines), n=len(ids))
    return depot, order, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()

    manifest = json.loads((args.sample_dir / "sample_manifest.json").read_text())
    routes = json.loads((args.sample_dir / "sample_route_data.json").read_text())
    travel = json.loads((args.sample_dir / "sample_travel_times.json").read_text())

    prompt_dir = args.out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT)

    index = []
    for r in manifest["route_ids"]:
        depot, order, prompt = build(r, routes[r], travel[r], args.seed)
        (prompt_dir / f"{r}.txt").write_text(prompt)
        index.append({"route_id": r, "depot": depot, "n_stops_including_depot": len(order),
                      "listed_order": order, "prompt_chars": len(prompt)})
    total = sum(x["prompt_chars"] for x in index)
    info = {
        "seed": args.seed,
        "listing_order": "random.Random(f'{seed}:prompt:{route_id}').shuffle(sorted stop IDs)",
        "neighbours": "5 nearest other locations by supplied travel time from the stop, ties by ID, seconds rounded",
        "inputs_read": ["sample_manifest.json (route IDs, order)", "sample_route_data.json (stops only)",
                        "sample_travel_times.json"],
        "not_read": ["sample_actual.json", "sample_invalid.json", "route_score labels"],
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "total_prompt_chars": total,
        "mean_prompt_chars": total / len(index),
        "tokens": "not counted locally; per-call input tokens are recorded from each API's usage fields",
        "routes": index,
    }
    (args.out_dir / "prompts_index.json").write_text(json.dumps(info, indent=1))
    print(f"{len(index)} prompts, {total} chars total, mean {total / len(index):.0f}, "
          f"min {min(x['prompt_chars'] for x in index)}, max {max(x['prompt_chars'] for x in index)}")


if __name__ == "__main__":
    sys.exit(main())
