#!/usr/bin/env python3
"""Offline checks for the "Opus direct + history" prompts. Makes no API or model calls.

  .venv/bin/python -I scripts/check_pilot_history.py

Checks:
  1. 50 prompts, one per pilot route, each starting with its original pilot prompt byte for byte.
  2. Every cited reference route is in TRAIN and not in pilot or val (by splits.json).
  3. Every zone order shown in a prompt equals that train route's first-visit zone order,
     recomputed here from its npz, and appears in the prompt exactly as in prompts_index.json.
  4. The appended text holds no route IDs and none of this route's stop IDs, so the only
     route data from outside TRAIN is the original prompt's unordered stop list.
  5. No zone order shown equals any pilot route's own driver zone order. (This is the only
     place pilot sequences are read, from sample_actual.json, and only for this comparison.
     Val sequences are never read; val is excluded by route ID in check 2.)
  6. The system prompt is the pilot's plus one sentence; prompt size ratios are under 1.5.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from v0_model import zone_membership  # noqa: E402

OUT = ROOT / "runs/pilot_history"
PILOT = ROOT / "runs/pilot"
SAMPLE = ROOT / "data/raw/sample"


def first_visit_zones(route):
    names, zone_of = zone_membership(route)
    seen, order = set(), []
    for i in route["actual_order"][1:]:
        z = names[int(zone_of[int(i)])]
        if z not in seen:
            seen.add(z)
            order.append(z)
    return order


def main():
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)

    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    train = {x["route_id"] for x in splits["train"]}
    pilot_ids = {x["route_id"] for x in splits["pilot"]}
    val = {x["route_id"] for x in splits["val"]}
    idx = json.loads((OUT / "prompts_index.json").read_text())
    manifest = json.loads((SAMPLE / "sample_manifest.json").read_text())["route_ids"]
    routes = json.loads((SAMPLE / "sample_route_data.json").read_text())

    # 1
    check([x["route_id"] for x in idx["routes"]] == manifest, "route list differs from the pilot manifest")
    check(len(list((OUT / "prompts").glob("*.txt"))) == 50, "not exactly 50 prompt files")

    # 5: pilot driver zone orders (raw zone IDs, first visit, missing zones skipped)
    actual = json.loads((SAMPLE / "sample_actual.json").read_text())
    pilot_orders = {}
    for r in manifest:
        seq = sorted(actual[r]["actual"], key=actual[r]["actual"].get)
        seen, order = set(), []
        for s in seq:
            z = routes[r]["stops"][s]["zone_id"]
            if isinstance(z, str) and z not in ("-", "") and z not in seen:
                seen.add(z)
                order.append(z)
        pilot_orders[r] = tuple(order)
    pilot_order_set = set(pilot_orders.values())

    n_refs = 0
    recomputed = {}
    for x in idx["routes"]:
        r = x["route_id"]
        original = (PILOT / "prompts" / f"{r}.txt").read_text()
        prompt = (OUT / "prompts" / f"{r}.txt").read_text()
        check(prompt.startswith(original), f"{r}: original prompt not an exact prefix")
        added = prompt[len(original):]
        check("RouteID" not in added, f"{r}: route ID in appended section")
        stop_ids = set(routes[r]["stops"])
        words = set(re.findall(r"\b[A-Z]{2}\b", added)) - {"ID", "IDs"}
        check(not (words & stop_ids), f"{r}: stop IDs in appended section: {sorted(words & stop_ids)}")
        shown = re.findall(r"zones in driver order: (.*)", added)
        check(len(shown) == len(x["references"]), f"{r}: shown {len(shown)} vs index {len(x['references'])}")
        for ref, line in zip(x["references"], shown):
            n_refs += 1
            rid = ref["route_id"]
            check(rid in train and rid not in pilot_ids and rid not in val, f"{r}: reference {rid} not train-only")
            check(rid != r, f"{r}: cites itself")
            if rid not in recomputed:
                with np.load(ROOT / "data/processed/routes" / f"{rid}.npz", allow_pickle=False) as d:
                    route = {k: d[k] for k in ("stop_ids", "travel_time", "zone_ids", "actual_order", "station")}
                recomputed[rid] = (first_visit_zones(route), str(route["station"]))
            zs, st = recomputed[rid]
            check(st == x["station"], f"{r}: reference {rid} from station {st}, not {x['station']}")
            check(line.split() == zs == ref["zone_order"], f"{r}: zone order for {rid} does not match its npz")
            check(tuple(zs) != pilot_orders[r], f"{r}: shows its own driver zone order")
            check(tuple(zs) not in pilot_order_set, f"{r}: shows some pilot route's zone order")
        # 6
        check(len(prompt) / len(original) < 1.5, f"{r}: size ratio {len(prompt) / len(original):.2f}")

    sys_orig = (PILOT / "system_prompt.txt").read_text()
    sys_new = (OUT / "system_prompt.txt").read_text()
    check(sys_new.startswith(sys_orig) and sys_new[len(sys_orig):].count(".") == 1,
          "system prompt is not the pilot's plus one sentence")

    print(f"checked {len(idx['routes'])} prompts, {n_refs} references ({len(recomputed)} distinct train routes)")
    print(f"size ratio min {idx['size_ratio_min']:.3f} mean {idx['size_ratio_mean']:.3f} max {idx['size_ratio_max']:.3f}")
    if fails:
        print(f"FAILED {len(fails)} checks:")
        for f in fails[:50]:
            print("  " + f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
