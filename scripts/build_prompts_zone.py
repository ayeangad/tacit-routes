#!/usr/bin/env python3
"""Build the zone-level "swap the brain" prompts: Opus picks only the ORDER OF ZONES.

  .venv/bin/python -I scripts/build_prompts_zone.py

For each of the 50 pilot routes (same routes and order as runs/pilot) this writes
runs/pilot_zone/prompts/<route>.txt, plus runs/pilot_zone/system_prompt.txt and
runs/pilot_zone/prompts_index.json. Afterwards it re-reads every prompt and runs the leak
checks listed at the bottom; it exits non-zero if any fails.

Each prompt holds:
  - the station's ID and lat/lon;
  - every zone: its ID, centroid lat/lon (mean of its stops' lat/lon) and stop count;
  - the zone-to-zone travel-time matrix: MEAN travel time over every (stop in zone A, stop in
    zone B) pair, in seconds. This is exactly v0's `prepare_route()["ztt"]`, the number the
    tiny model sees, not a centroid-to-centroid time;
  - station -> zone (mean over the zone's stops of station -> stop) and zone -> station
    (mean of stop -> station), also from prepare_route;
  - the zone sequences of the 3 most similar past routes (see below);
  - the request: a JSON list of zone IDs, each zone exactly once, starting after the station.

Zones use v0's definition (v0_model.zone_membership): a stop with no zone ID takes the zone of
its nearest stop by travel time (there and back). So the zones listed are exactly the zones
the inside-zone solver later fills in.

Similarity rule for past routes: the build_prompts_history.py rule, made TIME-RESPECTING.
  - candidates: TRAIN routes (splits.json "train"), same station, all quality labels, dated
    STRICTLY BEFORE the target route's date (so same-day routes are excluded);
  - score: number of distinct zone IDs shared with the target route;
  - ties: nearest date first, then route ID; must share at least 1 zone; top 3.
  - a past route's zone sequence = first visit to each zone in its driver's actual order,
    same zone definition as above (identical to build_prompts_history.train_zone_order).

What is read for the pilot route itself: its npz, but ONLY the keys stop_ids, lat, lng,
zone_ids, travel_time, station, date, route_id. Its actual_order is never loaded here.
Train npz files are read in full (their driver orders are the history). Val routes are never
opened.
"""
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from v0_model import prepare_route, zone_membership  # noqa: E402

ROUTES = ROOT / "data/processed/routes"
OUT = ROOT / "runs/pilot_zone"
K_REF = 3
PILOT_KEYS = ("stop_ids", "lat", "lng", "zone_ids", "travel_time", "station", "date", "route_id",
              "n_packages", "tw_end", "service_time")  # last 3: prepare_route needs them; not shown

SYSTEM = ("You are a delivery planner. A delivery route's stops are grouped into zones. "
          "Your job is to choose the order in which to visit the zones, the way experienced "
          "delivery drivers do. Answer with a JSON list only, no other text.")

TEMPLATE = """A delivery van leaves the station and must deliver to every zone below. All stops in a zone are delivered together, so you only choose the order of the zones. The order of stops inside each zone is decided separately.

Station: {station} at {slat:.6f} {slng:.6f}

Zones ({Z}). Each line: zone ID, centroid latitude, centroid longitude, number of stops. Lines are sorted by zone ID.
{zone_lines}

Travel times in seconds. Each number is the mean driving time over all pairs of stops (one stop in the first zone, one in the second).

From the station to each zone, in the zone order above:
{from_station}

From each zone back to the station, in the zone order above:
{to_station}

Zone-to-zone matrix. Row = from, column = to; columns are in the zone order above. The diagonal is the mean time between stops inside the same zone.
{matrix}

{history}

Output the order in which to visit the zones as a JSON list of zone IDs, starting with the first zone after leaving the station. Include each of the {Z} zones exactly once and nothing else. Do not include the station."""

HIST_HEAD = """Past routes from the same station, driven before this route's date. These are the {k} earlier routes that share the most zone IDs with this route. Each shows its date, how many of this route's {Z} zones it also visits, and its zones in the order its driver first visited them. Past routes may contain zones that are not on this route.
{lines}"""

HIST_NONE = "Past routes from the same station: no earlier route from this station visits any of this route's zones, so none is listed."


def load_pilot(route_id):
    with np.load(ROUTES / f"{route_id}.npz", allow_pickle=False) as d:
        assert "actual_order" in d.files
        return {k: d[k] for k in PILOT_KEYS}            # actual_order deliberately not loaded


def train_zone_order(route_id):
    with np.load(ROUTES / f"{route_id}.npz", allow_pickle=False) as d:
        route = {k: d[k] for k in ("stop_ids", "travel_time", "zone_ids", "actual_order")}
    names, zone_of = zone_membership(route)
    seen, order = set(), []
    for i in route["actual_order"][1:]:
        z = names[int(zone_of[int(i)])]
        if z not in seen:
            seen.add(z)
            order.append(z)
    return order


def zone_view(route):
    """Everything the prompt shows about one route's zones, from v0's prepare_route."""
    P = prepare_route(route)
    lat, lng = route["lat"].astype(np.float64), route["lng"].astype(np.float64)
    zones = [{"zone_id": P["names"][k], "lat": float(lat[m].mean()), "lng": float(lng[m].mean()),
              "n_stops": int(len(m))} for k, m in enumerate(P["members"])]
    Z = P["Z"]
    return P, zones, P["ztt"][:Z, :Z], P["ztt"][Z], P["to_station"]


def main():
    splits = json.loads((ROOT / "data/processed/splits.json").read_text())
    train_ids = {x["route_id"] for x in splits["train"]}
    pilot_ids = {x["route_id"] for x in splits["pilot"]}
    val_ids = {x["route_id"] for x in splits["val"]}
    assert not train_ids & (pilot_ids | val_ids) and not pilot_ids & val_ids
    manifest = json.loads((ROOT / "data/raw/sample/sample_manifest.json").read_text())["route_ids"]
    assert set(manifest) == pilot_ids

    meta = {}
    with open(ROOT / "data/processed/index.csv") as f:
        for row in csv.DictReader(f):
            meta[row["route_id"]] = (row["station"], date.fromisoformat(row["date"]))

    stations = {meta[r][0] for r in manifest}
    by_station = {}
    for rid in sorted(train_ids):
        st, d = meta[rid]
        if st in stations:
            by_station.setdefault(st, []).append((rid, d))
    cache = {}

    def zones_of(rid):
        assert rid in train_ids
        if rid not in cache:
            cache[rid] = train_zone_order(rid)
        return cache[rid]

    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    (OUT / "system_prompt.txt").write_text(SYSTEM)
    index = []
    for r in manifest:
        route = load_pilot(r)
        station, target_date = str(route["station"]), date.fromisoformat(str(route["date"]))
        assert (station, target_date) == meta[r]
        P, zones, zz, from_st, to_st = zone_view(route)
        Z = len(zones)
        mine = set(P["names"])
        ranked = []
        for rid, d in by_station.get(station, []):
            if d >= target_date:                       # time-respecting: strictly earlier only
                continue
            shared = len(mine & set(zones_of(rid)))
            if shared > 0:
                ranked.append(((-shared, (target_date - d).days, rid), rid, d, shared))
        ranked.sort()
        refs = ranked[:K_REF]
        if refs:
            lines = [f"{d.isoformat()}: shares {s} of {Z} zones ({s / Z:.0%}); {len(zones_of(rid))} zones "
                     f"in driver order: {' '.join(zones_of(rid))}" for _, rid, d, s in refs]
            history = HIST_HEAD.format(k=len(refs), Z=Z, lines="\n".join(lines))
        else:
            history = HIST_NONE
        ids = [z["zone_id"] for z in zones]
        w = max(len(i) for i in ids)
        prompt = TEMPLATE.format(
            station=station, slat=float(route["lat"][0]), slng=float(route["lng"][0]), Z=Z,
            zone_lines="\n".join(f"{z['zone_id']} {z['lat']:.6f} {z['lng']:.6f} {z['n_stops']}" for z in zones),
            from_station=" ".join(f"{z}:{round(float(v))}" for z, v in zip(ids, from_st)),
            to_station=" ".join(f"{z}:{round(float(v))}" for z, v in zip(ids, to_st)),
            matrix="\n".join(f"{ids[a]:<{w}} | " + " ".join(str(round(float(v))) for v in zz[a]) for a in range(Z)),
            history=history)
        (OUT / "prompts" / f"{r}.txt").write_text(prompt)
        index.append({"route_id": r, "station": station, "date": target_date.isoformat(),
                      "n_stops_including_station": int(len(route["stop_ids"])), "n_zones": Z,
                      "zone_ids": ids, "prompt_chars": len(prompt),
                      "train_candidates_same_station_before_date":
                          sum(1 for _, d in by_station.get(station, []) if d < target_date),
                      "references": [{"route_id": rid, "date": d.isoformat(), "shared_zones": s,
                                      "zone_order": zones_of(rid)} for _, rid, d, s in refs]})

    info = {"similarity_rule": ("TRAIN routes, same station, dated strictly before the target date; ranked by "
                                "shared distinct zone IDs, ties nearest date then route ID; >= 1 shared zone; top 3"),
            "zone_definition": "v0_model.zone_membership (no-zone stops take the zone of their nearest stop)",
            "travel_times": ("mean over all stop pairs (v0 prepare_route ztt); station->zone and zone->station are "
                             "means over the zone's stops; rounded to whole seconds"),
            "centroid": "mean lat/lon of the zone's stops (after the no-zone assignment)",
            "system_prompt_chars": len(SYSTEM), "routes": index}
    (OUT / "prompts_index.json").write_text(json.dumps(info, indent=1))
    n_refs = [len(x["references"]) for x in index]
    print(f"{len(index)} prompts; chars mean {np.mean([x['prompt_chars'] for x in index]):.0f} "
          f"max {max(x['prompt_chars'] for x in index)}; zones mean {np.mean([x['n_zones'] for x in index]):.1f} "
          f"max {max(x['n_zones'] for x in index)}; refs per route: "
          f"{ {k: n_refs.count(k) for k in sorted(set(n_refs))} }")
    return check(info, meta, train_ids, pilot_ids, val_ids)


def check(info, meta, train_ids, pilot_ids, val_ids):
    """Leak checks, run on the written files. Reads pilot actual_order HERE ONLY, to prove the
    prompts do not contain the pilot drivers' zone orders. Val npz files are never opened."""
    problems, same_run = [], []
    pilot_orders = {}
    for x in info["routes"]:
        with np.load(ROUTES / f"{x['route_id']}.npz", allow_pickle=False) as d:
            route = {k: d[k] for k in d.files}
        P = prepare_route(route)
        seen, order = set(), []
        for i in route["actual_order"][1:]:
            k = int(P["zone_of"][int(i)])
            if k not in seen:
                seen.add(k)
                order.append(P["names"][k])
        pilot_orders[x["route_id"]] = " ".join(order)
    for x in info["routes"]:
        r = x["route_id"]
        text = (OUT / "prompts" / f"{r}.txt").read_text()
        for ref in x["references"]:
            rid = ref["route_id"]
            if rid not in train_ids or rid in pilot_ids or rid in val_ids:
                problems.append(f"{r}: reference {rid} not a train-only route")
            st, d = meta[rid]
            if st != x["station"]:
                problems.append(f"{r}: reference {rid} from station {st}")
            if not d < date.fromisoformat(x["date"]):
                problems.append(f"{r}: reference {rid} dated {d}, not before {x['date']}")
            if ref["zone_order"] != train_zone_order(rid):
                problems.append(f"{r}: reference {rid} zone order does not recompute")
        if "RouteID" in text:
            problems.append(f"{r}: a route ID appears in the prompt")
        # Reference lines were just verified to come from earlier TRAIN routes. Remove them, then
        # no pilot driver's zone order may appear anywhere in what is left.
        outside = text
        for ref in x["references"]:
            tail = "zones in driver order: " + " ".join(ref["zone_order"]) + "\n"
            if tail not in outside + "\n":
                problems.append(f"{r}: reference {ref['route_id']} line not found verbatim")
            outside = (outside + "\n").replace(tail, "zones in driver order: <checked>\n", 1)
        for pr, order in pilot_orders.items():
            if len(order.split()) > 1 and order in outside:
                problems.append(f"{r}: pilot route {pr}'s driver zone order appears outside the references")
        # Information only, not a leak: an earlier train route can contain the same zone run.
        if any(pilot_orders[r] in " ".join(ref["zone_order"]) for ref in x["references"]):
            same_run.append(r)
        for z in x["zone_ids"]:
            if z not in text:
                problems.append(f"{r}: zone {z} missing from prompt")
    if problems:
        print("LEAK CHECK FAILED:\n  " + "\n  ".join(problems))
        return 1
    n = sum(len(x["references"]) for x in info["routes"])
    print(f"leak check passed: {len(info['routes'])} prompts, {n} references, all TRAIN, same station, "
          f"dated strictly before the target; no route IDs; no pilot driver zone order outside the "
          f"verified reference lines")
    print(f"info: on {len(same_run)} route(s) an earlier train route's zone sequence contains this route's "
          f"driver zone order as an unbroken run (real history, not a leak): {same_run}")
    info["leak_check"] = {"passed": True, "routes_where_a_reference_contains_driver_zone_run": same_run}
    (OUT / "prompts_index.json").write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
