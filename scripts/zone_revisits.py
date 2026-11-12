#!/usr/bin/env python3
"""Zone revisits on the 50 pilot routes: driver vs shortest route vs Opus. Descriptive only.

Run from tacit-routes/:
  .venv/bin/python -I scripts/zone_revisits.py --sample-dir data/raw/sample --results-dir results \
      --opus runs/pilot/opus.jsonl

Definition. Take a stop order, drop the station and every stop whose zone_id is missing, and
collapse consecutive stops in the same zone into one "visit". A route that never splits a zone
has exactly one visit per distinct zone. Every extra visit is a return to a zone the route had
already left, so

    zone_revisits = number of visits - number of distinct zones.

The count does not depend on direction (a reversed order has the same revisits).
"""
import argparse
import json
import random
import statistics
from pathlib import Path

SEED = 20261111
N_BOOT = 10_000


def revisits(order, stops):
    zones = [stops[s]["zone_id"] for s in order
             if stops[s]["type"] != "Station" and stops[s].get("zone_id") not in (None, "-")]
    visits = [z for i, z in enumerate(zones) if i == 0 or z != zones[i - 1]]
    return len(visits) - len(set(zones)), len(set(zones)), len(visits)


def check_perm(order, stops, rid, who):
    if sorted(order) != sorted(stops):
        raise SystemExit(f"{who} {rid}: not a permutation of the route's stops")


def boot_ci(diffs, rng):
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return [means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT) - 1]]


def summary(xs):
    return {"mean": sum(xs) / len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--opus", required=True, type=Path)
    a = ap.parse_args()

    manifest = json.load(open(a.sample_dir / "sample_manifest.json"))
    routes = json.load(open(a.sample_dir / "sample_route_data.json"))
    actual = json.load(open(a.sample_dir / "sample_actual.json"))
    shortest = json.load(open(a.results_dir / "proposed_shortest.json"))
    opus = {}
    for line in open(a.opus):
        rec = json.loads(line)
        if not rec["valid"]:
            raise SystemExit(f"opus {rec['route_id']} is not valid")
        opus[rec["route_id"]] = rec["parsed_sequence"]

    ids = manifest["route_ids"]
    if set(opus) != set(ids) or not set(ids) <= set(shortest):
        raise SystemExit("route sets differ between manifest, shortest and opus")

    per = []
    for rid in ids:
        stops = routes[rid]["stops"]
        orders = {
            "driver": sorted(actual[rid]["actual"], key=actual[rid]["actual"].get),
            "shortest": sorted(shortest[rid]["proposed"], key=shortest[rid]["proposed"].get),
            "opus": opus[rid],
        }
        row = {"route_id": rid, "n_stops": len(stops),
               "n_missing_zone_non_station": sum(1 for s in stops.values()
                                                 if s["type"] != "Station"
                                                 and s.get("zone_id") in (None, "-"))}
        for who, order in orders.items():
            # Opus may omit the station; the comparison ignores it either way.
            full = order
            if who == "opus" and len(order) == len(stops) - 1:
                station = next(s for s, v in stops.items() if v["type"] == "Station")
                full = [station] + order
            check_perm(full, stops, rid, who)
            rv, nz, nv = revisits(full, stops)
            row[f"{who}_revisits"] = rv
            row[f"{who}_visits"] = nv
            row["distinct_zones"] = nz
        per.append(row)

    rng = random.Random(SEED)
    arms = ["driver", "shortest", "opus"]
    out = {
        "definition": __doc__.split("Definition.")[1].strip(),
        "routes": len(per),
        "bootstrap": {"seed": SEED, "resamples": N_BOOT, "type": "percentile, paired by route"},
        "distinct_zones": summary([r["distinct_zones"] for r in per]),
        "revisits": {w: summary([r[f"{w}_revisits"] for r in per]) for w in arms},
        "routes_with_zero_revisits": {w: sum(r[f"{w}_revisits"] == 0 for r in per) for w in arms},
        "paired": {},
        "per_route": per,
    }
    for x, y in [("opus", "driver"), ("shortest", "driver"), ("opus", "shortest")]:
        d = [r[f"{x}_revisits"] - r[f"{y}_revisits"] for r in per]
        out["paired"][f"{x}_minus_{y}"] = {
            "mean": sum(d) / len(d), "ci95": boot_ci(d, rng), "median": statistics.median(d),
            f"{x}_more": sum(v > 0 for v in d), f"{x}_fewer": sum(v < 0 for v in d),
            "tied": sum(v == 0 for v in d)}
    with open(a.results_dir / "zone_revisits.json", "w") as f:
        json.dump(out, f, indent=1)

    L = ["# Zone revisits on the 50 pilot routes (descriptive, exploratory)", "",
         "Produced by `scripts/zone_revisits.py`; numbers from `results/zone_revisits.json`.", "",
         "**Definition.** Drop the station and stops with no zone, collapse consecutive stops in the "
         "same zone into one visit. Revisits = visits − distinct zones, i.e. how many times the route "
         "comes back to a zone it already left. 0 means every zone is done in one go. "
         "Direction does not matter.", "",
         "Orders: driver = `sample_actual.json`; shortest = `results/proposed_shortest.json` "
         "(OR-Tools, 10 s); Opus = `parsed_sequence` in `runs/pilot/opus.jsonl` (50/50 valid).", "",
         "## Summary (n = 50)", "",
         "| | Mean | Median | Min | Max | Routes with 0 |", "|---|---|---|---|---|---|"]
    for w in arms:
        s = out["revisits"][w]
        L.append(f"| {w} revisits | {s['mean']:.2f} | {s['median']:g} | {s['min']} | {s['max']} | "
                 f"{out['routes_with_zero_revisits'][w]} |")
    s = out["distinct_zones"]
    L.append(f"| distinct zones | {s['mean']:.2f} | {s['median']:g} | {s['min']} | {s['max']} | |")
    L += ["", "## Paired differences (bootstrap 95% CI, 10,000 resamples, seed 20261111)", "",
          "| Comparison | Mean diff | 95% CI | Median diff | first more / fewer / tied |",
          "|---|---|---|---|---|"]
    for k, p in out["paired"].items():
        x = k.split("_minus_")[0]
        L.append(f"| {k.replace('_minus_', ' − ')} | {p['mean']:+.2f} | [{p['ci95'][0]:+.2f}, "
                 f"{p['ci95'][1]:+.2f}] | {p['median']:+g} | {p[f'{x}_more']} / {p[f'{x}_fewer']} / "
                 f"{p['tied']} |")
    L += ["", "## Per route", "",
          "| Route | Stops | Zones | Driver | Shortest | Opus |", "|---|---|---|---|---|---|"]
    for r in per:
        L.append(f"| {r['route_id'][8:16]} | {r['n_stops']} | {r['distinct_zones']} | "
                 f"{r['driver_revisits']} | {r['shortest_revisits']} | {r['opus_revisits']} |")
    L += ["", "## Caveats", "",
          "- 50 routes, chosen before this question was asked; exploratory, no hypothesis was registered.",
          "- Zone IDs are Amazon's own planning zones; the count uses the full ID (e.g. `G-24.3G`).",
          "- The shortest route comes from a 10 s heuristic, not an optimal tour.", ""]
    (a.results_dir / "zone_revisits.md").write_text("\n".join(L))
    print(json.dumps({k: out[k] for k in ("distinct_zones", "revisits", "paired")}, indent=1))


if __name__ == "__main__":
    main()
