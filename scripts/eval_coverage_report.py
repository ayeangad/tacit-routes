#!/usr/bin/env python3
"""How much station history do the final-test routes have? Coverage of their zones in TRAIN history.

Approved in docs/plan.md item 5. Reads ONLY:
  - data/eval_frozen/routes/<id>.npz: fields station, date, zone_ids (apply-side inputs; never order)
  - runs/v0/history_train.json: train-split zone orders (station, date, zones)
  - data/processed/routes/<val id>.npz for the 300 val routes: station, date, zone_ids (for the same
    numbers on val, so the two can be compared)
Never opens data/eval_frozen/raw/ (asserted). Writes results/eval_coverage.json and .md.

For each route, against all train routes from the same station (the competition history rule; for val,
leave nothing out since val routes are not in train):
  - share of its zones seen in train at full ('A-2.2A'), micro ('A-2.2') and macro ('A-2') level
  - best single reference: largest share of its zones found in one train route
  - zones never seen at the station at any level ("new zones")
Grouped by: eval before training dates / inside / after, and by days of gap to the nearest train date.

  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/eval_coverage_report.py
"""
import collections
import datetime as dt
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval_frozen" / "routes"
FIELDS = ("station", "date", "zone_ids")
T0, T1 = dt.date(2018, 7, 19), dt.date(2018, 8, 26)


def lv(z, level):
    if level == 0:
        return z
    m = re.match(r"^([A-Z]+-\d+)\.(\d+)([A-Z]+)$", z)
    if not m:
        return z
    return m.group(1) + "." + m.group(2) if level == 1 else m.group(1)


def load(path):
    with np.load(path, allow_pickle=False) as d:
        assert "actual_order" not in d.files or "eval_frozen" not in str(path)
        return str(d["station"]), dt.date.fromisoformat(str(d["date"])[:10]), sorted({str(z) for z in d["zone_ids"] if str(z)})


def main():
    assert "raw" not in str(EVAL)
    rows = json.load(open(ROOT / "runs" / "v0" / "history_train.json"))["routes"]
    by_st = collections.defaultdict(list)
    for r in rows:
        by_st[r["station"]].append(set(r["zones"]))
    seen = {s: [set().union(*rs) if L == 0 else {lv(z, L) for x in rs for z in x} for L in (0, 1, 2)]
            for s, rs in by_st.items()}

    def cov(station, zones):
        if station not in by_st or not zones:
            return None
        zs = set(zones)
        out = {f"seen_L{L}": len({lv(z, L) for z in zs} & seen[station][L]) / len({lv(z, L) for z in zs}) for L in (0, 1, 2)}
        out["best_ref"] = max(len(zs & r) for r in by_st[station]) / len(zs)
        out["new_zones"] = sum(1 for z in zs if lv(z, 2) not in seen[station][2] and z not in seen[station][0])
        return out

    splits = json.load(open(ROOT / "data" / "processed" / "splits.json"))
    res = {"val": [], "eval": []}
    for x in splits["val"]:
        s, d, zs = load(ROOT / "data" / "processed" / "routes" / f"{x['route_id']}.npz")
        c = cov(s, zs)
        if c:
            res["val"].append({"station": s, "date": str(d), "gap": 0, "where": "inside", **c})
    no_hist = []
    for p in sorted(EVAL.glob("*.npz")):
        s, d, zs = load(p)
        gap = (T0 - d).days if d < T0 else (d - T1).days if d > T1 else 0
        where = "before" if d < T0 else "after" if d > T1 else "inside"
        c = cov(s, zs)
        if c is None:
            no_hist.append({"route_id": p.stem, "station": s})
            continue
        res["eval"].append({"station": s, "date": str(d), "gap": gap, "where": where, **c})

    def summ(g):
        keys = ("seen_L0", "seen_L1", "seen_L2", "best_ref")
        return {"n": len(g), **{k: round(st.mean(r[k] for r in g), 3) for k in keys},
                "best_ref_median": round(st.median(r["best_ref"] for r in g), 3),
                "new_zones_per_route": round(st.mean(r["new_zones"] for r in g), 2)}

    out = {"val": summ(res["val"]), "eval_all": summ(res["eval"]),
           "eval_by_where": {w: summ([r for r in res["eval"] if r["where"] == w]) for w in ("before", "inside", "after")},
           "eval_by_gap": {}, "eval_no_station_history": no_hist}
    for lo, hi in ((0, 0), (1, 7), (8, 14), (15, 21), (22, 28), (29, 38), (39, 48)):
        g = [r for r in res["eval"] if lo <= r["gap"] <= hi]
        if g:
            out["eval_by_gap"][f"{lo}-{hi}"] = summ(g)
    (ROOT / "results" / "eval_coverage.json").write_text(json.dumps(out, indent=1))

    lines = ["# History coverage of the final-test routes (inputs only; no answers read)", "",
             "Made by `scripts/eval_coverage_report.py`. Reads only station, date and zone IDs. Train history = all "
             "train routes at the same station (competition rule). Val shown for comparison.", "",
             "| Group | Routes | Zones seen: full | micro | macro | Best single reference (mean / median) | New zones per route |",
             "|---|---|---|---|---|---|---|"]
    def row(name, s):
        lines.append(f"| {name} | {s['n']} | {s['seen_L0']:.3f} | {s['seen_L1']:.3f} | {s['seen_L2']:.3f} | "
                     f"{s['best_ref']:.3f} / {s['best_ref_median']:.3f} | {s['new_zones_per_route']} |")
    row("Val (300, in-time)", out["val"])
    row("Eval, all", out["eval_all"])
    for w, s in out["eval_by_where"].items():
        row(f"Eval, {w} training dates", s)
    for k, s in out["eval_by_gap"].items():
        row(f"Eval, {k} days from training dates", s)
    lines += ["", f"Routes at a station with no train history: {len(no_hist)} ({', '.join(r['station'] for r in no_hist)})."]
    (ROOT / "results" / "eval_coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
