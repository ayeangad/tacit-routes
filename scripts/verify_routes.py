#!/usr/bin/env python3
"""Check processed .npz files against the raw JSON by an independent path (no ijson).

Run from tacit-routes/:
  .venv/bin/python -I scripts/verify_routes.py --raw-dir data/raw --out-dir data/processed --n 5

Picks n routes (first, last, and n-2 drawn with seed 20261111 from index.csv). For each raw file
it scans the file in 16 MB chunks, finds the route's key by byte search, cuts out that route's object by
brace matching and parses only that slice with the standard json module. It then checks that:
  - the travel-time matrix equals float32(raw) element for element (and reports max |diff|
    against the float64 raw values),
  - the actual order, lat/lng, zones, types and metadata match exactly,
  - package counts, service time and time windows match a fresh recomputation,
  - the file hash matches index.csv.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


CHUNK = 16 << 20


def find_all(f, key):
    """Byte offsets of key in the file, reading 16 MB at a time (small, bounded memory)."""
    hits, base, prev = [], 0, b""
    f.seek(0)
    while chunk := f.read(CHUNK):
        buf = prev + chunk
        i = buf.find(key)
        while i >= 0:
            hits.append(base - len(prev) + i)
            i = buf.find(key, i + 1)
        prev = buf[-(len(key) - 1):]
        base += len(chunk)
    return sorted(set(hits))


def route_slice(f, rid):
    key = f'"{rid}":'.encode()
    hits = find_all(f, key)
    if len(hits) != 1:
        raise SystemExit(f"{rid}: key found {len(hits)} times")
    f.seek(hits[0] + len(key))
    mm = f.read(4 * CHUNK)  # larger than any single route object
    start = mm.find(b"{")
    depth, j, in_str = 0, start, False
    while True:  # brace matching; strings here never contain braces or escaped quotes
        c = mm[j]
        if c == 0x22:
            in_str = not in_str
        elif not in_str:
            if c == 0x7B:
                depth += 1
            elif c == 0x7D:
                depth -= 1
                if depth == 0:
                    break
        j += 1
    text = mm[start:j + 1]
    return json.loads(re.sub(rb": NaN", b": null", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    index = list(csv.DictReader(open(a.out_dir / "index.csv")))
    middle = random.Random(20261111).sample(index[1:-1], a.n - 2)
    picks = [index[0]] + middle + [index[-1]]
    files = {}
    for name in ("route_data", "actual_sequences", "package_data", "travel_times"):
        files[name] = open(a.raw_dir / f"{name}.json", "rb")

    for row in picks:
        rid = row["route_id"]
        raw = open(a.out_dir / "routes" / f"{rid}.npz", "rb").read()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"], "hash mismatch"
        z = np.load(a.out_dir / "routes" / f"{rid}.npz", allow_pickle=False)
        route = route_slice(files["route_data"], rid)
        act = route_slice(files["actual_sequences"], rid)["actual"]
        pkg = route_slice(files["package_data"], rid)
        tt = route_slice(files["travel_times"], rid)
        ids = list(z["stop_ids"])
        n = len(ids)
        assert set(ids) == set(route["stops"]) and len(ids) == len(route["stops"])
        assert route["stops"][ids[0]]["type"] == "Station"

        raw_m = np.array([[tt[a_][b] for b in ids] for a_ in ids], dtype=np.float64)
        assert np.array_equal(z["travel_time"], raw_m.astype(np.float32)), "matrix mismatch"
        max_diff = float(np.abs(z["travel_time"].astype(np.float64) - raw_m).max())

        driver = sorted(act, key=act.get)
        assert [ids[i] for i in z["actual_order"]] == driver, "order mismatch"

        for i, s in enumerate(ids):
            st = route["stops"][s]
            assert z["lat"][i] == st["lat"] and z["lng"][i] == st["lng"]
            zid = st["zone_id"] if st["zone_id"] is not None else ""
            assert z["zone_ids"][i] == zid and z["stop_type"][i] == st["type"]
        for k, src in (("quality", "route_score"), ("date", "date_YYYY_MM_DD"),
                       ("station", "station_code"), ("departure_time_utc", "departure_time_utc")):
            assert str(z[k]) == route[src], k
        assert str(z["route_id"]) == rid

        dep = datetime.fromisoformat(f"{route['date_YYYY_MM_DD']} {route['departure_time_utc']}")
        n_tw = 0
        for i, s in enumerate(ids):
            p = pkg.get(s, {})
            assert z["n_packages"][i] == len(p)
            svc = sum(q["planned_service_time_seconds"] for q in p.values())
            assert math.isclose(z["service_time"][i], svc, rel_tol=1e-6, abs_tol=1e-3)
            st_ = [(datetime.fromisoformat(q["time_window"]["start_time_utc"]) - dep).total_seconds()
                   for q in p.values() if q["time_window"]["start_time_utc"]]
            en_ = [(datetime.fromisoformat(q["time_window"]["end_time_utc"]) - dep).total_seconds()
                   for q in p.values() if q["time_window"]["end_time_utc"]]
            for got, want in ((z["tw_start"][i], min(st_) if st_ else None),
                              (z["tw_end"][i], max(en_) if en_ else None)):
                assert (math.isnan(got) if want is None else got == np.float32(want)), (s, got, want)
            n_tw += bool(st_)
        print(f"OK {rid}: {n} stops, {int(z['n_packages'].sum())} packages, {n_tw} stops with a "
              f"time window, matrix max |float32 - raw| = {max_diff:.2e} s")
    print(f"all {len(picks)} routes match")


if __name__ == "__main__":
    sys.exit(main())
