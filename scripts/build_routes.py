#!/usr/bin/env python3
"""Convert the full training side into one compact .npz per route, streaming every file.

Run from tacit-routes/ with -I (the inputs are untrusted downloads):
  .venv/bin/python -I scripts/build_routes.py --raw-dir data/raw --out-dir data/processed

Inputs (data/raw/): route_data.json, actual_sequences.json, package_data.json, travel_times.json.
All four are top-level objects keyed by route ID in the same order, so they are streamed side by
side with ijson, one route at a time; any key mismatch aborts the run. Nothing is loaded whole.

Per route, data/processed/routes/<route_id>.npz holds (stop order: station first, then the other
stops sorted by stop ID):
  stop_ids      <U..   stop IDs
  lat, lng      float64
  zone_ids      <U..   "" when the raw zone_id is missing (NaN)
  stop_type     <U..   "Station" or "Dropoff"
  travel_time   float32 [n, n]  travel_time[i, j] = seconds from stop i to stop j
  n_packages    int16
  tw_start      float32  earliest package time-window start, seconds after departure; NaN if none
  tw_end        float32  latest package time-window end, seconds after departure; NaN if none
  service_time  float32  sum of planned_service_time_seconds over the stop's packages
  actual_order  int16    driver's visiting order as indices into stop_ids (starts with 0);
                         absent when convert() gets actual=None (frozen eval set, build_eval_frozen.py)
  route_id, quality, date, station, departure_time_utc  0-d <U arrays
  executor_capacity_cm3  float64 scalar
Loads with np.load(path) and allow_pickle=False.

Also writes data/processed/index.csv (route_id, quality, station, date, n_stops, sha256).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import ijson
import numpy as np

MEMORY_CAP_BYTES = 3 * 1024**3
NAN_TOKEN = b': NaN'
SENTINEL = "-"  # same length trick as fetch_sample.py; no real value is a bare "-"


class NaNToSentinel:
    """Rewrite bare NaN (invalid JSON) to the string "-" while streaming."""

    def __init__(self, raw):
        self.raw, self.tail = raw, b""

    def read(self, n=-1):
        if n == 0:
            return b""
        data = self.tail + self.raw.read(n)
        if len(data) == len(self.tail):
            self.tail = b""
            return data.replace(NAN_TOKEN, b': "-"')
        data = data.replace(NAN_TOKEN, b': "-"')
        self.tail = data[-(len(NAN_TOKEN) - 1):]
        return data[:-(len(NAN_TOKEN) - 1)]


def stream(path):
    f = open(path, "rb", buffering=1 << 20)
    return ijson.kvitems(NaNToSentinel(f), "", use_float=True)


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def convert(rid, route, actual, packages, travel):
    stops = route["stops"]
    station = [s for s, v in stops.items() if v["type"] == "Station"]
    if len(station) != 1:
        raise SystemExit(f"{rid}: {len(station)} stations")
    ids = station + sorted(s for s in stops if s != station[0])
    pos = {s: i for i, s in enumerate(ids)}
    n = len(ids)

    if set(travel) != set(stops) or any(len(travel[s]) != n for s in ids):
        raise SystemExit(f"{rid}: travel matrix does not match stop list")
    tt = np.empty((n, n), dtype=np.float32)
    for i, a in enumerate(ids):
        row = travel[a]
        tt[i] = [row[b] for b in ids]
    if not np.isfinite(tt).all() or (tt < 0).any():
        raise SystemExit(f"{rid}: non-finite or negative travel time")

    order = None
    if actual is not None:  # None for the evaluation apply side (no driver order there)
        seq = actual["actual"]
        if set(seq) != set(stops) or sorted(seq.values()) != list(range(n)):
            raise SystemExit(f"{rid}: actual sequence is not a permutation of the stops")
        order = np.array([pos[s] for s in sorted(seq, key=seq.get)], dtype=np.int16)
        if order[0] != 0:
            raise SystemExit(f"{rid}: driver order does not start at the station")

    if not set(packages) <= set(stops):
        raise SystemExit(f"{rid}: packages at unknown stops")
    dep = parse_ts(f"{route['date_YYYY_MM_DD']} {route['departure_time_utc']}")
    npk = np.zeros(n, dtype=np.int16)
    tws = np.full(n, np.nan, dtype=np.float32)
    twe = np.full(n, np.nan, dtype=np.float32)
    svc = np.zeros(n, dtype=np.float32)
    for s, pk in packages.items():
        i = pos[s]
        npk[i] = len(pk)
        starts, ends, total = [], [], 0.0
        for p in pk.values():
            w = p["time_window"]
            if w["start_time_utc"] not in (None, SENTINEL):
                starts.append((parse_ts(w["start_time_utc"]) - dep).total_seconds())
            if w["end_time_utc"] not in (None, SENTINEL):
                ends.append((parse_ts(w["end_time_utc"]) - dep).total_seconds())
            total += p["planned_service_time_seconds"]
        if starts:
            tws[i] = min(starts)
        if ends:
            twe[i] = max(ends)
        svc[i] = total

    zone = [stops[s].get("zone_id") for s in ids]
    out = dict(
        stop_ids=np.array(ids),
        lat=np.array([stops[s]["lat"] for s in ids], dtype=np.float64),
        lng=np.array([stops[s]["lng"] for s in ids], dtype=np.float64),
        zone_ids=np.array(["" if z in (None, SENTINEL) else z for z in zone]),
        stop_type=np.array([stops[s]["type"] for s in ids]),
        travel_time=tt, n_packages=npk, tw_start=tws, tw_end=twe, service_time=svc,
        actual_order=order,
        route_id=np.array(rid), quality=np.array(route.get("route_score", "")),
        date=np.array(route["date_YYYY_MM_DD"]), station=np.array(route["station_code"]),
        departure_time_utc=np.array(route["departure_time_utc"]),
        executor_capacity_cm3=np.array(float(route["executor_capacity_cm3"])),
    )
    if order is None:
        del out["actual_order"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many routes (testing)")
    a = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))

    routes_dir = a.out_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    streams = [stream(a.raw_dir / f) for f in
               ("route_data.json", "actual_sequences.json", "package_data.json", "travel_times.json")]
    rows, t0 = [], time.time()
    for k, items in enumerate(zip(*streams)):
        keys = {key for key, _ in items}
        if len(keys) != 1:
            raise SystemExit(f"route {k}: files out of step: {sorted(keys)}")
        rid = items[0][0]
        arrays = convert(rid, items[0][1], items[1][1], items[2][1], items[3][1])
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        data = buf.getvalue()
        path = routes_dir / f"{rid}.npz"
        with open(path.with_suffix(".part"), "wb") as f:
            f.write(data)
        os.replace(path.with_suffix(".part"), path)
        rows.append({"route_id": rid, "quality": str(arrays["quality"]), "station": str(arrays["station"]),
                     "date": str(arrays["date"]), "n_stops": len(arrays["stop_ids"]),
                     "sha256": hashlib.sha256(data).hexdigest()})
        if (k + 1) % 250 == 0:
            print(f"{k + 1} routes, {time.time() - t0:.0f}s, peak RSS "
                  f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
        if a.limit and k + 1 >= a.limit:
            break
    else:
        # zip stops at the shortest stream; make sure none had routes left over
        for s, name in zip(streams, ("route_data", "actual", "package", "travel")):
            if next(s, None) is not None:
                raise SystemExit(f"{name} has more routes than the others")

    with open(a.out_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} routes, {time.time() - t0:.0f}s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
