#!/usr/bin/env python3
"""Fetch a small, seeded sample of High-quality routes from the Amazon Last Mile
Routing Research Challenge (training side) without loading any large file whole.

Run with -I, since it reads untrusted downloads:
  .venv/bin/python -I scripts/fetch_sample.py --raw-dir data/raw --n 50 --seed 20261110

What it does:
  1. Downloads route_data.json, actual_sequences.json and invalid_sequence_scores.json.
     route_data is needed in full because the quality labels live there.
  2. Streams route_data with ijson to list every High route and sample n of them.
  3. Pulls only the sampled routes out of travel_times.json (1.8 GB) on S3 using HTTP
     byte-range requests. A full stream measured about 56 minutes on this connection.
     The routes appear in the same order as route_data.json and a route's size in bytes
     follows from its stop count, so each route's offset can be predicted and then
     confirmed by the route ID actually found there. Any mismatch aborts the run.
  4. Writes sample_*.json plus sample_manifest.json into <raw-dir>/sample/.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import resource
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ijson

BASE = ("https://amazon-last-mile-challenges.s3.amazonaws.com/"
        "almrrc2021/almrrc2021-data-training/model_build_inputs/")
MEMORY_CAP_BYTES = 3 * 1024**3  # crash this process rather than push the laptop into OOM


def download(name, dest):
    if dest.exists():
        print(f"have {name} ({dest.stat().st_size} bytes)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        expected = int(r.headers["Content-Length"])
        with open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    if tmp.stat().st_size != expected:
        raise SystemExit(f"short download for {name}")
    os.replace(tmp, dest)
    print(f"downloaded {name} ({expected} bytes)", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


NAN_TOKEN = b': NaN'
NAN_SENTINEL = '-'  # real zone IDs look like "A-1.2D", so a bare "-" cannot collide


class NaNToSentinel:
    """route_data.json contains bare NaN (zone_id of the station and some stops), which is
    not valid JSON and which ijson rejects. Rewrite it to the same-length string "-" while
    streaming, holding back a few bytes so a token split across two reads is still caught."""

    def __init__(self, raw):
        self.raw, self.tail = raw, b""

    def read(self, n=-1):
        if n == 0:
            return b""
        data = self.tail + self.raw.read(n)
        if len(data) == len(self.tail):  # end of file: flush what was held back
            self.tail = b""
            return data.replace(NAN_TOKEN, b': "-"')
        data = data.replace(NAN_TOKEN, b': "-"')
        self.tail = data[-(len(NAN_TOKEN) - 1):]
        return data[:-(len(NAN_TOKEN) - 1)]


def restore_nan(route):
    """Turn the sentinel back into None so downstream code sees a missing zone."""
    for stop in route.get("stops", {}).values():
        if stop.get("zone_id") == NAN_SENTINEL:
            stop["zone_id"] = None
    return route


def stream_items(path):
    with open(path, "rb") as f:
        yield from ijson.kvitems(NaNToSentinel(f), "", use_float=True)


def stream_filter(path, wanted):
    """Yield (route_id, value) from a top-level JSON object on disk, wanted keys only."""
    for key, value in stream_items(path):
        if key in wanted:
            yield key, value


KEY_RE = re.compile(rb'"(RouteID_[0-9a-f-]+)"\s*:\s*\{')
TT_URL = BASE + "travel_times.json"
PROBE_BYTES = 1_500_000  # wider than the largest route, so every window holds a key
MAX_PROBES = 12
bytes_fetched = [0]


def fetch_range(start, end):
    """Return bytes start..end inclusive, retrying transient network failures."""
    req = urllib.request.Request(TT_URL, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise SystemExit(f"server ignored the range request (HTTP {r.status})")
                data = r.read()
            bytes_fetched[0] += len(data)
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def route_weight(n_stops):
    """Rough JSON size of an n-by-n matrix. Only used to predict offsets."""
    return 15 * n_stops * n_stops + 12 * n_stops + 60


class RangeIndex:
    def __init__(self, order, n_stops, total_bytes):
        self.order = order
        self.rank = {r: i for i, r in enumerate(order)}
        self.cum = [0]
        for r in order:
            self.cum.append(self.cum[-1] + route_weight(n_stops[r]))
        self.scale = total_bytes / self.cum[-1]
        self.total = total_bytes

    def keys_in(self, data, base):
        """(absolute offset, rank) of each route key, checking file order as we go."""
        found = [(base + m.start(), m.group(1).decode()) for m in KEY_RE.finditer(data)]
        ranks = []
        for pos, rid in found:
            if rid not in self.rank:
                raise SystemExit(f"unknown route {rid} at byte {pos}")
            ranks.append((pos, self.rank[rid]))
        if any(a[1] >= b[1] for a, b in zip(ranks, ranks[1:])):
            raise SystemExit("travel_times order differs from route_data order")
        return ranks

    def locate(self, target):
        """Find the byte offset of the key for the route with rank target."""
        anchor_pos, anchor_rank = 0, 0
        for _ in range(MAX_PROBES):
            est = anchor_pos + self.scale * (self.cum[target] - self.cum[anchor_rank])
            start = max(0, min(int(est) - PROBE_BYTES // 2, self.total - PROBE_BYTES))
            data = fetch_range(start, start + PROBE_BYTES - 1)
            keys = self.keys_in(data, start)
            if not keys:
                raise SystemExit(f"no route key in window at byte {start}")
            for pos, rank in keys:
                if rank == target:
                    return pos
            anchor_pos, anchor_rank = min(keys, key=lambda k: abs(k[1] - target))
        raise SystemExit(f"could not locate {self.order[target]} in {MAX_PROBES} probes")

    def fetch_route(self, route_id):
        """Download and parse one route's matrix (about 1 MB, parsed in isolation)."""
        target = self.rank[route_id]
        pos = self.locate(target)
        size = int(self.scale * (self.cum[target + 1] - self.cum[target]) * 1.3) + 65536
        while True:
            end = min(pos + size, self.total) - 1
            data = fetch_range(pos, end)
            keys = list(KEY_RE.finditer(data))
            if keys[0].group(1).decode() != route_id:
                raise SystemExit(f"key mismatch for {route_id}")
            body_start = keys[0].end() - 1  # keep the opening brace
            if len(keys) > 1:
                body = data[body_start:keys[1].start()].rstrip().rstrip(b",").rstrip()
                break
            if end == self.total - 1:  # last route: drop the file's closing brace
                body = data[body_start:].rstrip()
                if not body.endswith(b"}"):
                    raise SystemExit("truncated end of travel_times.json")
                body = body[:-1].rstrip()
                break
            size *= 2
        return route_id, json.loads(body)


def write_json(obj, path):
    tmp = path.with_suffix(".part")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))
    out = args.raw_dir / "sample"
    out.mkdir(parents=True, exist_ok=True)

    for name in ("route_data.json", "actual_sequences.json", "invalid_sequence_scores.json"):
        download(name, args.raw_dir / name)

    # Pass 1: labels only. Each route is built, read, and dropped.
    labels, n_stops, order = {}, {}, []
    for route_id, route in stream_items(args.raw_dir / "route_data.json"):
        labels[route_id] = route["route_score"]
        n_stops[route_id] = len(route["stops"])
        order.append(route_id)
    counts = {k: sum(1 for v in labels.values() if v == k) for k in sorted(set(labels.values()))}
    print("quality labels:", counts, flush=True)

    # Sort before sampling so the draw depends only on the seed, not on file order.
    high = sorted(r for r, q in labels.items() if q == "High")
    sample_ids = sorted(random.Random(args.seed).sample(high, args.n))
    wanted = set(sample_ids)

    routes = {r: restore_nan(v) for r, v in stream_filter(args.raw_dir / "route_data.json", wanted)}
    actual = dict(stream_filter(args.raw_dir / "actual_sequences.json", wanted))
    invalid = dict(stream_filter(args.raw_dir / "invalid_sequence_scores.json", wanted))

    # Pass 2: byte-range requests for only the sampled routes.
    t0 = time.time()
    head = urllib.request.Request(TT_URL, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as r:
        total_bytes = int(r.headers["Content-Length"])
    index = RangeIndex(order, n_stops, total_bytes)
    with ThreadPoolExecutor(max_workers=6) as pool:
        travel = dict(pool.map(index.fetch_route, sample_ids))
    print(f"range-fetched {len(travel)} routes, {bytes_fetched[0] / 1e6:.0f} MB transferred, "
          f"{time.time() - t0:.0f}s", flush=True)

    # The sample is only usable if all four sources agree, so check before writing.
    missing = wanted - set(travel)
    if missing or wanted - set(actual) or wanted - set(invalid) or wanted - set(routes):
        raise SystemExit(f"sample incomplete, missing travel times for {sorted(missing)}")
    for rid in sample_ids:
        stops = set(routes[rid]["stops"])
        if set(travel[rid]) != stops or any(set(row) != stops for row in travel[rid].values()):
            raise SystemExit(f"{rid}: travel matrix does not match stop list")
        if not all(math.isfinite(t) and t >= 0 for row in travel[rid].values() for t in row.values()):
            raise SystemExit(f"{rid}: travel matrix has a non-finite or negative time")
        seq = actual[rid]["actual"]
        if set(seq) != stops or sorted(seq.values()) != list(range(len(stops))):
            raise SystemExit(f"{rid}: actual sequence is not a permutation of the stops")
        first = min(seq, key=seq.get)
        if routes[rid]["stops"][first]["type"] != "Station":
            raise SystemExit(f"{rid}: position 0 is not the station")

    write_json(routes, out / "sample_route_data.json")
    write_json(actual, out / "sample_actual.json")
    write_json(invalid, out / "sample_invalid.json")
    write_json(travel, out / "sample_travel_times.json")

    stop_counts = [len(routes[r]["stops"]) for r in sample_ids]
    stations = {}
    for r in sample_ids:
        code = routes[r]["station_code"]
        stations[code] = stations.get(code, 0) + 1
    manifest = {
        "source": BASE,
        "side": "training (model_build_inputs); evaluation side not touched",
        "quality_label_counts_all_training_routes": counts,
        "quality_filter": "High",
        "seed": args.seed,
        "n": args.n,
        "sampling": "random.Random(seed).sample(sorted High route IDs, n)",
        "route_ids": sample_ids,
        "stations": stations,
        "stops_min_mean_max": [min(stop_counts), sum(stop_counts) / len(stop_counts), max(stop_counts)],
        "travel_times_fetch": {"method": "HTTP byte ranges, offsets predicted then verified by route ID",
                               "file_bytes": total_bytes, "bytes_transferred": bytes_fetched[0],
                               "stored_on_disk": "sampled routes only"},
        "sha256": {p.name: sha256(p) for p in sorted(args.raw_dir.glob("*.json"))},
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    write_json(manifest, out / "sample_manifest.json")
    print(json.dumps({k: manifest[k] for k in
                      ("quality_label_counts_all_training_routes", "n", "stations",
                       "stops_min_mean_max", "peak_rss_mb")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
