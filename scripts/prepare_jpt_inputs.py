#!/usr/bin/env python3
"""Write rc-cli style inputs for jpt-amz (the 2021 winner) from our splits, streaming every file.

Run from tacit-routes/ with -I (the inputs are untrusted downloads):
  .venv/bin/python -I scripts/prepare_jpt_inputs.py --raw-dir data/raw \
      --splits data/processed/splits.json --out-dir runs/jpt/data

Writes (rc-cli layout, the one model_build.sh / model_apply.sh read):
  model_build_inputs/  route_data.json, actual_sequences.json, package_data.json, travel_times.json
                       -> the 5,762 "train" routes only
  model_apply_inputs/  new_route_data.json, new_package_data.json, new_travel_times.json
                       -> the 300 "val" routes only, no driver sequences, route_score removed
  model_score_inputs/  new_actual_sequences.json, new_invalid_sequence_scores.json
                       -> the 300 "val" routes, read only by our scoring script, never by jpt-amz
The 50 "pilot" routes are skipped everywhere. data/eval_frozen/ is never opened.

The four raw files are top-level objects keyed by route ID in the same order, so they are streamed
side by side with ijson, one route at a time (as in build_routes.py). Values are written back with
json.dumps(allow_nan=True), so bare NaN in the raw files comes out as bare NaN again.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import ijson

MEMORY_CAP_BYTES = 2 * 1024**3
NAN_TOKEN = b': NaN'
SENTINEL = "-"


class NaNToSentinel:
    """Rewrite bare NaN (invalid JSON for ijson) to the string "-" while streaming."""

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


def restore_nan(x):
    """Turn the sentinel back into float NaN so the output matches the raw file."""
    if isinstance(x, dict):
        return {k: restore_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [restore_nan(v) for v in x]
    if x == SENTINEL:
        return float("nan")
    return x


class ObjectWriter:
    """Write a top-level JSON object one key at a time."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.tmp = path, path.with_suffix(".part")
        self.f = open(self.tmp, "w")
        self.f.write("{")
        self.n = 0

    def add(self, key, value):
        self.f.write((", " if self.n else "") + json.dumps(key) + ": " + json.dumps(value, allow_nan=True))
        self.n += 1

    def close(self):
        self.f.write("}")
        self.f.close()
        self.tmp.replace(self.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))

    splits = json.load(open(a.splits))
    train = {r["route_id"] for r in splits["train"]}
    val = {r["route_id"] for r in splits["val"]}
    pilot = {r["route_id"] for r in splits["pilot"]}
    assert len(train) == 5762 and len(val) == 300 and len(pilot) == 50
    assert not (train & val) and not (train & pilot) and not (val & pilot)

    invalid = json.load(open(a.raw_dir / "invalid_sequence_scores.json"))  # 0.4 MB, small

    b, p, s = a.out_dir / "model_build_inputs", a.out_dir / "model_apply_inputs", a.out_dir / "model_score_inputs"
    w = {
        "b_route": ObjectWriter(b / "route_data.json"), "b_seq": ObjectWriter(b / "actual_sequences.json"),
        "b_pkg": ObjectWriter(b / "package_data.json"), "b_tt": ObjectWriter(b / "travel_times.json"),
        "a_route": ObjectWriter(p / "new_route_data.json"), "a_pkg": ObjectWriter(p / "new_package_data.json"),
        "a_tt": ObjectWriter(p / "new_travel_times.json"),
        "s_seq": ObjectWriter(s / "new_actual_sequences.json"),
        "s_inv": ObjectWriter(s / "new_invalid_sequence_scores.json"),
    }
    streams = [stream(a.raw_dir / f) for f in
               ("route_data.json", "actual_sequences.json", "package_data.json", "travel_times.json")]
    seen = {"train": 0, "val": 0, "pilot": 0, "other": 0}
    t0 = time.time()
    for k, items in enumerate(zip(*streams)):
        keys = {key for key, _ in items}
        if len(keys) != 1:
            raise SystemExit(f"route {k}: files out of step: {sorted(keys)}")
        rid = items[0][0]
        route, seq, pkg, tt = (restore_nan(v) for _, v in items)
        if rid in train:
            seen["train"] += 1
            w["b_route"].add(rid, route)
            w["b_seq"].add(rid, seq)
            w["b_pkg"].add(rid, pkg)
            w["b_tt"].add(rid, tt)
        elif rid in val:
            seen["val"] += 1
            route = {kk: vv for kk, vv in route.items() if kk != "route_score"}
            w["a_route"].add(rid, route)
            w["a_pkg"].add(rid, pkg)
            w["a_tt"].add(rid, tt)
            w["s_seq"].add(rid, seq)
            w["s_inv"].add(rid, invalid[rid])
        elif rid in pilot:
            seen["pilot"] += 1  # skipped on purpose
        else:
            seen["other"] += 1
        if (k + 1) % 500 == 0:
            print(f"{k + 1} routes, {time.time() - t0:.0f}s, peak RSS "
                  f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
    for s_ in streams:
        if next(s_, None) is not None:
            raise SystemExit("raw files have different route counts")
    for x in w.values():
        x.close()
    print(json.dumps({"seen": seen, "seconds": round(time.time() - t0),
                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))
    if seen != {"train": 5762, "val": 300, "pilot": 50, "other": 0}:
        raise SystemExit("unexpected split counts")


if __name__ == "__main__":
    sys.exit(main())
