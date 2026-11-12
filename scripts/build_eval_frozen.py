#!/usr/bin/env python3
"""Build the FROZEN final test set from the official evaluation apply inputs.

Run from tacit-routes/ with -I (the inputs are untrusted downloads):
  .venv/bin/python -I scripts/build_eval_frozen.py --eval-dir data/eval_frozen

Reads ONLY data/eval_frozen/raw/apply/ (eval_route_data, eval_package_data, eval_travel_times),
streamed side by side with build_routes.py's streaming and conversion code. Writes one .npz per
route to <eval-dir>/routes/ in the training format but with NO actual_order (the apply side has no
driver order), plus <eval-dir>/index.csv.

The score files in raw/score/ (driver sequences) are NEVER opened here. MANIFEST.json gets their
sha256 and byte size only, computed by hashing raw bytes (not parsing), and they are set to mode 444.
They are to be opened only once, by the pre-registered final scoring script.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import resource
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_routes as br  # noqa: E402

MEMORY_CAP_BYTES = 1400 * 1024**2
APPLY = ("eval_route_data.json", "eval_package_data.json", "eval_travel_times.json")
SCORE = ("eval_actual_sequences.json", "eval_invalid_sequence_scores.json")


def file_hash(path):
    """sha256 of raw bytes; no parsing, nothing printed about the contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    a = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP_BYTES, MEMORY_CAP_BYTES))

    apply_dir, score_dir = a.eval_dir / "raw" / "apply", a.eval_dir / "raw" / "score"
    routes_dir = a.eval_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    streams = [br.stream(apply_dir / f) for f in APPLY]
    rows, t0 = [], time.time()
    for k, items in enumerate(zip(*streams)):
        keys = {key for key, _ in items}
        if len(keys) != 1:
            raise SystemExit(f"route {k}: files out of step: {sorted(keys)}")
        rid = items[0][0]
        arrays = br.convert(rid, items[0][1], None, items[1][1], items[2][1])
        assert "actual_order" not in arrays
        buf = io.BytesIO()
        br.np.savez_compressed(buf, **arrays)
        data = buf.getvalue()
        path = routes_dir / f"{rid}.npz"
        with open(path.with_suffix(".part"), "wb") as f:
            f.write(data)
        os.replace(path.with_suffix(".part"), path)
        rows.append({"route_id": rid, "station": str(arrays["station"]), "date": str(arrays["date"]),
                     "n_stops": len(arrays["stop_ids"]), "quality": str(arrays["quality"]),
                     "sha256": hashlib.sha256(data).hexdigest()})
        if (k + 1) % 250 == 0:
            print(f"{k + 1} routes, {time.time() - t0:.0f}s, peak RSS "
                  f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
    for s, name in zip(streams, APPLY):
        if next(s, None) is not None:
            raise SystemExit(f"{name} has more routes than the others")

    with open(a.eval_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    manifest = {
        "what": "Amazon Last Mile Routing Research Challenge 2021, official evaluation set. FROZEN final test.",
        "source": "s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-evaluation/ (anonymous)",
        "n_routes": len(rows),
        "apply_inputs": {f: {"bytes": (apply_dir / f).stat().st_size, "sha256": file_hash(apply_dir / f)}
                         for f in APPLY},
        "score_inputs_SEALED": {f: {"bytes": (score_dir / f).stat().st_size, "sha256": file_hash(score_dir / f)}
                                for f in SCORE},
        "score_rule": "Score files are hashed as raw bytes only; never parsed. Open them once, in the "
                      "pre-registered final scoring script, after checking these hashes.",
        "index_csv_sha256": file_hash(a.eval_dir / "index.csv"),
    }
    with open(a.eval_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1)
    for fname in SCORE:
        os.chmod(score_dir / fname, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"done: {len(rows)} routes, {time.time() - t0:.0f}s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    sys.exit(main())
