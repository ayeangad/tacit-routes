#!/usr/bin/env python3
"""Build the sandbox folder for the Opus "fair fight" (opus_arena/).

  .venv/bin/python -I scripts/setup_opus_arena.py [arena_dir_name] [runs_subdir]   (defaults: opus_arena opus_fight)

Copies (never symlinks):
  - train-split routes, unchanged (driver order included)  -> opus_arena/data/train/<id>.npz
  - val-split routes with actual_order and quality REMOVED  -> opus_arena/data/val_inputs/<id>.npz
  - the official scorer (score.py, README, SOURCE_COMMIT)   -> opus_arena/scorer/
  - opus_arena/data/train_index.csv (route_id, quality, station, date, n_stops)
  - opus_arena/data/val_ids.txt
Pilot routes are never copied. Writes runs/opus_fight/arena_manifest.json (outside the arena).
"""
import csv
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "routes"
ARENA = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "opus_arena")
RUNDIR = ROOT / "runs" / (sys.argv[2] if len(sys.argv) > 2 else "opus_fight")
REMOVED = ("actual_order", "quality")


def main():
    splits = json.loads((ROOT / "data" / "processed" / "splits.json").read_text())
    ids = {k: [x["route_id"] for x in splits[k]] for k in ("train", "val", "pilot")}
    train, val, pilot = ids["train"], ids["val"], set(ids["pilot"])
    assert not (pilot & set(train)) and not (pilot & set(val)) and not (set(train) & set(val))
    (ARENA / "data" / "train").mkdir(parents=True, exist_ok=True)
    (ARENA / "data" / "val_inputs").mkdir(parents=True, exist_ok=True)
    (ARENA / "scorer").mkdir(parents=True, exist_ok=True)

    rows = []
    for rid in train:
        dst = ARENA / "data" / "train" / f"{rid}.npz"
        shutil.copyfile(SRC / f"{rid}.npz", dst)
        with np.load(dst, allow_pickle=False) as z:
            rows.append({"route_id": rid, "quality": str(z["quality"]), "station": str(z["station"]),
                         "date": str(z["date"]), "n_stops": len(z["stop_ids"])})
    with open(ARENA / "data" / "train_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    val_hashes = {}
    for rid in val:
        with np.load(SRC / f"{rid}.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k not in REMOVED}
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        dst = ARENA / "data" / "val_inputs" / f"{rid}.npz"
        dst.write_bytes(buf.getvalue())
        with np.load(dst, allow_pickle=False) as z:
            bad = [k for k in REMOVED if k in z.files]
            if bad:
                raise SystemExit(f"{rid}: {bad} still present")
        val_hashes[rid] = hashlib.sha256(buf.getvalue()).hexdigest()
    (ARENA / "data" / "val_ids.txt").write_text("\n".join(val) + "\n")

    for name in ("score.py", "README.md", "SOURCE_COMMIT.txt"):
        shutil.copyfile(ROOT / "scripts" / "official_rc_cli" / name, ARENA / "scorer" / name)

    # Final checks over what is actually on disk.
    on_disk_train = {p.stem for p in (ARENA / "data" / "train").glob("*.npz")}
    on_disk_val = {p.stem for p in (ARENA / "data" / "val_inputs").glob("*.npz")}
    assert on_disk_train == set(train) and on_disk_val == set(val)
    assert not (pilot & (on_disk_train | on_disk_val))
    for p in (ARENA / "data" / "val_inputs").glob("*.npz"):
        with np.load(p, allow_pickle=False) as z:
            assert "actual_order" not in z.files and "quality" not in z.files, p
    with np.load(next((ARENA / "data" / "val_inputs").glob("*.npz")), allow_pickle=False) as z:
        val_fields = sorted(z.files)

    out = RUNDIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "arena_manifest.json").write_text(json.dumps({
        "n_train": len(on_disk_train), "n_val": len(on_disk_val), "pilot_in_arena": 0,
        "val_fields_kept": val_fields, "val_fields_removed": list(REMOVED),
        "val_npz_sha256": val_hashes}, indent=1))
    print(f"train {len(on_disk_train)}, val {len(on_disk_val)}, val fields {val_fields}")


if __name__ == "__main__":
    sys.exit(main())
