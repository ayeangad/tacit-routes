#!/usr/bin/env python3
"""Inputs-only validity check of the stage-1 predictions, BEFORE any answer is opened.
For every arm in runs/final/predictions/MANIFEST.json (smoke_* skipped): every expected route present
(all eval routes; the 500-route sample for opus_model_* arms; the 50 pilot routes for pilot_* arms),
each order a permutation of the route's stop IDs that starts at the station, and the file's sha256
still matching the manifest. Writes runs/final/validity.json; problems also go to incidents.log.
  .venv/bin/python -I scripts/final_validity_check.py"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_common as fc  # noqa: E402

man = json.load(open(fc.PRED / "MANIFEST.json"))
all_ids = fc.eval_ids()
sample = json.load(open(fc.SAMPLE_FILE))["route_ids"] if fc.SAMPLE_FILE.exists() else []
splits = json.load(open(fc.ROOT / "data" / "processed" / "splits.json"))
pilot = sorted(r["route_id"] for r in splits["pilot"])
stops = {}


def route_stops(rid, kind):
    if rid not in stops:
        path = (fc.ROOT / "data" / "processed" / "routes" if kind == "pilot" else fc.EVAL_ROUTES) / f"{rid}.npz"
        with np.load(path, allow_pickle=False) as d:
            stops[rid] = [str(s) for s in d["stop_ids"]]
    return stops[rid]


report, bad_total = {}, 0
for arm, info in sorted(man.items()):
    if arm.startswith("smoke_"):
        continue
    kind = "pilot" if arm.startswith("pilot_") else "eval"
    # option B (2026-11-11): repeat 2 of each Opus-built system runs on the 500 sample only -> expect the sample
    expected = pilot if kind == "pilot" else sample if (arm.startswith("opus_model_") or re.match(r"^opus_run\d_rep2$", arm)) else all_ids
    f = fc.PRED / info["file"]
    sha_ok = fc.sha256_file(f) == info["sha256"]
    orders = json.load(open(f))["orders"]
    missing = [r for r in expected if r not in orders]
    extra = [r for r in orders if r not in set(expected)]
    invalid = []
    for r in expected:
        if r in orders:
            s = route_stops(r, kind)
            o = orders[r]
            if not (o and o[0] == s[0] and sorted(o) == sorted(s)):
                invalid.append(r)
    report[arm] = {"expected": len(expected), "present": len(expected) - len(missing), "missing": missing,
                   "invalid": invalid, "extra": len(extra), "sha256_matches_manifest": sha_ok}
    n_bad = len(missing) + len(invalid) + (0 if sha_ok else 1)
    bad_total += n_bad
    flag = "OK" if n_bad == 0 else "PROBLEM"
    if n_bad and not arm.startswith("opus_model_strict"):
        fc.incident(arm, f"validity: {len(missing)} missing, {len(invalid)} invalid, sha ok {sha_ok}",
                    " ".join((missing + invalid)[:10]))
    print(f"{flag:7s} {arm:34s} present {len(expected) - len(missing):5d}/{len(expected):5d}  invalid {len(invalid):3d}  sha {'ok' if sha_ok else 'CHANGED'}")
(fc.FINAL / "validity.json").write_text(json.dumps(report, indent=1))
print(f"routes missing/invalid across arms: {bad_total} (STRICT OPUS-MODEL gaps are expected and scored invalid by design)")
