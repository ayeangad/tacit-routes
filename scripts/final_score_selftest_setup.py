#!/usr/bin/env python3
"""Build the stage-2 self-test fixture from 3 VAL routes (never eval): fake 'sealed' answers, fake
predictions (v0-hist val orders, Opus run 3 val orders, one arm with a missing route and one with a
broken route), a fake PREREGISTRATION.md and a locked config naming its sha256.
  .venv/bin/python -I scripts/final_score_selftest_setup.py && .venv/bin/python -I scripts/final_score.py --test-fake"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_common as fc  # noqa: E402

T = fc.FINAL / "score_selftest"
(T / "answers").mkdir(parents=True, exist_ok=True)
splits = json.load(open(fc.ROOT / "data" / "processed" / "splits.json"))
ids = sorted(r["route_id"] for r in splits["val"])[:3]
inv_all = json.load(open(fc.ROOT / "data" / "raw" / "invalid_sequence_scores.json"))
actual, stops = {}, {}
for r in ids:
    with np.load(fc.ROOT / "data" / "processed" / "routes" / f"{r}.npz", allow_pickle=False) as d:
        s = [str(x) for x in d["stop_ids"]]
        actual[r] = {"actual": {s[i]: p for p, i in enumerate(d["actual_order"])}}
        stops[r] = s
json.dump(actual, open(T / "answers" / "eval_actual_sequences.json", "w"))
json.dump({r: inv_all[r] for r in ids}, open(T / "answers" / "eval_invalid_sequence_scores.json", "w"))
v0 = json.load(open(fc.ROOT / "runs" / "v0" / "val_orders.json"))["hist_high_medium/beam5+travel(lambda=0.5)"]["stop_orders"]
op = json.load(open(fc.ROOT / "opus_arena_run3" / "val_predictions.json"))
arms = {"ours": {r: v0[r] for r in ids}, "opus_a": {r: op[r] for r in ids}, "opus_b": {r: op[r] for r in ids},
        "missing_one": {r: v0[r] for r in ids[:2]},
        "broken_one": {**{r: v0[r] for r in ids}, ids[0]: stops[ids[0]][:-1]}}
pred = T / "predictions"
pred.mkdir(exist_ok=True)
man = {}
for a, o in arms.items():
    (pred / f"{a}.json").write_text(json.dumps({"arm": a, "meta": {"selftest": True}, "orders": o}))
    man[a] = {"sha256": fc.sha256_file(pred / f"{a}.json"), "meta": {"selftest": True}, "n_routes": len(o)}
(pred / "MANIFEST.json").write_text(json.dumps(man))
(T / "PREREGISTRATION.md").write_text("# fake pre-registration for the stage-2 self-test\n")
cfg = {"locked": True, "preregistration_sha256": fc.sha256_file(T / "PREREGISTRATION.md"),
       "primary_sample_file": "unused", "sample_only_arms": [], "arms": list(arms),
       "comparisons": [
           {"name": "P", "role": "primary", "routes": "sample", "a": "ours", "b": "opus_a"},
           {"name": "S1", "role": "secondary", "routes": "all", "a": "ours", "b": "mean:opus_a+opus_b"},
           {"name": "S2", "role": "secondary", "routes": "all", "a": "missing_one", "b": "ours"},
           {"name": "R", "role": "reference", "routes": "all", "a": "broken_one", "b": "ours"}],
       "claims": {"P": {"members": ["P"], "rule": "all"}, "S": {"members": ["S1", "S2"], "rule": "all"}}}
(T / "PREREG_CONFIG.json").write_text(json.dumps(cfg, indent=1))
print("fixture:", ids)
