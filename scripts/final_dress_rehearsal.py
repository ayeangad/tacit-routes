#!/usr/bin/env python3
"""Dress rehearsal helper for scripts/final_dress_rehearsal.sh (VALIDATION routes only).

  setup: lay out the 300 VAL routes exactly like data/eval_frozen under runs/final/dress/:
         eval_frozen/routes/*.npz (inputs only: actual_order and quality dropped), index.csv,
         raw/apply/eval_*.json (the val rc-cli apply files jpt used), a MOCK sealed
         raw/score/ (val answers + invalid scores) with a MOCK MANIFEST.json holding their sha256,
         a mock PREREGISTRATION.md + locked PREREG_CONFIG.json, the primary "sample" = the 40 val
         routes of the frozen zone_rich dev run (runs/zone_rich/dev_A, version A = FROZEN), and a copy
         of that run re-pointed at the mock eval folder (to rehearse zone-rich-collect).
  check: compare the rehearsal's per-route scores with the stored val results.
         EXACT (deterministic): v0-hist vs results/val_comparison.json. Reported (time-limited
         searches, so small differences are expected): v1 (no stored val), jpt, R2, shortest (val
         used 10 s, rehearsal 3 s), Opus run 3 (two repeats vs results/opus_fight_run3.json),
         OPUS-MODEL repaired (greedy) vs runs/zone_rich/dev_A/scores_greedy.json.
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
D = ROOT / "runs" / "final" / "dress"
E = D / "eval_frozen"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup():
    if D.exists():
        shutil.rmtree(D)
    (E / "routes").mkdir(parents=True)
    (E / "raw" / "apply").mkdir(parents=True)
    (E / "raw" / "score").mkdir(parents=True)
    splits = json.load(open(ROOT / "data" / "processed" / "splits.json"))
    val = sorted(r["route_id"] for r in splits["val"])
    rows = []
    for rid in val:
        with np.load(ROOT / "data" / "processed" / "routes" / f"{rid}.npz", allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files if k not in ("actual_order",)}
        arrays["quality"] = np.array("")
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        (E / "routes" / f"{rid}.npz").write_bytes(buf.getvalue())
        rows.append({"route_id": rid, "station": str(arrays["station"]), "date": str(arrays["date"]),
                     "n_stops": len(arrays["stop_ids"]), "quality": "", "sha256": sha(E / "routes" / f"{rid}.npz")})
    with open(E / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    jd = ROOT / "runs" / "jpt" / "data"
    for src, dst in (("model_apply_inputs/new_route_data.json", "apply/eval_route_data.json"),
                     ("model_apply_inputs/new_package_data.json", "apply/eval_package_data.json"),
                     ("model_apply_inputs/new_travel_times.json", "apply/eval_travel_times.json"),
                     ("model_score_inputs/new_actual_sequences.json", "score/eval_actual_sequences.json"),
                     ("model_score_inputs/new_invalid_sequence_scores.json", "score/eval_invalid_sequence_scores.json")):
        shutil.copyfile(jd / src, E / "raw" / dst)
    json.dump({"what": "MOCK manifest for the dress rehearsal (val routes)",
               "score_inputs_SEALED": {n: sha(E / "raw" / "score" / n) for n in
                                       ("eval_actual_sequences.json", "eval_invalid_sequence_scores.json")}},
              open(E / "MANIFEST.json", "w"), indent=1)

    # OPUS-MODEL rehearsal input: the frozen-version dev run, re-pointed at the mock eval folder.
    zsrc, zdst = ROOT / "runs" / "zone_rich" / "dev_A", D / "zone_rich_dev_A"
    zdst.mkdir()
    info = json.load(open(zsrc / "prompts_index.json"))
    info["input_dir"] = str(E / "routes")
    (zdst / "prompts_index.json").write_text(json.dumps(info))
    shutil.copyfile(zsrc / "opus.jsonl", zdst / "opus.jsonl")
    sample = sorted(x["route_id"] for x in info["routes"])
    (D / "primary_sample_500.json").write_text(json.dumps(
        {"NOTE": "dress rehearsal: the 40 val routes of runs/zone_rich/dev_A", "route_ids": sample,
         "sha256_of_route_ids": hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()}))

    (D / "PREREGISTRATION.md").write_text("# MOCK pre-registration for the dress rehearsal (val routes)\n")
    cfg = {"locked": True, "preregistration_sha256": sha(D / "PREREGISTRATION.md"),
           "primary_sample_file": str((D / "primary_sample_500.json").relative_to(ROOT)),
           "sample_only_arms": ["opus_model_repaired_greedy", "opus_model_strict_greedy"],
           "arms": ["v1_exact", "v1", "v0hist", "opus_model_repaired_greedy", "opus_model_strict_greedy",
                    "opus_run3_rep1", "opus_run3_rep2", "jpt", "rules_R2", "rules_S3"],
           "comparisons": [
               {"name": "PRIMARY", "role": "primary", "routes": "sample", "a": "v1", "b": "opus_model_repaired_greedy"},
               {"name": "PRIMARY_strict_view", "role": "reference", "routes": "sample", "a": "v1", "b": "opus_model_strict_greedy"},
               {"name": "A_vs_run3", "role": "secondary", "routes": "all", "a": "v0hist", "b": "mean:opus_run3_rep1+opus_run3_rep2"},
               {"name": "B_vs_run3", "role": "secondary", "routes": "all", "a": "v1_exact", "b": "mean:opus_run3_rep1+opus_run3_rep2"},
               {"name": "ref_jpt", "role": "reference", "routes": "all", "a": "v1_exact", "b": "jpt"},
               {"name": "ref_R2", "role": "reference", "routes": "all", "a": "v1_exact", "b": "rules_R2"},
               {"name": "ref_shortest_3s", "role": "reference", "routes": "all", "a": "v1_exact", "b": "rules_S3"},
               {"name": "exact_vs_greedy", "role": "reference", "routes": "all", "a": "v1_exact", "b": "v1"}],
           "claims": {"PRIMARY": {"members": ["PRIMARY"], "rule": "all"},
                      "ENGINEER": {"members": ["A_vs_run3", "B_vs_run3"], "rule": "all"}}}
    (D / "PREREG_CONFIG.json").write_text(json.dumps(cfg, indent=1))
    print(f"dress setup: {len(val)} val routes laid out under {E}; sample {len(sample)} routes")


def check():
    res = json.load(open(D / "results" / "final_test.json"))
    per = res["per_route"]
    vc = {x["route_id"]: x for x in json.load(open(ROOT / "results" / "val_comparison.json"))["per_route"]}
    o3 = {x["route_id"]: x["opus_score"] for x in json.load(open(ROOT / "results" / "opus_fight_run3.json"))["per_route"]}
    zr = {r: v.get("score_repaired", v.get("score_fallback")) for r, v in json.load(open(ROOT / "runs" / "zone_rich" / "dev_A" / "scores_greedy.json"))["per_route"].items()}
    lines, fail = [], []

    def cmp(name, arm, stored, exact):
        ids = sorted(set(per[arm]) & set(stored))
        d = np.array([per[arm][r] - stored[r] for r in ids])
        same = int((np.abs(d) < 1e-12).sum())
        m_new, m_old = float(np.mean([per[arm][r] for r in ids])), float(np.mean([stored[r] for r in ids]))
        lines.append(f"| {name} | {len(ids)} | {m_new:.4f} | {m_old:.4f} | {same}/{len(ids)} | {np.abs(d).max():.4g} | {'EXACT required' if exact else 'report'} |")
        if exact and same != len(ids):
            fail.append(name)

    cmp("v0-hist (pre-registered)", "v0hist", {r: vc[r]["v0:hist_high_medium/beam5+travel(lambda=0.5)"] for r in vc}, True)
    cmp("jpt (re-run, LKH 1 s + 1 s)", "jpt", {r: vc[r]["jpt"] for r in vc}, False)
    cmp("R2 (re-run)", "rules_R2", {r: vc[r]["rules:R2"] for r in vc}, False)
    cmp("shortest: 3 s now vs 10 s stored", "rules_S3", {r: vc[r]["rules:S"] for r in vc if "rules:S" in vc[r]}, False)
    cmp("Opus run 3, repeat 1 (sandboxed re-run)", "opus_run3_rep1", o3, False)
    cmp("Opus run 3, repeat 2 (sandboxed re-run)", "opus_run3_rep2", o3, False)
    cmp("OPUS-MODEL repaired (greedy), dev_A", "opus_model_repaired_greedy", zr, False)
    v1m = np.mean(list(per["v1"].values()))
    v1e = np.mean(list(per["v1_exact"].values()))
    out = ["| Arm | Routes | Rehearsal mean | Stored mean | Identical routes | Max abs diff | Rule |", "|---|---|---|---|---|---|---|"] + lines
    out += ["", f"v1 (greedy) val mean {v1m:.4f}; v1_exact val mean {v1e:.4f} (no stored val numbers for v1 on all 300 routes)."]
    out += ["", "REHEARSAL " + ("PASSED" if not fail else "FAILED: " + ", ".join(fail))]
    (D / "results" / "dress_check.md").write_text("# Dress rehearsal check (val routes)\n\n" + "\n".join(out) + "\n")
    print("\n".join(out))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    {"setup": setup, "check": check}[sys.argv[1]]()
