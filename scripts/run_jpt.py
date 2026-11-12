#!/usr/bin/env python3
"""Run jpt-amz's own model_build.sh or model_apply.sh under a memory cap and record cost.

Run from tacit-routes/ (no -I needed: this file only launches their scripts):
  python3 scripts/run_jpt.py build --run-dir runs/jpt
  python3 scripts/run_jpt.py apply --run-dir runs/jpt --t1 1 --t2 1 --tag t1_1
Expects runs/jpt/src -> vendor/jpt-amz and runs/jpt/data/ from prepare_jpt_inputs.py.

What it records (runs/jpt/logs/<phase>_<tag>.json):
  wall seconds, total CPU seconds of all child processes (user + sys, from wait4),
  peak resident memory of the whole process tree (sampled every 0.5 s from /proc),
  and for apply, LKH's own per-route CPU time ("<name>: Cost = ..., Time = x sec." lines).
Each child process is capped at --mem-cap-gb of address space (RLIMIT_AS).
"""
import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024


def tree_rss_mb(root_pid):
    """Sum of RSS over root_pid and all its descendants, from /proc."""
    children, rss = {}, {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            ppid, pages = int(parts[1]), int(parts[21])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(d))
        rss[int(d)] = pages * PAGE_KB / 1024
    total, stack = 0.0, [root_pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0.0)
        stack.extend(children.get(p, []))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "apply"])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--t1", type=int, help="LKH time limit, first constraint set (s)")
    ap.add_argument("--t2", type=int, help="LKH time limit, second constraint set (s)")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--mem-cap-gb", type=float, default=2.4)
    a = ap.parse_args()

    run_dir = a.run_dir.resolve()
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    name = f"{a.phase}_{a.tag}" if a.phase == "apply" else "build"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(run_dir / "pydeps")  # natsort 8.4.0, the only non-stdlib dependency
    if a.phase == "apply":
        if a.t1:
            env["JPT_TIME_LIMIT_1"] = str(a.t1)
        if a.t2:
            env["JPT_TIME_LIMIT_2"] = str(a.t2)
        cmd = ["sh", "src/model_apply.sh"]
    else:
        cmd = ["sh", "src/model_build.sh"]

    cap = int(a.mem_cap_gb * 1024**3)

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        os.nice(5)

    peak = [0.0]
    done = threading.Event()
    t0 = time.time()
    with open(logs / f"{name}.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=limit)

        def sample():
            while not done.is_set():
                peak[0] = max(peak[0], tree_rss_mb(proc.pid))
                time.sleep(0.5)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        _, status, ru = os.wait4(proc.pid, 0)
        done.set()
        th.join()
    wall = time.time() - t0
    code = os.waitstatus_to_exitcode(status)

    rec = {"phase": a.phase, "tag": a.tag, "command": " ".join(cmd), "exit_code": code,
           "time_limits_s": [a.t1, a.t2] if a.phase == "apply" else None,
           "wall_seconds": wall, "cpu_seconds_user": ru.ru_utime, "cpu_seconds_sys": ru.ru_stime,
           "peak_tree_rss_mb_sampled": peak[0],
           "mem_cap_per_process_gb": a.mem_cap_gb, "nice": 5,
           "lkh_threads": 3}
    if a.phase == "apply":
        text = (logs / f"{name}.log").read_text()
        # LKH prints "<instance>: Cost = a_b, Runs = n, Time = x sec." once per instance, per pass.
        times = [float(m) for m in re.findall(r": Cost = \S+, Runs = \d+, Time = ([\d.]+) sec\.", text)]
        rec["lkh_instances_logged"] = len(times)
        rec["lkh_cpu_seconds_per_instance"] = times
        out = run_dir / "data" / "model_apply_outputs" / "proposed_sequences.json"
        if out.exists():
            keep = run_dir / f"out_{a.tag}"
            keep.mkdir(exist_ok=True)
            shutil.copy2(out, keep / "proposed_sequences.json")
            rec["proposals"] = str(keep / "proposed_sequences.json")
    with open(logs / f"{name}.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "lkh_cpu_seconds_per_instance"}, indent=1))
    return 0 if code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
