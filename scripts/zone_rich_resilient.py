#!/usr/bin/env python3
"""Scheduler around the FROZEN scripts/zone_rich.py for the OPUS-MODEL eval run. Added 2026-11-12 ~02:35 IST
after a safeguard false positive (runs/final/incidents.log). Nothing that shapes an answer changes:
prompts, system prompt, the claude command, effort, parsing and the record format are zone_rich.py's own
(it is imported unchanged; its sha256 is checked against runs/zone_rich/FROZEN.json). Only the order of
calls and the reaction to one error type differ:

  - zone_rich.py run stops the whole run on ANY error, and a resume retries the failed route first. A route
    whose call is refused by the API safeguard ("safeguards flagged this message") would then block every
    resume. Here such a refusal is recorded (same record as zone_rich.py writes) and the run CONTINUES.
  - Routes with earlier safeguard refusals are scheduled LAST, and a route is retried at most
    MAX_SAFEGUARD_ATTEMPTS times in total (across resumes). After that it is final: "refused".
  - Every other error (usage limit, network, anything unknown) still stops the run, exactly as before;
    the launcher waits and resumes.
  - Only status "ok" replies are ever used as Opus answers (unchanged).

  - Concurrency (added 2026-11-12 ~02:55 IST, Angad: "maximize and use my laptop hardware ... without breaking
    anything"): up to MAX_WORKERS parallel calls. If the file DIR/WORKERS exists, its number (1..MAX_WORKERS) is
    re-read every 30 s and becomes the target, so it can be lowered or raised without a restart. Concurrency changes
    no answer: each call is an independent `claude -p` in its own empty temp folder with the same frozen inputs.

  .venv/bin/python -I scripts/zone_rich_resilient.py --out DIR --workers 5 [--effort max] --i-have-approval
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
MAX_SAFEGUARD_ATTEMPTS = 3
MAX_WORKERS = 6
SAFEGUARD = "safeguards flagged"


def load_zone_rich():
    path = HERE / "zone_rich.py"
    frozen = json.load(open(ROOT / "runs" / "zone_rich" / "FROZEN.json"))
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h == frozen["script_sha256"], f"zone_rich.py changed: {h} != FROZEN {frozen['script_sha256']}"
    spec = importlib.util.spec_from_file_location("zone_rich", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def is_safeguard(rec):
    return rec.get("status") == "error" and SAFEGUARD in (rec.get("error") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default=None)
    ap.add_argument("--i-have-approval", action="store_true")
    a = ap.parse_args()
    assert 1 <= a.workers <= MAX_WORKERS
    zr = load_zone_rich()
    out = Path(a.out).resolve()
    info = json.loads((out / "prompts_index.json").read_text())
    system_file = out / "system_prompt.txt"
    log = out / "opus.jsonl"
    done = zr.rp.done_routes(log)
    refusals = collections.Counter()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if is_safeguard(r):
                refusals[r["route_id"]] += 1
    todo = [x for x in info["routes"] if x["route_id"] not in done and refusals[x["route_id"]] < MAX_SAFEGUARD_ATTEMPTS]
    final_refused = [x["route_id"] for x in info["routes"] if x["route_id"] not in done and refusals[x["route_id"]] >= MAX_SAFEGUARD_ATTEMPTS]
    todo.sort(key=lambda x: refusals[x["route_id"]] > 0)  # stable: never-refused routes first, in the original order
    print(f"{len(done)} done, {len(todo)} to call ({sum(refusals[x['route_id']] > 0 for x in todo)} earlier refusals, last), "
          f"{len(final_refused)} final refusals", flush=True)
    if not a.i_have_approval:
        print("refusing without --i-have-approval")
        return 3
    stop = False
    queue = collections.deque(todo)
    wfile = out / "WORKERS"

    def target():
        try:
            return max(1, min(MAX_WORKERS, int(wfile.read_text().split()[0])))
        except (OSError, ValueError, IndexError):
            return a.workers

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}

        def submit():
            x = queue.popleft()
            futs[ex.submit(zr.one_call, out, x["route_id"], x["zone_ids"], system_file, a.effort)] = x

        def top_up():
            while not stop and queue and len(futs) < target():
                submit()

        top_up()
        while futs:
            finished, _ = cf.wait(futs, timeout=30, return_when=cf.FIRST_COMPLETED)
            if not finished:
                top_up()
                continue
            f = finished.pop()
            x = futs.pop(f)
            rec = f.result()
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{rec['route_id'][-12:]} {rec['status']} valid={rec.get('valid')} {rec['wall_seconds']:.0f}s "
                  f"${rec.get('total_cost_usd')}", flush=True)
            if is_safeguard(rec):
                refusals[x["route_id"]] += 1
                if refusals[x["route_id"]] < MAX_SAFEGUARD_ATTEMPTS:
                    queue.append(x)  # retry at the very end
                    print(f"   safeguard refusal {refusals[x['route_id']]}/{MAX_SAFEGUARD_ATTEMPTS}: retry at the end", flush=True)
                else:
                    print(f"   safeguard refusal {MAX_SAFEGUARD_ATTEMPTS}/{MAX_SAFEGUARD_ATTEMPTS}: FINAL (refused)", flush=True)
            elif rec["status"] == "error":
                stop = True
                print("ERROR -> stopping:", rec["error"][:200], flush=True)
            top_up()
    return 2 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
