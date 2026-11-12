#!/usr/bin/env python3
"""Zone-level "swap the brain" arm: ask Opus for the ORDER OF ZONES, one call per pilot route.

  .venv/bin/python -I scripts/run_pilot_zone.py --dry-run            # prints the command, no call
  .venv/bin/python -I scripts/run_pilot_zone.py --i-have-approval    # runs the calls

Prompts come from scripts/build_prompts_zone.py (runs/pilot_zone/prompts/). The `claude -p`
command, flags, fresh empty temp directory per call and JSONL record fields are taken from
run_pilot.py (opus_command, call_opus), so they are identical to the Opus direct pilot:

  env -u ANTHROPIC_API_KEY -u CLAUDE_CODE_EFFORT_LEVEL claude -p --model claude-opus-5-5
      --no-session-persistence --max-turns 1 --tools "" --strict-mcp-config
      --system-prompt-file runs/pilot_zone/system_prompt.txt --output-format json < prompt

Sequential, one call per route, in sample_manifest order. Resumable: a route with an "ok" line
in runs/pilot_zone/opus.jsonl is skipped. A wrong answer is never retried. Network-type
failures are retried at most 2 times. A usage-limit, rate-limit or auth error stops the run at
once (exit code 2); any other error is logged and stops the run (exit code 1).

Each line records: total_cost_usd, duration_ms, usage, modelUsage, the raw output text, the
parsed zone list, validity and why it is invalid. Converting zones to stops and scoring happen
later, in score_pilot_zone.py.
"""
import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT_DIR = ROOT / "runs/pilot_zone"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rp = _load("run_pilot", HERE / "run_pilot.py")

STOP_PATTERNS = re.compile(r"usage limit|rate limit|rate_limit|limit reached|429|quota|credit|"
                           r"authenticat|unauthori|401|403|log ?in|oauth|api key|forbidden", re.I)
NETWORK_PATTERNS = re.compile(r"connection|ECONNRESET|ETIMEDOUT|timed? ?out|network|socket|"
                              r"overloaded|529|500|502|503|504|internal server", re.I)


def parse_zone_order(text, zone_ids):
    """First JSON list in the text. Returns (list or None, valid, reason, details)."""
    decoder = json.JSONDecoder()
    seq, i = None, text.find("[")
    while i != -1:
        try:
            obj, _ = decoder.raw_decode(text, i)
            if isinstance(obj, list):
                seq = obj
                break
        except json.JSONDecodeError:
            pass
        i = text.find("[", i + 1)
    expected = set(zone_ids)
    if seq is None:
        return None, False, "no JSON list found", {"missing": sorted(expected), "unknown": [], "duplicated": []}
    items = [x if isinstance(x, str) else json.dumps(x) for x in seq]
    missing = sorted(expected - set(items))
    unknown = sorted(set(items) - expected)
    dupes = sorted({x for x in items if items.count(x) > 1})
    reasons = []
    if not all(isinstance(x, str) for x in seq):
        reasons.append("non-string items")
    if missing:
        reasons.append(f"missing {len(missing)} zones")
    if unknown:
        reasons.append(f"{len(unknown)} unknown IDs: {unknown[:5]}")
    if dupes:
        reasons.append(f"{len(dupes)} duplicated zones")
    return items, not reasons, "; ".join(reasons) or "valid", {"missing": missing, "unknown": unknown,
                                                               "duplicated": dupes}


def classify(text):
    if STOP_PATTERNS.search(text or ""):
        return "stop"
    if NETWORK_PATTERNS.search(text or ""):
        return "network"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-have-approval", action="store_true", help="Angad approved these Opus calls")
    ap.add_argument("--dry-run", action="store_true", help="print the planned command, call nothing")
    ap.add_argument("--limit", type=int, default=None, help="only the first N routes in manifest order")
    args = ap.parse_args()

    system_file = PILOT_DIR / "system_prompt.txt"
    system = system_file.read_text()
    index = {x["route_id"]: x for x in json.loads((PILOT_DIR / "prompts_index.json").read_text())["routes"]}
    manifest = json.loads((ROOT / "data/raw/sample/sample_manifest.json").read_text())["route_ids"]
    route_ids = manifest[: args.limit]
    out_path = PILOT_DIR / "opus.jsonl"
    done = rp.done_routes(out_path)
    todo = [r for r in route_ids if r not in done]

    if args.dry_run or not args.i_have_approval:
        cmd = " ".join(a if a else '""' for a in rp.opus_command(system_file))
        print(f"{len(todo)} Opus calls planned ({len(done)} routes already done). Example:\n"
              f"{cmd} < {PILOT_DIR / 'prompts' / (todo or route_ids)[0]}.txt\n(cwd: fresh empty temp dir)")
        if not args.dry_run:
            print("refusing to call Opus without --i-have-approval")
            return 3
        return 0

    for n, r in enumerate(route_ids, 1):
        if r in done:
            continue
        prompt_file = PILOT_DIR / "prompts" / f"{r}.txt"
        prompt = prompt_file.read_text()
        zone_ids = index[r]["zone_ids"]
        rec = {"route_id": r, "model_arm": "opus_zone", "n_zones": len(zone_ids),
               "n_stops_including_station": index[r]["n_stops_including_station"],
               "prompt_chars": len(prompt), "system_prompt_chars": len(system), "attempts": 0,
               "started_utc": rp.now()}
        t0 = time.monotonic()
        error, kind, resp, failures = None, None, None, []
        for attempt in range(3):                        # first try + at most 2 network retries
            rec["attempts"] = attempt + 1
            try:
                resp, code, leftover = rp.call_opus(system_file, prompt_file)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                error = f"{type(e).__name__}: {e}"
                kind = classify(error)
            else:
                if code != 0 or resp.get("is_error"):
                    error = (f"claude exit {code}, is_error {resp.get('is_error')}, subtype {resp.get('subtype')}, "
                             f"result {str(resp.get('result'))[:300]!r}")
                    kind = classify(error)
                else:
                    error = kind = None
                    break
            failures.append(error)
            if kind != "network":
                break
            time.sleep(10 * (attempt + 1))
        rec["finished_utc"] = rp.now()
        rec["wall_seconds"] = time.monotonic() - t0
        if failures:
            rec["failed_attempts"] = failures
        if resp is not None:
            rec.update(model_requested=rp.OPUS_MODEL,
                       model_returned=sorted((resp.get("modelUsage") or {}).keys()),
                       settings={"command": " ".join(a if a else '""' for a in rp.opus_command(system_file))
                                 + " < <prompt file>", "cwd": "fresh empty temp dir per call"},
                       exit_code=code, is_error=resp.get("is_error"), subtype=resp.get("subtype"),
                       stop_reason=resp.get("stop_reason"), num_turns=resp.get("num_turns"),
                       usage=resp.get("usage"), model_usage=resp.get("modelUsage"),
                       total_cost_usd=resp.get("total_cost_usd"), duration_ms=resp.get("duration_ms"),
                       duration_api_ms=resp.get("duration_api_ms"), temp_dir_files_after_call=leftover)
        if error is not None:
            rec.update(status="error", error=error, error_kind=kind,
                       raw_output=(resp or {}).get("result"))
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{n}] {r}: ERROR ({kind}) {error}; stopping", flush=True)
            return 2 if kind == "stop" else 1
        text = resp.get("result") or ""
        zones, valid, reason, details = parse_zone_order(text, zone_ids)
        rec.update(status="ok", raw_output=text, output_chars=len(text), parsed_zone_order=zones,
                   valid=valid, invalid_reason=None if valid else reason, invalid_details=details)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{n}] {r}: valid={valid} ({reason}) {rec['wall_seconds']:.1f}s "
              f"cost={rec.get('total_cost_usd')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
