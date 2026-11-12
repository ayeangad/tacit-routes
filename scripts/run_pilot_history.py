#!/usr/bin/env python3
"""Run the "Opus direct + history" arm: 50 Opus calls, one per pilot route. NEEDS APPROVAL.

  # see what would run, no calls:
  .venv/bin/python -I scripts/run_pilot_history.py --dry-run
  # the real run, only after Angad approves the extra Opus calls:
  .venv/bin/python -I scripts/run_pilot_history.py --i-have-approval

This is a thin wrapper around scripts/run_pilot.py. It calls run_pilot.main() with
--model opus --pilot-dir runs/pilot_history --sample-dir data/raw/sample, so the `claude -p`
command and flags, the fresh empty temp directory per call, the parsing, the validity rules,
the retry rules, resume-on-rerun and the JSONL logging are exactly the pilot's. Output goes
to runs/pilot_history/opus.jsonl. The only differences from the pilot are the prompt files
and system prompt inside runs/pilot_history/.

Without --i-have-approval it prints the number of calls it would make and exits with code 3.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT_DIR = ROOT / "runs/pilot_history"
SAMPLE_DIR = ROOT / "data/raw/sample"


def load_run_pilot():
    spec = importlib.util.spec_from_file_location("run_pilot", HERE / "run_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-have-approval", action="store_true",
                    help="Angad has approved these extra Opus calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned command for the first pending route; run nothing")
    ap.add_argument("--limit", type=int, default=None, help="only the first N routes in manifest order")
    args = ap.parse_args()

    rp = load_run_pilot()
    manifest = json.loads((SAMPLE_DIR / "sample_manifest.json").read_text())
    route_ids = manifest["route_ids"][: args.limit]
    missing = [r for r in route_ids if not (PILOT_DIR / "prompts" / f"{r}.txt").exists()]
    if missing:
        print(f"{len(missing)} prompts missing; run scripts/build_prompts_history.py first")
        return 2
    done = rp.done_routes(PILOT_DIR / "opus.jsonl")
    pending = [r for r in route_ids if r not in done]
    print(f"Opus calls this run would make: {len(pending)} "
          f"({len(route_ids)} routes selected, {len(route_ids) - len(pending)} already done) "
          f"-> {PILOT_DIR / 'opus.jsonl'}")

    if args.dry_run:
        if pending:
            r = pending[0]
            system_file = PILOT_DIR.resolve() / "system_prompt.txt"
            cmd = " ".join(a if a else '""' for a in rp.opus_command(system_file))
            print(f"Planned command for {r} (NOT executed), from a fresh empty temp dir:")
            print(f"  {cmd} < {PILOT_DIR / 'prompts' / (r + '.txt')}")
        return 0

    if not args.i_have_approval:
        print("Refusing to run: these are extra Opus calls and need Angad's approval. "
              "Pass --i-have-approval once he has given it.")
        return 3

    argv = ["run_pilot.py", "--model", "opus", "--pilot-dir", str(PILOT_DIR), "--sample-dir", str(SAMPLE_DIR)]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    sys.argv = argv
    return rp.main()


if __name__ == "__main__":
    sys.exit(main())
