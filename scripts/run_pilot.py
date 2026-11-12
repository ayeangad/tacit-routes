#!/usr/bin/env python3
"""Ask a model to order a route's stops directly, one call per route. Resumable.

  .venv/bin/python -I scripts/run_pilot.py --model deepseek --pilot-dir runs/pilot \
      --sample-dir data/raw/sample --cap-usd 1.00
  .venv/bin/python -I scripts/run_pilot.py --model opus --pilot-dir runs/pilot \
      --sample-dir data/raw/sample --limit 5

Writes one JSON line per call to <pilot-dir>/<model>.jsonl. A route with a line whose
status is "ok" is skipped on rerun. Wrong answers are never retried; only network errors
are retried, at most 2 times. Any other error is recorded and stops the run.

DeepSeek: the API key is read at runtime from ~/.deepseek_key, the same place atelier's
client (atelier/poc/atelier_poc/llm.py) reads it. It is never printed, logged or copied.

Opus: `claude -p` with exactly the approved flags, from a fresh empty temp
directory per call, sequentially.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# DeepSeek, same model ID and peak list prices (USD per 1M tokens) as atelier's llm.py.
DS_URL = "https://api.deepseek.com/chat/completions"
DS_MODEL = "deepseek-flash"
DS_PRICE = {"hit": 0.006, "miss": 0.30, "out": 1.20}
DS_SETTINGS = {"temperature": 0, "max_tokens": 4096, "thinking": {"type": "disabled"}, "stream": False}

OPUS_MODEL = "claude-opus-5-5"


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_sequence(text, depot, stop_ids):
    """Take the first JSON list in the text. Return (sequence or None, valid, reason)."""
    decoder = json.JSONDecoder()
    seq = None
    i = text.find("[")
    while i != -1:
        try:
            obj, _ = decoder.raw_decode(text, i)
            if isinstance(obj, list):
                seq = obj
                break
        except json.JSONDecodeError:
            pass
        i = text.find("[", i + 1)
    if seq is None:
        return None, False, "no JSON list found"
    if not all(isinstance(x, str) for x in seq):
        return seq, False, "list contains non-string items"
    reasons = []
    expected = set(stop_ids)
    missing = expected - set(seq)
    extra = sorted(set(seq) - expected)
    dupes = sorted({x for x in seq if seq.count(x) > 1})
    if missing:
        reasons.append(f"missing {len(missing)} stops")
    if extra:
        reasons.append(f"{len(extra)} unknown IDs: {extra[:5]}")
    if dupes:
        tail = " (incl. depot repeated)" if depot in dupes else ""
        reasons.append(f"{len(dupes)} duplicated IDs{tail}")
    if not seq or seq[0] != depot:
        reasons.append(f"does not start at depot (starts with {seq[0] if seq else None!r})")
    return seq, not reasons, "; ".join(reasons) or "valid"


class NetworkError(Exception):
    pass


def call_deepseek(system, prompt):
    key = (Path.home() / ".deepseek_key").read_text().strip()
    body = {"model": DS_MODEL, "messages": [{"role": "system", "content": system},
                                            {"role": "user", "content": prompt}], **DS_SETTINGS}
    req = urllib.request.Request(DS_URL, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    del key
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (429, 500, 502, 503, 504):
            raise NetworkError(f"HTTP {e.code}") from None
        # Body may describe the error; it never contains the key.
        raise RuntimeError(f"DeepSeek HTTP {e.code}: {e.read()[:300]!r}") from None
    except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
        raise NetworkError(type(e).__name__ + ": " + str(e)[:200]) from None


def ds_cost(usage):
    hit = int(usage.get("prompt_cache_hit_tokens", 0))
    miss = int(usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit))
    out = int(usage.get("completion_tokens", 0))
    return (hit * DS_PRICE["hit"] + miss * DS_PRICE["miss"] + out * DS_PRICE["out"]) / 1e6


def ds_worst_case(prompt_chars):
    """Upper bound on one call's cost: every character a fresh input token, full max_tokens out."""
    return (prompt_chars * DS_PRICE["miss"] + DS_SETTINGS["max_tokens"] * DS_PRICE["out"]) / 1e6


def opus_command(system_file):
    return ["env", "-u", "ANTHROPIC_API_KEY", "-u", "CLAUDE_CODE_EFFORT_LEVEL",
            "claude", "-p", "--model", OPUS_MODEL, "--no-session-persistence", "--max-turns", "1",
            "--tools", "", "--strict-mcp-config", "--system-prompt-file", str(system_file),
            "--output-format", "json"]


def call_opus(system_file, prompt_file):
    workdir = tempfile.mkdtemp(prefix="opus_pilot_")
    try:
        with open(prompt_file, "rb") as stdin:
            proc = subprocess.run(opus_command(system_file), stdin=stdin, cwd=workdir,
                                  capture_output=True, timeout=1800)
    finally:
        leftover = os.listdir(workdir)
        shutil.rmtree(workdir, ignore_errors=True)
    out = proc.stdout.decode(errors="replace")
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude exit {proc.returncode}, non-JSON stdout {out[:300]!r}, "
                           f"stderr {proc.stderr.decode(errors='replace')[:300]!r}") from None
    return data, proc.returncode, leftover


def done_routes(path):
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("status") == "ok":
                done.add(rec["route_id"])
    return done


def spent(path):
    total = 0.0
    if path.exists():
        for line in path.read_text().splitlines():
            total += json.loads(line).get("cost_usd_list_price") or 0.0
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["deepseek", "opus"])
    ap.add_argument("--pilot-dir", required=True, type=Path)
    ap.add_argument("--sample-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="only the first N routes in manifest order")
    ap.add_argument("--cap-usd", type=float, default=1.00, help="DeepSeek hard cap at list prices")
    args = ap.parse_args()

    pilot = args.pilot_dir.resolve()
    system_file = pilot / "system_prompt.txt"
    system = system_file.read_text()
    index = json.loads((pilot / "prompts_index.json").read_text())
    manifest = json.loads((args.sample_dir / "sample_manifest.json").read_text())
    route_ids = manifest["route_ids"][: args.limit]
    info = {x["route_id"]: x for x in index["routes"]}
    out_path = pilot / f"{args.model}.jsonl"
    done = done_routes(out_path)

    for n, r in enumerate(route_ids, 1):
        if r in done:
            continue
        prompt_file = pilot / "prompts" / f"{r}.txt"
        prompt = prompt_file.read_text()
        depot, stop_ids = info[r]["depot"], info[r]["listed_order"]

        if args.model == "deepseek":
            so_far = spent(out_path)
            worst = ds_worst_case(len(prompt) + len(system))
            if so_far + worst > args.cap_usd:
                print(f"STOP: spent ${so_far:.4f}; next call could cost up to ${worst:.4f}, "
                      f"cap ${args.cap_usd:.2f}", flush=True)
                return 2

        rec = {"route_id": r, "model_arm": args.model, "n_stops_including_depot": len(stop_ids),
               "prompt_chars": len(prompt), "system_prompt_chars": len(system), "attempts": 0}
        rec["started_utc"] = now()
        t0 = time.monotonic()
        error = None
        for attempt in range(3):  # first try + at most 2 retries, network errors only
            rec["attempts"] = attempt + 1
            try:
                if args.model == "deepseek":
                    resp = call_deepseek(system, prompt)
                else:
                    resp, code, leftover = call_opus(system_file, prompt_file)
                error = None
                break
            except NetworkError as e:
                error = f"network: {e}"
                time.sleep(5 * (attempt + 1))
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                error = f"{type(e).__name__}: {e}"
                break
        rec["finished_utc"] = now()
        rec["wall_seconds"] = time.monotonic() - t0

        if error is not None:
            rec.update(status="error", error=error)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{n}] {r}: ERROR {error}; stopping", flush=True)
            return 1

        if args.model == "deepseek":
            choice = resp["choices"][0]
            text = choice["message"].get("content") or ""
            usage = resp.get("usage", {})
            rec.update(model_requested=DS_MODEL, model_returned=resp.get("model"),
                       settings=DS_SETTINGS, finish_reason=choice.get("finish_reason"),
                       usage=usage, cost_usd_list_price=ds_cost(usage),
                       price_basis="peak list price per 1M tokens " + json.dumps(DS_PRICE),
                       response_id=resp.get("id"), system_fingerprint=resp.get("system_fingerprint"))
        else:
            text = resp.get("result") or ""
            rec.update(model_requested=OPUS_MODEL,
                       model_returned=sorted((resp.get("modelUsage") or {}).keys()),
                       settings={"command": " ".join(a if a else '""' for a in opus_command(system_file))
                                 + " < <prompt file>", "cwd": "fresh empty temp dir per call"},
                       exit_code=code, is_error=resp.get("is_error"), subtype=resp.get("subtype"),
                       stop_reason=resp.get("stop_reason"), num_turns=resp.get("num_turns"),
                       usage=resp.get("usage"), model_usage=resp.get("modelUsage"),
                       total_cost_usd=resp.get("total_cost_usd"), duration_ms=resp.get("duration_ms"),
                       duration_api_ms=resp.get("duration_api_ms"),
                       temp_dir_files_after_call=leftover)
            if code != 0 or resp.get("is_error"):
                rec.update(status="error", error=f"claude exit {code}, is_error {resp.get('is_error')}, "
                                                 f"subtype {resp.get('subtype')}", raw_output=text)
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{n}] {r}: ERROR from claude; stopping", flush=True)
                return 1

        seq, valid, reason = parse_sequence(text, depot, stop_ids)
        rec.update(status="ok", raw_output=text, output_chars=len(text), parsed_sequence=seq,
                   valid=valid, invalid_reason=None if valid else reason)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        cost = rec.get("cost_usd_list_price", rec.get("total_cost_usd"))
        print(f"[{n}] {r}: valid={valid} ({reason}) {rec['wall_seconds']:.1f}s cost={cost}", flush=True)
    if args.model == "deepseek":
        print(f"DeepSeek spend so far ${spent(out_path):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
