#!/usr/bin/env python3
"""DeepSeek arm of the pilot rerun with thinking ON, same prompts. Resumable.

  .venv/bin/python -I scripts/run_deepseek_thinking.py --pilot-dir runs/pilot \
      --sample-dir data/raw/sample --cap-usd 2.00 --workers 4

Reuses run_pilot.py's parse_sequence (parsing and validity rules), ds_cost, done_routes,
spent and NetworkError unchanged; only the request settings differ. run_pilot.py itself
is not modified, so its default (thinking off) is unchanged.

Settings (DeepSeek API docs, read 2026-11-10: guides/thinking_mode, api/create-chat-completion,
quick_start/pricing):
  - model `deepseek-flash` (DeepSeek-V4.1-Flash-1017), same as the thinking-off run
  - thinking {"type": "enabled"}, reasoning_effort "high" (the documented default)
  - temperature not sent: the docs say it has no effect in thinking mode; top_p not sent
    (default); no other sampling parameters
  - max_tokens 393216, the documented maximum (384K)
  - stream false
Cost at PEAK list prices (conservative): $0.006 / $0.30 / $1.20 per 1M tokens for cache-hit
input / cache-miss input / output (reasoning tokens counted as output).

Spend guard: a call is launched only if spend so far plus the worst case of every call in
flight plus this call's worst case (all input uncached, full max_tokens out) stays within the
cap, so the cap cannot be exceeded. After the first 3 calls the run stops if the mean cost of
those calls times 50 projects above the cap.

The API key is read at runtime from ~/.deepseek_key; it is never printed, logged or copied.
Reasoning text is stored in full only if at most 20,000 characters; otherwise its length
plus the first and last 2,000 characters.
"""
import argparse
import http.client
import importlib.util
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_pilot", HERE / "run_pilot.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

DS_URL = rp.DS_URL
DS_MODEL = rp.DS_MODEL  # "deepseek-flash", same as the thinking-off run
DS_PRICE = rp.DS_PRICE  # peak list prices, same as the thinking-off run
DS_SETTINGS = {"thinking": {"type": "enabled"}, "reasoning_effort": "high",
               "max_tokens": 393216, "stream": False}
REASONING_KEEP = 20000
ARM = "deepseek_thinking"


def call_deepseek(system, prompt, timeout):
    key = (Path.home() / ".deepseek_key").read_text().strip()
    body = {"model": DS_MODEL, "messages": [{"role": "system", "content": system},
                                            {"role": "user", "content": prompt}], **DS_SETTINGS}
    req = urllib.request.Request(DS_URL, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    del key
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except (http.client.HTTPException, ConnectionError) as e:
        # e.g. IncompleteRead: connection dropped mid-response (added after the 2026-11-10 crash)
        raise rp.NetworkError(type(e).__name__ + ": " + str(e)[:200]) from None
    except urllib.error.HTTPError as e:
        if e.code in (429, 500, 502, 503, 504):
            raise rp.NetworkError(f"HTTP {e.code}") from None
        raise RuntimeError(f"DeepSeek HTTP {e.code}: {e.read()[:300]!r}") from None
    except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
        raise rp.NetworkError(type(e).__name__ + ": " + str(e)[:200]) from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise rp.NetworkError(f"truncated or non-JSON body ({len(raw)} bytes): {e}") from None


def worst_case(chars):
    return (chars * DS_PRICE["miss"] + DS_SETTINGS["max_tokens"] * DS_PRICE["out"]) / 1e6


def run_one(r, system, prompt, depot, stop_ids, timeout):
    rec = {"route_id": r, "model_arm": ARM, "n_stops_including_depot": len(stop_ids),
           "prompt_chars": len(prompt), "system_prompt_chars": len(system), "attempts": 0,
           "started_utc": rp.now()}
    t0 = time.monotonic()
    error, resp = None, None
    for attempt in range(3):  # first try + at most 2 retries, network errors only
        rec["attempts"] = attempt + 1
        try:
            resp = call_deepseek(system, prompt, timeout)
            error = None
            break
        except rp.NetworkError as e:
            error = f"network: {e}"
            time.sleep(5 * (attempt + 1))
        except Exception as e:  # anything else: record and stop, never crash unlogged
            error = f"{type(e).__name__}: {str(e)[:300]}"
            break
    rec["finished_utc"] = rp.now()
    rec["wall_seconds"] = time.monotonic() - t0
    rec["network_retries_note"] = ("any failed attempts may have been billed; not counted in cost"
                                   if rec["attempts"] > 1 else None)
    if error is not None:
        rec.update(status="error", error=error)
        return rec
    choice = resp["choices"][0]
    msg = choice["message"]
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = resp.get("usage", {})
    rec.update(model_requested=DS_MODEL, model_returned=resp.get("model"), settings=DS_SETTINGS,
               finish_reason=choice.get("finish_reason"), usage=usage,
               reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
               reasoning_chars=len(reasoning),
               cost_usd_list_price=rp.ds_cost(usage),
               price_basis="peak list price per 1M tokens " + json.dumps(DS_PRICE),
               response_id=resp.get("id"), system_fingerprint=resp.get("system_fingerprint"))
    if len(reasoning) <= REASONING_KEEP:
        rec["reasoning_content"] = reasoning
    else:
        rec["reasoning_head"], rec["reasoning_tail"] = reasoning[:2000], reasoning[-2000:]
    seq, valid, reason = rp.parse_sequence(text, depot, stop_ids)
    rec.update(status="ok", raw_output=text, output_chars=len(text), parsed_sequence=seq,
               valid=valid, invalid_reason=None if valid else reason)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True, type=Path)
    ap.add_argument("--sample-dir", required=True, type=Path)
    ap.add_argument("--cap-usd", type=float, default=2.00)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--unlogged-usd", type=float, default=0.0,
                    help="spend from calls that were billed but never logged, counted against the cap")
    ap.add_argument("--probe", type=int, default=3, help="calls used for the cost projection")
    args = ap.parse_args()
    if args.workers > 4:
        raise SystemExit("at most 4 in parallel")

    pilot = args.pilot_dir.resolve()
    system = (pilot / "system_prompt.txt").read_text()
    index = json.loads((pilot / "prompts_index.json").read_text())
    manifest = json.loads((args.sample_dir / "sample_manifest.json").read_text())
    route_ids = manifest["route_ids"][: args.limit]
    info = {x["route_id"]: x for x in index["routes"]}
    out_path = pilot / f"{ARM}.jsonl"
    done = rp.done_routes(out_path)
    todo = [r for r in route_ids if r not in done]
    lock = threading.Lock()

    def record(rec):
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def project():
        costs = [json.loads(l)["cost_usd_list_price"] for l in out_path.read_text().splitlines()
                 if json.loads(l).get("status") == "ok"] if out_path.exists() else []
        return costs

    def launch(pool, r):
        prompt = (pilot / "prompts" / f"{r}.txt").read_text()
        return pool.submit(run_one, r, system, prompt, info[r]["depot"], info[r]["listed_order"],
                           args.timeout), worst_case(len(prompt) + len(system))

    # Phase 1: projection from the first `probe` successful calls (run in parallel).
    costs = project()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if len(costs) < args.probe:
            batch = todo[: args.probe - len(costs)]
            todo = todo[len(batch):]
            futs = [launch(pool, r)[0] for r in batch]
            for f in futs:
                rec = f.result()
                record(rec)
                print(f"{rec['route_id'][:16]}: {rec['status']} valid={rec.get('valid')} "
                      f"({rec.get('invalid_reason') or rec.get('error')}) {rec['wall_seconds']:.0f}s "
                      f"out={rec.get('usage', {}).get('completion_tokens')} "
                      f"reason_tok={rec.get('reasoning_tokens')} cost={rec.get('cost_usd_list_price')}",
                      flush=True)
                if rec["status"] != "ok":
                    print("ERROR; stopping", flush=True)
                    return 1
            costs = project()
        mean = sum(costs) / len(costs)
        proj = mean * len(route_ids)
        print(f"Projection: mean ${mean:.4f}/call x {len(route_ids)} = ${proj:.3f} "
              f"(cap ${args.cap_usd:.2f})", flush=True)
        if proj > args.cap_usd:
            print("STOP: projection exceeds cap", flush=True)
            return 2

        # Phase 2: the rest, at most `workers` in flight, under the hard worst-case guard.
        inflight = {}
        stopped = None
        while todo or inflight:
            while todo and len(inflight) < args.workers and stopped is None:
                prompt_chars = len((pilot / "prompts" / f"{todo[0]}.txt").read_text()) + len(system)
                reserve = sum(inflight.values()) + worst_case(prompt_chars)
                if rp.spent(out_path) + args.unlogged_usd + reserve > args.cap_usd:
                    if not inflight:
                        stopped = (f"STOP: spent ${rp.spent(out_path):.4f}; next call worst case "
                                   f"${worst_case(prompt_chars):.4f} would exceed cap")
                    break
                fut, wc = launch(pool, todo.pop(0))
                inflight[fut] = wc
            if not inflight:
                break
            finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for f in finished:
                del inflight[f]
                rec = f.result()
                record(rec)
                print(f"{rec['route_id'][:16]}: {rec['status']} valid={rec.get('valid')} "
                      f"({rec.get('invalid_reason') or rec.get('error')}) {rec['wall_seconds']:.0f}s "
                      f"out={rec.get('usage', {}).get('completion_tokens')} "
                      f"reason_tok={rec.get('reasoning_tokens')} cost={rec.get('cost_usd_list_price')}",
                      flush=True)
                if rec["status"] != "ok":
                    stopped = f"ERROR on {rec['route_id']}: {rec['error']}"
                    todo = []
        if stopped:
            print(stopped, flush=True)
    print(f"DeepSeek thinking spend ${rp.spent(out_path):.4f}", flush=True)
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
