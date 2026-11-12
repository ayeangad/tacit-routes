#!/usr/bin/env python3
"""PRIMARY comparison, frontier arm: Opus 5.5 picks the ZONE ORDER from a zone summary plus RICH
station history; our inside-zone solver (v0_decode.stops_from_zone_order) fills in the stops.
Plan of record: docs/plan.md item 1.

Differences from the pilot zone arm (scripts/build_prompts_zone.py, past-only, 3 references):
  - history rule = COMPETITION RULE: any TRAIN route at the same station, whatever its date
    (the same rule our model uses on the final test);
  - richer evidence, counted over ALL same-station train routes, restricted to THIS route's zones
    (each past route is projected onto this route's zone set, keeping its driver's first-visit order):
      * usual first zone: how often each of this route's zones came first (top 5, with counts);
      * for every zone: its most common next zones (top 3, with counts);
      * a zone never seen exactly at this station: the same counts at the coarser zone-ID levels
        ('A-2.2', then 'A-2'), clearly labelled;
      * the K most similar train routes (most shared zones; ties: route ID), each projected onto
        this route's zones.
Zone definition, centroids and travel times are identical to the pilot zone arm (v0 prepare_route).

Generic over a route list and an input folder, so the SAME frozen prompt runs later on pilot or
eval npz files. Target npz files are read ONLY for: stop_ids, lat, lng, zone_ids, travel_time,
station, date, route_id, n_packages, tw_end, service_time (never actual_order). Driver orders of
targets are read only by `score` (val/pilot, after the calls) and by the val leak check.

  build:  .venv/bin/python -I scripts/zone_rich.py build --version A --routes R.txt --input-dir DIR --out OUT
  run:    .venv/bin/python -I scripts/zone_rich.py run --out OUT --workers 3 --i-have-approval
  score:  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/zone_rich.py score --out OUT [--solver greedy|exact]
  run --effort LEVEL adds `--effort LEVEL` to the approved claude command (default: not set).
  score writes STRICT and REPAIRED (= fallback) per route; the primary comparison uses REPAIRED.

Invalid answers: same two views as score_pilot_zone.py.
  strict:   an invalid zone list (missing, duplicate or unknown zones, or no list) gets the
            challenge's invalid-route score;
  fallback: keep the valid zones in Opus's order (first occurrence, unknown IDs dropped), then
            insert each missing zone by cheapest insertion on v0 zone travel times.
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from v0_model import prepare_route  # noqa: E402
import v0_decode  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rp = _load("run_pilot", HERE / "run_pilot.py")
rz = _load("run_pilot_zone", HERE / "run_pilot_zone.py")
spz = _load("score_pilot_zone", HERE / "score_pilot_zone.py")

TARGET_KEYS = ("stop_ids", "lat", "lng", "zone_ids", "travel_time", "station", "date", "route_id",
               "n_packages", "tw_end", "service_time")
HIST = ROOT / "runs" / "v0" / "history_train.json"
K_REF = 8
TOP_NEXT = 3
TOP_FIRST = 5

SYSTEM = ("You are a delivery planner. A delivery route's stops are grouped into zones. "
          "Your job is to choose the order in which to visit the zones, the way experienced "
          "delivery drivers at this station actually do. Answer with a JSON list only, no other text.")

GUIDE_B = """How to read this: drivers at a station are creatures of habit. The history below counts what this station's drivers actually did with these same zones, and it is the strongest evidence you have: follow the usual first zone and the usual next-zone moves wherever they are consistent, and use the most similar past routes to settle the overall sweep. Zone IDs are hierarchical (e.g. `A-2.3B` = area A-2, row 3, column B); drivers usually finish one area before moving to the next and sweep neighbouring rows and columns in order. Fall back on travel times only where the history is silent or contradictory.

"""

TEMPLATE = """A delivery van leaves the station and must deliver to every zone below. All stops in a zone are delivered together, so you only choose the order of the zones. The order of stops inside each zone is decided separately.

{guide}Station: {station} at {slat:.6f} {slng:.6f}

Zones ({Z}). Each line: zone ID, centroid latitude, centroid longitude, number of stops. Lines are sorted by zone ID.
{zone_lines}

Travel times in seconds. Each number is the mean driving time over all pairs of stops (one stop in the first zone, one in the second).

From the station to each zone, in the zone order above:
{from_station}

From each zone back to the station, in the zone order above:
{to_station}

Zone-to-zone matrix. Row = from, column = to; columns are in the zone order above. The diagonal is the mean time between stops inside the same zone.
{matrix}

History of this station: {n_hist} past routes driven by this station's drivers. Each past route was reduced to the zones of THIS route that it visits, kept in its driver's order.

Usual FIRST zone (how many past routes started with each of this route's zones):
{first_lines}

Usual NEXT zone. For each zone of this route: the zones its drivers most often went to next, with counts ("seen in N past routes" = how many past routes visited it).
{next_lines}

The {k} most similar past routes (most zones in common with this route), reduced to this route's zones in driver order:
{ref_lines}

Output the order in which to visit the zones as a JSON list of zone IDs, starting with the first zone after leaving the station. Include each of the {Z} zones exactly once and nothing else. Do not include the station."""


def level(z, L):
    if L == 0:
        return z
    m = re.match(r"^([A-Z]+-\d+)\.(\d+)([A-Z]+)$", z)
    if not m:
        return z
    return f"{m.group(1)}.{m.group(2)}" if L == 1 else m.group(1)


def load_target(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in TARGET_KEYS}


def history_by_station():
    rows = json.load(open(HIST))["routes"]
    splits = json.load(open(ROOT / "data" / "processed" / "splits.json"))
    train = {x["route_id"] for x in splits["train"]}
    assert {r["route_id"] for r in rows} == train, "history table must be exactly the train split"
    by = collections.defaultdict(list)
    for r in rows:
        by[r["station"]].append(r)
    return by, train


def evidence(names, past):
    """Counted evidence restricted to this route's zones. past: list of {route_id, zones}."""
    mine = set(names)
    proj = [(r["route_id"], [z for z in r["zones"] if z in mine]) for r in past]
    proj = [(rid, seq) for rid, seq in proj if seq]
    first = collections.Counter(seq[0] for _, seq in proj)
    nxt = collections.defaultdict(collections.Counter)
    seen = collections.Counter()
    for _, seq in proj:
        seen.update(set(seq))
        for a, b in zip(seq, seq[1:]):
            nxt[a][b] += 1
    # coarse levels: project past routes onto this route's coarse groups
    coarse = {}
    for L in (1, 2):
        groups = {level(z, L) for z in names}
        cn = collections.defaultdict(collections.Counter)
        cseen = collections.Counter()
        for r in past:
            seq = []
            for z in r["zones"]:
                g = level(z, L)
                if g in groups and (not seq or seq[-1] != g):
                    seq.append(g)
            cseen.update(set(seq))
            for a, b in zip(seq, seq[1:]):
                cn[a][b] += 1
        coarse[L] = (cn, cseen)
    refs = sorted(((-len(mine & set(r["zones"])), r["route_id"], r) for r in past
                   if mine & set(r["zones"])))[:K_REF]
    return first, nxt, seen, coarse, refs, len(proj)


def build(a):
    by, train = history_by_station()
    ids = [x.strip() for x in Path(a.routes).read_text().split() if x.strip()]
    out = Path(a.out).resolve()
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "system_prompt.txt").write_text(SYSTEM)
    index = []
    for rid in ids:
        assert rid not in train
        route = load_target(Path(a.input_dir) / f"{rid}.npz")
        station = str(route["station"])
        P, zones, zz, from_st, to_st = _zone_view(route)
        names = [z["zone_id"] for z in zones]
        Z = len(names)
        past = by.get(station, [])
        first, nxt, seen, coarse, refs, n_hist = evidence(names, past)
        first_lines = ", ".join(f"{z} ({c})" for z, c in first.most_common(TOP_FIRST)) or "none of this route's zones was ever first"
        nl = []
        for z in names:
            if seen[z]:
                tops = ", ".join(f"{b} ({c})" for b, c in nxt[z].most_common(TOP_NEXT)) or "it was always the last of these zones"
                nl.append(f"{z}: seen in {seen[z]}; next: {tops}")
            else:
                done = False
                for L in (1, 2):
                    g = level(z, L)
                    cn, cs = coarse[L]
                    if cs[g]:
                        tops = ", ".join(f"{b} ({c})" for b, c in cn[g].most_common(TOP_NEXT)) or "always last"
                        nl.append(f"{z}: never seen at this station; its group {g} seen in {cs[g]}; group next: {tops}")
                        done = True
                        break
                if not done:
                    nl.append(f"{z}: never seen at this station, nor its groups {level(z, 1)} / {level(z, 2)}")
        ref_lines = "\n".join(
            f"{r['date']}: shares {-neg} of {Z} zones: " + " ".join(z for z in r["zones"] if z in set(names))
            for neg, _, r in refs) or "none"
        w = max(len(i) for i in names)
        prompt = TEMPLATE.format(
            guide=GUIDE_B if a.version == "B" else "", station=station,
            slat=float(route["lat"][0]), slng=float(route["lng"][0]), Z=Z,
            zone_lines="\n".join(f"{z['zone_id']} {z['lat']:.6f} {z['lng']:.6f} {z['n_stops']}" for z in zones),
            from_station=" ".join(f"{z}:{round(float(v))}" for z, v in zip(names, from_st)),
            to_station=" ".join(f"{z}:{round(float(v))}" for z, v in zip(names, to_st)),
            matrix="\n".join(f"{names[i]:<{w}} | " + " ".join(str(round(float(v))) for v in zz[i]) for i in range(Z)),
            n_hist=n_hist, first_lines=first_lines, next_lines="\n".join(nl), k=len(refs), ref_lines=ref_lines)
        (out / "prompts" / f"{rid}.txt").write_text(prompt)
        index.append({"route_id": rid, "station": station, "date": str(route["date"]), "zone_ids": names,
                      "n_zones": Z, "n_stops_including_station": int(len(route["stop_ids"])),
                      "prompt_chars": len(prompt), "n_hist_routes_with_overlap": n_hist,
                      "references": [r["route_id"] for _, _, r in refs]})
    info = {"version": a.version, "input_dir": str(a.input_dir), "history_rule": "competition: all same-station TRAIN routes, any date",
            "k_ref": K_REF, "top_next": TOP_NEXT, "top_first": TOP_FIRST,
            "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "template_sha256": hashlib.sha256((TEMPLATE + (GUIDE_B if a.version == "B" else "")).encode()).hexdigest(),
            "routes": index}
    (out / "prompts_index.json").write_text(json.dumps(info, indent=1))
    print(f"{len(index)} prompts in {out}; chars mean {np.mean([x['prompt_chars'] for x in index]):.0f} "
          f"max {max(x['prompt_chars'] for x in index)}")
    return leak_check(out, info, train)


def _zone_view(route):
    P = prepare_route(route)
    lat, lng = route["lat"].astype(np.float64), route["lng"].astype(np.float64)
    zones = [{"zone_id": P["names"][k], "lat": float(lat[m].mean()), "lng": float(lng[m].mean()),
              "n_stops": int(len(m))} for k, m in enumerate(P["members"])]
    Z = P["Z"]
    return P, zones, P["ztt"][:Z, :Z], P["ztt"][Z], P["to_station"]


def leak_check(out, info, train):
    """References must be TRAIN routes; no route ID in any prompt. For routes whose npz carries a
    driver order (val/pilot), that order must not appear outside the reference block."""
    problems = []
    for x in info["routes"]:
        text = (out / "prompts" / f"{x['route_id']}.txt").read_text()
        if "RouteID" in text:
            problems.append(f"{x['route_id']}: route ID in prompt")
        if any(r not in train for r in x["references"]):
            problems.append(f"{x['route_id']}: non-train reference")
        p = Path(info["input_dir"]) / f"{x['route_id']}.npz"
        with np.load(p, allow_pickle=False) as d:
            if "actual_order" not in d.files:
                continue
            route = {k: d[k] for k in d.files}
        P = prepare_route(route)
        seen, order = set(), []
        for i in route["actual_order"][1:]:
            k = int(P["zone_of"][int(i)])
            if k not in seen:
                seen.add(k)
                order.append(P["names"][k])
        head = text.split("most zones in common with this route), reduced to")[0]
        assert len(head) < len(text), "reference header not found"
        if len(order) > 2 and " ".join(order) in head:
            problems.append(f"{x['route_id']}: driver zone order appears outside the reference block")
    if problems:
        print("LEAK CHECK FAILED:\n  " + "\n  ".join(problems))
        return 1
    print(f"leak check passed: {len(info['routes'])} prompts; all references TRAIN; no route IDs; "
          f"no target driver zone order outside the reference block")
    return 0


def opus_cmd(system_file, effort=None):
    """The approved command (run_pilot.opus_command); with --effort <level> added after the model if set."""
    cmd = rp.opus_command(system_file)
    if effort:
        i = cmd.index(rp.OPUS_MODEL) + 1
        cmd = cmd[:i] + ["--effort", effort] + cmd[i:]
    return cmd


def call_opus(system_file, prompt_file, effort=None):
    """run_pilot.call_opus with the optional --effort flag; fresh empty temp dir per call."""
    import os
    import shutil
    import tempfile
    workdir = tempfile.mkdtemp(prefix="opus_zone_rich_")
    try:
        with open(prompt_file, "rb") as stdin:
            proc = subprocess.run(opus_cmd(system_file, effort), stdin=stdin, cwd=workdir,
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


def one_call(out, rid, zone_ids, system_file, effort=None):
    prompt_file = out / "prompts" / f"{rid}.txt"
    rec = {"route_id": rid, "started_utc": rp.now(), "n_zones": len(zone_ids), "effort": effort or "default"}
    t0 = time.monotonic()
    error = kind = resp = code = None
    for attempt in range(3):
        rec["attempts"] = attempt + 1
        try:
            resp, code, leftover = call_opus(system_file, prompt_file, effort)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            error, kind = f"{type(e).__name__}: {e}", None
            kind = rz.classify(error)
        else:
            if code != 0 or resp.get("is_error"):
                error = f"claude exit {code}, is_error {resp.get('is_error')}, result {str(resp.get('result'))[:300]!r}"
                kind = rz.classify(error)
            else:
                error = kind = None
                break
        if kind != "network":
            break
        time.sleep(10 * (attempt + 1))
    rec["wall_seconds"] = time.monotonic() - t0
    if resp is not None:
        rec.update(model_returned=sorted((resp.get("modelUsage") or {}).keys()), total_cost_usd=resp.get("total_cost_usd"),
                   duration_ms=resp.get("duration_ms"), usage=resp.get("usage"),
                   command=" ".join(x if x else '""' for x in opus_cmd(system_file, effort)) + " < <prompt file>")
    if error:
        rec.update(status="error", error=error, error_kind=kind)
        return rec
    text = resp.get("result") or ""
    zones, valid, reason, details = rz.parse_zone_order(text, zone_ids)
    rec.update(status="ok", raw_output=text, parsed_zone_order=zones, valid=valid,
               invalid_reason=None if valid else reason, invalid_details=details)
    return rec


def run(a):
    out = Path(a.out).resolve()
    info = json.loads((out / "prompts_index.json").read_text())
    system_file = out / "system_prompt.txt"
    log = out / "opus.jsonl"
    done = rp.done_routes(log)
    todo = [x for x in info["routes"] if x["route_id"] not in done]
    if not a.i_have_approval:
        print(f"{len(todo)} calls planned; refusing without --i-have-approval")
        return 3
    stop = False
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {}
        it = iter(todo)
        for x in [next(it, None) for _ in range(a.workers)]:
            if x:
                futs[ex.submit(one_call, out, x["route_id"], x["zone_ids"], system_file, a.effort)] = x
        while futs:
            f = next(cf.as_completed(futs))
            futs.pop(f)
            rec = f.result()
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{rec['route_id'][-12:]} {rec['status']} valid={rec.get('valid')} {rec['wall_seconds']:.0f}s "
                  f"${rec.get('total_cost_usd')}", flush=True)
            if rec["status"] == "error":
                stop = True
                print("ERROR -> stopping:", rec["error"][:200])
            if not stop:
                x = next(it, None)
                if x:
                    futs[ex.submit(one_call, out, x["route_id"], x["zone_ids"], system_file, a.effort)] = x
    return 2 if stop else 0


def stops_for(route, order, solver):
    if solver == "greedy":
        return v0_decode.stops_from_zone_order(route, order)
    if solver == "exact":
        import exact_zone
        return exact_zone.stops_from_zone_order_exact(route, order)
    raise SystemExit(f"unknown --solver {solver}")


def score(a):
    out = Path(a.out).resolve()
    info = json.loads((out / "prompts_index.json").read_text())
    recs = {}
    for line in (out / "opus.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "ok":
            recs[r["route_id"]] = r
    strict, fallback, per = {}, {}, {}
    for x in info["routes"]:
        rid = x["route_id"]
        if rid not in recs:
            continue
        route = v0_decode.load_npz(rid) if Path(info["input_dir"]).resolve() == v0_decode.ROUTES_DIR.resolve() else None
        assert route is not None, "scoring only for routes with driver orders in data/processed"
        P = prepare_route(route)
        names = x["zone_ids"]
        assert P["names"] == names
        rec = recs[rid]
        k = {z: i for i, z in enumerate(names)}
        kept = list(dict.fromkeys(z for z in (rec["parsed_zone_order"] or []) if z in k))
        full, inserted = spz.cheapest_insertion(P, [k[z] for z in kept])
        t0 = time.process_time()
        stops = stops_for(route, [names[z] for z in full], a.solver)
        cpu = time.process_time() - t0
        fallback[rid] = stops
        if rec["valid"]:
            strict[rid] = stops
        per[rid] = {"valid": rec["valid"], "inserted": [names[z] for z in inserted], "solver_cpu_s": cpu,
                    "cost_usd": rec.get("total_cost_usd"), "wall_s": rec.get("wall_seconds")}
    invalid = json.load(open(v0_decode.INVALID))
    fs, _ = v0_decode.official_scores(fallback, invalid)
    ss = {}
    if strict:
        ss, _ = v0_decode.official_scores(strict, invalid)
    for rid in per:
        per[rid]["score_repaired"] = per[rid]["score_fallback"] = fs[rid]   # repaired = fallback view
        per[rid]["score_strict"] = ss.get(rid, float(invalid[rid]) if rid in invalid else None)
    vals = lambda key: [per[r][key] for r in per]
    summary = {"n": len(per), "valid": sum(p["valid"] for p in per.values()),
               "mean_strict": float(np.mean(vals("score_strict"))), "mean_repaired": float(np.mean(vals("score_repaired"))),
               "mean_fallback": float(np.mean(vals("score_fallback"))),
               "mean_cost_usd": float(np.mean([c for c in vals("cost_usd") if c is not None])),
               "mean_wall_s": float(np.mean(vals("wall_s"))), "solver": a.solver}
    json.dump({"summary": summary, "per_route": per}, open(out / f"scores_{a.solver}.json", "w"), indent=1)
    print(json.dumps(summary))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--version", choices=("A", "B"), required=True)
    b.add_argument("--routes", required=True)
    b.add_argument("--input-dir", default=str(ROOT / "data" / "processed" / "routes"))
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--i-have-approval", action="store_true")
    r.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default=None,
                   help="passed to claude as --effort; default: not set (the CLI default)")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--solver", choices=("greedy", "exact"), default="greedy")
    a = ap.parse_args()
    if a.cmd == "run":
        assert a.workers <= 3
    return {"build": build, "run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
