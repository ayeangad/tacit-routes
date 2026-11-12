#!/usr/bin/env python3
"""FINAL TEST, STAGE 1: predictions on the 3,052 frozen evaluation routes. NO ANSWERS ARE READ.

Plan of record: docs/plan.md (+ amendments), DECISION_LOG.md, and the
harness amendments (review 5). Reads only apply-side inputs
(data/eval_frozen/routes/*.npz, data/eval_frozen/raw/apply/) plus train-side models and history.
Never opens data/eval_frozen/raw/score/. Writes runs/final/predictions/<arm>.json + MANIFEST.json
(sha256 per arm); every harness incident goes to runs/final/incidents.log.

History rule for every arm: the COMPETITION rule (any train route, whatever its date), except
v0-hist-past (train routes strictly before the route's date).

DBO6 (the one eval station with no train routes) is always its own batch: our models and the rules
decode one route per job anyway; jpt-amz and the Opus systems get a separate single-route batch.
Our models give an unknown station empty same-station history, so only POOLED history is used.

Subcommands (run from tacit-routes/):
  sample                          write the fixed-seed 500-route primary sample (+ .txt for zone_rich.py)
  models --arms v0hist,v0past,v1,v1_exact
  pilot                           v0-hist, v1 (and v1_exact) on the 50 pilot routes (descriptive)
  rules --arms R2,R4,S3           train-chosen rules; S3 = shortest route at 3 s/route (val used 10 s)
  jpt-prepare / jpt-apply         jpt-amz (built on train, runs/jpt) in 300-route chunks
  opus --run N --rep K            Opus fair-fight system N on eval, sandboxed, in 300-route batches
  zone-rich-collect --out DIR [--solver exact|greedy]
                                  OPUS-MODEL arm: stop orders (REPAIRED and STRICT) from a
                                  scripts/zone_rich.py run on eval npz; no answers used
  status
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_common as fc  # noqa: E402

LAMBDAS = {"v0hist": 0.5, "v0past": 1.0, "v1": 0.5, "v1_exact": 0.5}
EXCLUDE = {"v0hist": (), "v0past": ("past",), "v1": (), "v1_exact": ()}
BEAM = 5
BATCH = int(os.environ.get("TR_OPUS_BATCH", "300"))
BATCH_TIMEOUT = os.environ.get("TR_OPUS_BATCH_TIMEOUT", "90m")
SINGLE_TIMEOUT = "15m"
S_TIME_LIMIT = 3


def tag(name, limit):
    return ("smoke_" if limit else "") + name


def subset(ids, limit, extra=""):
    return ids[:limit] + ([extra] if extra and extra not in ids[:limit] else []) if limit else ids


# ------------------------------------------------------------------------------- our models
def _model_init(loader_kind):
    import v1_model
    if loader_kind == "eval":
        v1_model.load_npz = fc.load_eval_npz            # same layout; no actual_order
    v1_model.set_ensemble("main", v1_model.SEEDS)


def _model_job(job):
    import v1_model
    rid, arm = job
    base = "v1" if arm == "v1_exact" else arm
    c0 = time.process_time()
    _, order, names = v1_model.decode_job((rid, base, EXCLUDE[arm], LAMBDAS[arm], BEAM))
    if arm == "v1_exact":
        import exact_zone
        order = exact_zone.stops_from_zone_order_exact(v1_model.load_npz(rid), names)
    return rid, order, names, time.process_time() - c0


def run_models(arms, ids, workers, loader_kind, prefix=""):
    import numpy as np  # noqa: F401
    loader = fc.load_eval_npz if loader_kind == "eval" else None
    for arm in arms:
        t0 = time.time()
        with mp.Pool(workers, initializer=_model_init, initargs=(loader_kind,)) as pool:
            done = pool.map(_model_job, [(r, arm) for r in ids], chunksize=4)
        orders = {r: o for r, o, _, _ in done}
        if loader:
            bad = [r for r in ids if not fc.check_permutation(loader(r), orders[r])]
            if bad:
                fc.incident(prefix + arm, f"{len(bad)} invalid orders", " ".join(bad[:10]))
        meta = {"history_rule": "past-only" if arm == "v0past" else "competition (all train routes)",
                "in_zone_solver": "exact (scripts/exact_zone.py)" if arm == "v1_exact" else "greedy descent (v0_decode)",
                "beam": BEAM, "lambda": LAMBDAS[arm], "routes": len(ids), "wall_s": time.time() - t0,
                "cpu_s_per_route_mean": sum(c for *_, c in done) / len(done), "workers": workers,
                "zone_orders": {r: z for r, _, z, _ in done}}
        fc.write_predictions(prefix + arm, orders, meta)
        print(f"{prefix}{arm}: {len(ids)} routes, {time.time() - t0:.0f}s", flush=True)


def cmd_models(a):
    run_models(a.arms.split(","), subset(fc.eval_ids(), a.limit, a.with_route), a.workers, "eval",
               prefix="smoke_" if a.limit else "")


def cmd_pilot(a):
    splits = json.load(open(fc.ROOT / "data" / "processed" / "splits.json"))
    ids = subset(sorted(r["route_id"] for r in splits["pilot"]), a.limit)
    run_models(a.arms.split(","), ids, a.workers, "processed", prefix=("smoke_" if a.limit else "") + "pilot_")


# ------------------------------------------------------------------------------------ rules
_RB = {}


def _rules_init(hist):
    import rules_baseline as rb
    rb.worker_init(hist)
    _RB["rb"] = rb


def _eval_route_for_rules(rid):
    import numpy as np
    z = fc.load_eval_npz(rid)
    assert str(z["stop_type"][0]) == "Station"
    return {"rid": rid, "ids": [str(s) for s in z["stop_ids"]],
            "tt": np.round(z["travel_time"].astype(np.float64), 1),
            "raw_zones": [str(s) for s in z["zone_ids"]], "station": str(z["station"])}


def _rules_job(job):
    rid, arm, rule = job
    rb = _RB["rb"]
    route = _eval_route_for_rules(rid)
    c0 = time.process_time()
    if rule == "shortest":
        order = rb.shortest_route(route, time_limit_s=S_TIME_LIMIT)
    else:
        order = rb.blocks_route(route, rule)[0]
    assert order[0] == 0 and sorted(order) == list(range(len(route["ids"])))
    return rid, [route["ids"][i] for i in order], time.process_time() - c0


def cmd_rules(a):
    import rules_baseline as rb
    r2, _, _ = rb.choose_rules()
    alpha, _ = rb.choose_alpha()
    splits = json.load(open(rb.SPLITS))
    hist = rb.build_hist(sorted(r["route_id"] for r in splits["train"]))   # same counts as the val run
    rules = {"S3": "shortest", "R2": r2, "R4": f"hist_{alpha}"}
    ids = subset(fc.eval_ids(), a.limit, a.with_route)
    for arm in a.arms.split(","):
        t0 = time.time()
        with mp.Pool(a.workers, initializer=_rules_init, initargs=(hist,)) as pool:
            done = pool.map(_rules_job, [(r, arm, rules[arm]) for r in ids], chunksize=2)
        meta = {"rule": rules[arm], "routes": len(ids), "wall_s": time.time() - t0,
                "cpu_s_per_route_mean": sum(c for *_, c in done) / len(done)}
        if arm == "S3":
            meta["note"] = f"shortest route at {S_TIME_LIMIT} s per route (reference only); the val run used 10 s"
        fc.write_predictions(tag(f"rules_{arm}", a.limit), {r: o for r, o, _ in done}, meta)
        print(f"{tag('rules_' + arm, a.limit)}: {len(ids)} routes, {time.time() - t0:.0f}s", flush=True)


# -------------------------------------------------------------------------------------- jpt
def jpt_dir():
    return fc.FINAL / "jpt"


def _jpt_write_chunks(route_filter, chunk_size, name_prefix, limit=0):
    """Stream the three eval apply files side by side; write rc-cli chunk folders. DBO6-type
    routes (no train history at their station) always go to their own single-route chunk."""
    import prepare_jpt_inputs as pj
    files = ("eval_route_data.json", "eval_package_data.json", "eval_travel_times.json")
    streams = [pj.stream(fc.EVAL_APPLY / f) for f in files]
    solo = set(fc.no_history_ids())
    writers, n_in, chunk, k, made = None, 0, -1, 0, []

    def open_chunk(name):
        d = jpt_dir() / name / "data" / "model_apply_inputs"
        made.append(name)
        return {"r": pj.ObjectWriter(d / "new_route_data.json"), "p": pj.ObjectWriter(d / "new_package_data.json"),
                "t": pj.ObjectWriter(d / "new_travel_times.json")}

    for items in zip(*streams):
        keys = {key for key, _ in items}
        assert len(keys) == 1, keys
        rid = items[0][0]
        if route_filter is not None and rid not in route_filter:
            continue
        if limit and k == limit:
            break
        route, pkg, tt = (pj.restore_nan(v) for _, v in items)
        route = {kk: vv for kk, vv in route.items() if kk != "route_score"}
        if rid in solo or route_filter is not None and chunk_size == 1:
            w = open_chunk(f"{name_prefix}single_{rid[-12:]}")
            w["r"].add(rid, route), w["p"].add(rid, pkg), w["t"].add(rid, tt)
            for x in w.values():
                x.close()
        else:
            if writers is None or n_in == chunk_size:
                if writers:
                    for x in writers.values():
                        x.close()
                chunk += 1
                n_in = 0
                writers = open_chunk(f"{name_prefix}chunk_{chunk:02d}")
            writers["r"].add(rid, route), writers["p"].add(rid, pkg), writers["t"].add(rid, tt)
            n_in += 1
        k += 1
    if writers:
        for x in writers.values():
            x.close()
    return k, made


def cmd_jpt_prepare(a):
    import resource
    import prepare_jpt_inputs as pj
    resource.setrlimit(resource.RLIMIT_AS, (pj.MEMORY_CAP_BYTES, pj.MEMORY_CAP_BYTES))
    k, made = _jpt_write_chunks(None, a.chunk, "", a.limit)
    if not a.limit:
        assert k == len(fc.eval_ids()), (k, "routes in apply files vs index")
    print(f"jpt inputs: {k} routes in {len(made)} chunks under {jpt_dir()}")


def _jpt_run_chunk(c):
    for name, target in (("src", fc.ROOT / "vendor" / "jpt-amz"), ("pydeps", fc.ROOT / "runs" / "jpt" / "pydeps")):
        if not (c / name).exists():
            (c / name).symlink_to(target)
    bo = c / "data" / "model_build_outputs"
    if not bo.exists():
        shutil.copytree(fc.ROOT / "runs" / "jpt" / "data" / "model_build_outputs", bo, symlinks=True)
    (c / "data" / "model_apply_outputs").mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(HERE / "run_jpt.py"), "apply", "--run-dir", str(c),
                        "--t1", "1", "--t2", "1", "--tag", "final"], cwd=fc.ROOT)
    out = c / "out_final" / "proposed_sequences.json"
    if r.returncode != 0 or not out.exists():
        return None
    return {rid: sorted(v["proposed"], key=v["proposed"].get) for rid, v in json.load(open(out)).items()}


def cmd_jpt_apply(a):
    import resource
    import prepare_jpt_inputs as pj
    chunks = sorted(p for p in jpt_dir().glob("*") if p.is_dir() and (p.name.startswith("chunk_") or p.name.startswith("single_")))
    if a.limit_chunks:
        chunks = chunks[:a.limit_chunks]
    orders, t0 = {}, time.time()
    for c in chunks:
        got = _jpt_run_chunk(c)
        want = list(json.load(open(c / "data" / "model_apply_inputs" / "new_route_data.json")))
        if got is None or set(want) - set(got):
            fc.incident("jpt", f"{c.name}: failed or incomplete; re-running its routes one by one",
                        (c / "logs" / "apply_final.log").read_text() if (c / "logs" / "apply_final.log").exists() else "")
            got = got or {}
            redo = [r for r in want if r not in got]
            resource.setrlimit(resource.RLIMIT_AS, (pj.MEMORY_CAP_BYTES, pj.MEMORY_CAP_BYTES))
            _, singles = _jpt_write_chunks(set(redo), 1, f"redo_{c.name}_")
            for s in singles:
                g = _jpt_run_chunk(jpt_dir() / s)
                if g:
                    got.update(g)
                else:
                    fc.incident("jpt", f"{s}: fails on its own -> route left invalid")
        orders.update(got)
    fc.write_predictions(tag("jpt", a.limit_chunks), orders,
                         {"lkh_limits_s": [1, 1], "built_on": "5,762 train routes (runs/jpt)", "chunks": len(chunks),
                          "wall_s": time.time() - t0, "cpu_s_per_route_mean": None})
    print(f"{tag('jpt', a.limit_chunks)}: {len(orders)} routes, {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------- opus
def _prepare_copy(src, copy):
    """Real copies of code/models/small files (writes cannot reach the original arena); data/train
    and .venv are hard links, mounted read-only by final_bwrap.sh. Stale outputs removed."""
    if copy.exists():
        subprocess.run(["chmod", "-R", "u+w", str(copy)], check=True)
        shutil.rmtree(copy)
    (copy / "data").mkdir(parents=True)
    for e in src.iterdir():
        if e.name in ("data", ".venv", ".claude"):
            continue
        (shutil.copytree(e, copy / e.name, symlinks=True) if e.is_dir() else shutil.copy2(e, copy / e.name))
    for e in (src / "data").iterdir():
        if e.name in ("train", "val_inputs"):
            continue
        (shutil.copytree(e, copy / "data" / e.name) if e.is_dir() else shutil.copy2(e, copy / "data" / e.name))
    subprocess.run(["cp", "-al", str(src / "data" / "train"), str(copy / "data" / "train")], check=True)
    subprocess.run(["cp", "-al", str(src / ".venv"), str(copy / ".venv")], check=True)
    for stale in ("val_predictions.json", "eval_predictions.json"):
        (copy / stale).unlink(missing_ok=True)


def _write_inputs(dirpath, ids):
    import io
    import numpy as np
    if dirpath.exists():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True)
    for rid in ids:
        with np.load(fc.EVAL_ROUTES / f"{rid}.npz", allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files if k not in ("actual_order", "quality")}
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        (dirpath / f"{rid}.npz").write_bytes(buf.getvalue())


def _run_batch(src, copy, ids, mode, logfile, timeout):
    """Run Opus's unchanged predict.py on exactly `ids`. mode 'blind': swap the inputs into
    data/val_inputs (+ val_ids.txt), no args; mode 'informed': predict.py <in_dir> <out_path>.
    Returns ({route: order} parsed from its output, exit code)."""
    if mode == "blind":
        _write_inputs(copy / "data" / "val_inputs", ids)
        (copy / "data" / "val_ids.txt").write_text("\n".join(ids) + "\n")
        out = copy / "val_predictions.json"
        cmd = [".venv/bin/python", "predict.py"]
    else:
        _write_inputs(copy / "eval_inputs_batch", ids)
        out = copy / "eval_predictions_batch.json"
        cmd = [".venv/bin/python", "predict.py", str(src / "eval_inputs_batch"), str(src / "eval_predictions_batch.json")]
    out.unlink(missing_ok=True)
    with open(logfile, "a") as lf:
        lf.write(f"\n===== {time.strftime('%FT%TZ', time.gmtime())} mode={mode} routes={len(ids)}\n")
        lf.flush()
        r = subprocess.run(["sh", str(HERE / "final_bwrap.sh"), str(copy), str(src), timeout, *cmd],
                           stdout=lf, stderr=subprocess.STDOUT)
    got = {}
    if out.exists():
        try:
            got = json.load(open(out))
        except json.JSONDecodeError:
            got = {}
    return {k: v for k, v in got.items() if k in set(ids)}, r.returncode


def _valid(rid, order):
    try:
        return fc.check_permutation(fc.load_eval_npz(rid), order)
    except Exception:
        return False


def cmd_opus(a):
    """Batches of 300 (the scale the systems were built for), DBO6-type routes alone. A failed batch
    (crash/OOM/timeout/partial or invalid output) -> its failed routes are re-run one by one, so a
    failure invalidates ONLY routes that fail on their own. Informed runs (5-6) use
    predict.py <in> <out>; if that interface fails on a batch where the blind method works, the
    harness switches to the blind method for that system (logged). Opus's code is never edited."""
    assert a.run in (2, 3, 4, 5, 6) and a.rep in (1, 2)
    arm = tag(f"opus_run{a.run}_rep{a.rep}", a.limit)
    src = fc.ROOT / f"opus_arena_run{a.run}"
    work = fc.FINAL / "opus" / f"run{a.run}_rep{a.rep}"
    work.mkdir(parents=True, exist_ok=True)
    copy = work / "arena"
    _prepare_copy(src, copy)
    ids = subset(fc.eval_ids(), a.limit, a.with_route)
    if getattr(a, "sample", False):  # option B (Angad): repeat 2 only on the 500-route sample
        samp = set(json.load(open(fc.SAMPLE_FILE))["route_ids"])
        ids = [r for r in ids if r in samp]
        assert a.limit or len(ids) == 500, len(ids)
    solo = [r for r in ids if r in set(fc.no_history_ids())]
    rest = [r for r in ids if r not in set(solo)]
    batches = [rest[i:i + BATCH] for i in range(0, len(rest), BATCH)] + [[r] for r in solo]
    mode = "informed" if a.run >= 5 else "blind"
    log = work / "log.txt"
    orders, t0, batch_walls = {}, time.time(), []
    for bi, b in enumerate(batches):
        tb = time.time()
        got, code = _run_batch(src, copy, b, mode, log, BATCH_TIMEOUT)
        if mode == "informed" and (code != 0 or not got):
            got_b, code_b = _run_batch(src, copy, b, "blind", log, BATCH_TIMEOUT)
            if code_b == 0 and got_b:
                fc.incident(arm, f"batch {bi}: predict.py <in> <out> failed (exit {code}); blind method works -> "
                                 "using the blind method (inputs swapped into data/val_inputs) for this system")
                mode, got, code = "blind", got_b, code_b
        ok = {r: o for r, o in got.items() if _valid(r, o)}
        failed = [r for r in b if r not in ok]
        if failed:
            fc.incident(arm, f"batch {bi} ({len(b)} routes): exit {code}, {len(failed)} routes missing/invalid; "
                             "re-running those routes one by one")
            for r in failed:
                g, c = _run_batch(src, copy, [r], mode, log, SINGLE_TIMEOUT)
                if r in g and _valid(r, g[r]):
                    ok[r] = g[r]
                else:
                    fc.incident(arm, f"route {r} fails on its own (exit {c}) -> scored as invalid")
        orders.update(ok)
        batch_walls.append({"batch": bi, "routes": len(b), "wall_s": time.time() - tb, "exit": code})
        print(f"{arm}: batch {bi + 1}/{len(batches)} ({len(b)} routes) ok {len(ok)} in {time.time() - tb:.0f}s", flush=True)
    meta = {"design": "B informed + literature" if a.run >= 5 else "A blind", "mode_used": mode,
            "batch_size": BATCH, "routes_expected": len(ids), "missing_routes": sorted(set(ids) - set(orders)),
            "wall_s": time.time() - t0, "batches": batch_walls, "log": str(log)}
    fc.write_predictions(arm, orders, meta)
    print(f"{arm}: {len(orders)}/{len(ids)} routes in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------ OPUS-MODEL arm
def cmd_zone_rich_collect(a):
    """Turn a scripts/zone_rich.py run on eval npz into stop orders, without answers.
    REPAIRED = zone_rich's fallback view (valid zones in Opus's order + cheapest insertion of the
    missing ones); STRICT = only answers whose zone list was valid. Routes with no OK call are absent
    (scored as invalid in both views). Solver = the final model's in-zone solver."""
    import numpy as np  # noqa: F401
    import score_pilot_zone as spz
    import zone_rich as zr
    from v0_model import prepare_route
    out = Path(a.out).resolve()
    info = json.loads((out / "prompts_index.json").read_text())
    assert Path(info["input_dir"]).resolve() == fc.EVAL_ROUTES.resolve(), "zone_rich run must be on the eval npz folder"
    recs, refused, spent = {}, collections.defaultdict(list), collections.defaultdict(float)
    for line in (out / "opus.jsonl").read_text().splitlines():
        r = json.loads(line)
        spent[r["route_id"]] += r.get("total_cost_usd") or 0.0
        if r.get("status") == "ok":
            recs[r["route_id"]] = r
        elif "safeguards flagged" in (r.get("error") or ""):
            refused[r["route_id"]].append(r)
    # REFUSAL RULE (fixed 2026-11-12 ~02:40 IST, before any answer was opened; runs/final/incidents.log): a route whose
    # call was only ever refused by the API safeguard has no Opus answer. STRICT: absent (invalid score), as before.
    # REPAIRED: treated as an empty zone list, so the fallback (cheapest insertion of ALL zones) answers for it,
    # exactly as for any invalid list. This is the view that is kinder to Opus, and it is the primary view.
    repaired, strict, per = {}, {}, {}
    for x in info["routes"]:
        rid = x["route_id"]
        if rid not in recs and rid not in refused:
            continue
        route = fc.load_eval_npz(rid)
        P = prepare_route(route)
        names = x["zone_ids"]
        assert P["names"] == names
        rec = recs.get(rid)
        k = {z: i for i, z in enumerate(names)}
        kept = list(dict.fromkeys(z for z in ((rec or {}).get("parsed_zone_order") or []) if z in k))
        full, inserted = spz.cheapest_insertion(P, [k[z] for z in kept])
        stops = zr.stops_for(route, [names[z] for z in full], a.solver)
        repaired[rid] = stops
        if rec is not None and rec["valid"]:
            strict[rid] = stops
        if rec is not None:
            per[rid] = {"valid": rec["valid"], "inserted": [names[z] for z in inserted],
                        "cost_usd": rec.get("total_cost_usd"), "wall_s": rec.get("wall_seconds"),
                        "cost_usd_all_attempts": spent[rid], "safeguard_refusals_before_ok": len(refused.get(rid, []))}
        else:
            per[rid] = {"valid": False, "refused": True, "inserted": [names[z] for z in inserted],
                        "cost_usd": spent[rid], "wall_s": sum(r.get("wall_seconds") or 0 for r in refused[rid]),
                        "cost_usd_all_attempts": spent[rid], "safeguard_refusals": len(refused[rid])}
    sample = json.load(open(fc.SAMPLE_FILE))
    base = {"solver": a.solver, "zone_rich_out": str(out), "frozen": json.load(open(fc.ROOT / "runs" / "zone_rich" / "FROZEN.json")),
            "sample_sha256": sample["sha256_of_route_ids"], "calls_ok": len(recs), "routes_in_run": len(info["routes"]),
            "routes_refused_by_safeguard": sorted(r for r in refused if r not in recs),
            "refusal_rule": "refused-only routes: STRICT absent (invalid); REPAIRED = fallback from an empty zone list",
            "per_route": per}
    # Arm names follow the effort recorded in each OK call (fix 2026-11-12, see runs/final/incidents.log):
    # max (the frozen primary) -> opus_model_*; any other effort -> opus_model_<effort>_*, so the
    # descriptive default-effort run can never overwrite the primary arm.
    efforts = {r.get("effort") for r in recs.values()}
    if len(efforts) != 1:
        raise SystemExit(f"REFUSING: mixed or missing effort in {out}/opus.jsonl: {efforts}")
    effort = efforts.pop()
    # None = records written before zone_rich.py stored the effort (the dev runs, used by the dress
    # rehearsal): keep the original arm names for them, exactly as before the fix.
    stem = "opus_model" if effort in ("max", None) else f"opus_model_{effort}"
    base["effort"] = effort
    view = "REPAIRED (primary)" if effort == "max" else "REPAIRED (descriptive)"
    fc.write_predictions(f"{stem}_repaired_{a.solver}", repaired, {**base, "view": view})
    fc.write_predictions(f"{stem}_strict_{a.solver}", strict, {**base, "view": "STRICT"})
    print(f"{stem} ({a.solver}, effort {effort}): repaired {len(repaired)}, strict {len(strict)}, calls ok {len(recs)}")


def cmd_sample(a):
    s = fc.select_primary_sample(write=True)
    (fc.FINAL / "primary_sample_500.txt").write_text("\n".join(s["route_ids"]) + "\n")
    print(f"sample: {s['n']} routes, sha256 {s['sha256_of_route_ids']}")


def cmd_status(a):
    man = fc.PRED / "MANIFEST.json"
    for k, v in (json.load(open(man)).items() if man.exists() else []):
        print(f"{k:34s} {v['n_routes']:5d}  {v['sha256'][:12]}  {v['written_utc']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    sub.add_parser("status")
    m = sub.add_parser("models")
    m.add_argument("--arms", default="v0hist,v0past,v1,v1_exact")
    m.add_argument("--workers", type=int, default=3)
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--with-route", default="")
    p = sub.add_parser("pilot")
    p.add_argument("--arms", default="v0hist,v1,v1_exact")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("rules")
    r.add_argument("--arms", default="R2,R4,S3")
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--with-route", default="")
    j = sub.add_parser("jpt-prepare")
    j.add_argument("--chunk", type=int, default=300)
    j.add_argument("--limit", type=int, default=0)
    ja = sub.add_parser("jpt-apply")
    ja.add_argument("--limit-chunks", type=int, default=0)
    o = sub.add_parser("opus")
    o.add_argument("--run", type=int, required=True)
    o.add_argument("--rep", type=int, required=True)
    o.add_argument("--limit", type=int, default=0)
    o.add_argument("--with-route", default="")
    o.add_argument("--sample", action="store_true", help="option B: run only the 500-route primary sample (repeat 2)")
    z = sub.add_parser("zone-rich-collect")
    z.add_argument("--out", required=True)
    z.add_argument("--solver", choices=("exact", "greedy"), default="exact")
    a = ap.parse_args()
    {"sample": cmd_sample, "status": cmd_status, "models": cmd_models, "pilot": cmd_pilot, "rules": cmd_rules,
     "jpt-prepare": cmd_jpt_prepare, "jpt-apply": cmd_jpt_apply, "opus": cmd_opus,
     "zone-rich-collect": cmd_zone_rich_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
