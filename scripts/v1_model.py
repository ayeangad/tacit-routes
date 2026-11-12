#!/usr/bin/env python3
"""v1: v0-hist + coarser zone-ID history + history dropout, as a 5-seed average.

Agreed plan (docs/plan.md item 3, fixed before any rehearsal number):
  - features: v0's 33 base features + v0's 18 history features + 7 history features at each of
    two coarser zone-ID levels (micro 'A-2.2', macro 'A-2'): 65 in total (v1_history.py);
  - history dropout in training: for each training route, with probability DROP_P the history
    is thinned by removing every train route dated within ±D days of it, D uniform in
    0..DROP_MAX_DAYS (drawn once per route per seed); otherwise leave-one-out, as v0-hist;
  - architecture, optimiser, batch size, learning rate, early stopping, training data
    (High+Medium train routes outside the fixed 10% held-out split) are v0-hist's exactly
    (v0_model.ZonePointer, split_train, make_batch, evaluate). Early stopping uses the held-out
    High routes with FULL (leave-one-out) history, as v0-hist did;
  - 5 seeds (SEED + k, k = 0..4); prediction = log_softmax of the mean of the 5 models'
    log-probabilities (a normalised product of experts);
  - decoding: v0-hist's pre-registered decoder, beam 5 + travel lambda = 0.5, kept FIXED (not
    re-tuned) so v1 differs from v0-hist only in the model.

Train (heavy: run only after Opus runs 3-4 finish):
  OMP_NUM_THREADS=3 .venv/bin/python -I scripts/v1_model.py train --seeds 0,1,2,3,4
Smoke test (light):
  OMP_NUM_THREADS=1 .venv/bin/python -I scripts/v1_model.py train --seeds 0 --limit 40 --epochs 2 --tag smoke
Models go to runs/v1/<tag>/seed<k>/. Never touches val, pilot or data/eval_frozen.
"""
import argparse
import csv
import json
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v0_model import (N_FEATURES, SEED, SPLITS, ZonePointer, count_parameters, driver_zone_order,  # noqa: E402
                      evaluate, load_npz, make_batch, prepare_route, split_train, step_features)
from v0_history import N_HIST_FEATURES  # noqa: E402
from v1_history import N_LEVEL_FEATURES, ExclHistory, v1_hist_features  # noqa: E402

ROOT = HERE.parent
V1_DIR = ROOT / "runs" / "v1"
N_V1_FEATURES = N_FEATURES + N_HIST_FEATURES + N_LEVEL_FEATURES       # 33 + 18 + 14 = 65
DROP_P = 0.5
DROP_MAX_DAYS = 28
LAMBDA = 0.5
BEAM = 5
SEEDS = (0, 1, 2, 3, 4)


def features_v1(P, cur, prev, visited, t):
    return np.concatenate([step_features(P, cur, prev, visited, t),
                           v1_hist_features(P, cur, visited)], axis=1)


def teacher_forced_steps(P, order, feat):
    """As v0_model.teacher_forced_steps, with the feature function passed in."""
    Z = P["Z"]
    X, M, Y = [], [], []
    visited = np.zeros(Z, dtype=bool)
    cur, prev = -1, -1
    for t, target in enumerate(order):
        if Z - t >= 2:
            X.append(feat(P, cur, prev, visited, t))
            M.append(~visited)
            Y.append(target)
        visited[target] = True
        prev, cur = cur, target
    return np.stack(X).astype(np.float16), np.stack(M), np.array(Y, dtype=np.int64)


def build_examples(route_ids, eh, exclude_for, log_every=500):
    """exclude_for(route_id, route) -> set of train route IDs hidden from this route's history
    (the route itself is always hidden: leave-one-out)."""
    data, t0 = [], time.time()
    for k, r in enumerate(route_ids):
        route = load_npz(r)
        P = prepare_route(route)
        if P["Z"] < 2:
            continue
        eh.attach(P, str(route["station"]), r, exclude=exclude_for(r, route))
        X, M, Y = teacher_forced_steps(P, driver_zone_order(P, route), features_v1)
        data.append((r, X, M, Y))
        if log_every and (k + 1) % log_every == 0:
            print(f"  features for {k + 1}/{len(route_ids)} routes, {time.time() - t0:.0f}s", flush=True)
    return data


def dropout_rule(eh, seed):
    """History dropout, drawn once per route for this seed (deterministic given the seed)."""
    def exclude_for(r, route):
        rng = random.Random(f"{SEED}:{seed}:dropout:{r}")
        if rng.random() < DROP_P:
            return eh.window(str(route["date"]), rng.randint(0, DROP_MAX_DAYS))
        return set()
    return exclude_for


def train_one(seed, train_ids, held_ids, eh, out, args):
    torch.manual_seed(SEED + seed)
    random.seed(SEED + seed)
    out.mkdir(parents=True, exist_ok=True)
    t0, cpu0 = time.time(), time.process_time()
    train = build_examples(train_ids, eh, dropout_rule(eh, seed))
    held = build_examples(held_ids, eh, lambda r, route: set(), log_every=0)
    feat_seconds = time.time() - t0
    model = ZonePointer(N_V1_FEATURES)
    rows = np.concatenate([X[M] for _, X, M, _ in train]).astype(np.float64)
    model.feat_mean.copy_(torch.tensor(rows.mean(0), dtype=torch.float32))
    model.feat_std.copy_(torch.tensor(np.maximum(rows.std(0), 1e-3), dtype=torch.float32))
    del rows
    n_params = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    curve, best, best_epoch, waited = [], float("inf"), 0, 0
    rng = random.Random(SEED + seed)
    t_train, cpu_train = time.time(), time.process_time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        loss_sum, step_sum = 0.0, 0
        for k in range(0, len(order), args.batch_routes):
            x, m, y = make_batch([train[i] for i in order[k:k + args.batch_routes]])
            logp = model(x, m)
            loss = -logp[torch.arange(len(y)), y].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            step_sum += len(y)
        held_loss, held_acc, _ = evaluate(model, held)
        curve.append({"epoch": epoch, "train_loss": loss_sum / step_sum, "heldout_loss": held_loss,
                      "heldout_top1": held_acc, "seconds": time.time() - t_train})
        print(f"[seed {seed}] epoch {epoch:2d} train {loss_sum / step_sum:.4f} held-out {held_loss:.4f} "
              f"top-1 {held_acc:.3f} {time.time() - t_train:.0f}s", flush=True)
        if held_loss < best - 1e-4:
            best, best_epoch, waited = held_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "n_features": N_V1_FEATURES,
                        "n_params": n_params, "seed": seed}, out / "model.pt")
        else:
            waited += 1
            if waited >= args.patience:
                break
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    summary = {"seed": seed, "n_params": n_params, "n_features": N_V1_FEATURES,
               "train_routes": len(train), "heldout_high_routes": len(held),
               "best_epoch": best_epoch, "best_heldout_loss": best,
               "drop_p": DROP_P, "drop_max_days": DROP_MAX_DAYS,
               "feature_seconds": feat_seconds, "train_seconds": time.time() - t_train,
               "train_cpu_seconds": time.process_time() - cpu_train,
               "total_cpu_seconds": time.process_time() - cpu0,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "curve": curve}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    return summary


def cmd_train(args):
    torch.set_num_threads(args.threads)
    splits = json.load(open(SPLITS))
    fit_ids, held_ids, quality = split_train(splits)
    train_ids = sorted(r for r in fit_ids if quality[r] in {"High", "Medium"})
    held_high = sorted(r for r in held_ids if quality[r] == "High")
    if args.limit:
        train_ids, held_high = train_ids[:args.limit], held_high[:max(10, args.limit // 4)]
    forbidden = {r["route_id"] for r in splits["val"]} | {r["route_id"] for r in splits["pilot"]}
    assert not (set(train_ids) | set(held_high)) & forbidden
    assert not set(train_ids) & set(held_ids)               # the 234 held-out routes never train
    eh = ExclHistory()
    for s in [int(x) for x in args.seeds.split(",")]:
        summ = train_one(s, train_ids, held_high, eh, V1_DIR / args.tag / f"seed{s}", args)
        print(json.dumps({k: v for k, v in summ.items() if k != "curve"}), flush=True)


# ---- prediction: ensemble + decoding ------------------------------------------------------

class Ensemble(torch.nn.Module):
    """log_softmax(mean_k log p_k): a normalised product of the seed models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x, mask):
        lp = torch.stack([m(x, mask) for m in self.models]).mean(0)
        return torch.log_softmax(lp, dim=-1)


def load_ensemble(tag="main", seeds=SEEDS):
    models = []
    for s in seeds:
        ck = torch.load(V1_DIR / tag / f"seed{s}" / "model.pt", weights_only=False)
        m = ZonePointer(ck["n_features"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return Ensemble(models).eval()


def beam_decode(model, P, beam, lam, feat):
    """v0_decode.beam_decode with the feature function passed in (identical logic)."""
    Z = P["Z"]
    states = [(0.0, [], np.zeros(Z, dtype=bool), -1, -1)]
    for t in range(Z):
        x = np.stack([feat(P, cur, prev, vis, t) for _, _, vis, cur, prev in states])
        m = np.stack([~vis for _, _, vis, _, _ in states])
        with torch.no_grad():
            logp = model(torch.from_numpy(x), torch.from_numpy(m)).numpy()
        expansions = []
        for s, (score, order, vis, cur, prev) in enumerate(states):
            secs = P["ztt"][cur if cur >= 0 else Z]
            for z in np.where(~vis)[0]:
                expansions.append((score + float(logp[s, z]) - lam * float(secs[z]) / P["tt_scale"], s, int(z)))
        expansions.sort(key=lambda e: -e[0])
        new_states = []
        for total, s, z in expansions[:beam]:
            _, order, vis, cur, _ = states[s]
            vis2 = vis.copy()
            vis2[z] = True
            new_states.append((total, order + [z], vis2, z, cur))
        states = new_states
    return states[0][1]


def sequence_objective(model, P, order, lam, feat):
    """The decoder's objective for a full zone order: sum of log-probabilities minus
    lam * travel / tt_scale, step by step from the station."""
    Z = P["Z"]
    vis, cur, prev, total = np.zeros(Z, dtype=bool), -1, -1, 0.0
    for t, z in enumerate(order):
        x = feat(P, cur, prev, vis, t)[None]
        with torch.no_grad():
            lp = model(torch.from_numpy(x), torch.from_numpy((~vis)[None])).numpy()[0]
        total += float(lp[z]) - lam * float(P["ztt"][cur if cur >= 0 else Z][z]) / P["tt_scale"]
        vis[z] = True
        prev, cur = cur, z
    return total


# ---- one decoding job: (route, arm, exclusion) -> stop order --------------------------------
_W = {}


def _worker_state():
    if not _W:
        torch.set_num_threads(1)
        _W["eh"] = ExclHistory()
    return _W


_ENS = ["main", SEEDS]


def set_ensemble(tag, seeds):
    """Which v1 ensemble the workers load (pool initializer)."""
    _ENS[0], _ENS[1] = tag, tuple(seeds)


def arm_model(arm):
    """'v0hist' (frozen v0-hist), 'v0past' (frozen v0-hist-past), 'v1' (5-seed ensemble)."""
    W = _worker_state()
    if arm not in W:
        if arm == "v1":
            W[arm] = load_ensemble(_ENS[0], _ENS[1])
        else:
            import v0_decode
            W[arm] = v0_decode.load_model({"v0hist": "hist_high_medium", "v0past": "past_high_medium"}[arm])[0]
    return W[arm]


def resolve_exclude(eh, route, spec):
    """Exclusion spec -> set of train route IDs.
      ()                      nothing beyond the route itself (full history, leave-one-out)
      ("window", D)           train routes within +-D days of the route's date
      ("past",)               train routes dated on/after the route's date (v0-hist-past's rule)
      ("past+window", D)      both
      ("ids", (id, ...))      an explicit set (e.g. the volume-matched random control)
      ("past+ids", (...))     past-only plus an explicit set"""
    if not spec:
        return set()
    kind, date = spec[0], str(route["date"])
    if kind == "window":
        return eh.window(date, spec[1])
    if kind == "past":
        return eh.on_or_after(date)
    if kind == "past+window":
        return eh.on_or_after(date) | eh.window(date, spec[1])
    if kind == "ids":
        return set(spec[1])
    if kind == "past+ids":
        return eh.on_or_after(date) | set(spec[1])
    raise ValueError(spec)


def decode_job(job):
    """job = (route_id, arm, exclusion spec (see resolve_exclude), lam, beam). Returns (key, stop order, zone names, objective info)."""
    from v0_decode import stops_from_zone_order
    route_id, arm, exclude, lam, beam = job[:5]
    W = _worker_state()
    route = load_npz(route_id)
    P = prepare_route(route)
    W["eh"].attach(P, str(route["station"]), route_id, exclude=resolve_exclude(W["eh"], route, exclude),
                   levels=(arm == "v1"))
    model = arm_model(arm)
    if arm == "v1":
        zones = beam_decode(model, P, beam, lam, features_v1)
    else:                                                  # v0 arms: v0's own decoder, unchanged
        import v0_decode
        zones = v0_decode.beam_decode(model, P, beam, lam)
    names = [P["names"][z] for z in zones]
    return job, stops_from_zone_order(route, names), names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seeds", default="0,1,2,3,4")
    t.add_argument("--tag", default="main")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--batch-routes", type=int, default=16)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
