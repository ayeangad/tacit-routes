#!/usr/bin/env python3
"""v0 tiny model: learn which zone a delivery driver visits next.

Run from tacit-routes/ (the -I flag keeps Python from importing stray files from the data folders):
  .venv/bin/python -I scripts/v0_model.py --quality High           --tag high
  .venv/bin/python -I scripts/v0_model.py --quality High,Medium    --tag high_medium
  add --history all  (tag hist_*) for v0-hist, or --history past (tag past_*) for v0-hist-past;
  both need the history table first: .venv/bin/python -I scripts/v0_history.py

The big picture
---------------
A route has about 150 stops grouped into about 20 zones (Amazon's planning areas, IDs like
"A-2.3B"). Drivers almost never come back to a zone they have left, so the route is mostly
"pick a zone order, then do each zone in one go". This file learns the zone order.

At each step the driver is standing in some zone (at the start: the station) and has already
done some zones. Every zone not yet done is a *candidate*. We describe each candidate with
33 numbers (its *features*: how far, how long to drive there, how many stops, how its ID
relates to the current zone's ID, ...). The model turns each candidate's features into one
number, a *score*. Softmax turns the scores into probabilities that add up to 1. Training
nudges the weights so the zone the driver really went to next gets a higher probability.

The model ("pointer" scorer, 3 parts):
  1. encoder:   each candidate's 33 features -> 64 numbers (a small two-layer network)
  2. attention: every candidate looks at every other candidate, so it can tell
                "I am the nearest one left" or "my neighbours are still open"
  3. head:      64 numbers -> 1 score per candidate
It is called a pointer network because its output "points" at one of the inputs
(one candidate zone) rather than choosing from a fixed list of classes.

Data rules: trains on the train split only. 10% of train routes are held out for early
stopping (we stop when the held-out loss stops improving). val and pilot are never touched here.
"""
import argparse
import csv
import json
import math
import random
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))       # -I leaves the script folder off the path; v0_history lives there
ROOT = HERE.parent
ROUTES_DIR = ROOT / "data" / "processed" / "routes"
SPLITS = ROOT / "data" / "processed" / "splits.json"
RUN_DIR = ROOT / "runs" / "v0"
SEED = 20261111           # same seed as the splits, for one fewer number to remember
HELDOUT_FRACTION = 0.10   # share of train routes kept aside for early stopping and tuning

# Zone IDs look like "A-2.3B": letter A, number 2, sub-number 3, sub-letter B.
ZONE_RE = re.compile(r"^([A-Z])-(\d+)\.(\d+)([A-Z])$")

FEATURE_NAMES = [
    # --- about the candidate zone itself (same at every step) ---
    "cand_dx_km_from_station", "cand_dy_km_from_station", "cand_km_from_station",
    "cand_log_stops", "cand_log_packages", "cand_share_stops_with_time_window",
    "cand_earliest_window_end_h", "cand_service_time_h",
    # --- where the candidate is relative to the current zone ---
    "dx_km_from_current", "dy_km_from_current", "km_from_current",
    # --- travel time (the solver's view of the world) ---
    "minutes_from_current", "minutes_to_station", "time_vs_nearest_candidate",
    "time_rank_among_candidates", "is_nearest_candidate",
    # --- how the zone ID relates to the current zone's ID ---
    "current_is_station", "same_letter", "same_number", "abs_number_diff",
    "same_subnumber", "subnumber_diff", "same_subletter", "subletter_diff", "id_neighbour",
    # --- does this move continue the direction of the last move? ---
    "continues_subnumber_direction", "continues_subletter_direction",
    # --- progress through the bigger area (letter + number, e.g. "A-2") ---
    "share_of_its_area_done", "current_area_unfinished", "same_area_as_current",
    # --- where we are in the route ---
    "step_fraction", "share_of_zones_left", "log_zone_count",
]
N_FEATURES = len(FEATURE_NAMES)


# ----------------------------------------------------------------------------------------------
# Part 1: turn one route (stops) into zones and zone features
# ----------------------------------------------------------------------------------------------

def load_npz(route_id):
    """One route as saved by build_routes.py. Index 0 is always the station."""
    with np.load(ROUTES_DIR / f"{route_id}.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def parse_zone(zone_id):
    """'A-2.3B' -> (ord('A'), 2, 3, ord('B'), True). Unusual IDs -> zeros and False."""
    m = ZONE_RE.match(zone_id)
    if not m:
        return 0, 0, 0, 0, False
    return ord(m.group(1)), int(m.group(2)), int(m.group(3)), ord(m.group(4)), True


def zone_membership(route):
    """Which zone each stop belongs to. Returns (sorted zone names, zone index per stop).

    The station gets -1. Stops with no zone ID (about 2% of stops) are attached to the zone
    of the stop nearest to them by travel time (there and back)."""
    n = len(route["stop_ids"])
    tt = route["travel_time"].astype(np.float64)
    raw_zone = [str(z) for z in route["zone_ids"]]
    names = sorted({z for z in raw_zone[1:] if z})
    zindex = {z: k for k, z in enumerate(names)}
    zone_of = np.full(n, -1, dtype=np.int64)            # -1 = station
    for i in range(1, n):
        if raw_zone[i]:
            zone_of[i] = zindex[raw_zone[i]]
    has_zone = np.where(zone_of >= 0)[0]
    for i in range(1, n):
        if zone_of[i] < 0:                               # no zone: borrow the nearest stop's zone
            both_ways = tt[i, has_zone] + tt[has_zone, i]
            zone_of[i] = zone_of[has_zone[np.argmin(both_ways)]]
    return names, zone_of


def prepare_route(route):
    """Everything about a route that does not change while we drive it.

    Returns a dict with the zones, which stop belongs to which zone, zone-to-zone travel
    times and the per-zone features."""
    tt = route["travel_time"].astype(np.float64)        # tt[i, j] = seconds from stop i to j
    names, zone_of = zone_membership(route)
    Z = len(names)
    members = [np.where(zone_of == k)[0] for k in range(Z)]

    # --- positions: zone centres in km, relative to the station -------------------------------
    lat, lng = route["lat"], route["lng"]
    km_per_deg_lat = 111.2
    km_per_deg_lng = 111.2 * math.cos(math.radians(float(lat[0])))
    x = (lng - lng[0]) * km_per_deg_lng
    y = (lat - lat[0]) * km_per_deg_lat
    centre = np.array([[x[m].mean(), y[m].mean()] for m in members])   # [Z, 2]

    # --- zone-to-zone travel time: average over every pair of stops ---------------------------
    # Row k = from zone k; the extra last row (index Z) = from the station.
    ztt = np.zeros((Z + 1, Z))
    for a in range(Z):
        for b in range(Z):
            ztt[a, b] = tt[np.ix_(members[a], members[b])].mean()
    for b in range(Z):
        ztt[Z, b] = tt[0, members[b]].mean()
    to_station = np.array([tt[members[k], 0].mean() for k in range(Z)])

    # --- per-zone facts -----------------------------------------------------------------------
    npk = route["n_packages"].astype(np.float64)
    tw_end = route["tw_end"].astype(np.float64)
    svc = route["service_time"].astype(np.float64)
    static = np.zeros((Z, 8))
    for k, m in enumerate(members):
        has_tw = np.isfinite(tw_end[m])
        static[k] = [
            centre[k, 0], centre[k, 1], np.hypot(*centre[k]),
            math.log1p(len(m)), math.log1p(npk[m].sum()), has_tw.mean(),
            (tw_end[m][has_tw].min() / 3600.0) if has_tw.any() else 0.0,
            svc[m].sum() / 3600.0,
        ]

    parts = np.array([parse_zone(z) for z in names], dtype=np.int64)    # [Z, 5]
    area_key = [(p[0], p[1]) if p[4] else ("?", k) for k, p in enumerate(parts)]
    area_names = sorted(set(area_key), key=str)
    area = np.array([area_names.index(a) for a in area_key])

    # Typical zone-to-zone travel time on this route; used to put travel time on the same
    # scale as log-probabilities in the combined decoder.
    off = ztt[:Z, :Z][~np.eye(Z, dtype=bool)]
    scale = float(np.median(off)) if off.size else 1.0

    return {"route_id": str(route["route_id"]), "Z": Z, "names": names, "zone_of": zone_of,
            "members": members, "centre": centre, "ztt": ztt, "to_station": to_station,
            "static": static, "parts": parts, "area": area, "tt_scale": scale}


def driver_zone_order(P, route):
    """The zones in the order the driver first entered them (station excluded).

    Drivers revisit a zone about once per route; we keep only the first visit, so the
    target is always a zone not yet done."""
    seen, order = set(), []
    for i in route["actual_order"][1:]:
        k = int(P["zone_of"][i])
        if k not in seen:
            seen.add(k)
            order.append(k)
    return order


def step_features(P, cur, prev, visited, t):
    """Features for every zone at one moment of the drive.

    cur:     zone the driver is in now (-1 = still at the station)
    prev:    zone before that (-1 if none)
    visited: bool array [Z], True for zones already done (cur included)
    t:       how many zones are done
    Returns a float32 array [Z, N_FEATURES]. Rows of visited zones are filled too, but the
    model masks them out, so they never matter."""
    Z = P["Z"]
    cand = ~visited
    K = int(cand.sum())                                   # number of candidates left
    f = np.zeros((Z, N_FEATURES), dtype=np.float64)
    f[:, 0:8] = P["static"]

    here = P["centre"][cur] if cur >= 0 else np.zeros(2)
    d = P["centre"] - here
    f[:, 8:10] = d
    f[:, 10] = np.hypot(d[:, 0], d[:, 1])

    secs = P["ztt"][cur if cur >= 0 else Z]               # seconds from here to each zone
    f[:, 11] = secs / 60.0
    f[:, 12] = P["to_station"] / 60.0
    nearest = secs[cand].min() if K else 1.0
    f[:, 13] = np.minimum(secs / max(nearest, 1.0), 10.0)
    rank = np.zeros(Z)
    order = np.where(cand)[0][np.argsort(secs[cand], kind="stable")]
    rank[order] = np.arange(K) / max(K - 1, 1)            # 0 = nearest, 1 = farthest candidate
    f[:, 14] = rank
    f[:, 15] = 0.0
    if K:
        f[order[0], 15] = 1.0

    pa = P["parts"]                                       # columns: letter, num, sub, subletter, ok
    if cur < 0:
        f[:, 16] = 1.0                                    # at the station: no ID relations
    else:
        c = pa[cur]
        ok = (pa[:, 4] == 1) & bool(c[4])
        same_letter = ok & (pa[:, 0] == c[0])
        f[:, 17] = same_letter
        f[:, 18] = same_letter & (pa[:, 1] == c[1])
        f[:, 19] = np.where(same_letter, np.minimum(np.abs(pa[:, 1] - c[1]), 5) / 5.0, 1.0)
        dsub = np.clip(pa[:, 2] - c[2], -3, 3)
        dsl = np.clip(pa[:, 3] - c[3], -3, 3)
        f[:, 20] = ok & (dsub == 0)
        f[:, 21] = np.where(ok, dsub / 3.0, 0.0)
        f[:, 22] = ok & (dsl == 0)
        f[:, 23] = np.where(ok, dsl / 3.0, 0.0)
        f[:, 24] = f[:, 18] * ((np.abs(dsub) + np.abs(dsl)) == 1)
        if prev >= 0 and pa[prev, 4] and c[4]:
            last_sub = np.sign(c[2] - pa[prev, 2])
            last_sl = np.sign(c[3] - pa[prev, 3])
            f[:, 25] = ok & (last_sub != 0) & (np.sign(dsub) == last_sub)
            f[:, 26] = ok & (last_sl != 0) & (np.sign(dsl) == last_sl)

    area = P["area"]
    for a in np.unique(area):
        in_a = area == a
        f[in_a, 27] = visited[in_a].mean()
    if cur >= 0:
        unfinished = bool((cand & (area == area[cur])).any())
        f[:, 28] = unfinished
        f[:, 29] = area == area[cur]
    f[:, 30] = t / Z
    f[:, 31] = K / Z
    f[:, 32] = math.log(Z) / 4.0
    return f.astype(np.float32)


def features(P, cur, prev, visited, t):
    """All features the model sees: the 33 base features, plus the 18 station-history
    features (v0-hist only) when prepare_route's dict carries history ("hist")."""
    base = step_features(P, cur, prev, visited, t)
    if "hist" not in P:
        return base
    from v0_history import step_hist_features
    return np.concatenate([base, step_hist_features(P["hist"], cur, visited)], axis=1)


def teacher_forced_steps(P, order):
    """All training examples from one route: walk the driver's real zone order and, at
    every step with at least 2 candidates, record (features, which zones are candidates,
    which one the driver picked). Steps with one candidate left teach nothing."""
    Z = P["Z"]
    X, M, Y = [], [], []
    visited = np.zeros(Z, dtype=bool)
    cur, prev = -1, -1
    for t, target in enumerate(order):
        if Z - t >= 2:
            X.append(features(P, cur, prev, visited, t))
            M.append(~visited)
            Y.append(target)
        visited[target] = True
        prev, cur = cur, target
    return np.stack(X).astype(np.float16), np.stack(M), np.array(Y, dtype=np.int64)


def build_examples(route_ids, log_every=500, history=None, mode="all"):
    """Teacher-forced examples for many routes; each route is kept as its own block.
    With a History object, history features are added: mode "all" = every train route except
    this one (leave-one-out); mode "past" = only train routes from earlier days."""
    data, t0 = [], time.time()
    for k, r in enumerate(route_ids):
        route = load_npz(r)
        P = prepare_route(route)
        if P["Z"] < 2:
            continue
        if history is not None:
            history.attach(P, str(route["station"]), r, mode=mode, date=str(route["date"]))
        X, M, Y = teacher_forced_steps(P, driver_zone_order(P, route))
        data.append((r, X, M, Y))
        if log_every and (k + 1) % log_every == 0:
            print(f"  features for {k + 1}/{len(route_ids)} routes, {time.time() - t0:.0f}s", flush=True)
    return data


# ----------------------------------------------------------------------------------------------
# Part 2: the model
# ----------------------------------------------------------------------------------------------

class ZonePointer(nn.Module):
    """Scores every candidate zone; softmax over the scores = probability of going there next."""

    def __init__(self, n_features=N_FEATURES, d=64, heads=4):
        super().__init__()
        # 1. encoder: a little two-layer network applied to each candidate separately.
        self.encoder = nn.Sequential(nn.Linear(n_features, d), nn.ReLU(), nn.Linear(d, d))
        # 2. one attention layer: each candidate mixes in information from the others.
        self.attention = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        # 3. head: one score per candidate.
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        # Standardisation constants (mean and spread of each feature over the training data),
        # stored inside the model so decoding uses exactly the same scaling.
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

    def forward(self, x, mask):
        """x: [batch, zones, features]; mask: [batch, zones], True = candidate.
        Returns log-probabilities [batch, zones]; non-candidates get -inf (probability 0)."""
        h = self.encoder((x - self.feat_mean) / self.feat_std)
        # key_padding_mask=True means "ignore": candidates only look at other candidates.
        mixed, _ = self.attention(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = self.norm(h + mixed)                          # residual: keep the original too
        scores = self.head(h).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(scores, dim=-1)


def count_parameters(model):
    """Number of trainable numbers in the model (buffers such as feat_mean are not counted)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_batch(blocks):
    """Stack the steps of several routes into one padded batch.
    Routes have different zone counts, so shorter ones are padded with non-candidates."""
    Zmax = max(X.shape[1] for _, X, _, _ in blocks)
    T = sum(X.shape[0] for _, X, _, _ in blocks)
    x = np.zeros((T, Zmax, blocks[0][1].shape[2]), dtype=np.float32)
    m = np.zeros((T, Zmax), dtype=bool)
    y = np.zeros(T, dtype=np.int64)
    row = 0
    for _, X, M, Y in blocks:
        t, z = X.shape[0], X.shape[1]
        x[row:row + t, :z] = X
        m[row:row + t, :z] = M
        y[row:row + t] = Y
        row += t
    return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y)


def evaluate(model, data, batch_routes=64):
    """Mean cross-entropy and top-1 accuracy over all steps (teacher-forced)."""
    model.eval()
    total_loss, correct, steps = 0.0, 0, 0
    with torch.no_grad():
        for k in range(0, len(data), batch_routes):
            x, m, y = make_batch(data[k:k + batch_routes])
            logp = model(x, m)
            total_loss += -logp[torch.arange(len(y)), y].sum().item()
            correct += (logp.argmax(-1) == y).sum().item()
            steps += len(y)
    return total_loss / steps, correct / steps, steps


# ----------------------------------------------------------------------------------------------
# Part 3: training
# ----------------------------------------------------------------------------------------------

def split_train(splits):
    """Fixed 90/10 split of the train routes (all qualities), so every model trained here
    uses the same held-out routes. Early stopping and tuning use the High routes in the 10%,
    because val and the final test contain only High routes."""
    ids = sorted(r["route_id"] for r in splits["train"])
    random.Random(f"{SEED}:heldout").shuffle(ids)
    n_held = int(round(HELDOUT_FRACTION * len(ids)))
    quality = {r["route_id"]: r["quality"] for r in splits["train"]}
    return set(ids[n_held:]), set(ids[:n_held]), quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="High", help="comma list of train qualities, e.g. High,Medium")
    ap.add_argument("--tag", required=True, help="output folder name under runs/v0/")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5, help="stop after this many epochs without improvement")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-routes", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="use only N train routes (testing)")
    ap.add_argument("--history", choices=["none", "all", "past"], default="none",
                    help="none = v0; all = v0-hist (leave-one-out); past = v0-hist-past (earlier days only)")
    args = ap.parse_args()

    torch.set_num_threads(3)            # the machine is shared; the brief allows 3 threads
    torch.manual_seed(SEED)
    random.seed(SEED)
    out = RUN_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)

    splits = json.load(open(SPLITS))
    fit_ids, held_ids, quality = split_train(splits)
    wanted = set(args.quality.split(","))
    train_ids = sorted(r for r in fit_ids if quality[r] in wanted)
    held_high = sorted(r for r in held_ids if quality[r] == "High")
    if args.limit:
        train_ids, held_high = train_ids[:args.limit], held_high[:max(20, args.limit // 10)]
    # Guard: nothing from val or pilot may appear here.
    forbidden = {r["route_id"] for r in splits["val"]} | {r["route_id"] for r in splits["pilot"]}
    assert not (set(train_ids) | set(held_high)) & forbidden

    t0, cpu0 = time.time(), time.process_time()
    print(f"building features: {len(train_ids)} training routes, {len(held_high)} held-out High routes")
    history = None
    if args.history != "none":
        from v0_history import History
        history = History()                               # counts from the train split only
    train = build_examples(train_ids, history=history, mode=args.history)
    held = build_examples(held_high, log_every=0, history=history, mode=args.history)
    feat_seconds = time.time() - t0
    del history

    from v0_history import N_HIST_FEATURES
    n_features = N_FEATURES + (N_HIST_FEATURES if args.history != "none" else 0)
    model = ZonePointer(n_features)
    # Standardise features with the training data's mean and spread (candidate rows only).
    rows = np.concatenate([X[M] for _, X, M, _ in train]).astype(np.float64)
    model.feat_mean.copy_(torch.tensor(rows.mean(0), dtype=torch.float32))
    model.feat_std.copy_(torch.tensor(np.maximum(rows.std(0), 1e-3), dtype=torch.float32))
    del rows
    n_params = count_parameters(model)
    print(f"model parameters: {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    curve, best, best_epoch, waited = [], float("inf"), 0, 0
    rng = random.Random(SEED)
    t_train, cpu_train = time.time(), time.process_time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)                                # new route order every epoch
        loss_sum, step_sum = 0.0, 0
        for k in range(0, len(order), args.batch_routes):
            x, m, y = make_batch([train[i] for i in order[k:k + args.batch_routes]])
            logp = model(x, m)
            # Cross-entropy: -log(probability given to the zone the driver really chose).
            loss = -logp[torch.arange(len(y)), y].mean()
            opt.zero_grad()
            loss.backward()                               # backprop: how each weight moves the loss
            opt.step()                                    # nudge every weight a little downhill
            loss_sum += loss.item() * len(y)
            step_sum += len(y)
        train_loss = loss_sum / step_sum
        held_loss, held_acc, _ = evaluate(model, held)
        curve.append({"epoch": epoch, "train_loss": train_loss, "heldout_loss": held_loss,
                      "heldout_top1": held_acc, "seconds": time.time() - t_train})
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  held-out loss {held_loss:.4f}  "
              f"held-out top-1 {held_acc:.3f}  {time.time() - t_train:.0f}s", flush=True)
        if held_loss < best - 1e-4:                       # better: save this version
            best, best_epoch, waited = held_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "n_features": n_features, "history": args.history, "n_params": n_params,
                        "quality": sorted(wanted)}, out / "model.pt")
        else:
            waited += 1
            if waited >= args.patience:                   # early stopping
                break

    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    summary = {
        "tag": args.tag, "train_qualities": sorted(wanted), "n_params": n_params,
        "history_features": args.history, "n_features": n_features,
        "train_routes": len(train), "train_steps": int(sum(len(Y) for *_, Y in train)),
        "heldout_high_routes": len(held), "heldout_steps": int(sum(len(Y) for *_, Y in held)),
        "best_epoch": best_epoch, "best_heldout_loss": best,
        "best_heldout_top1": next(c["heldout_top1"] for c in curve if c["epoch"] == best_epoch),
        "epochs_run": len(curve), "feature_seconds": feat_seconds,
        "train_seconds": time.time() - t_train,
        "train_cpu_seconds": time.process_time() - cpu_train,
        "total_cpu_seconds_features_plus_training": time.process_time() - cpu0,
        "total_wall_seconds_features_plus_training": time.time() - t0,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "settings": {"lr": args.lr, "batch_routes": args.batch_routes, "patience": args.patience,
                     "max_epochs": args.epochs, "optimizer": "Adam", "d_model": 64, "heads": 4,
                     "heldout": "10% of all train routes (seeded), High ones only used"},
        "curve": curve,
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
