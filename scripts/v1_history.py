#!/usr/bin/env python3
"""Station history for v1 (and for re-scoring v0-hist with thinned history). TRAIN ROUTES ONLY.

Agreed plan: docs/plan.md, item 3.

What this adds to v0_history.py (which is NOT modified; v0-hist stays byte-for-byte reproducible):

1. Exclusion by route set. attach() takes `exclude`: a set of train route IDs whose driving must
   be treated as unknown. Counts are the full train counts (v0's sorted-code arrays, every day)
   minus the excluded routes' own events. Uses:
     - leave-one-out: exclude = {the route itself}           (== v0 mode "all")
     - past-only:     exclude = routes dated on/after the date (== v0 mode "past")
     - date window:   exclude = routes within ±D days of the target date (thinned history,
                      history dropout in training, the adoption test, the rehearsal)
     - volume control: exclude = a random set of routes (rehearsal control)
   The route itself is always excluded if it is a train route. `self_test()` checks that the
   first two reproduce v0_history.History.attach exactly.

2. Coarser zone-ID levels. Zone 'A-2.2A' -> micro 'A-2.2' (level 1) -> macro 'A-2' (level 2),
   parsed like a one-off analysis script. Each train route's zone order is
   mapped to a level and de-duplicated (first visit of each group), and the same
   transition / precedence counts are kept per level, same-station and pooled. IDs that do not
   match the pattern stay as they are at every level.

Run `.venv/bin/python -I scripts/v1_history.py --self-test N` to check equivalence on N train
routes (light: 1 thread).
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v0_history import TABLE, History, SMOOTH_K, step_hist_features  # noqa: E402
from v0_model import RUN_DIR  # noqa: E402

LEVELS = (1, 2)
LEVEL_RE = re.compile(r"^([A-Z]+-\d+)\.(\d+)([A-Z]+)$")
LEVEL_FEATURE_NAMES = [
    "same_group", "logp_next_group_same_station", "logp_next_group_pooled",
    "log_count_group_pair_same_station", "log_count_group_pair_pooled",
    "share_group_before_same_station", "share_group_before_pooled",
]
N_LEVEL_FEATURES = len(LEVEL_FEATURE_NAMES) * len(LEVELS)        # 14


def level_name(z, level):
    """'A-2.2A' -> level 0 'A-2.2A', level 1 'A-2.2', level 2 'A-2'."""
    if level == 0:
        return z
    m = LEVEL_RE.match(z)
    if not m:
        return z
    return f"{m.group(1)}.{m.group(2)}" if level == 1 else m.group(1)


def level_order(zones, level):
    """Zone order mapped to a level, keeping only the first visit of each group."""
    seen, out = set(), []
    for z in zones:
        g = level_name(z, level)
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def level_table(level):
    """Write (once) the train history table mapped to a zone-ID level; returns its path.
    Derived only from runs/v0/history_train.json (train split)."""
    if level == 0:
        return TABLE
    path = RUN_DIR.parent / "v1" / f"history_train_L{level}.json"
    if not path.exists():
        rows = json.load(open(TABLE))["routes"]
        out = [{**r, "zones": level_order(r["zones"], level)} for r in rows]
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"source": f"train split only, zone level {level}", "routes": out}, open(path, "w"))
    return path


class ExclHistory:
    """History at levels 0, 1, 2 with arbitrary route exclusion (see module docstring)."""

    def __init__(self, levels=(0,) + LEVELS):
        rows = json.load(open(TABLE))["routes"]
        self.station = {r["route_id"]: r["station"] for r in rows}
        self.date = {r["route_id"]: dt.date.fromisoformat(r["date"]) for r in rows}
        self.ids = sorted(self.station)
        self.h, self.zones, self.zset = {}, {}, {}
        for L in levels:
            self.h[L] = History(level_table(L))
            self.zones[L] = {r["route_id"]: level_order(r["zones"], L) for r in rows}
            self.zset[L] = {k: set(v) for k, v in self.zones[L].items()}

    # ---- which routes to exclude --------------------------------------------------------
    def window(self, date, days):
        """Train routes dated within ±days of `date` (str or date). days=0 -> same day only."""
        d = dt.date.fromisoformat(str(date)) if not isinstance(date, dt.date) else date
        return {r for r, x in self.date.items() if abs((x - d).days) <= days}

    def on_or_after(self, date):
        d = dt.date.fromisoformat(str(date))
        return {r for r, x in self.date.items() if x >= d}

    # ---- raw counts with exclusion ------------------------------------------------------
    def raw(self, L, names, station, exclude):
        """Count arrays for the zone (or group) list `names` of one route, every train day,
        minus the excluded routes. Same layout as v0_history.History.attach:
        c_* [Z+1, Z] transitions (row Z = START), b_* [Z, Z] precedence, n_* [Z+1] out-counts."""
        H = self.h[L]
        Z, V = len(names), H.V
        s = H.sid.get(station, -1)
        g = np.array([H.zid.get(z, -1) for z in names], dtype=np.int64)
        frm = np.concatenate([g, [H.START]])
        ok = (frm[:, None] >= 0) & (g[None, :] >= 0)
        A, B = frm[:, None], g[None, :]

        def pairs(table, same):
            if same and s < 0:
                return np.zeros(ok.shape)
            keys = ((s * V + A) * V + B) if same else (A * V + B)
            return H._count(table, np.where(ok, keys, -1).ravel(), H.D).reshape(ok.shape)

        c_same, c_pool = pairs("trans_same", True), pairs("trans_pool", False)
        b_same, b_pool = pairs("before_same", True)[:Z], pairs("before_pool", False)[:Z]
        n_pool = H._count("from_pool", np.where(frm >= 0, frm, -1), H.D)
        n_same = (H._count("from_same", np.where(frm >= 0, s * V + frm, -1), H.D)
                  if s >= 0 else np.zeros(Z + 1))

        local = {z: k for k, z in enumerate(names)}
        lset = set(names)
        for r in exclude:
            if r not in self.station:
                continue                                       # not a train route: nothing to remove
            same = self.station[r] == station and s >= 0
            n_pool[Z] -= 1                                     # every route leaves START once
            if same:
                n_same[Z] -= 1
            if not (self.zset[L][r] & lset):
                continue
            o = self.zones[L][r]
            prev = [None] + o[:-1]
            for a, b in zip(prev, o):
                ai = Z if a is None else local.get(a)
                bi = local.get(b)
                if a is not None and ai is not None:
                    n_pool[ai] -= 1
                    if same:
                        n_same[ai] -= 1
                if ai is not None and bi is not None:
                    c_pool[ai, bi] -= 1
                    if same:
                        c_same[ai, bi] -= 1
            f = [local[z] for z in o if z in local]
            for x in range(len(f)):
                for y in range(x + 1, len(f)):
                    b_pool[f[x], f[y]] -= 1
                    if same:
                        b_same[f[x], f[y]] -= 1
        return c_same, c_pool, b_same, b_pool, n_same, n_pool

    def reference(self, station, zone_set, exclude):
        """As History.reference (same order, same tie-break), skipping excluded routes."""
        best, allowed = None, 0
        for rid, _, zs, order in self.h[0].station_routes.get(station, []):
            if rid in exclude:
                continue
            allowed += 1
            shared = len(zone_set & zs)
            key = (shared, shared / len(zone_set | zs))
            if best is None or key > best[0]:
                best = (key, order)
        if best is None:
            return 0, [], 0
        return best[0][0], best[1], allowed

    # ---- attach to a prepared route ------------------------------------------------------
    def attach(self, P, station, route_id, exclude=(), levels=True):
        """P["hist"] in exactly v0's format (so v0-hist can be scored with thinned history), and,
        if levels, P["hist_levels"] for v1's coarser features. The route itself is always
        excluded when it is a train route."""
        exclude = set(exclude)
        if route_id in self.station:
            exclude.add(route_id)
        names, Z = P["names"], P["Z"]
        c_same, c_pool, b_same, b_pool, n_same, n_pool = self.raw(0, names, station, exclude)
        shared, order, allowed = self.reference(station, set(names), exclude)
        ref_pos = np.full(Z, -1.0)
        ref_len, ref_share = max(len(order), 1), 0.0
        if shared > 0:
            local = {z: k for k, z in enumerate(names)}
            for p, z in enumerate(order):
                if z in local:
                    ref_pos[local[z]] = p
            ref_share = shared / Z
        P["hist"] = {
            "c_same": c_same, "c_pool": c_pool, "n_same": n_same, "n_pool": n_pool,
            "share_same": (b_same + 0.5) / (b_same + b_same.T + 1.0),
            "share_pool": (b_pool + 0.5) / (b_pool + b_pool.T + 1.0),
            "co_same": b_same + b_same.T, "co_pool": b_pool + b_pool.T,
            "ref_pos": ref_pos, "ref_len": ref_len, "ref_share": ref_share,
            "ref_exists": float(ref_share > 0), "mode": "excl", "leave_one_out": route_id in exclude,
            "same_station_routes_used": allowed, "excluded_routes": len(exclude),
        }
        if levels:
            P["hist_levels"] = {}
            for L in LEVELS:
                groups = sorted({level_name(z, L) for z in names})
                gidx = np.array([groups.index(level_name(z, L)) for z in names], dtype=np.int64)
                c_s, c_p, b_s, b_p, _, _ = self.raw(L, groups, station, exclude)
                P["hist_levels"][L] = {"gidx": gidx, "G": len(groups), "c_same": c_s, "c_pool": c_p,
                                       "share_same": (b_s + 0.5) / (b_s + b_s.T + 1.0),
                                       "share_pool": (b_p + 0.5) / (b_p + b_p.T + 1.0)}
        return P


def step_level_features(HL, cur, visited):
    """Coarser-level history features for every zone at one moment: [Z, 7] for one level.
    The driver is in zone `cur` (group A); for candidate zone b in group B:
      same_group = 1 if B == A (then the other six are neutral: 0, 0, 0, 0, 0.5, 0.5);
      otherwise the smoothed log-probability that group B comes next after A (normalised over
      the distinct candidate groups other than A), the raw counts, and the share of past routes
      that did A before B. At the station, A = START (no 'same group', shares neutral)."""
    g, G = HL["gidx"], HL["G"]
    Z = len(visited)
    cand = ~visited
    A = g[cur] if cur >= 0 else G
    B = g
    other = (B != A) if cur >= 0 else np.ones(Z, dtype=bool)
    cand_groups = np.unique(B[cand & other])
    f = np.zeros((Z, 7))
    f[:, 0] = ~other
    for col, key in ((1, "c_same"), (2, "c_pool")):
        c = HL[key][A]
        if len(cand_groups):
            denom = (c[cand_groups] + SMOOTH_K).sum()
            f[:, col] = np.where(other, np.log((c[B] + SMOOTH_K) / denom), 0.0)
    f[:, 3] = np.where(other, np.log1p(HL["c_same"][A][B]), 0.0)
    f[:, 4] = np.where(other, np.log1p(HL["c_pool"][A][B]), 0.0)
    if cur >= 0:
        f[:, 5] = np.where(other, HL["share_same"][A][B], 0.5)
        f[:, 6] = np.where(other, HL["share_pool"][A][B], 0.5)
    else:
        f[:, 5:7] = 0.5
    return f


def v1_hist_features(P, cur, visited):
    """v0's 18 history features + 7 per coarser level (14) = 32 columns."""
    parts = [step_hist_features(P["hist"], cur, visited)]
    for L in LEVELS:
        parts.append(step_level_features(P["hist_levels"][L], cur, visited))
    return np.concatenate(parts, axis=1).astype(np.float32)


def self_test(n):
    """Check ExclHistory reproduces v0's History.attach for leave-one-out and past-only."""
    from v0_model import load_npz, prepare_route
    import random
    eh = ExclHistory(levels=(0,))
    rng = random.Random(1)
    worst = 0.0
    for rid in rng.sample(eh.ids, n):
        route = load_npz(rid)
        st, date = str(route["station"]), str(route["date"])
        for mode, excl in (("all", set()), ("past", eh.on_or_after(date))):
            P1 = eh.h[0].attach(prepare_route(route), st, rid, mode=mode, date=date)["hist"]
            P2 = eh.attach(prepare_route(route), st, rid, exclude=excl, levels=False)["hist"]
            for k in ("c_same", "c_pool", "n_same", "n_pool", "share_same", "share_pool",
                      "co_same", "co_pool", "ref_pos"):
                worst = max(worst, float(np.abs(np.asarray(P1[k]) - np.asarray(P2[k])).max()))
            for k in ("ref_len", "ref_share", "ref_exists"):
                worst = max(worst, abs(float(P1[k]) - float(P2[k])))
    print(f"self-test on {n} train routes x 2 modes: max abs difference {worst}")
    return worst == 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", type=int, default=0)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test(a.self_test) else 1)
