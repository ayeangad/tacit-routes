#!/usr/bin/env python3
"""Station history for v0-hist: what past drivers did, counted from TRAIN routes only.

Why: zone names repeat from day to day. If drivers out of station DLA9 nearly always go from
zone A-2.2A to A-2.1A, a new DLA9 route that contains both zones will probably do the same.
The 2021 challenge winners got their biggest gain from exactly this kind of history.

What we count, once, over every train route's zone order (first visit to each zone):
  - transitions:  how often a driver went straight from zone a to zone b
                  (the station counts as a zone called START, so we also learn first zones)
  - precedence:   in how many routes zone a was visited before zone b (anywhere later)
Each is kept twice: for the same station only, and pooled over all stations.
  - reference route: the past train route at the same station that shares the most zones
                  with this route. Its zone order is a ready-made guess.

Two leakage rules, one per model family:
  v0-hist      (mode "all"):  leave-one-out. When a train route's own features are built, its
               own transitions and precedences are subtracted and it can never be its own
               reference. Val routes are not in the table, so nothing needs subtracting.
  v0-hist-past (mode "past"): time-respecting. A route (train or val) only sees train routes
               dated strictly before its own date. Dates are whole days, so same-day routes are
               excluded as well; the route itself is therefore excluded automatically.
Pilot and val sequences are never read here.

The 18 history features for a candidate zone b, when the driver is in zone a (see
step_hist_features): smoothed probability that b comes next after a (same station / pooled),
the raw counts behind it, the share of past routes with a before b, how often b came before
the other zones still left, and where b sits in the reference route relative to a.

Build the table once (about a minute):
  .venv/bin/python -I scripts/v0_history.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v0_model import ROUTES_DIR, RUN_DIR, SPLITS, zone_membership  # noqa: E402

TABLE = RUN_DIR / "history_train.json"
SMOOTH_K = 1.0          # add-k smoothing for transition probabilities

HIST_FEATURE_NAMES = [
    "hist_logp_next_same_station", "hist_logp_next_pooled",
    "hist_log_count_pair_same_station", "hist_log_count_from_current_same_station",
    "hist_log_count_pair_pooled", "hist_log_count_from_current_pooled",
    "hist_share_current_before_cand_same_station", "hist_share_current_before_cand_pooled",
    "hist_share_cand_before_other_cands_same_station", "hist_share_cand_before_other_cands_pooled",
    "hist_log_routes_with_cand_and_others_same_station", "hist_log_routes_with_cand_and_others_pooled",
    "ref_has_cand", "ref_has_current", "ref_position_after_current", "ref_is_next",
    "ref_share_of_zones_shared", "ref_exists",
]
N_HIST_FEATURES = len(HIST_FEATURE_NAMES)


def driver_zone_order_from_npz(route):
    """Zone names in first-visit order, using the same zone definition as the model."""
    names, zone_of = zone_membership(route)
    seen, order = set(), []
    for i in route["actual_order"][1:]:
        z = names[int(zone_of[i])]
        if z not in seen:
            seen.add(z)
            order.append(z)
    return order


def build_table():
    """One pass over the TRAIN split: each route's station, date and zone order. Saved as JSON."""
    splits = json.load(open(SPLITS))
    rows, t0 = [], time.time()
    for k, r in enumerate(sorted(x["route_id"] for x in splits["train"])):
        with np.load(ROUTES_DIR / f"{r}.npz", allow_pickle=False) as d:
            route = {key: d[key] for key in ("stop_ids", "travel_time", "zone_ids", "actual_order",
                                             "station", "date")}
        rows.append({"route_id": r, "station": str(route["station"]), "date": str(route["date"]),
                     "zones": driver_zone_order_from_npz(route)})
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1} train routes, {time.time() - t0:.0f}s", flush=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"source": "train split only", "routes": rows}, open(TABLE, "w"))
    print(f"history table: {len(rows)} train routes in {time.time() - t0:.0f}s -> {TABLE}")


class History:
    """Counts over the train routes, able to answer two kinds of question:

      mode "all"  (v0-hist):      use every train route except the target route itself
                                  (leave-one-out).
      mode "past" (v0-hist-past): use only train routes dated strictly BEFORE the target
                                  route's date. Dates are whole days, so routes from the same
                                  day are excluded too (we cannot tell which came first).

    How the counting works: every event (a transition a -> b, or "a came before b") becomes
    one integer code = pair_key * D + day_index, where D is the number of distinct days. All
    codes go into one sorted array. "How many times did pair k happen before day c?" is then
    the number of codes in [k*D, k*D + c), which two binary searches (np.searchsorted) find
    instantly. Mode "all" just uses c = D (every day)."""

    def __init__(self, path=TABLE):
        rows = json.load(open(path))["routes"]
        names = sorted({z for r in rows for z in r["zones"]})
        self.zid = {z: i for i, z in enumerate(names)}
        self.START = len(names)
        self.V = V = len(names) + 1
        self.stations = sorted({r["station"] for r in rows})
        self.sid = {s: i for i, s in enumerate(self.stations)}
        self.days = sorted({r["date"] for r in rows})
        self.D = D = len(self.days)
        day = {d: i for i, d in enumerate(self.days)}
        assert (len(self.stations) * V + V) * V * D < 2**62    # codes fit in int64
        self.by_id = {}
        ev = {k: [] for k in ("trans_same", "trans_pool", "before_same", "before_pool",
                              "from_same", "from_pool")}
        for r in rows:
            s, d = self.sid[r["station"]], day[r["date"]]
            o = np.array([self.zid[z] for z in r["zones"]], dtype=np.int64)
            self.by_id[r["route_id"]] = r
            frm = np.concatenate([[self.START], o[:-1]])          # START -> first zone, then a -> b
            i, j = np.triu_indices(len(o), 1)                    # every pair with a before b
            ev["trans_pool"].append((frm * V + o) * D + d)
            ev["trans_same"].append(((s * V + frm) * V + o) * D + d)
            ev["before_pool"].append((o[i] * V + o[j]) * D + d)
            ev["before_same"].append(((s * V + o[i]) * V + o[j]) * D + d)
            ev["from_pool"].append(frm * D + d)                  # transitions out of a (any b)
            ev["from_same"].append((s * V + frm) * D + d)
        self.events = {k: np.sort(np.concatenate(v)) for k, v in ev.items()}
        # Reference-route search: per station, every past route's zone set and day.
        self.station_routes = {}
        for r in rows:
            self.station_routes.setdefault(r["station"], []).append(
                (r["route_id"], day[r["date"]], set(r["zones"]), r["zones"]))

    def cutoff(self, before_date):
        """Day index c: events with day index < c are allowed. None = all days."""
        if before_date is None:
            return self.D
        return int(np.searchsorted(np.array(self.days), before_date, side="left"))

    def _count(self, table, keys, c):
        """How often each pair key happened on allowed days (keys < 0 = unknown -> 0)."""
        arr = self.events[table]
        k = np.asarray(keys, dtype=np.int64)
        lo = np.searchsorted(arr, k * self.D)
        hi = np.searchsorted(arr, k * self.D + c)
        return np.where(k >= 0, hi - lo, 0).astype(np.float64)

    def reference(self, station, zone_set, route_id, c):
        """The allowed same-station train route sharing the most zones (ties: higher Jaccard,
        then smaller route ID). Returns (shared zone count, its zone order, allowed routes)."""
        best, allowed = None, 0
        for rid, d, zs, order in self.station_routes.get(station, []):
            if rid == route_id or d >= c:
                continue                                       # never itself, never the future
            allowed += 1
            shared = len(zone_set & zs)
            key = (shared, shared / len(zone_set | zs))
            if best is None or key > best[0]:
                best = (key, order)
        if best is None:
            return 0, [], 0
        return best[0][0], best[1], allowed

    def attach(self, P, station, route_id, mode="all", date=None):
        """Per-route history arrays, stored in P["hist"]; see step_hist_features for their use.

        mode "all":  all train routes; if route_id is a train route, its own contribution is
                     removed (leave-one-out).
        mode "past": only train routes dated before `date` (the route itself is same-day, so
                     it is excluded automatically)."""
        names, Z, V = P["names"], P["Z"], self.V
        c = self.cutoff(date if mode == "past" else None)
        s = self.sid.get(station, -1)
        g = np.array([self.zid.get(z, -1) for z in names], dtype=np.int64)   # -1 = never seen
        frm = np.concatenate([g, [self.START]])                 # rows: zones, then START (index Z)
        ok = (frm[:, None] >= 0) & (g[None, :] >= 0)
        A, B = frm[:, None], g[None, :]

        def pairs(table, same):
            if same and s < 0:
                return np.zeros(ok.shape)
            keys = ((s * V + A) * V + B) if same else (A * V + B)
            return self._count(table, np.where(ok, keys, -1).ravel(), c).reshape(ok.shape)

        c_same = pairs("trans_same", True)                      # [Z+1, Z] transitions a -> b
        c_pool = pairs("trans_pool", False)
        b_same = pairs("before_same", True)[:Z]                 # [Z, Z] routes with a before b
        b_pool = pairs("before_pool", False)[:Z]
        # Transitions out of a, counted to any zone (also zones not on this route).
        n_pool = self._count("from_pool", np.where(frm >= 0, frm, -1), c)
        n_same = (self._count("from_same", np.where(frm >= 0, s * V + frm, -1), c)
                  if s >= 0 else np.zeros(Z + 1))

        loo = mode == "all" and route_id in self.by_id
        if loo:                                                 # leave-one-out
            local = {z: k for k, z in enumerate(names)}
            o = [local[z] for z in self.by_id[route_id]["zones"]]
            prev = [Z] + o[:-1]
            for a, b in zip(prev, o):
                c_same[a, b] -= 1
                c_pool[a, b] -= 1
                n_same[a] -= 1
                n_pool[a] -= 1
            for x in range(len(o)):
                for y in range(x + 1, len(o)):
                    b_same[o[x], o[y]] -= 1
                    b_pool[o[x], o[y]] -= 1

        shared, order, allowed = self.reference(station, set(names), route_id, c)
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
            "ref_exists": float(ref_share > 0), "mode": mode, "leave_one_out": loo,
            "same_station_routes_used": allowed,
        }
        return P


def step_hist_features(H, cur, visited):
    """History features for every zone at one moment of the drive: array [Z, N_HIST_FEATURES]."""
    Z = len(visited)
    cand = ~visited
    row = cur if cur >= 0 else Z
    f = np.zeros((Z, N_HIST_FEATURES))
    for col, key in ((0, "c_same"), (1, "c_pool")):
        c = H[key][row]
        denom = (c[cand] + SMOOTH_K).sum()
        f[:, col] = np.log((c + SMOOTH_K) / denom)            # add-k smoothed, over candidates
    f[:, 2] = np.log1p(H["c_same"][row])
    f[:, 3] = np.log1p(H["n_same"][row])
    f[:, 4] = np.log1p(H["c_pool"][row])
    f[:, 5] = np.log1p(H["n_pool"][row])
    if cur >= 0:
        f[:, 6] = H["share_same"][cur]
        f[:, 7] = H["share_pool"][cur]
    else:
        f[:, 6:8] = 0.5                                        # at the station: no information
    others = cand.sum() - cand                                 # other candidates per row
    for col, key in ((8, "share_same"), (9, "share_pool")):
        s = (H[key] * cand[None, :]).sum(1) - np.where(cand, np.diag(H[key]), 0)
        f[:, col] = np.where(others > 0, s / np.maximum(others, 1), 0.5)
    for col, key in ((10, "co_same"), (11, "co_pool")):
        s = (H[key] * cand[None, :]).sum(1) - np.where(cand, np.diag(H[key]), 0)
        f[:, col] = np.log1p(np.where(others > 0, s / np.maximum(others, 1), 0))
    pos = H["ref_pos"]
    here = pos[cur] if cur >= 0 else -1.0                      # the station sits before position 0
    has = pos >= 0
    f[:, 12] = has
    f[:, 13] = 1.0 if (cur < 0 or here >= 0) else 0.0
    if f[0, 13]:
        f[:, 14] = np.where(has, (pos - here) / H["ref_len"], 0.0)
        ahead = cand & has & (pos > here)
        pool = ahead if ahead.any() else (cand & has)
        if pool.any():
            nxt = np.where(pool)[0][np.argmin(pos[pool])]
            f[nxt, 15] = 1.0
    f[:, 16] = H["ref_share"]
    f[:, 17] = H["ref_exists"]
    return f.astype(np.float32)


if __name__ == "__main__":
    build_table()
