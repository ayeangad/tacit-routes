# FROZEN: richer-history Opus zone-order prompt (primary comparison, frontier arm)

Frozen 2026-11-11T17:13:51.795461+00:00 after development on 40 validation routes (`runs/zone_rich/dev_val40.txt`, seed 20261112). After this, the prompt, template, constants and builder are not changed. Pilot and eval runs use `scripts/zone_rich.py build --version A` unchanged.

- Version: **A** (evidence only, no guidance paragraph)
- **Effort: `--effort max`**, frozen 2026-11-11T18:02:20.388836+00:00 (review 5, point 6)
- system_prompt sha256: `8130bb2daca43586c656381c89ba1d65e7dbd9fb7ad03e04548228000e8c9c82`
- template sha256: `fa48f8bdc3001f81a0053cc1fd75b258e1f253cbfe84c457e726b80a605d616e`
- scripts/zone_rich.py sha256: `e6d2400ac9eb8a18343701e5382d69b2268885bebbe60e18ddb31c51237cbe37` (after adding `--effort`, `--solver exact` and the strict/repaired views; before that it was `a43794560f4bec36b7b484b7d24e01eee4f442acf5ff7f3536ce9cc84dd4d2ab`; the prompt and template are unchanged)
- Constants: K_REF = 8 references, TOP_NEXT = 3, TOP_FIRST = 5
- History rule: competition (all same-station TRAIN routes, any date), restricted to the route's own zones; coarse zone-ID levels (`A-2.2`, then `A-2`) for zones never seen exactly
- Opus command: the approved `claude -p` flags (scripts/run_pilot.py `opus_command`), fresh empty temp dir per call, at most 3 calls at a time
- Invalid answers: strict (challenge invalid score) and fallback (cheapest insertion of missing zones)

## Version comparison (40 val routes, greedy in-zone solver)

| Version | Mean score | Valid | Mean cost per call | Mean wall time per call |
|---|---|---|---|---|
| **A: evidence only (frozen)** | **0.0319** | 40/40 | $0.041 | 5.2 s |
| B: A + guidance paragraph | 0.0335 | 40/40 | $0.043 | 5.6 s |

B − A: +0.0016 [+0.0002, +0.0033] (paired bootstrap, seed 20261111), with B better on 3 routes and worse on 8. A was chosen by mean.

## System prompt

```
You are a delivery planner. A delivery route's stops are grouped into zones. Your job is to choose the order in which to visit the zones, the way experienced delivery drivers at this station actually do. Answer with a JSON list only, no other text.
```

## Template (version A: `{guide}` is empty)

```
A delivery van leaves the station and must deliver to every zone below. All stops in a zone are delivered together, so you only choose the order of the zones. The order of stops inside each zone is decided separately.

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

Output the order in which to visit the zones as a JSON list of zone IDs, starting with the first zone after leaving the station. Include each of the {Z} zones exactly once and nothing else. Do not include the station.
```

## Effort comparison (frozen version A, identical prompts, same 40 val routes, greedy solver, REPAIRED view)

The approved launch removes CLAUDE_CODE_EFFORT_LEVEL, so the first development calls ran at the CLI's default effort. Rule: freeze whichever scores better for Opus; ties go to max.

| Effort | Mean (repaired) | Strict | Valid | Cost per call | Wall time per call |
|---|---|---|---|---|---|
| default | 0.0319 | 0.0319 | 40/40 | $0.041 | 5.2 s |
| **max (frozen)** | **0.0310** | 0.0310 | 40/40 | $0.451 | 196.1 s |

max − default: −0.0009 [−0.0046, +0.0026] (paired bootstrap, seed 20261111). max was better on 9 routes, worse on 2 and tied on 29. That is not significant, but the rule picks max.

**Cost of max effort:** about 11 times the money and 38 times the latency per call. At the frozen setting, 500 eval routes would take about $225 by the CLI list-price field and about 9 hours with 3 calls at a time.

**Solver option:** `score --solver exact` calls `scripts/exact_zone.stops_from_zone_order_exact`. On default-effort A's zone orders it gives 0.0318, against 0.0319 for greedy.

**Score views:** `score` writes STRICT (an invalid list gets the challenge's invalid score) and REPAIRED (missing zones inserted by cheapest insertion) for every route. The primary comparison uses REPAIRED, which is conservative toward Opus. All 80 development answers were valid, so the two views are identical here.
