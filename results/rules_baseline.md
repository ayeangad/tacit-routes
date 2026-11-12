# Rules baselines: zone rules vs the plain shortest route (validation set)

Run on 2026-11-11. Every number here comes from `results/rules_baseline.json` (val) or the
`results/rules_baseline_tuning*.json` files (train-only rule selection), all written by
`scripts/rules_baseline.py`.

## The question

The plain shortest route ignores how drivers actually think. Drivers work through Amazon's
**zones** (small areas with IDs like `A-27.3J`) one at a time and almost never come back to a zone
they have left. Can a few hand-written rules about zones, with no learning, get much closer to
the driver than the shortest route?

**Short answer: yes, by a lot.** The best rule scores **0.0305** on the 300 validation routes,
against **0.0894** for the shortest route, about a third of the error. It beats the shortest route
on 288 of 300 routes. This is now the baseline any learned model has to beat.

## Results (300 val routes, official scorer, lower is better, 0 = the driver's order)

| Arm | What it does | Mean | Bootstrap 95% CI | Median | Zone revisits (driver: 0.96) | Tour time / driver's |
|---|---|---|---|---|---|---|
| **S** | Plain shortest route | 0.0894 | 0.0835 – 0.0955 | 0.0752 | 15.84 | 0.907 |
| **R1** | Zones as blocks, zone order = shortest tour over zones | 0.0467 | 0.0432 – 0.0504 | 0.0424 | 0 | 0.926 |
| **R2** | Zones as blocks, zone order read from zone IDs ("snake") | 0.0305 | 0.0276 – 0.0337 | 0.0243 | 0 | 0.945 |
| **R3** | Hybrid: shortest tour over zones, with a penalty for leaving the snake (λ = 2) | **0.0305** | 0.0276 – 0.0337 | 0.0245 | 0 | 0.945 |
| **R4-hist** | Shortest tour over zones plus a penalty from counted driver history | 0.0331 | 0.0294 – 0.0370 | 0.0232 | 0 | 0.945 |

All 1,500 predicted routes were valid (every stop exactly once).

### Paired comparisons (same route, one rule minus the other; negative = first is better)

| Comparison | Mean difference | Bootstrap 95% CI | First better / worse / tied | Exact sign test p |
|---|---|---|---|---|
| R1 − S | −0.0427 | −0.0472 – −0.0384 | 275 / 25 / 0 | 2.1e−54 |
| R2 − S | −0.0588 | −0.0637 – −0.0541 | 287 / 13 / 0 | 2.0e−68 |
| R3 − S | −0.0588 | −0.0637 – −0.0541 | 288 / 12 / 0 | 9.1e−70 |
| R4-hist − S | −0.0563 | −0.0611 – −0.0514 | 283 / 17 / 0 | 2.4e−63 |
| R4-hist − R3 (best of R1–R3) | +0.0026 | −0.0003 – +0.0056 | 114 / 118 / 68 | 0.84 |
| R3 − R2 | −0.00002 | −0.0002 – +0.0001 | 23 / 32 / 245 | 0.28 |

How to read this:
- Every rule beats the shortest route on almost every route. That is not luck (the p-values are tiny).
- **R2 and R3 are the same rule in practice.** R3 matches R2's zone order on 245 of 300 routes, and the
  average difference is 0.00002. The "hybrid" adds nothing over the plain snake rule.
- **R4-hist (counting) does not beat the snake rule.** It is slightly worse on average (+0.0026), and
  the confidence interval just touches 0. Its median is the lowest (0.0232), so it does well on
  most routes but badly on a few. Per-route wins are 114 vs 118: a coin toss.
- For scale: the plain shortest route on the 50 pilot routes was 0.0784, and Opus ordering stops
  directly was 0.069. Those numbers come from **different routes**, so treat them as rough
  context only. The published best, 0.0374 (Wu et al.), is on the official evaluation set, also a
  different set. A 0.0305 here does not mean "beats the state of the art".

## The idea behind every rule

Every rule (R1–R4) has the same two layers:

1. **Zone order (the big decision).** Decide the order in which to visit the zones. Each zone is
   visited exactly once, as one block. The rules differ only in this step.
2. **Stop order inside each zone (the small decision).** Within a zone, find the shortest path
   with OR-Tools. It starts from the last stop of the previous zone and ends heading towards the
   next zone: the path ends at a pretend "exit" stop, whose distance from each real stop is the
   travel time to the nearest stop of the next zone (to the station after the last zone).

Stops with no zone: 122 of 300 val routes have at least one stop with a missing zone ID (0.95 per
route on average). Each such stop joins the zone of its nearest stop that has a zone, where "nearest"
means the shortest round-trip travel time. Two zone IDs in all of train (`LOA1`, `RTED3`) do not fit
the `A-27.3J` pattern; they are handled the same way.

**Why blocks help so much.** The driver almost never re-enters a zone (0.96 re-entries per route).
The shortest route re-enters zones 15.8 times per route, because hopping into a neighbouring zone
for one stop is quicker. The metric punishes those hops, since they put stops far from where the
driver had them. Blocking zones cuts re-entries to 0. That costs about 4% more driving time (0.945
against 0.907 of the driver's time) and gets much closer to what the driver actually did.

## R1: zones as blocks, shortest tour over zones

- **Zone distance.** The travel time from zone a to zone b is the average travel time over every
  pair (stop in a, stop in b). Think of it as the distance between the zones' centres, measured in
  driving seconds. The station is a one-stop "zone".
- **Zone order.** A shortest closed tour (a TSP) through the station and all zones, solved with
  OR-Tools.
- **Result:** 0.0467. That halves the shortest route's error, purely from "finish a zone before
  you leave it".
- **Its weakness:** the shortest tour over zones is not the order the driver uses. In the worked
  example below, R1 visits the 18 zones in an order that zigzags; the driver sweeps across them
  row by row.

## R2: zone-ID order ("snake")

**What the IDs look like.** A zone ID such as `A-27.3J` has four parts: `A` (a big area), `27` (a
block of zones), `3` (a row, 1 to 3) and `J` (a column letter). The letters skip `F` and `I`.

**What drivers do (measured on 500 random High-quality train routes, 9,786 zone-to-zone moves):**

| Next zone differs from the current one by... | Share |
|---|---|
| row ±1, same column (e.g. `3J → 2J`) | 54.2% |
| column one letter on, same row (e.g. `1J → 1K`) | 18.5% |
| both row and column | 7.8% |
| column two letters on, same row (mostly skipped `F`/`I` or absent columns) | 6.9% |
| block ±1 (e.g. `D-23 → D-22`) | 6.7% |
| anything else | 5.9% |

So drivers walk the grid like a lawnmower: down a column (`3J, 2J, 1J`), step to the next column,
up it (`1K, 2K, 3K`), step across, down again, and so on. Routes cover about 2 blocks on average.

**The rule ("snake"):**
1. Group zones by block (`A-27`) and sort the blocks by ID.
2. Inside a block, take the columns in letter order. Inside each column, take the rows in order,
   **reversing the row direction at every new column** so consecutive zones always touch.
3. At each new block, reverse the column direction, so the walk continues from where it ended.
4. That leaves 8 versions (block order up or down × first column direction × first row direction).
   Each version's reverse is also one of the 8, so "ascending or descending" is covered. Each is
   costed as a closed tour from the station on the zone distance matrix, and **the cheapest is
   used, per route**.

**Chosen on train, not val.** The alternative was a plain sort by (area, block, row, column),
ascending or descending, whichever is cheaper. It scored 0.0962 on the 300 train routes, worse
than the shortest route: sorting jumps from `1J` to `1K` to `1L`... and then back to `2J`. The
snake scored 0.0310 there.

## R3: the hybrid (chosen on train only)

Candidates were scored on 300 random High-quality **train** routes
(`random.Random(20261111).sample`), never on val or pilot.

| Round | Candidate | Idea | Train mean |
|---|---|---|---|
| 1 | R1 (`tsp`) | for reference | 0.0489 |
| 1 | R2 `id_snake` | for reference | **0.0310** |
| 1 | R2 `id_lex` | plain ID sort | 0.0962 |
| 1 | `adj_pen_0.5 / 1 / 2` | zone TSP; a move to a zone that is not a grid neighbour costs extra (λ × median zone distance) | 0.0361 / 0.0371 / 0.0377 |
| 1 | `group_tsp` | zone TSP, but each block (`A-27`) must be visited in one go | 0.0438 |
| 1 | `group_tsp+adj_pen_1` | both | 0.0343 |
| 2 | `snake_pen_0.25 / 0.5 / 1 / 2` | zone TSP; any move that does not follow the snake (either way) costs λ × median zone distance | 0.0352 / 0.0327 / 0.0316 / **0.0310** |

**Disclosure:** round 2 was added **after** round 1 showed every round-1 hybrid losing to the
plain snake. It used the same 300 train routes. The chosen rule is the best hybrid across both
rounds, `snake_pen_2`. With λ = 0 this is exactly R1; with a very large λ it is exactly R2. At λ = 2
it leaves the snake only where the snake is very expensive in driving time. On train it was no
better than R2 (difference +0.00006, 51 better / 33 worse / 216 tied), and on val it was the same
again. Honest conclusion: **no hybrid we tried improves on the pure snake rule.**

## R4-hist: a counting control (history lookup table)

**Purpose.** A later neural model will predict "which zone comes next". Before giving the network
credit, we need to know how much simple counting already explains. This mirrors Ghosh et al. 2023
("Learn global and optimize local", Computers & OR), reported at about 0.030 on the official
evaluation set without a neural network. That is a different set, so the comparison is indicative
only.

**How it works.**
- **Counting.** For every train route, list the driver's zones in visiting order (drop the
  station and zone-less stops, merge repeats), then count each move a → b, plus station → first
  zone. Counts are kept per station, because the same ID at two stations is a different place.
- **Probability of "b next after a"** among the zones in *this* route (other than a):
  P(b | a) = (count(a→b) + k) / Σ over candidate zones b′ of (count(a→b′) + k), with **k = 1**
  (add-one smoothing: a move never seen still gets a small probability).
- **Back-off.** If this station's counts for row a, summed over this route's candidate zones, come
  to fewer than 5, add the counts pooled across all stations for the same zone-ID pair. On val, 13.1
  of about 20.8 rows per route backed off. Station tables are thinner than expected for these exact
  zone pairs.
- **Cost used by the zone TSP:**
  cost(a → b) = zone distance(a, b) + α × scale × m(a) × (−log P(b | a)), where
  - scale = the route's median zone-to-zone distance (this makes α unit-free);
  - m(a) = **3 for station → first zone** (history weighs more there, since the start is hard
    to guess from distances), **1 for zone → zone**;
  - **zone → station uses travel time only** (no history term).
  −log P is 0 for a certain move and grows as the move gets rarer. So common moves are cheap and
  rare moves are expensive, and α sets how much history matters against driving time.
- **Tuning α on train only.** The same 300 train routes. Their own moves were **left out of the
  counts while tuning**: the table came from the other 5,462 train routes, of all quality labels.
  α ∈ {0.25, 0.5, 1, 2, 4, 8} first; 0.25 was best but at the edge of the grid, so {0.05, 0.1, 0.15}
  was added. Train means: 0.05 → 0.0368, 0.1 → 0.0330, 0.15 → 0.0318, **0.25 → 0.0306**, 0.5 →
  0.0338, 1 → 0.0357, 2 → 0.0390, 4 → 0.0400, 8 → 0.0414. **α = 0.25** was used; k and the start
  multiplier were fixed, not tuned.
- **Val run.** The table was rebuilt from all 5,762 train routes, never from val or pilot, and
  scored once.

**Result.** On train, R4-hist was a hair better than the snake (0.0306 vs 0.0310). On val, it was
a little worse (0.0331 vs 0.0305), with a 114/118 split. The counting table mostly rediscovers the
snake, because drivers' history *is* the snake. Where history is thin or unusual it adds noise.
For the neural model, the counting control to beat is about **0.033**, and the hand-written snake
is about **0.0305**.

## Worked example

Route `RouteID_554fc365-f78e-4224-8ad1-72c1bfb2e910`, station DBO2, 99 stops including the
station, 18 zones. It was chosen as **the val route with the median S score**, so it is neither a
flattering nor an unflattering pick.

Scores on this route: S 0.0757, R1 0.0420, R2 0.0047, **R3 0.0047**, R4-hist 0.0051. Zone re-entries:
driver 0, S 15, every rule 0.

First 15 stops after the station (both S and R3 are shown in the direction produced; neither
needed flipping):

| # | Driver | Zone | S (shortest) | Zone | R3 (best rule) | Zone |
|---|---|---|---|---|---|---|
| 1 | QU | A-27.3J | TB | A-27.1L | EB | A-27.3J |
| 2 | SM | A-27.3J | AF | A-27.1M | LV | A-27.3J |
| 3 | UI | A-27.3J | HK | A-27.1M | HW | A-27.3J |
| 4 | ZO | A-27.3J | KO | A-27.1M | LM | A-27.3J |
| 5 | FF | A-27.3J | ZQ | A-27.2M | ZO | A-27.3J |
| 6 | LM | A-27.3J | BP | A-27.2M | FF | A-27.3J |
| 7 | HW | A-27.3J | LP | A-27.2M | QU | A-27.3J |
| 8 | EB | A-27.3J | IS | A-27.2M | SM | A-27.3J |
| 9 | LV | A-27.3J | ZV | A-27.2M | UI | A-27.3J |
| 10 | YR | A-27.2J | HU | A-27.1M | YR | A-27.2J |
| 11 | TC | A-27.2J | FB | A-27.2M | TC | A-27.2J |
| 12 | QM | A-27.2J | TU | A-27.2M | QM | A-27.2J |
| 13 | KC | A-27.2J | JL | A-27.3M | KC | A-27.2J |
| 14 | BC | A-27.1J | SP | A-27.3M | BC | A-27.1J |
| 15 | WF | A-27.1J | QF | A-27.3M | WF | A-27.1J |

Whole route, zone by zone:
- **Driver:** 3J 2J 1J · 1K 2K 3K · 3L 2L 1L · 1M 2M 3M · 3P 2P 1P · 1Q 2Q 3Q. That is a perfect snake.
- **R2/R3:** exactly the same 18-zone order. The only errors left are *inside* zone 3J. The driver
  went QU SM UI ZO FF LM HW EB LV; the rule went EB LV HW LM ZO FF QU SM UI. It covered the same
  nine stops, roughly in the driver's order run backwards.
- **R1 (shortest tour over zones):** 1L 2L 1M 2M 3M 3K 3L 2P 1P 1K 1J 2J 3J 2K 3P 1Q 2Q 3Q. Each
  zone is one block, but the order zigzags, so it scores 0.042.
- **S:** 33 zone visits for 18 zones (15 re-entries). It starts in 1L, jumps between M and K, and
  comes back to J three times.

This one route sums up the result: the zone order is the big decision. A simple ID rule gets it
exactly right here. What remains is the stop order inside each zone.

## Every setting

| Setting | Value |
|---|---|
| Data | `data/processed/routes/<id>.npz`. Val = the 300 `val` routes in `splits.json` (High, seed 20261111). Pilot routes never touched. |
| Cost matrix for scoring | npz float32 travel times rounded to 0.1 s; this reproduces the raw JSON exactly (checked cell by cell on one route) |
| S | OR-Tools 9.15 routing, 1 vehicle, closed tour from the station, PATH_CHEAPEST_ARC → GUIDED_LOCAL_SEARCH, **10 s per route**; costs in tenths of a second. Identical to `run_plain_baseline.py`. |
| Zone-order TSP (R1, R3, R4) | Same OR-Tools settings, closed tour station → zones → station, **1,000 ms per route** |
| Inside-zone path (R1–R4) | Same OR-Tools settings, open path from the previous stop to an exit dummy, **100 ms per zone**; one-stop zones placed without a solver call |
| Why 100 ms inside zones | Pre-check on 106 train zones (4–18 stops) against a 1 s GLS run: 50 ms was worse on 5 zones (mean excess 0.02%), 200 ms on 2 (< 0.01%), greedy descent without GLS on 19 (0.09%). 100 ms sits between the two GLS settings tested; it was not tested by itself |
| R2 variant choice | 8 snake versions, cheapest closed zone tour on the mean-distance matrix |
| R3 | `snake_pen_2`: λ = 2 × median zone distance for every move not following the cheapest snake |
| R4-hist | α = 0.25, k = 1, start multiplier 3, back-off threshold 5, zone → station history-free |
| Scorer | Official unmodified rc-cli `score.py` (commit `fdafa50e…`), called per route exactly as `evaluate()` does (isinvalid check, then `score()`). On 3 val routes × 5 arms, `evaluate()` on real JSON files gave identical scores, and the driver's own order scored 0. (`evaluate()` was not used for all 300 because it needs the whole cost-matrix JSON in memory.) numpy 2.5.4. |
| Statistics | Bootstrap 95% CI: 10,000 percentile resamples of routes, numpy `default_rng(20261111)`. Exact two-sided sign test on non-tied routes. |
| Resources | 3 worker processes, niced. Peak memory about 160 MB main + about 150 MB per worker. |
| Wall time | train round 1: 3,349 s; round 2: 1,651 s; R4 α round 1: 2,367 s; α round 2: 1,333 s; val: 2,555 s. Total about 3.1 h. |

## How to rerun

From `tacit-routes/`; `-I` stops Python importing anything from the working or data folders.

```
.venv/bin/python -I scripts/rules_baseline.py tune --workers 3              # train: R1, R2 candidates, round-1 hybrids  (~56 min)
.venv/bin/python -I scripts/rules_baseline.py tune --round 2 --workers 3    # train: snake-penalty hybrids               (~28 min)
.venv/bin/python -I scripts/rules_baseline.py tune-hist --workers 3         # train: R4-hist alpha round 1               (~40 min)
.venv/bin/python -I scripts/rules_baseline.py tune-hist --round 2 --workers 3  # train: R4-hist alpha round 2           (~22 min)
.venv/bin/python -I scripts/rules_baseline.py val --workers 3               # val: S, R1, R2, R3, R4 once                (~43 min)
```

`val` reads the train choices from the four tuning files. It stops with an error if any of them
is missing. OR-Tools runs with time limits, so a rerun can return slightly different tours; the
exact orders used are saved per route in `results/rules_baseline.json` (`per_route[].arms[].order`).
Logs are in `runs/rules_baseline/`.

## Caveats

- Val is 300 High-quality routes from the training side of the challenge, split at random rather
  than by date. Routes from the same station and the same week appear in both train and val. That
  helps R4-hist more than the ID rules, because the ID rules use no counts.
- The rule-selection subset is 300 train routes. Two rounds of candidates were tried there; the
  second round was added after seeing the first (disclosed above). Val was scored once, after
  every choice was fixed.
- The comparator "best of R1–R3" for R4-hist was picked on val. R3 and R2 are nearly identical,
  so this makes no practical difference.
- The solvers are heuristics with time limits, not proofs of optimality.
- What remains is mostly **stop order inside zones**. On the worked example the zone order is
  perfect, yet the score is 0.0047, not 0. A learned model must improve within-zone order, or the
  rare routes where drivers break the snake, to beat 0.0305.
