# Tiny model vs Opus, the 2021 winner and simple rules, route by route (validation set)

Run on 2026-11-11 by `scripts/val_comparison.py`. All numbers come from `results/val_comparison.json`. This is the **300 validation routes only**. Pilot routes and `data/eval_frozen/` were not touched.

## What this compares, in plain words

Every method gets the same 300 delivery routes and proposes an order for the stops. The official 2021 challenge scorer compares that order with the order the real driver used. **Lower is better. 0 means the driver's exact order.**

Because every method is scored on the *same* routes, we can compare them **route by route** ("paired"). On each route we take method A's score minus method B's score. A negative difference means A was closer to the driver on that route. Then:

- **Mean difference [95% CI]:** the average of those 300 differences. The bracket comes from a **bootstrap**: we redraw the 300 routes at random with replacement 10,000 times (seed 20261111) and keep the middle 95% of the averages. If the bracket does not include 0, the average gap is unlikely to be luck in which routes we happened to get.
- **Wins / losses / ties:** on how many routes A beat B, lost, or tied exactly.
- **Sign test p:** if A and B were really equally good, how surprising would a win/loss split this lopsided be? Small p (say below 0.05) means "probably not a coin flip". Ties are dropped. It ignores *how much* each route was won by, so it can disagree with the mean.
- **Holm correction:** we test 3 tiny-model variants against Opus, and testing several things raises the chance of one lucky "win". Holm adjusts the p-values for that. The adjusted p is the one to quote.

## The contestants

| Name here | What it is |
|---|---|
| **v0** | Our tiny model (27,329 weights): picks the order of zones, OR-Tools orders stops inside each zone. No history. |
| **v0-hist** | Same model plus 18 "station history" inputs (28,481 weights) counted from all training routes. |
| **v0-hist-past** | Same as v0-hist, but history only from training routes dated **before** the route's own day. The version that cannot be accused of using the future. |
| **Opus run 2** | Opus 5.5 wrote and ran its own predictor in a locked sandbox (`results/opus_fight_run2.md`). The best Opus result, and the main opponent. |
| **Opus run 1** | The earlier Opus attempt, cut short by our memory setup (`results/opus_fight.md`). |
| **jpt** | The 2021 challenge winner's code (jpt-amz), run locally (`results/jpt_baseline.md`). A reference, not the goal. |
| **R2** | Hand-written rule: visit zones in the order of their IDs, "snaking" (`results/rules_baseline.md`). The best rule, chosen on training data. |
| **R4-hist** | Counting control: zone tour by travel time plus a penalty from counted past zone-to-zone moves. History without a learned model. |
| **Shortest** | Plain shortest route (OR-Tools, 10 s). |

### Which v0 variants count (fixed before looking at validation)

18 v0 variants were scored on validation (3 families × 2 training sets × 3 decoders). To avoid picking the luckiest, the **pre-registered** variant per family is chosen using training data only:

- **Training data:** the one with the lower held-out-train loss, which is High+Medium in all three families (0.560 / 0.455 / 0.487).
- **Decoding:** beam 5 plus travel time, at the λ tuned on 234 held-out train routes (λ = 0.5 / 0.5 / 1).

| Family | Pre-registered variant | Best on validation (**optimistic**, chosen by looking at val) |
|---|---|---|
| v0 | High+Medium, beam 5 + travel λ=0.5 (0.0356) | High+Medium, beam 5 (0.0351) |
| v0-hist | High+Medium, beam 5 + travel λ=0.5 (0.0229) | the same variant |
| v0-hist-past | High+Medium, beam 5 + travel λ=1 (0.0261) | High+Medium, beam 5 (0.0255) |

All comparisons below use the pre-registered variants unless marked "optimistic".

## Headline table

Each row is "first minus second" on the same 300 routes. Negative = the first one is closer to the drivers.

| Comparison | Means (first / second) | Mean difference [95% CI] | Wins / losses / ties | Sign p | Holm p |
|---|---|---|---|---|---|
| **v0-hist − Opus run 2** | 0.0229 / 0.0268 | **−0.0039 [−0.0063, −0.0016]** | **182 / 118 / 0** | 0.00026 | **0.0008** |
| **v0-hist-past − Opus run 2** | 0.0261 / 0.0268 | −0.0007 [−0.0031, +0.0016] | 172 / 128 / 0 | 0.013 | 0.013 |
| **v0 (no history) − Opus run 2** | 0.0356 / 0.0268 | **+0.0088 [+0.0058, +0.0118]** | 126 / 174 / 0 | 0.0066 | 0.013 |
| *Does the model learn more than rules and counting?* | | | | | |
| v0-hist − R2 (best rule) | 0.0229 / 0.0305 | **−0.0076 [−0.0099, −0.0054]** | 170 / 108 / 22 | 0.00024 | — |
| v0-hist-past − R2 | 0.0261 / 0.0305 | **−0.0044 [−0.0067, −0.0021]** | 157 / 127 / 16 | 0.085 | — |
| v0 − R2 | 0.0356 / 0.0305 | +0.0051 [+0.0024, +0.0079] | 112 / 168 / 20 | 0.00098 | — |
| Opus run 2 − R2 | 0.0268 / 0.0305 | −0.0037 [−0.0060, −0.0014] | 164 / 136 / 0 | 0.12 | — |
| v0-hist − R4-hist (counting) | 0.0229 / 0.0331 | **−0.0101 [−0.0131, −0.0074]** | 163 / 109 / 28 | 0.0013 | — |
| v0-hist-past − R4-hist | 0.0261 / 0.0331 | **−0.0070 [−0.0098, −0.0043]** | 150 / 125 / 25 | 0.15 | — |
| v0 − R4-hist | 0.0356 / 0.0331 | +0.0026 [−0.0005, +0.0056] | 122 / 164 / 14 | 0.015 | — |
| Opus run 2 − R4-hist | 0.0268 / 0.0331 | −0.0063 [−0.0091, −0.0036] | 158 / 142 / 0 | 0.39 | — |
| *Reference: the 2021 winner* | | | | | |
| v0-hist − jpt | 0.0229 / 0.0213 | +0.0017 [−0.0005, +0.0036] | 114 / 185 / 1 | 5e-05 | — |

Holm p applies only to the three rows against Opus run 2. Every other p is unadjusted and should be read as descriptive.

**Reading it:**

- **v0-hist beats Opus run 2** by 0.0039 on average and on 182 of 300 routes. The CI excludes 0 and the Holm-corrected sign test gives p ≈ 0.0008. This is the clearest result.
- **v0-hist-past against Opus run 2 is a draw on the average.** The mean gap is −0.0007 and its CI spans 0. It wins more routes (172 / 128, Holm p = 0.013), but by small margins, while its losses are larger. **Do not claim "the past-only model beats Opus" from this.** Claim "it matches Opus".
- **v0 without history loses to Opus run 2** (+0.0088, 174 of 300 routes for Opus).
- **Against rules and counting:**
  - Both history models beat the best rule R2 and the counting control R4-hist on the average, with CIs well away from 0. So the learned model adds something beyond "follow zone IDs" and "count past moves".
  - For v0-hist-past the route-by-route sign tests are weak (p 0.085 and 0.15). Its average win comes partly from avoiding a minority of bad routes, not from winning most routes.
  - **v0 without history is worse than the R2 rule.** Without history the model does not beat a hand-written rule.
  - Opus run 2 also beats R2 and R4-hist on the average, but by less than v0-hist does.
- **The 2021 winner (jpt) is still the best method here.** v0-hist's mean is 0.0017 higher; the CI just touches 0, and jpt wins 185 of 300 routes (p = 5e-05). Opus run 2 is further behind jpt (+0.0055, jpt better on 220 routes).

## Other comparisons

| Comparison | Mean difference [95% CI] | W / L / T | Sign p |
|---|---|---|---|
| v0-hist − Opus run 1 | −0.0119 [−0.0146, −0.0095] | 241 / 59 / 0 | 3e-27 |
| v0-hist-past − Opus run 1 | −0.0088 [−0.0112, −0.0064] | 229 / 71 / 0 | 1e-20 |
| v0 − Opus run 1 | +0.0008 [−0.0023, +0.0039] | 159 / 141 / 0 | 0.33 |
| v0-hist-past − jpt | +0.0048 [+0.0029, +0.0067] | 98 / 201 / 1 | 3e-09 |
| Opus run 2 − jpt | +0.0055 [+0.0038, +0.0075] | 80 / 220 / 0 | 3e-16 |
| Opus run 2 − Opus run 1 | −0.0080 [−0.0105, −0.0056] | 217 / 83 / 0 | 6e-15 |
| jpt − R2 | −0.0093 [−0.0112, −0.0073] | 225 / 72 / 3 | 2e-19 |
| v0-hist / v0-hist-past / v0 − shortest | −0.0668 / −0.0637 / −0.0541 | 293/7, 295/5, 292/8 | < 1e-74 |
| Opus run 2 − shortest | −0.0630 [−0.0676, −0.0582] | 291 / 9 / 0 | 5e-74 |
| *Optimistic:* v0-hist-past (beam 5, best on val) − Opus run 2 | −0.0013 [−0.0038, +0.0011] | 176 / 124 / 0 | 0.003 |
| *Optimistic:* v0 (beam 5, best on val) − Opus run 2 | +0.0083 [+0.0053, +0.0112] | 125 / 175 / 0 | 0.005 |

R3 (the train-chosen hybrid rule) behaves the same as R2 here, to within 0.0001 (in the JSON).

All arms on their own (mean [95% CI], median):

| Arm | Mean [95% CI] | Median |
|---|---|---|
| jpt (2021 winner) | 0.0213 [0.0183, 0.0246] | 0.0105 |
| **v0-hist** | **0.0229 [0.0200, 0.0262]** | 0.0118 |
| v0-hist-past | 0.0261 [0.0229, 0.0296] | 0.0143 |
| **Opus run 2** | **0.0268 [0.0236, 0.0303]** | 0.0153 |
| R2 zone-ID rule | 0.0305 [0.0276, 0.0337] | 0.0243 |
| R4-hist counting | 0.0331 [0.0294, 0.0370] | 0.0232 |
| Opus run 1 | 0.0349 [0.0309, 0.0392] | 0.0220 |
| v0 (no history) | 0.0356 [0.0322, 0.0394] | 0.0275 |
| Shortest route | 0.0898 [0.0840, 0.0959] | 0.0766 |

## Direction: does the route run the driver's way round?

The official score barely notices if a route is driven exactly backwards. So we also check direction. From each method's stop order we take the order in which it first enters each zone, then compare that with the driver's zone order using **Kendall tau**. Tau is +1 for the same order, −1 for exactly reversed, and about 0 for unrelated. Zone-less stops borrow the nearest stop's zone, as in v0.

As a check, rebuilding v0's zone order from its stop order gave back its stored zone order on 300 of 300 routes.

| Arm | Mean tau | Median tau | Routes mostly backwards (tau < 0) | Exact zone order | Exact reverse |
|---|---|---|---|---|---|
| jpt | +0.78 | +0.96 | 13 | 111 | 2 |
| **Opus run 2** | **+0.76** | +0.96 | 19 | 111 | 1 |
| Opus run 1 | +0.75 | +0.95 | 19 | 111 | 1 |
| **v0-hist** | **+0.75** | +0.97 | 21 | 113 | 1 |
| v0-hist-past | +0.63 | +0.92 | 41 | 99 | 5 |
| R4-hist counting | +0.46 | +0.70 | 71 | 64 | 7 |
| R2 zone-ID rule | +0.08 | +0.22 | 141 | 64 | 54 |
| v0 (no history) | +0.07 | +0.13 | 138 | 42 | 32 |
| Shortest | −0.00 | +0.00 | 148 | 1 | 1 |

v0-hist and Opus run 2 get direction right about equally often: the paired tau difference is −0.009 [−0.051, +0.034], 40 vs 49 routes, p = 0.40. v0-hist-past is worse on direction than Opus run 2 (−0.12 [−0.18, −0.07]). **v0 without history and the R2 rule are close to a coin flip on direction.** Part of their score comes from the metric forgiving reversed routes.

## Cost and speed

| Arm | Per route | One-off / session | Money |
|---|---|---|---|
| v0 (pre-registered) | 0.27 CPU-s (1 thread) | Training: 1,964 CPU-s | $0 |
| v0-hist | 0.31 CPU-s | Training: 1,274 CPU-s + 45 s history table | $0 |
| v0-hist-past | 0.30 CPU-s | Training: 1,160 CPU-s | $0 |
| Opus run 2 | `predict.py` about 2.5 min wall on 3 processes for 300 routes (about 0.5 s wall per route). This is Opus's own statement, not timed by us. | Session 71 min 38 s wall, 2 h 31 min CPU, 34,580 output tokens | **$1.37** (CLI list-price cost field) |
| Opus run 1 | `predict.py` 12.6 min wall on 3 processes | Session 67 min | unknown (killed before the cost line) |
| jpt | 2.70 CPU-s | Build 313 s wall; apply 570 s wall for 300 routes | $0 |
| Shortest | 10 s search per route | — | $0 |

v0's numbers are from `results/v0.json` (one worker process, scoring not included).

## How the scoring was checked

- **Same routes, same scoring path.** Every arm except the rules was re-scored here from its saved stop orders with `v0_decode.official_scores`: the official, unmodified `evaluate()`, travel times rounded to 0.1 s (the precision of the raw data), 20 routes per call. The rules arms were re-scored the same way from the orders stored in `results/rules_baseline.json`.
- **Stored scores reproduced exactly.** All 14 arms are 300/300 valid. Every recomputed per-route score equals the stored one (maximum difference 0):
  - v0 against `runs/v0/cache` and the means in `v0.json`;
  - Opus runs 1 and 2 against their results files, so run 1 was already scored with the 0.1 s rounding fix;
  - jpt, even though `score_jpt_baseline.py` read the raw JSON travel times instead of the rounded npz;
  - the shortest route;
  - all five rules arms.

  So there was no convention mismatch, and nothing needed a correction.
- **Two shortest routes.** The rules file's own shortest route (S, 0.0894) is a separate 10 s OR-Tools run, not the saved v0 shortest tours (0.0898). The tours are identical on 172 of 300 routes, and on those the scores are equal, so the scoring convention matches. The other routes differ only because the search is time-limited. Comparisons here use the v0 shortest tours.

## Caveats (read before quoting any number)

- **Validation set only.** These 300 routes are the ones we developed on. **The verdict that counts is the frozen evaluation test** (`data/eval_frozen/`, not yet opened), with both v0 and Opus run on it.
- **Many variants tried.** 18 v0 variants were scored on these same routes; the rules had several candidates; Opus ran twice. The pre-registered rule above limits the damage for v0, but the whole project has looked at validation many times. Expect the true gaps to be somewhat smaller than shown.
- **One Opus run each.** Run 1 → run 2 alone moved Opus by 0.008, which is twice the v0-hist margin. Another Opus run could land on either side of v0-hist. We have no estimate of Opus's run-to-run spread.
- **Random-by-date split.** Validation routes are mixed in time with training routes. v0-hist's history table includes routes from *later* days than some val routes; only v0-hist-past avoids that. Opus run 2 also built per-station history from all training routes, so on this point it is in the same position as v0-hist. The frozen test is mostly outside the training dates, where history will be weaker for everyone.
- **"Tiny" needs a footnote.** v0-hist has 28,481 learned weights, but at prediction time it also looks things up in a history table: the zone orders of 5,762 training routes (115,518 route-zone entries, 1.9 MB as JSON). That table is data the method carries, beyond its weights. The same is true of R4-hist, Opus run 2 and jpt.
- **Mean versus wins can disagree.** v0-hist-past wins more routes than Opus run 2 but has about the same mean. Quote both, and do not call it a win.
- **The metric forgives reversed routes.** See the direction table. The no-history model and the R2 rule look better on the score than their direction accuracy would suggest.
- **Opus's `predict.py` runtime is its own report,** not our measurement. Opus's searches are time-limited, so a re-run could change its predictions slightly.

## How to rerun

```
OMP_NUM_THREADS=1 .venv/bin/python -I scripts/val_comparison.py --workers 3
```

Per-arm scores are cached in `runs/val_comparison/`; delete that folder to re-score everything (about 15 min on 3 workers, peak RSS 0.42 GB). The run log is in `runs/val_comparison.log`.
