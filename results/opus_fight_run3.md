# Opus fair fight, run 3 (blind: no information about the final test)

Run on 2026-11-11 (17:30–19:04 IST, 94 min, 55 turns, $1.76 by the CLI cost field). The setup is identical to run 2 apart from the folder name (`runs/opus_fight_run3/`). The audit is clean: 0 sandbox violations, 0 paths outside the arena, 0 pattern flags, and 1 permission refusal (a harmless log read). `scripts/audit_opus_fight.py` was fixed to take each run's own arena; run 2's audit is unchanged under the fix. Numbers come from `results/opus_fight_run3.json` (`scripts/score_opus_fight.py --tag opus_fight_run3`).

## Score (300 validation routes, lower is better)
- **Opus run 3: 0.0219 [0.0190, 0.0250]**, median 0.0110, 300/300 valid. Shortest route: 0.0898.

## Paired against the same 300 routes
Each row is "first minus Opus run 3". Bootstrap: 10,000 resamples, seed 20261111. Sign test: exact.

| First | Mean diff [95% CI] | First better / worse | Sign p |
|---|---|---|---|
| v0-hist (pre-registered) | **+0.0010 [−0.0005, +0.0027]** | 118 / 175 | 0.001 |
| v0-hist-past | +0.0042 [+0.0028, +0.0057] | 99 / 194 | 3e-08 |
| Opus run 2 | +0.0049 [+0.0032, +0.0070] | 97 / 203 | 9e-10 |
| jpt (2021 winner) | −0.0006 [−0.0021, +0.0010] | 169 / 129 | 0.024 |

**Reading it:**
- On validation, Opus run 3 **matches v0-hist on the mean** (the interval spans 0) and **wins more routes** (175 to 118).
- It is level with the 2021 winner.
- Opus's fair-fight runs so far: 0.0349 (run 1, cut short), 0.0268 and 0.0219. **The run-to-run spread (0.005 between the two full runs) is larger than any margin v0-hist ever had.**

## What it built (its APPROACH.md)
- **Zone order:** a pairwise zone-transition MLP with 31 features, including same-station history at three zone-ID levels (full, `A-2`, `A-2.2`) with leave-one-out. It is blended with a hand cost, then solved as a zone TSP by OR-Tools.
- **Stops:** an exact shortest open path inside each zone (Held–Karp for zones of up to 11 stops).
- **Its own holdout:** 0.0232 on 300 High train routes. With the oracle zone order it reaches 0.0066, so nearly all its remaining error is in the zone order.
