# Plain shortest-route baseline (no learning)

Run on 2026-11-10. All numbers come from `results/plain_baseline.json`.

## What ran

- **Data:** the training side of the Amazon Last Mile Routing Research Challenge (`s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/model_build_inputs/`), downloaded anonymously. The evaluation side was not touched.
- **Sample:** 50 routes labelled **High** quality. Labels across all 6,112 training routes: High 2,718, Medium 3,292, Low 102. High was chosen because the official evaluation set contains only High-quality routes (Wu et al., arXiv 2205.04001). The draw was `random.Random(20261110).sample(sorted High route IDs, 50)`. It covers 13 stations, with 63 to 192 stops per route (mean 144.6).
- **Shortest route:** OR-Tools 9.15 routing with one vehicle, a closed tour that starts and ends at the station, PATH_CHEAPEST_ARC followed by GUIDED_LOCAL_SEARCH, and **10 s per route** (3 routes solved in parallel). Costs are the supplied travel times, rounded to 0.1 s. The tour is closed because the scorer appends the return to the station.
- **Random control:** a uniform shuffle of every non-station stop, seeded per route with `random.Random(f"20261110:{route_id}")`.
- **Scorer:** the **official, unmodified** `score.py` from MIT-CAVE/rc-cli, commit `fdafa50e8b8a18d4cbff112ec432011952b92c7d` (`scripts/official_rc_cli/`). It was called through its own `evaluate()` on JSON files in the challenge format. It pins numpy 1.20.1, which does not install on Python 3.12, so it ran on numpy 2.5.4. It only uses `np.mean` and `np.std`.

## Results (50 routes, lower is better)

| Method | Mean score | Bootstrap 95% CI | Median |
|---|---|---|---|
| Plain shortest route | **0.0784** | 0.0690 – 0.0882 | 0.0750 |
| Random order | 0.9014 | 0.8149 – 0.9908 | 0.8830 |

- Paired difference, random minus shortest: 95% CI 0.739 – 0.909.
- Sanity check: the challenge's own "invalid route" scores average 0.9011 on these routes. They are defined as a typical random shuffle, and our random control matches them.
- The solver's tours average 0.916 of the driver's tour time (median 0.912). Drivers take about 9% longer than the shortest tour, so they are not minimising time. On 1 of 50 routes the 10 s solver tour was 0.7% longer than the driver's own. That shows the heuristic is not optimal everywhere.
- **Compared with Wu et al.'s 0.0374:** the plain shortest route scores about twice as high (worse). The comparison is only indicative. Their 0.0374 is on the 3,052-route official evaluation set, while this is 50 training-side routes.

## What the metric is

For each route, score = SD × (ERP_norm / ERP_edits). The submission score is the unweighted mean over routes.

- **SD (sequence deviation)** looks at the stops in the driver's order and asks where each of our consecutive stops sits in it. It adds up how far each step jumps beyond 1 and scales the total by 2/(n(n−1)).
- **ERP (edit distance with real penalty)** turns our sequence into the driver's. Substituting stop a for stop b costs the normalised travel time from a to b. ERP_norm / ERP_edits is the average cost per edit.
- **Direction:** 0 is a perfect match and lower is better. The scorer's README says full random shuffles typically score 0.8–1.2.
- **Quirk (tested):** the exact reverse of the driver's route has SD = 0, so it **scores 0** even though it is the opposite order. SD only checks which stops are adjacent, not the direction.

`scripts/test_scorer.py` checks the official functions against hand-computed cases: an exact match gives 0, a swap gives 1/3, a reverse gives 0, and a missing stop gives the invalid score.

## Worked example

This route is the median of the 50 by shortest-route score, so it is neither a flattering nor an unflattering pick.

- Route `RouteID_d85ecd2b-e6a1-4abf-8f70-4c4346e946c4`, station DLA9 (stop `YY`), **160 stops** including the station.

| # | Driver stop | Driver zone | Shortest stop | Shortest zone |
|---|---|---|---|---|
| 1 | MH | G-24.3G | BG | G-24.2C |
| 2 | OK | G-24.3G | MA | G-24.3G |
| 3 | MJ | G-24.3G | MJ | G-24.3G |
| 4 | MA | G-24.3G | MH | G-24.3G |
| 5 | DK | G-24.3E | OK | G-24.3G |
| 6 | XH | G-24.3E | XH | G-24.3E |
| 7 | RP | G-24.3E | DK | G-24.3E |
| 8 | CA | G-24.3E | RP | G-24.3E |
| 9 | QD | G-24.3E | CA | G-24.3E |
| 10 | CC | G-24.3E | QD | G-24.3E |

- **Score 0.0763**, from SD 0.0538 × ERP per edit 1.418. The random order scores 0.825 on the same route.
- The driver's tour takes 16,473 s and the shortest tour 15,510 s, so the driver spends 6% longer.
- What to notice:
  - The solver first visits stop BG in a different zone (G-24.2C), which the driver leaves for later.
  - Then both visit the same four G-24.3G stops in a different order.
  - Then both move into zone G-24.3E, where the order is almost the same.
  - The broad zone order is right; the errors are detours to nearby stops and small swaps within a zone.

## How to rerun

Run from `tacit-routes/`. The `-I` flag keeps Python from importing anything from the working directory or the data folders.

```
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install numpy ortools ijson
.venv/bin/python -I scripts/test_scorer.py
.venv/bin/python -I scripts/fetch_sample.py --raw-dir data/raw --n 50 --seed 20261110
.venv/bin/python -I scripts/run_plain_baseline.py --sample-dir data/raw/sample --results-dir results --time-limit 10 --workers 3
```

- **The fetch** downloads route_data (79 MB), actual_sequences (10 MB) and invalid_sequence_scores (0.4 MB). It never loads the 1.8 GB travel-times file. Instead it predicts each sampled route's byte offset from stop counts, confirms it with the route ID found at that offset, and pulls only those routes (about 97 MB transferred, 68 s). Any mismatch in order, stop sets or permutations aborts the run.
- **Data quirk:** `route_data.json` contains bare `NaN` for some `zone_id` values, which is invalid JSON. The fetch rewrites it to null while streaming.
- **The baseline run** takes about 3.5 minutes. With a 10 s time limit, re-solving can return slightly different tours. The exact tours used are saved in `results/proposed_shortest.json`.

## Caveats

- 50 routes is a small sample, so the CI is about ±0.01.
- These are training routes. Any later model must not be trained on them if this sample is reused for comparison, or a fresh held-out split must be defined.
- The 10 s solver is not optimal, as the one route where the driver's tour was shorter shows. A stronger solver (LKH) could shift the score in either direction, because a shorter tour is not necessarily closer to the driver.
