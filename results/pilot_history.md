# Opus direct + history: results on 50 of 50 pilot routes

Produced by `scripts/score_pilot_history.py` from `runs/pilot_history/opus.jsonl`; every number here is in `results/pilot_history.json`.

## What was tested, from zero

Each pilot route is one real van trip: a depot and 60–190 stops. The driver visited the stops in some order, and we know that order. Each competitor guesses the order, and the official challenge scorer compares the guess with the driver's real order. **0 means a perfect match, and lower is better.**

- **Opus direct** (already run, `results/pilot.md`): Opus sees only this route's stops: their IDs, map positions, zone labels and nearest neighbours.
- **Opus + history** (this run): the same prompt, word for word, plus a short list of up to 3 other routes from the same depot that share the most zone labels. Each one shows the order its driver visited those zones in. This is the same kind of history our own model learns from.
- **Shortest route:** a solver that just finds a short tour and knows nothing about drivers.

The question: does showing Opus how drivers at this depot usually move through the zones make its guess closer to the real driver?

## Main results (lower is better)

| Arm | Routes | Mean score | Bootstrap 95% CI | Median | Valid answers |
|---|---|---|---|---|---|
| Opus + history | 50 | 0.0355 | 0.0279 – 0.0437 | 0.0252 | 50 / 50 |
| Opus direct | 50 | 0.0693 | 0.0581 – 0.0814 | 0.0672 | — |
| Shortest route | 50 | 0.0784 | 0.0690 – 0.0882 | 0.0750 | — |

**How to read the CI.** We resample the routes 10,000 times. The range holds the middle 95% of the resampled means. A wide range means the mean could easily have come out differently with other routes.

## Paired comparisons (first minus second; negative = first is better)

Every route is scored under both arms, so we subtract the two scores route by route and average the differences. This removes the fact that some routes are simply harder than others.

The **sign test** asks: if neither arm were really better, how surprising is this win/loss split? It is like flipping a fair coin once per route. A small p (below about 0.05) means the split would be rare by luck alone.

| Comparison | Mean difference | 95% CI | Median | First better / worse / tied | Sign test p |
|---|---|---|---|---|---|
| Opus+history − Opus direct | -0.0338 | [-0.0435, -0.0245] | -0.0254 | 44 / 6 / 0 | 3.24e-08 |
| Opus+history − shortest | -0.0428 | [-0.0511, -0.0343] | -0.0441 | 45 / 5 / 0 | 4.21e-09 |
| Opus direct − shortest (same routes) | -0.0090 | [-0.0192, +0.0015] | -0.0128 | 30 / 20 / 0 | 0.203 |

## Zone revisits

A zone is a small area with a label such as `D-16.2C`. A **revisit** means the route leaves a zone and later comes back to it. Real drivers almost never do this. The method is `scripts/zone_revisits.py`'s: drop the depot and stops without a zone, merge back-to-back stops in the same zone into one visit, and count visits minus distinct zones.

Routes where both Opus answers are valid: 50.

| Order | Mean revisits | Median | Routes with 0 |
|---|---|---|---|
| Driver | 0.96 | 0 | 32 |
| Opus direct | 10.66 | 10 | 1 |
| Opus + history | 1.98 | 1 | 24 |

Opus+history − Opus direct revisits: mean -8.68, 95% CI [-9.98, -7.38], fewer on 48, more on 1, tied on 1 (sign test p = 1.78e-13).

## Worked route

`RouteID_ae7545f8-6535-4710-b6ea-30b4fb64cc02` (station DLA9, 130 locations). Chosen as the route with the median (Opus+history − Opus direct) score difference, so it is a typical route for this comparison, not the best or the worst.

| | Score | Zone revisits |
|---|---|---|
| Driver (the answer) | 0 | 0 |
| Shortest route | 0.0720 | — |
| Opus direct | 0.0598 | 10 |
| Opus + history | 0.0349 | 0 |

Zones in the order each one first enters them:

- **Past route #1 shown to Opus (2018-08-08, shares 19 zones):** G-7.2B G-7.1B G-7.1C G-7.2C G-7.3C G-7.3D G-7.2D G-7.1D G-7.1E G-7.2E G-7.3E G-7.3G G-7.2G G-7.1G G-7.1H G-7.2H G-7.3H G-7.3J G-7.2J
- **Real driver:** G-7.3C G-7.3D G-7.1C G-7.2C G-7.2B G-7.1B G-7.2D G-7.1D G-7.1E G-7.2E G-7.3E G-7.3G G-7.2G G-7.1J G-7.3H G-7.2H G-7.1G G-7.1H G-7.3J G-7.2J
- **Opus direct:** G-7.1J G-7.2J G-7.3J G-7.3H G-7.2H G-7.1G G-7.1H G-7.2G G-7.3G G-7.3E G-7.2E G-7.1E G-7.1D G-7.2D G-7.1B G-7.2B G-7.2C G-7.1C G-7.3C G-7.3D
- **Opus + history:** G-7.2B G-7.1B G-7.1C G-7.2C G-7.3C G-7.3D G-7.2D G-7.1D G-7.1E G-7.2E G-7.3E G-7.3G G-7.2G G-7.1G G-7.1H G-7.2H G-7.3H G-7.3J G-7.2J G-7.1J

Things to compare: does Opus + history follow the past route's zone order more closely than Opus direct does? Is the past route itself close to what the real driver did?

## Cost and time per call

- Cost field from `claude -p` (`total_cost_usd`): total $6.05, mean $0.121 per call.
- Mean tokens per call: input 8247, output 2754 (of which thinking 2294).
- Mean duration: 23.0 s reported by the CLI, 25.0 s wall clock.
- Models returned: ["claude-opus-5-5"]; stop reasons: end_turn.

## Caveats

- 50 routes. The comparisons were not pre-registered as a final test, and the pilot routes are the frontier-comparison sample.
- Opus uses extended thinking. Each arm is one sample per route, so run-to-run randomness in Opus is part of the difference.
- The past routes come from the TRAIN split, picked by shared zones. Some are dated after the pilot route (the split is random, not by date). Our v0-hist model has the same information.
- One DLA4 route had no past route sharing any zone, so its prompt contained no history.
