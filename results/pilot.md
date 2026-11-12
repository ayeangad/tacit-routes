# Pilot: DeepSeek and Opus ordering stops directly

Runs on 2026-11-10 (DeepSeek, all 50 routes; Opus, first 5) and 2026-11-11 (Opus, remaining 45). All numbers come from `results/pilot.json`, built by `scripts/score_pilot.py` from the raw call logs `runs/pilot/deepseek.jsonl` and `runs/pilot/opus.jsonl`.

## What ran

- **Routes:** the same 50 High-quality training routes as in `plain_baseline.md`. Both DeepSeek and Opus ran all 50, one call per route.
- **Prompt:** one per route, identical for both models (`scripts/build_prompts.py`, saved in `runs/pilot/prompts/`).
  - It contains the depot ID and, for every location, its ID, latitude, longitude and zone ID, plus its 5 nearest other locations by supplied travel time (in seconds).
  - It asks for the order an experienced driver would visit every stop in, starting at the depot, as a JSON list of IDs and nothing else.
  - The system prompt is frozen in `runs/pilot/system_prompt.txt`.
- **Leakage controls:** locations are listed in a seeded random order (`random.Random(f"20261110:prompt:{route_id}")`). The prompt builder never opens the actual sequences, invalid scores or quality labels.
- **Prompt size:** 504,203 characters across the 50 prompts, mean 10,084 (range 4,779 to 13,088). Input tokens reported by the APIs, mean per route: DeepSeek 5,308; Opus 7,596 (prompt plus system prompt).
- **DeepSeek:**
  - Model `deepseek-flash` (the API returned `deepseek-flash`), the same ID atelier used.
  - Temperature 0, max_tokens 4096, thinking disabled (atelier's setting).
  - Cost is at atelier's peak list prices.
  - A thinking-on rerun was approved on 2026-11-11 and is running separately. Its results are not in this file.
- **Opus:** `claude -p --model claude-opus-5-5` with the approved flags, run one call at a time, each from a fresh empty temp directory.
  - All 50 calls returned model `claude-opus-5-5` with `stop_reason` `end_turn`. Each needed one attempt, and every temp directory was empty afterwards.
  - Opus used extended thinking on every call. The arms are therefore **not** matched on reasoning.
- **Parsing:** the first JSON list in the output, with stop IDs only. An answer is invalid if it is missing stops, has extra or duplicate stops, or does not start at the depot. An invalid answer is left out of the proposal file, so the official `evaluate()` gives it the challenge's invalid-sequence score. Our validity flag matched the official feasibility flag on every route.
- **Scorer:** the same official, unmodified rc-cli `score.py`, called through `evaluate()` as in `run_plain_baseline.py`. Re-scoring the saved shortest tours (`results/proposed_shortest.json`) reproduces every baseline route score exactly.

## Results, all 50 routes (lower is better)

| Arm | Routes | Mean score | Bootstrap 95% CI | Median | Valid answers |
|---|---|---|---|---|---|
| Opus, direct order | 50 | **0.0693** | 0.0581 – 0.0814 | 0.0672 | 50 / 50 |
| Shortest route (plain baseline) | 50 | 0.0784 | 0.0690 – 0.0882 | 0.0750 | 50 / 50 |
| DeepSeek, direct order (thinking off) | 50 | 0.8864 | 0.7945 – 0.9806 | 0.8771 | 1 / 50 |

Paired comparisons on the same 50 routes (arm minus comparison, negative = arm better). The sign test is exact, two-sided, with ties dropped.

| Comparison | Mean difference | Bootstrap 95% CI | Median difference | Arm better / worse / tied | Sign test p |
|---|---|---|---|---|---|
| Opus − shortest | −0.0090 | −0.0192 – +0.0015 | −0.0128 | 30 / 20 / 0 | 0.203 |
| DeepSeek − shortest | +0.8081 | +0.7185 – +0.8994 | — | 1 / 49 / 0 | — |
| Opus − DeepSeek | −0.8171 | −0.9080 – −0.7272 | −0.7987 | 49 / 1 / 0 | 9.1e-14 |

### Opus vs shortest by stop count (descriptive only)

The split is at the median stop count over the 50 routes, 147.5. It was not pre-registered, and no correction is made for looking at two groups.

| Group | Routes | Stops | Opus mean | Shortest mean | Mean diff [95% CI] | Opus better / worse | Sign test p |
|---|---|---|---|---|---|---|---|
| ≤ 147 stops | 25 | 63 – 147 | 0.0612 | 0.0786 | −0.0174 [−0.0311, −0.0046] | 18 / 7 | 0.043 |
| > 147 stops | 25 | 148 – 192 | 0.0775 | 0.0781 | −0.0007 [−0.0149, +0.0162] | 12 / 13 | 1.0 |

## Validity

**Opus:** 50 of 50 valid.

**DeepSeek, 49 of 50 invalid:**

| Reason | Routes |
|---|---|
| No JSON list found: the output hit max_tokens (4,096) before closing the list | 34 |
| Missing stops and duplicated stops | 11 (3 of them also repeat the depot) |
| Missing 1 to 4 stops, no duplicates | 3 |
| One duplicated stop | 1 |

- In all 34 truncated outputs, a stop ID repeats before the list has covered every stop. The output then cycles through the same IDs, so a larger max_tokens would not have made these valid.
- 46 of 50 DeepSeek outputs repeat at least one stop ID.
- DeepSeek's one valid answer (route `974820df…`) scored 0.0837. On the same route, the shortest route scored 0.0891 and Opus 0.0851.
- DeepSeek's mean is essentially the challenge's invalid scores, which average 0.9028 on its 49 invalid routes.

## Size, length, latency and cost

| | DeepSeek (50) | Opus (50) |
|---|---|---|
| Mean prompt characters | 10,084 | 10,084 |
| Mean input tokens | 5,308 | 7,596 |
| Mean output tokens | 2,899 | 3,766 (of which thinking 3,305) |
| Mean output characters | 6,437 | 724 |
| Mean wall time per call | 9.9 s | 31.9 s (CLI `duration_ms` mean 31.1 s) |
| Cost | **$0.2535** total at list price (cap $1.00) | **$6.80** total in the `total_cost_usd` field |

Opus usage over all 50 calls (from the CLI's JSON):

- **Input:** 379,790 tokens in total. 100 were uncached, 379,690 were written to the cache and 0 were read from it. Per call: mean 7,596, range 3,770 to 9,797.
- **Output:** 188,287 tokens in total, of which 165,272 were thinking. Per call: mean 3,766, range 1,243 to 8,124.
- **Cost field:** $6.8037 in total. Per call: mean $0.136, range $0.055 to $0.238. This is the CLI's list-price estimate (`costBasis: "list"`), not necessarily what the account is billed.
- **Duration:** `duration_ms` totals 1,555.8 s. Per call: mean 31.1 s, range 10.8 to 68.3 s.
- Per-call figures are in `results/pilot.json` under `arms.opus.per_route`.

## Earlier pilot: Opus on the first 5 routes only (2026-11-10)

These numbers were reported before the other 45 routes were run. They are kept for the record. The 5 routes are the first 5 in the manifest (sorted by route ID), not a random draw, and n = 5 is too small for conclusions.

| | Routes | Mean | Bootstrap 95% CI |
|---|---|---|---|
| Opus | 5 | 0.0524 | 0.0193 – 0.0885 |
| Shortest, same 5 routes | 5 | 0.0698 | 0.0359 – 0.1094 |
| DeepSeek, same 5 routes | 5 | 0.8071 | 0.4895 – 1.1247 |

- Opus minus shortest: −0.0174 [−0.0286, −0.0026]. Opus was better on 4 routes and worse on 1 (sign test p = 0.375).
- Opus minus DeepSeek: −0.7547 [−1.0584, −0.4510]. Opus was better on 5 routes and worse on 0.
- The Opus mean on all 50 routes (0.0693) is higher than on these first 5 (0.0524).

| Route | Stops | Input tokens | Output tokens (thinking) | `total_cost_usd` | `duration_ms` | Opus | Shortest |
|---|---|---|---|---|---|---|---|
| 13ad9c2a | 63 | 3,770 | 1,243 (1,040) | 0.0550 | 10,774 | 0.0041 | 0.0382 |
| 1a4903de | 150 | 7,870 | 3,116 (2,637) | 0.1253 | 25,921 | 0.0388 | 0.0273 |
| 1d2aa8bd | 100 | 5,508 | 2,913 (2,597) | 0.1023 | 24,035 | 0.0726 | 0.0982 |
| 21662e0b | 131 | 6,953 | 3,316 (2,899) | 0.1219 | 27,112 | 0.0247 | 0.0483 |
| 221006e8 | 146 | 7,673 | 3,856 (3,385) | 0.1385 | 31,278 | 0.1217 | 0.1368 |

## Worked example

This is the same median route as in `plain_baseline.md`: route `RouteID_d85ecd2b-e6a1-4abf-8f70-4c4346e946c4`, station DLA9 (depot `YY`), **160 stops** including the depot.

- Scores: driver 0, shortest 0.0763, **Opus 0.1057**, DeepSeek 0.8776 (invalid; the challenge's invalid score for this route).
- DeepSeek's answer was cut off at the token limit after repeating stops, so it has no parsed list. Its column shows the first 10 IDs after the depot from its raw output, for illustration only. They were not scored.

| # | Driver | Zone | Shortest | Zone | DeepSeek (raw, invalid) | Zone | Opus | Zone |
|---|---|---|---|---|---|---|---|---|
| 1 | MH | G-24.3G | BG | G-24.2C | MA | G-24.3G | OK | G-24.3G |
| 2 | OK | G-24.3G | MA | G-24.3G | OK | G-24.3G | MH | G-24.3G |
| 3 | MJ | G-24.3G | MJ | G-24.3G | MH | G-24.3G | MJ | G-24.3G |
| 4 | MA | G-24.3G | MH | G-24.3G | MJ | G-24.3G | MA | G-24.3G |
| 5 | DK | G-24.3E | OK | G-24.3G | DK | G-24.3E | IK | G-23.1D |
| 6 | XH | G-24.3E | XH | G-24.3E | XH | G-24.3E | TJ | G-23.2D |
| 7 | RP | G-24.3E | DK | G-24.3E | RP | G-24.3E | PQ | G-23.2D |
| 8 | CA | G-24.3E | RP | G-24.3E | CA | G-24.3E | KX | G-23.2D |
| 9 | QD | G-24.3E | CA | G-24.3E | QD | G-24.3E | DP | G-23.2D |
| 10 | CC | G-24.3E | QD | G-24.3E | CC | G-24.3E | WQ | G-23.2D |

- Opus starts with the driver's first 4 stops, in a slightly different order.
- Opus's stops 5 to 10 are among the driver's last stops (driver positions 147 to 152).
- The driver's stops 5 to 10 come at Opus positions 83 to 88.
- On this route Opus scores worse than the shortest route.
- DeepSeek's raw first 10 IDs are the driver's first 10 stops, with the first four swapped around. Its full output repeats stops and was never completed, so it scored as invalid.

## Caveats

- **The arms are not matched on reasoning.** Opus thought before answering (mean 3,305 thinking tokens), while DeepSeek ran with thinking off at temperature 0. The DeepSeek thinking-on rerun is running separately.
- **The DeepSeek result is mostly a validity result.** 49 of 50 answers were invalid.
- **Opus vs shortest is not significant on 50 routes.** The paired CI includes 0 and the sign test gives p = 0.20. The stop-count split is descriptive and was not pre-registered.
- **One call per route, no retries.** Repeat calls could give different answers. That applies to Opus especially, since its temperature is not under our control.
- **Inherited environment:** all 50 Opus calls inherited the parent Claude Code session's environment variables, including `CLAUDE_EFFORT=medium`. The approved command unsets only `ANTHROPIC_API_KEY` and `CLAUDE_CODE_EFFORT_LEVEL`. This was kept identical across all 50 calls so they are comparable, but Opus's effort level may have been medium.
- **The sample is from the training side** and does not compare directly with published evaluation-set scores (see `plain_baseline.md`).

## How to rerun

Run from `tacit-routes/`:

```
.venv/bin/python -I scripts/build_prompts.py --sample-dir data/raw/sample --out-dir runs/pilot
.venv/bin/python -I scripts/run_pilot.py --model deepseek --pilot-dir runs/pilot --sample-dir data/raw/sample --cap-usd 1.00
.venv/bin/python -I scripts/run_pilot.py --model opus --pilot-dir runs/pilot --sample-dir data/raw/sample
.venv/bin/python -I scripts/score_pilot.py --sample-dir data/raw/sample --pilot-dir runs/pilot --results-dir results
```

`run_pilot.py` is resumable: it skips any route that already has a successful line in the JSONL. The DeepSeek key is read at runtime from `~/.deepseek_key`, where atelier's client reads it.
