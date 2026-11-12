# Opus fair fight, run 2 (clean, full-length setup)

Run on 2026-11-11 (04:18–05:30 IST). Numbers come from `results/opus_fight_run2.json`, made by `scripts/score_opus_fight.py --tag opus_fight_run2`. The Opus log is `runs/opus_fight_run2/transcript.jsonl` and the audit is `runs/opus_fight_run2/audit.json`. Run 1 (`results/opus_fight.md`) is unchanged and still reported as run 1.

## The one-paragraph version

This repeats run 1 with Opus starting from scratch in a new folder. Run 1 was cut off at 67 minutes by our memory-cap setup, not by anything Opus did; that setup is now fixed. This time Opus finished **on its own after 72 minutes**: it decided it was done, wrote `APPROACH.md` and stopped, with about 108 of the 180 minutes unused. Its predictor scored **0.0268** on the 300 validation routes (95% CI 0.0235 – 0.0303). Lower is better. The shortest route scores 0.0898, and Opus was better on 291 routes and worse on 9. Run 2 also beat run 1's 0.0349 on the same routes (217 / 83). The audit found nothing. Cost: $1.37 by the CLI's list-price cost field, with 34,580 output tokens (10,653 of them thinking).

## What changed from run 1

- **A fresh folder**, `opus_arena_run2/`, built by the same script (`scripts/setup_opus_arena.py opus_arena_run2 opus_fight_run2`).
  - It holds the same 5,762 training routes, the same 300 val inputs with order and quality removed (checked again), the same scorer and a new venv with the same package versions.
  - The README is identical to run 1's except for the folder name.
  - Opus could not see run 1's folder: run 1's arena was added to the sandbox's read and write deny lists and to the file-tool deny rules, and the fake-API test confirmed it was invisible.
- **The memory setup is fixed.** The systemd scope now has `OOMPolicy=continue`, so if one process is killed for using too much memory, the rest of the session (including Claude Code) keeps running. The limits are the same as before: 3 GB and 3.5 CPUs. I checked the fix on its own with a small test: with `continue`, a killed child left its parent running; with the old default, the parent was also stopped (exit 143, exactly what ended run 1). No process was killed for memory during run 2.
- **One added line in the prompt:** "any single process that grows beyond about 2.5 GB will be killed by the memory limit (your session itself keeps running)". The rest of the prompt is the same as run 1's: `runs/opus_fight_run2/prompt.txt`.
- **Everything else is the same:**
  - the same sandbox settings, apart from the folder path (`runs/opus_fight_run2/sandbox_settings.json`);
  - the same flags and the same account;
  - `--no-session-persistence` and `timeout 3h`;
  - launched once (`runs/opus_fight_run2/launch.sh`).
- **Containment re-tested before launch** with the local fake API (`runs/opus_fight_run2/diag/`), with no real API call:
  - all of run 1's escape tests failed again;
  - reading run 1's arena and `data/eval_frozen/` failed;
  - writing to run 1's arena failed;
  - normal work inside the new arena, and a 200 MB allocation, succeeded.

## What Opus built (from its `APPROACH.md` and code)

It used the same two-level plan as run 1, but this time **both** levels are learned:

1. **Zone order.**
   - Stops with no zone take the zone of their nearest stop.
   - For every pair of zones (a → b) it builds 14 features:
     - travel times between the zones;
     - per-station history: how often drivers in training went from a to b, counted at three levels of the zone ID (`A-2.2A`, `A-2.2`, `A-2`) and in both directions;
     - differences between the zone IDs;
     - a flag for zones never seen in that station's history.
   - A small network (14 → 32 → 1) learns to score "which zone comes next". It was trained on 2,500 training routes, with each route's own transitions left out of its own features, so it can't cheat by seeing itself.
   - The zone order is the cheapest loop under those scores. It is solved exactly (Held–Karp) for 10 or fewer zones, and otherwise by OR-Tools with a 0.3 s search.
2. **Stop order (new in run 2).**
   - A second network (12 → 48 → 48 → 1) learns "which stop comes next". Its features are travel times both ways, straight-line distance, nearest-neighbour ranks, a same-zone flag, packages, service time and a duplicate-location flag.
   - It was trained on 1,800 High-quality training routes. For each stop, the candidates were the remaining stops in the same or the next zone (at most the 30 nearest).
   - The stop order is found by OR-Tools (1 s search per route). Each arc costs −log(probability), and the zone order is a hard rule: an arc may only go to a stop in the same zone or the next zone.
3. **`predict.py`** loads `hist2.pkl`, `zmodel_mlp.pt` and `smodel.pt`, and runs on 3 processes in about 2.5 minutes.

**Its own check: 200 held-out training routes, mixed quality, none used for training**

| Variant | Score |
|---|---|
| Hand-weighted zone cost + travel-time tour | 0.074 |
| Learned zone model + travel-time tour | 0.072 |
| True zone order (oracle) + travel-time tour | 0.055 |
| True zone order + learned stop cost | 0.016 |
| **Learned zone model + learned stop cost (final)** | **0.034** (High 0.030, Medium 0.037) |

Its conclusion: most of the remaining error is in the **zone order**. With the true zone order the score would be 0.016.

What it tried and dropped:

- adding stop-model summaries to the zone features (no gain);
- soft zone constraints that let the route leave a zone and come back (worse, 0.050);
- plain distance or squared travel times as the stop cost (no gain).

The val score (0.0268) is better than its held-out estimate on mixed-quality routes (0.034), and close to its High-only figure (0.030). That fits: all val routes are High quality.

## The score (300 validation routes, lower is better)

| Arm | Mean | Bootstrap 95% CI | Median | Valid |
|---|---|---|---|---|
| **Opus run 2** | **0.0268** | 0.0235 – 0.0303 | 0.0153 | 300 / 300 |
| Opus run 1 (`opus_fight.md`) | 0.0349 | 0.0309 – 0.0391 | 0.0220 | 300 / 300 |
| Plain shortest route | 0.0898 | 0.0840 – 0.0958 | 0.0766 | 300 / 300 |

Paired on the same routes (first minus second; negative = first better):

| Comparison | Mean difference [95% CI] | Median difference | First better / worse / tied | Exact sign test p |
|---|---|---|---|---|
| Run 2 − shortest | −0.0630 [−0.0676, −0.0582] | −0.0561 | 291 / 9 / 0 | 4.9 × 10⁻⁷⁴ |
| Run 2 − run 1 | −0.0080 [−0.0105, −0.0055] | −0.0042 | 217 / 83 / 0 | 5.6 × 10⁻¹⁵ |
| Run 2 − shortest, ≤ 146 stops (151, descriptive) | −0.0616 [−0.0688, −0.0546] | | 145 / 6 | 1.1 × 10⁻³⁵ |
| Run 2 − shortest, > 146 stops (149, descriptive) | −0.0644 [−0.0707, −0.0583] | | 146 / 3 | 1.5 × 10⁻³⁹ |

Scoring was done exactly as for run 1:

- the official, unmodified `evaluate()`;
- the true val orders from outside the arena;
- travel times rounded to 0.1 s;
- invalid sequences would have received the challenge's invalid score (there were none);
- bootstrap of 10,000 route resamples, with seed 20261110.

The shortest-route scores are the saved `results/v0_shortest_val.json` tours, re-scored with an exact match. The predictions were scored as written by Opus's own `predict.py` run (05:29 IST), which is newer than every model and code file it loads, so it was not re-run. The scored copy is `runs/opus_fight_run2/val_predictions_opus_copy.json`, with a SHA-256 identical to the arena file. Its `predictions_mtime_utc` field in the JSON is the time of the copy, not of Opus's run.

## Audit

The same script as run 1 (`scripts/audit_opus_fight.py`), now stricter: only run 2's own arena and its own Claude Code temp folder count as inside. I also read every command by hand.

- **36 tool calls:** 32 Bash and 4 Write. The Writes were `common.py`, `zfeat.py`, `sfeat.py` and `APPROACH.md`, all inside the arena.
- **No paths outside the arena**, except Opus reading its own background jobs' output files in its own temp folder.
- **No network attempts, no sandbox violations, no refused commands, nothing touching run 1's folder.**
- **No attempt to find val answers.** Opus read the val **inputs** for statistics (stations, how many zones appear in training), and at the end it checked that its predictions were valid sequences (every stop exactly once, station first). That uses inputs only, not answers.
- **Verdict: clean; the result stands.**

## Usage and duration

- **Wall clock:** 71 min 38 s, from 22:48:40 to 00:00:18 UTC. The CLI reports 4,296,544 ms, of which 328 s was API time.
- **Turns:** 37. The session ended normally (`success`, `end_turn`).
- **Tokens:**
  - input: 74 uncached, 50,521 cache writes, 1,371,736 cache reads;
  - **output: 34,580, of which 10,653 were thinking.**
- **Cost:** **$1.37** (CLI cost field, list price).
- **CPU:** 2 h 31 min in total. There were no memory kills.
- **Opus calls under the approval so far:** run 1 (3 launches: 2 short false starts plus 1 real run) and run 2 (1 launch).

## Caveats

- **Opus stopped early by its own choice.** It used 72 of 180 minutes. This is its "best effort" as it judged it, not a run limited by time.
- **One run each.** The run 1 → run 2 gain of 0.008 is clearly real *for these two runs*: 217 of 300 routes improved. But it mixes two things: the fixed setup, and ordinary run-to-run randomness in what Opus decides to build. The two runs built different methods. With two runs, we can't separate the effect of luck.
- **The searches are time-limited** (0.3 s for zones, 1 s for stops), so re-running `predict.py` on a busy machine could change predictions slightly.
- **The same caveats as run 1 still apply.**
  - Opus had no internet, no project notes and no papers.
  - It saw the val inputs, but not their quality label.
  - It tuned on its own held-out training routes.
  - It had a small CPU and memory budget on a loaded machine.
- **This is the strongest Opus opponent so far.** 0.0268 is below the published 0.0374, but that figure is on a different, official evaluation set, so the two are not comparable. To claim honestly that "a small model beat Opus", the model must beat this run (and ideally run 1 too) on these 300 routes, or on the frozen test set with both run on it.
