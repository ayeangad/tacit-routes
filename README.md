# tacit-routes

Amazon delivery drivers often drive their stops in a different order from the shortest route, because they know things the map doesn't. This repo trains a small model on 5,762 of their past routes to predict that order. It then compares the model with Claude Opus 5.5 on 3,052 routes the model never saw.

Data: Amazon Last Mile Routing Research Challenge (2021). Metric: the challenge's official score; 0 means the driver's exact order, and lower is better.

## Result

The test was pre-registered, locked by hash, and scored once. Each difference below is my model minus the opponent, with a 95% paired bootstrap interval, so negative means mine was better.

| Comparison | Routes | Difference | Outcome |
|---|---|---|---|
| **Primary.** Opus 5.5 as the model: it picks the zone order, given history drawn from the same training routes (summarised in its prompt), and uses the same in-zone solver | 500 | −0.0023 [−0.0042, −0.0005]; better on 149 routes, worse on 94, tied on 257 | Win |
| Cost per route, for the same comparison | 500 | Opus (max effort): $0.51 and 211 s. Mine: 2.3 CPU-seconds | About 8,000× cheaper (assumptions below) |
| Design B. Opus 5.5 as an engineer: told the test facts, it wrote its own program in a sandbox with a 3-hour limit (it used about 1.5). My final model against runs 5 and 6 | 3,052 | +0.0015 [+0.0010, +0.0020] and +0.0010 [+0.0005, +0.0015] | Loss |
| Design A. Opus 5.5 as an engineer, not told the test facts (runs 2, 3, 4). My earlier model (v0-hist), frozen before the test | 3,052 | −0.0009 [−0.0016, −0.0002], +0.0032 [+0.0026, +0.0038], +0.0031 [+0.0026, +0.0037] | Loss (a win had to hold against all three) |
| The 2021 winner's code, run locally | 3,052 | +0.0011 [+0.0005, +0.0017] | Reference |

Mean scores: mine 0.0259, Opus as the model 0.0284 (0.0324 at default effort), Opus's programs 0.0240–0.0281, 2021 winner 0.0248.

So a model with about 147k weights beats a frontier model used directly on this task, at a fraction of the cost.

**Cost assumptions.** Opus's $0.51 is the per-call cost reported by the `claude` CLI at list price, for the answering call only; it is $0.52 with retries. Mine is inference only, priced at $0.10 per CPU-hour. Training is extra and happens once: about 2.4 CPU-hours across 5 seeds. Each Opus engineer run cost $2.74 (run 5) and $3.47 (run 6), also once.

Two post-hoc checks. They are descriptive and were computed after the test (`scripts/readme_posthoc.py`):
- **Opus as the model is not a weak opponent.** On the same 500 routes it beats both rule baselines: −0.0035 [−0.0058, −0.0011] against R2 and −0.0062 [−0.0084, −0.0040] against R4. It is level with my earlier v0-hist (+0.0006 [−0.0013, +0.0026]).
- **The size of the win depends on a few routes; its direction doesn't.** About half the mean gap comes from my 5 biggest wins, and without them the mean is −0.0012. Dropping the 25 most negative and the 25 most positive differences still leaves −0.0017 [−0.0025, −0.0009].

Full tables: `results/final_test.md`. Breakdowns, which are descriptive only (no claims): `results/final_descriptive.md`.

## Caveats

- **Route size.** Against Opus as the model, the advantage is −0.0041 on routes with few zones, −0.0030 on medium ones and +0.0010 on the 137 largest (35 better, 26 worse). There's no interval for these: the advantage disappears on large routes.
- **Time period.** Every method scores better on test routes from before the training period than after it, so the earlier routes are simply easier. Compared with the 2021 winner within each period: on routes after the training period my model ties it (0.0000), and on routes before it, the winner is ahead by 0.0022.
- **The 2021 leaderboard.** On its scale (1st 0.0248, 2nd 0.0353), 0.0259 falls between 1st and 2nd. My local run of the winner's code reproduces its published 0.0248, so the scale is comparable. But this was built in 2026 with the winners' papers available, and the model was hardened knowing the test is out of time. It is a calibration point, not a competition result.
- **Opus as the model was tested on 500 routes, not all 3,052,** because each route costs a $0.51 call. The 500 were drawn with a fixed seed, stratified by station and period, and chosen before any answer was opened.
- **Deviations from the pre-registration** are logged in `results/final_test.md` and `runs/final/incidents.log`: one listed arm (v1-past) was never built, and two descriptive arms were built while the machine was under load.

## How the test was kept honest

1. Every arm, comparison, win rule and fallback was written down before scoring, in `PREREGISTRATION.md`, with the plan and amendments in `docs/plan.md`.
2. `runs/final/LOCK_RECORD.json` holds the sha256 of the pre-registration, the config, the scoring scripts and every prediction file. `final_score.py` refuses to run unless the hashes match.
3. All predictions were made and hashed before the answer files were opened. The scorer ran once. Its output hash is in the lock record.
4. For design B, anything I knew about the test (dates, quality, that it is out of time) was also given to Opus runs 5 and 6 in their brief (`runs/opus_informed_brief/`). Runs 2–4 were blind by design, and they are compared only with my model that was frozen before I knew those facts.

To check the hashes:

```
python3 -c "import json,hashlib; r=json.load(open('runs/final/LOCK_RECORD.json')); [print(hashlib.sha256(open(f,'rb').read()).hexdigest()==h, f) for f,h in r['sha256'].items()]"
```

## The model

- **Per-zone features:** 65 features for each candidate next zone. 33 describe the route itself: distance, drive time, number of stops, and how the zone's ID relates to the current one. 32 come from history: how often drivers at this station went from one zone to another in training, at three levels of the zone ID.
- **Network:** a small pointer network (29,377 weights) scores each candidate. It has an encoder, one attention layer and a scoring head. Five seeds are averaged.
- **Zone order:** a beam search (width 5) builds the zone order.
- **Stops inside each zone:** exact Held–Karp for up to 12 stops, otherwise OR-Tools guided local search with a 1 s limit.
- **Training:** about 30 minutes of CPU per seed on a laptop.
- **History tables:** `runs/v1/history_train_L*.json`. They are counted separately from the weights.

## Layout

```
PREREGISTRATION.md      locked test plan
DECISION_LOG.md         why each choice was made
docs/plan.md            final-test plan and amendments, written before scoring
scripts/                data pipeline, models, baselines, Opus harnesses, scoring
results/                every reported number (.md for reading, .json for the numbers)
runs/final/             lock record, config, predictions, 500-route sample, incident log
runs/v1/                trained weights and history tables
runs/zone_rich/         frozen prompt for Opus as the model
```

## Reproducing

The data is not included; see the licence section below. It is public on S3 (`s3://amazon-last-mile-challenges/almrrc2021/`, anonymous access).

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -I scripts/build_routes.py --raw-dir data/raw --out-dir data/processed
.venv/bin/python -I scripts/make_splits.py ...        # see each script's docstring for arguments
.venv/bin/python -I scripts/build_eval_frozen.py --eval-dir data/eval_frozen
.venv/bin/python scripts/v1_model.py ...
.venv/bin/python scripts/final_predict.py ...
```

The in-zone solver's OR-Tools fallback has a 1 s time limit, so a rebuild can differ from the locked predictions on a few percent of routes. Two builds of the same arm differed on 90 of 3,052 routes, one of them on a loaded machine. The scored files are the hashed ones in `runs/final/predictions/`.

Each script's docstring gives its exact command. The Opus arms need the `claude` CLI and a subscription. The calls cost about $0.51 per route at max effort.

## Why the programs Opus wrote beat mine

A route's stops are grouped into zones. My model decides the zone order, then a solver orders the stops inside each zone.

To find which half the gap came from, I took each Opus program's zone order and ran it through my in-zone solver. With my solver, every Opus zone order still beat mine, by +0.0010 to +0.0023. Opus's own in-zone ordering was never better than mine. **The gap is entirely in the zone order.**

My model picks the next zone one step at a time, using a beam search. All five Opus programs do it differently, according to their own design notes (`APPROACH.md` in each sandbox). They score every pair of zones (run 2 with a 513-weight network) and then solve for the best tour over all zones at once with an ATSP solver (Held–Karp or OR-Tools). Run 3's notes say it tried a sequential model with beam search, like mine, and dropped it because it was worse. Run 5, the strongest, adds a step-by-step re-ranking on top of the tour. Many of my worst routes are ones where my zone order runs the driver's route backwards (15% of routes, against 11–13% for Opus's programs). That is the kind of error that choosing the whole tour at once should prevent. This is the next thing I'd build. It hasn't been tested.

## Licence

- **Code:** MIT (`LICENSE`).
- **Data:** Amazon's Last Mile Routing Research Challenge data is licensed CC BY-NC 4.0 (per its AWS Open Data registry page).
- **Weights and history tables (`runs/v1/`):** derived from the challenge data, so they carry the same CC BY-NC 4.0 terms (non-commercial). The MIT licence does not cover them. It isn't redistributed here. Predictions and results derived from it are shared for non-commercial research only.
- **Official scorer:** `scripts/official_rc_cli/` is the challenge's own `score.py` (rc-cli, commit fdafa50). It is included unchanged so scores are comparable, and the upstream licence applies.
- **2021 winner's code:** `jpt-amz` by Cook, Held and Helsgaun (MIT, commit dc4ebc0; paper arXiv 2112.15192). It was used for the reference arm and is not included here; `scripts/run_jpt.py` expects it in `vendor/jpt-amz`.
