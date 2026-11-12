# The 2021 winner (jpt-amz), run on our 300 validation routes

Numbers come from `results/jpt_baseline.json`. Lower scores are better, and 0 means the driver's exact order.

## The result in one paragraph

On our 300 validation routes, the 2021 winning code scores **0.0213** (95% CI 0.0183 to 0.0246; median 0.0105). All 300 of its answers are valid. The plain shortest route scores **0.0898** on the same routes. The winner beats it on **294 of 300 routes** (6 worse, exact sign test p ≈ 1e-78), by a mean of −0.0685 (CI −0.0732 to −0.0639). It costs about **2.7 CPU-seconds per route** on this laptop, with no GPU and no API calls. This is the number any model of ours has to beat or add to.

## What the method does (from zero)

A driver's day is a list of about 140 stops. We have to guess the order the driver actually drove them in. The official score compares our order with the driver's order: how far each stop is out of place, weighted by how far apart the stops are.

The winner (team "Just Passing Through": Cook, Held, Helsgaun) uses **no neural network**. It has two halves.

1. **Build (reading the past).** It reads past routes and their driver orders. Every stop has a zone ID such as `A-2.3D`. The name has levels, like a postal address: `A` is a big area, `2` a sub-area, and `3D` a small block. From past routes it learns three things:
   - which past routes from the same station look like a new route (they share the most zones);
   - in which order drivers swept through each area's zones (often sorted or reverse-sorted by ID);
   - how big areas ("super-clusters") were chained together.
   It also fills in missing zone IDs. This is counting and bookkeeping, not training.
2. **Apply (solving each new route).** Each new route becomes a puzzle for **LKH**, a very strong solver for the travelling salesman problem (find the shortest loop through all stops). The puzzle adds rules:
   - finish a zone before leaving it;
   - visit zones in an order consistent with the most similar past route at the same station;
   - follow the zone-ID sweep pattern.
   LKH then finds the shortest tour that obeys the rules. It does this twice with two slightly different rule sets. It keeps the first tour unless the second is more than 1% shorter.

The intuition is that drivers are close to shortest *within the habits they have*. The habits are "do zones as blocks, in a familiar order", and history plus zone names reveal those habits.

## What we ran

- **Code:** github.com/heldstephan/jpt-amz at commit `dc4ebc0a`, MIT licence, cloned into `vendor/jpt-amz/`. I read the scripts and C sources first: there is no network access and nothing is written outside the run folder.
- **Local changes:** threads cut from 16 to 3, `gawk` replaced by `awk`, and an optional time-limit override. The method itself is untouched (`runs/jpt/LOCAL_CHANGES.txt`, `runs/jpt/local_changes.diff`). The solver was built with the system gcc 13.3, with no source edits. natsort 8.4.0 was installed into `runs/jpt/pydeps/` only.
- **Build data:** only the 5,762 **train** routes, with all four files.
- **Apply data:** only the 300 **val** routes, with no driver orders and the quality label removed.
- **Excluded:** the 50 pilot routes appear nowhere, and `data/eval_frozen/` was not touched. Checked by listing the route IDs in every input file (`runs/jpt/logs/input_checks.json`).
- **Settings:** their own `model_build.sh` and `model_apply.sh`, with LKH limited to **1 s + 1 s per route**. That is the setting of the paper's best training-side result. In the contest, their script gave each route about 42 s + 20 s; we did not run that (see "Not done").
- **Scoring:** the unmodified official scorer, exactly as in `run_plain_baseline.py`. As a wiring check, the driver's own order scores 0.

## Scores

| On our 300 val routes | Mean [95% CI] | Median | Valid |
|---|---|---|---|
| jpt-amz, 1 s + 1 s | **0.0213** [0.0183, 0.0246] | 0.0105 | 300/300 |
| Shortest route (OR-Tools, 10 s) | 0.0898 [0.0840, 0.0959] | 0.0766 | 300/300 |

Paired, on the same routes:
- jpt minus shortest: −0.0685 [−0.0732, −0.0639]; jpt better on 294, worse on 6, p ≈ 1e-78.
- The shortest-route scores are the stored orders from `results/v0_shortest_val.json`, re-scored here. They match the stored scores to 1e-16.
- **Best v0 variant and best rules arm: not compared yet.** `results/v0.json` and `results/rules_baseline.json` did not exist when this was scored. Rerun the scoring command below once they exist; it adds those comparisons automatically.

The error sits in a tail. The median route is about half the mean, 37 of 300 routes score above 0.05, and the worst 15 routes (5%) carry 25% of the total error. That is the gap a learned model could attack.

## Against the paper's numbers (do not mix these two)

| Where | Score | What it is |
|---|---|---|
| **Our val (this run)** | **0.0213 mean / 0.0105 median** | 300 High training-side routes, held out from build, 1 s + 1 s |
| Paper, training side | 0.0198 mean / 0.0073 median | 1,107 High, fully delivered training routes, 1 s + 1 s (from a summary of arXiv 2112.15192, not read verbatim) |
| Official evaluation side | 0.0248 (2nd 0.0353, 3rd 0.0391) | Hidden evaluation set of about 3,000 routes, contest time limits (team page, confirmed) |

- **Same kind of data, slightly worse.** Our val routes are like the paper's training-side set (High routes from the training side). We get 0.0213 against their 0.0198. Our routes were left out of the build data, while theirs probably were not, which may explain part of the gap. Our set is also smaller and drawn differently.
- **Not comparable to 0.0248.** That number comes from a different set of routes under different time limits. The frozen evaluation set (`data/eval_frozen/`) is where a like-for-like number would come from, in the single pre-registered final run.

## Cost and latency

| | Value |
|---|---|
| Build (5,762 routes) | 313 s wall, 151 CPU-s, peak 575 MB (includes recompiling LKH) |
| Apply (300 routes), total | 570 s wall on 3 threads, 810 CPU-s, peak 1.9 GB for the whole process tree |
| Per route, CPU | 2.7 s: LKH 1.0 s + 1.0 s (both passes), plus about 0.7 s of Python setup and loading |
| Per route, wall | about 1.9 s at 3 threads, on a laptop already busy with other jobs |
| Money | $0, no API calls |

## Not done

- **Contest-length time limits.** The contest gave about 42 s + 20 s per route. This would take about 2 hours of 3 threads here, so we did not run it. If Angad wants it, the rerun command is below.
- **Comparisons with v0 and the rules arms**, which wait on their results files.

## How to rerun

From `tacit-routes/`:

```
.venv/bin/python -I scripts/prepare_jpt_inputs.py --raw-dir data/raw --splits data/processed/splits.json --out-dir runs/jpt/data   # about 16 min
mkdir -p runs/jpt/data/model_build_outputs runs/jpt/data/model_apply_outputs
python3 scripts/run_jpt.py build --run-dir runs/jpt                                   # about 5 min
python3 scripts/run_jpt.py apply --run-dir runs/jpt --t1 1 --t2 1 --tag t1_1          # about 10 min
.venv/bin/python -I scripts/score_jpt_baseline.py --run-dir runs/jpt --tags t1_1 --primary t1_1 --results-dir results   # about 6 min
```

For contest-length limits, use `--t1 42 --t2 20 --tag t42_20`, then score with `--tags t1_1 t42_20 --primary t1_1`.

`runs/jpt/src` is a symlink to `vendor/jpt-amz`, because their scripts expect `src/` and `data/` side by side. LKH is seeded (`SEED = 1`), and its time limit counts CPU time, so reruns should give the same tours, as long as each LKH run gets the same amount of work done in its 1 CPU-second.
