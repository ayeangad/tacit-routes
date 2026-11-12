<!--
DRAFT. The informed README = the blind README (runs 2–4) word for word, with ONLY these three
sections inserted before "## Rules and limits" (agreed rule, docs/plan.md
item 4). Arena folder name substituted as before. The coverage bullet below was filled from
results/eval_coverage.md (zone IDs and stations only); nothing else may be added.
-->

## Addition 1: what your predictor will finally be tested on

After this session, your `predict.py` will be run once, unchanged, on a separate final test set that you cannot see. These are facts about it:

- **About 3,000 routes** (3,052), from the same challenge: the official evaluation routes of the 2021 Amazon Last Mile Routing Research Challenge. They are scored with the same official scorer, as a plain mean over routes.
- **All routes are High quality.**
- **Out of time.** The final-test routes are dated 2018-06-01 to 2018-10-02. Your training routes cover only 2018-07-19 to 2018-08-26. About half the final-test routes are dated before all training routes (up to about 48 days before), about half after (up to about 37 days after), and only 36 fall inside the training dates.
- **Stations.** 18 stations: the 17 that appear in training, plus 1 station with no training routes at all (1 route).
- **Size.** 19 to 225 stops per route, mean 141.9.
- **History coverage** (computed from the final-test inputs, zone IDs only): against all training routes at the same station, a final-test route has on average 81% of its zone IDs seen in training (validation routes: 90%); at the `A-2.2` level 93% (val 97%); at the `A-2` level 93% (val 97%). The single most similar training route shares on average 72% of its zones (val 82%). Routes dated after the training period are the thinnest: 74% of zone IDs seen, 65% best single reference, and about 1.7 zones per route never seen at that station.

The 300 validation routes in this folder are dated inside the training period, so a score on them will likely be better than on the final test. Plan for the final test.

## Addition 2: reading list

`READING_LIST.md` in this folder summarises the published work on this challenge: what the winning and later methods did and how they scored. You cannot open its links (no internet); the summaries are what you have.

## Addition 3: predict.py interface

`predict.py` must accept an input folder and an output path:

```
.venv/bin/python predict.py <input_dir> <output_path>
```

`<input_dir>` contains route files in exactly the format of `data/val_inputs/` (no driver order, no quality field). It must write `<output_path>`: a JSON object mapping each route ID found in `<input_dir>` to its predicted stop sequence (same rules as before). With no arguments it must default to `data/val_inputs/` and `val_predictions.json`, as before. It may load only files you saved in this folder. It must give a valid sequence for every route, including routes from a station it has never seen. It should process about 3,000 routes within about 3 hours on 3 CPU threads (about 3.5 s per route).
