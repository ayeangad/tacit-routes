# Out-of-time rehearsal on validation routes (DIAGNOSTIC ONLY)

Made by `scripts/rehearsal_oot.py`; numbers in `results/rehearsal_oot.json`; exclusion sets in `runs/rehearsal/exclusions.json`. **Not a decision tool and not a test between methods.** 76 targets, so paired intervals are wide. Read how much each method degrades.

Targets: val routes from the first and last week of the training window. Thinned: train routes within the 11-day gap (group-wise) removed, leaving every target at least 12 days from its history. Control: the same number of train routes removed at random (same volume, no gap). Lower is better.

| Arm | Full | Control (volume only) | Thinned (gap) | Thinned − full [95% CI] | Thinned − control: the gap effect [95% CI] |
|---|---|---|---|---|---|
| v0hist | 0.0229 | 0.0288 | 0.0288 | +0.0059 [+0.0022, +0.0099] | +0.0000 [-0.0051, +0.0044] |
| v0past | 0.0282 | 0.0306 | 0.0338 | +0.0056 [+0.0014, +0.0103] | +0.0032 [-0.0011, +0.0081] |
| v1 | 0.0221 | 0.0253 | 0.0260 | +0.0039 [+0.0011, +0.0072] | +0.0007 [-0.0048, +0.0056] |

Groups: {"early": {"n_targets": 55, "cut_date": "2018-08-05", "train_routes_removed": 2776}, "late": {"n_targets": 21, "cut_date": "2018-08-09", "train_routes_removed": 2415}}
