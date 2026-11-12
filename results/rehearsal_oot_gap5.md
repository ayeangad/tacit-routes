# Out-of-time rehearsal on validation routes (DIAGNOSTIC ONLY)

Made by `scripts/rehearsal_oot.py`; numbers in `results/rehearsal_oot.json`; exclusion sets in `runs/rehearsal/exclusions.json`. **Not a decision tool and not a test between methods.** 76 targets, so paired intervals are wide. Read how much each method degrades.

Targets: val routes from the first and last week of the training window. Thinned: train routes within the 5-day gap (group-wise) removed, leaving every target at least 6 days from its history. Control: the same number of train routes removed at random (same volume, no gap). Lower is better.

| Arm | Full | Control (volume only) | Thinned (gap) | Thinned − full [95% CI] | Thinned − control: the gap effect [95% CI] |
|---|---|---|---|---|---|
| v0hist | 0.0229 | 0.0260 | 0.0254 | +0.0024 [-0.0000, +0.0056] | -0.0006 [-0.0056, +0.0038] |
| v0past | 0.0282 | 0.0306 | 0.0335 | +0.0052 [+0.0011, +0.0100] | +0.0029 [-0.0008, +0.0074] |
| v1 | 0.0221 | 0.0243 | 0.0228 | +0.0007 [-0.0007, +0.0030] | -0.0014 [-0.0061, +0.0022] |

Groups: {"early": {"n_targets": 55, "cut_date": "2018-07-30", "train_routes_removed": 1735}, "late": {"n_targets": 21, "cut_date": "2018-08-15", "train_routes_removed": 1161}}
