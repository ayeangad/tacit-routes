# History coverage of the final-test routes (inputs only; no answers read)

Made by `scripts/eval_coverage_report.py`. Reads only station, date and zone IDs. Train history = all train routes at the same station (competition rule). Val shown for comparison.

| Group | Routes | Zones seen: full | micro | macro | Best single reference (mean / median) | New zones per route |
|---|---|---|---|---|---|---|
| Val (300, in-time) | 300 | 0.897 | 0.965 | 0.969 | 0.822 / 0.909 | 0.56 |
| Eval, all | 3051 | 0.812 | 0.926 | 0.932 | 0.718 / 0.812 | 1.19 |
| Eval, before training dates | 1524 | 0.878 | 0.959 | 0.962 | 0.783 / 0.857 | 0.7 |
| Eval, inside training dates | 36 | 0.924 | 0.972 | 0.972 | 0.843 / 0.923 | 0.39 |
| Eval, after training dates | 1491 | 0.743 | 0.892 | 0.900 | 0.649 / 0.737 | 1.71 |
| Eval, 0-0 days from training dates | 36 | 0.924 | 0.972 | 0.972 | 0.843 / 0.923 | 0.39 |
| Eval, 1-7 days from training dates | 470 | 0.856 | 0.945 | 0.949 | 0.765 / 0.848 | 1.03 |
| Eval, 8-14 days from training dates | 538 | 0.824 | 0.923 | 0.927 | 0.712 / 0.800 | 1.24 |
| Eval, 15-21 days from training dates | 541 | 0.821 | 0.945 | 0.949 | 0.725 / 0.818 | 0.82 |
| Eval, 22-28 days from training dates | 502 | 0.776 | 0.908 | 0.918 | 0.692 / 0.795 | 1.47 |
| Eval, 29-38 days from training dates | 677 | 0.768 | 0.904 | 0.910 | 0.680 / 0.792 | 1.48 |
| Eval, 39-48 days from training dates | 287 | 0.857 | 0.946 | 0.949 | 0.757 / 0.840 | 0.98 |

Routes at a station with no train history: 1 (DBO6).
