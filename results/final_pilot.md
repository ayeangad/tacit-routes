# Our models on the 50 pilot routes (descriptive; written after the lock)

Made once by `scripts/final_pilot_score.py`. n = 50, so intervals are wide. Opus numbers are the stored pilot runs. Lower is better.

| Arm | Mean |
|---|---|
| pilot_v0hist | 0.0224 |
| pilot_v1 | 0.0220 |
| pilot_v1_exact | 0.0216 |
| Opus direct (pilot.json arm) | 0.0693 |
| Opus + 3 past routes | 0.0355 |
| Opus zone-level (fallback view) | 0.0355 |
| Shortest route (OR-Tools) | 0.0784 |

| Comparison (a − b) | Mean diff [95% CI] | a better / worse / tied | Sign p |
|---|---|---|---|
| pilot_v0hist − Opus direct (pilot.json arm) | -0.0469 [-0.0558, -0.0383] | 48 / 2 / 0 | 2.3e-12 |
| pilot_v0hist − Opus + 3 past routes | -0.0132 [-0.0195, -0.0076] | 37 / 13 / 0 | 0.00094 |
| pilot_v0hist − Opus zone-level (fallback view) | -0.0132 [-0.0201, -0.0072] | 33 / 12 / 5 | 0.0025 |
| pilot_v0hist − Shortest route (OR-Tools) | -0.0560 [-0.0640, -0.0478] | 49 / 1 / 0 | 9.1e-14 |
| pilot_v1 − Opus direct (pilot.json arm) | -0.0473 [-0.0566, -0.0378] | 48 / 2 / 0 | 2.3e-12 |
| pilot_v1 − Opus + 3 past routes | -0.0135 [-0.0204, -0.0074] | 37 / 13 / 0 | 0.00094 |
| pilot_v1 − Opus zone-level (fallback view) | -0.0135 [-0.0205, -0.0074] | 34 / 10 / 6 | 0.00039 |
| pilot_v1 − Shortest route (OR-Tools) | -0.0563 [-0.0638, -0.0487] | 49 / 1 / 0 | 9.1e-14 |
| pilot_v1_exact − Opus direct (pilot.json arm) | -0.0477 [-0.0569, -0.0384] | 48 / 2 / 0 | 2.3e-12 |
| pilot_v1_exact − Opus + 3 past routes | -0.0139 [-0.0207, -0.0079] | 37 / 13 / 0 | 0.00094 |
| pilot_v1_exact − Opus zone-level (fallback view) | -0.0139 [-0.0210, -0.0079] | 37 / 13 / 0 | 0.00094 |
| pilot_v1_exact − Shortest route (OR-Tools) | -0.0568 [-0.0641, -0.0492] | 49 / 1 / 0 | 9.1e-14 |
