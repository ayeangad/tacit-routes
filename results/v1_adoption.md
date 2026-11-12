# v1 adoption test (held-out TRAIN routes; validation and the final test not touched)

Rule (written before any number): `docs/plan.md`, item 3. Numbers: `results/v1_adoption.json`, made by `scripts/v1_adoption_test.py`.

**Verdict: ADOPT.**

Targets: the 234 held-out High training routes. Neither model trained on them. Both used them for early stopping, and v0's travel weight λ was tuned on them; the decoder is fixed at beam 5 + λ = 0.5 for both. Lower is better.

| History | v0-hist | v1 (5-seed) | v1 − v0-hist [95% CI] | v1 better / worse | Needed | Pass? |
|---|---|---|---|---|---|---|
| **Matched (mean of ±3 and ±9 days removed; DECIDES)** | 0.0305 | 0.0290 | -0.0015 [-0.0025, -0.0006] | 48 / 25 | upper < 0 | yes |
| ±3 days removed (report) | 0.0298 | 0.0280 | -0.0018 [-0.0033, -0.0004] | 33 / 14 | — | — |
| ±6 days removed (report) | 0.0307 | 0.0283 | -0.0024 [-0.0040, -0.0010] | 39 / 19 | — | — |
| ±9 days removed (report) | 0.0313 | 0.0299 | -0.0013 [-0.0025, -0.0002] | 36 / 17 | — | — |
| ±14 days removed (report) | 0.0341 | 0.0321 | -0.0020 [-0.0036, -0.0004] | 48 / 24 | — | — |
| Full (leave-one-out) | 0.0292 | 0.0274 | -0.0018 [-0.0032, -0.0006] | 28 / 16 | upper < +0.002 | yes |

Rows marked "report" do not decide. ±14 days is a stress test, far thinner than the final test (see the amendment in the script's docstring and the joint decisions file).

Bootstrap: 10,000 route resamples, seed 20261111. Runtime 914 s.
