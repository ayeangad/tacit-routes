# Adopted v1 on the 300 validation routes

Made by `scripts/v1_val.py`; numbers in `results/v1_val.json`. Validation only; it has been looked at many times (see `results/val_comparison.md` caveats), so treat every gap as optimistic. The verdict is the frozen final test. v1 was adopted on held-out TRAIN routes (`results/v1_adoption.md`), not on these numbers. "v1 exact" uses the exact in-zone solver, whose adoption is decided by `results/exact_solver_gate.md`, not here.

| Arm | Mean |
|---|---|
| v1 greedy | 0.0228 |
| v1 exact | 0.0224 |
| v0-hist (pre-registered) | 0.0229 |
| Opus run 2 (blind) | 0.0268 |
| jpt (2021 winner) | 0.0213 |
| Opus run 3 (blind) | 0.0219 |
| Opus run 4 (blind) | 0.0215 |
| Opus run 5 (informed + literature) | 0.0214 |
| Opus run 6 (informed + literature) | 0.0216 |

Paired, "first minus second" on the same routes (negative = first closer to the drivers). Bootstrap 10,000, seed 20261111; exact sign test. Unadjusted; descriptive.

| Comparison | Means | Mean difference [95% CI] | W / L / T | Sign p |
|---|---|---|---|---|
| v1 greedy − v0-hist (pre-registered) | 0.0228 / 0.0229 | -0.0001 [-0.0014, +0.0011] | 34 / 25 / 241 | 0.3 |
| v1 greedy − Opus run 2 (blind) | 0.0228 / 0.0268 | -0.0040 [-0.0063, -0.0019] | 188 / 112 / 0 | 1.3e-05 |
| v1 greedy − jpt (2021 winner) | 0.0228 / 0.0213 | +0.0015 [-0.0001, +0.0031] | 112 / 187 / 1 | 1.7e-05 |
| v1 greedy − Opus run 3 (blind) | 0.0228 / 0.0219 | +0.0009 [-0.0003, +0.0021] | 117 / 175 / 8 | 0.00082 |
| v1 greedy − Opus run 4 (blind) | 0.0228 / 0.0215 | +0.0013 [-0.0001, +0.0026] | 126 / 171 / 3 | 0.011 |
| v1 greedy − Opus run 5 (informed + literature) | 0.0228 / 0.0214 | +0.0015 [+0.0001, +0.0028] | 136 / 146 / 18 | 0.59 |
| v1 greedy − Opus run 6 (informed + literature) | 0.0228 / 0.0216 | +0.0012 [-0.0001, +0.0025] | 144 / 138 / 18 | 0.77 |
| v1 exact − v0-hist (pre-registered) | 0.0224 / 0.0229 | -0.0005 [-0.0018, +0.0007] | 175 / 123 / 2 | 0.0031 |
| v1 exact − Opus run 2 (blind) | 0.0224 / 0.0268 | -0.0044 [-0.0066, -0.0023] | 200 / 100 / 0 | 8e-09 |
| v1 exact − jpt (2021 winner) | 0.0224 / 0.0213 | +0.0012 [-0.0005, +0.0028] | 131 / 167 / 2 | 0.042 |
| v1 exact − Opus run 3 (blind) | 0.0224 / 0.0219 | +0.0005 [-0.0006, +0.0017] | 147 / 136 / 17 | 0.55 |
| v1 exact − Opus run 4 (blind) | 0.0224 / 0.0215 | +0.0009 [-0.0004, +0.0023] | 130 / 165 / 5 | 0.048 |
| v1 exact − Opus run 5 (informed + literature) | 0.0224 / 0.0214 | +0.0011 [-0.0003, +0.0025] | 168 / 130 / 2 | 0.032 |
| v1 exact − Opus run 6 (informed + literature) | 0.0224 / 0.0216 | +0.0008 [-0.0005, +0.0021] | 169 / 129 / 2 | 0.024 |
| v1 exact − v1 greedy | 0.0224 / 0.0228 | -0.0004 [-0.0006, -0.0002] | 177 / 120 / 3 | 0.0011 |

Runtime 310 s.
