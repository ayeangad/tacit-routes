# Exact in-zone solver gate (held-out TRAIN routes; val and the final test not touched)

Rule fixed before any number: `docs/plan.md`, "Amendment after Opus run 3", P2. Made by `scripts/exact_solver_gate.py`; numbers in `results/exact_solver_gate.json`. Prompted by Opus fair-fight run 3, which solves each zone path exactly.

**Verdict: ADOPT.**

Targets: the 234 held-out High training routes. Zone order = the driver's own, so only the order of stops inside zones differs. Old = OR-Tools greedy descent (every v0/v1 number so far). New = Held–Karp dynamic programming for zones of up to 12 stops, OR-Tools guided local search for 1 s above. Lower is better.

| | Score |
|---|---|
| Old (greedy descent) | 0.0079 |
| New (exact) | 0.0077 |
| **New − old [95% CI]** | **-0.00023 [-0.00045, -0.00000]** |
| New better / worse / tied | 134 / 95 / 5 (sign p 0.012) |
| CPU per route (fill only) | old 0.090 s, new 1.839 s |

Rule: ADOPT iff the CI's upper end is below 0. Bootstrap: 10,000 route resamples, seed 20261111. Runtime 228 s.
