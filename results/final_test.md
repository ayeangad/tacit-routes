# Final test result

Config: `runs/final/PREREG_CONFIG.json` (locked, preregistration sha256 a8eccd3f611b…). Lower is better.

| Arm | Mean | Routes | Missing |
|---|---|---|---|
| v1_exact | 0.0259 | 3052 | 0 |
| v1 | 0.0263 | 3052 | 0 |
| v0hist | 0.0272 | 3052 | 0 |
| v0past | 0.0331 | 3052 | 0 |
| opus_model_repaired_exact | 0.0284 | 500 | 0 |
| opus_model_strict_exact | 0.0284 | 500 | 0 |
| opus_run2_rep1 | 0.0281 | 3052 | 0 |
| opus_run2_rep2 | 0.0272 | 500 | 0 |
| opus_run3_rep1 | 0.0240 | 3052 | 0 |
| opus_run3_rep2 | 0.0243 | 500 | 0 |
| opus_run4_rep1 | 0.0241 | 3052 | 0 |
| opus_run4_rep2 | 0.0243 | 500 | 0 |
| opus_run5_rep1 | 0.0244 | 3052 | 0 |
| opus_run5_rep2 | 0.0247 | 500 | 0 |
| opus_run6_rep1 | 0.0250 | 3052 | 0 |
| opus_run6_rep2 | 0.0254 | 500 | 0 |
| jpt | 0.0248 | 3052 | 0 |
| rules_R2 | 0.0322 | 3052 | 0 |
| rules_R4 | 0.0340 | 3052 | 0 |
| rules_S3 | 0.0901 | 3052 | 0 |
| samesolver_run2 | 0.0250 | 3052 | 0 |
| samesolver_run3 | 0.0240 | 3052 | 0 |
| samesolver_run4 | 0.0242 | 3052 | 0 |
| samesolver_run5 | 0.0237 | 3052 | 0 |
| samesolver_run6 | 0.0242 | 3052 | 0 |
| opus_model_default_repaired_exact | 0.0324 | 500 | 0 |
| opus_model_default_strict_exact | 0.0324 | 500 | 0 |

| Comparison (a − b) | Role | Routes | Mean diff [95% CI] | a better / worse | Sign p | Holm p | Win, all routes |
|---|---|---|---|---|---|---|---|
| PRIMARY: v1_exact − opus_model_repaired_exact | primary | 500 | -0.0023 [-0.0042, -0.0005] | 149 / 94 | 0.00051 | nan | YES |
| PRIMARY_strict_view: v1_exact − opus_model_strict_exact | reference | 500 | -0.0023 [-0.0042, -0.0005] | 149 / 94 | 0.00051 | nan | YES |
| B_vs_run5: v1_exact − opus_run5_rep1 | secondary | 3052 | +0.0015 [+0.0010, +0.0020] | 1753 / 1295 | 1.1e-16 | 1.1e-16 | no |
| B_vs_run6: v1_exact − opus_run6_rep1 | secondary | 3052 | +0.0010 [+0.0005, +0.0015] | 1774 / 1274 | 1.3e-19 | 2.6e-19 | no |
| A_vs_run2: v0hist − opus_run2_rep1 | secondary | 3052 | -0.0009 [-0.0016, -0.0002] | 1778 / 1274 | 7.1e-20 | 2.1e-19 | YES |
| A_vs_run3: v0hist − opus_run3_rep1 | secondary | 3052 | +0.0032 [+0.0026, +0.0038] | 1148 / 1869 | 1.3e-39 | 6.4e-39 | no |
| A_vs_run4: v0hist − opus_run4_rep1 | secondary | 3052 | +0.0031 [+0.0026, +0.0037] | 1168 / 1875 | 7.9e-38 | 3.2e-37 | no |
| ref_jpt: v1_exact − jpt | reference | 3052 | +0.0011 [+0.0005, +0.0017] | 1443 / 1602 | 0.0042 | nan | no |
| ref_R2: v1_exact − rules_R2 | reference | 3052 | -0.0062 [-0.0070, -0.0054] | 1848 / 1134 | 2.5e-39 | nan | YES |
| ref_R4: v1_exact − rules_R4 | reference | 3052 | -0.0081 [-0.0089, -0.0073] | 1935 / 1060 | 3.2e-58 | nan | YES |
| exact_vs_greedy: v1_exact − v1 | reference | 3052 | -0.0003 [-0.0004, -0.0002] | 1758 / 1248 | 1.3e-20 | nan | YES |
| past_only: v0past − v0hist | reference | 3052 | +0.0059 [+0.0052, +0.0066] | 506 / 1197 | 1.5e-64 | nan | no |
| ref_shortest_3s: v1_exact − rules_S3 | reference | 3052 | -0.0642 [-0.0658, -0.0625] | 2950 / 102 | 0 | nan | YES |
| ref_opus_model_default: v1_exact − opus_model_default_repaired_exact | reference | 500 | -0.0063 [-0.0085, -0.0043] | 201 / 87 | 1.5e-11 | nan | YES |
| samesolver_zone_run2: v1_exact − samesolver_run2 | reference | 3052 | +0.0010 [+0.0004, +0.0016] | 453 / 643 | 1e-08 | nan | no |
| opus_own_solver_run2: opus_run2_rep1 − samesolver_run2 | reference | 3052 | +0.0032 [+0.0029, +0.0034] | 987 / 2065 | 1.9e-86 | nan | no |
| samesolver_zone_run3: v1_exact − samesolver_run3 | reference | 3052 | +0.0019 [+0.0014, +0.0025] | 381 / 608 | 5.3e-13 | nan | no |
| opus_own_solver_run3: opus_run3_rep1 − samesolver_run3 | reference | 3052 | +0.0000 [-0.0000, +0.0000] | 1367 / 1365 | 0.98 | nan | no |
| samesolver_zone_run4: v1_exact − samesolver_run4 | reference | 3052 | +0.0018 [+0.0013, +0.0023] | 416 / 669 | 1.5e-14 | nan | no |
| opus_own_solver_run4: opus_run4_rep1 − samesolver_run4 | reference | 3052 | -0.0001 [-0.0002, +0.0000] | 1563 / 1460 | 0.064 | nan | no |
| samesolver_zone_run5: v1_exact − samesolver_run5 | reference | 3052 | +0.0023 [+0.0018, +0.0027] | 318 / 595 | 3.4e-20 | nan | no |
| opus_own_solver_run5: opus_run5_rep1 − samesolver_run5 | reference | 3052 | +0.0007 [+0.0006, +0.0008] | 1061 / 1981 | 2.7e-63 | nan | no |
| samesolver_zone_run6: v1_exact − samesolver_run6 | reference | 3052 | +0.0017 [+0.0012, +0.0022] | 424 / 632 | 1.7e-10 | nan | no |
| opus_own_solver_run6: opus_run6_rep1 − samesolver_run6 | reference | 3052 | +0.0007 [+0.0006, +0.0008] | 1065 / 1978 | 2.6e-62 | nan | no |

Same comparisons excluding routes at a station with no train history (DBO6), and the combined verdict (a win needs BOTH views):

| Comparison | Routes | Mean diff [95% CI] | a better / worse | Holm p | Win, excl. | Views agree | WIN (both) |
|---|---|---|---|---|---|---|---|
| PRIMARY | 500 | -0.0023 [-0.0042, -0.0005] | 149 / 94 | nan | YES | yes | YES |
| PRIMARY_strict_view | 500 | -0.0023 [-0.0042, -0.0005] | 149 / 94 | nan | YES | yes | YES |
| B_vs_run5 | 3051 | +0.0015 [+0.0010, +0.0020] | 1753 / 1294 | 9.4e-17 | no | yes | no |
| B_vs_run6 | 3051 | +0.0010 [+0.0005, +0.0015] | 1774 / 1273 | 2.2e-19 | no | yes | no |
| A_vs_run2 | 3051 | -0.0009 [-0.0017, -0.0002] | 1778 / 1273 | 1.8e-19 | YES | yes | YES |
| A_vs_run3 | 3051 | +0.0032 [+0.0026, +0.0038] | 1148 / 1868 | 7.9e-39 | no | yes | no |
| A_vs_run4 | 3051 | +0.0031 [+0.0026, +0.0037] | 1168 / 1874 | 3.9e-37 | no | yes | no |
| ref_jpt | 3051 | +0.0011 [+0.0005, +0.0017] | 1443 / 1601 | nan | no | yes | no |
| ref_R2 | 3051 | -0.0062 [-0.0070, -0.0055] | 1848 / 1133 | nan | YES | yes | YES |
| ref_R4 | 3051 | -0.0081 [-0.0089, -0.0073] | 1935 / 1059 | nan | YES | yes | YES |
| exact_vs_greedy | 3051 | -0.0003 [-0.0004, -0.0002] | 1758 / 1247 | nan | YES | yes | YES |
| past_only | 3051 | +0.0059 [+0.0052, +0.0066] | 506 / 1196 | nan | no | yes | no |
| ref_shortest_3s | 3051 | -0.0642 [-0.0658, -0.0626] | 2950 / 101 | nan | YES | yes | YES |
| ref_opus_model_default | 500 | -0.0063 [-0.0085, -0.0043] | 201 / 87 | nan | YES | yes | YES |
| samesolver_zone_run2 | 3051 | +0.0010 [+0.0003, +0.0016] | 453 / 642 | nan | no | yes | no |
| opus_own_solver_run2 | 3051 | +0.0032 [+0.0029, +0.0034] | 987 / 2064 | nan | no | yes | no |
| samesolver_zone_run3 | 3051 | +0.0019 [+0.0014, +0.0024] | 381 / 607 | nan | no | yes | no |
| opus_own_solver_run3 | 3051 | +0.0000 [-0.0000, +0.0000] | 1366 / 1365 | nan | no | yes | no |
| samesolver_zone_run4 | 3051 | +0.0018 [+0.0013, +0.0023] | 416 / 668 | nan | no | yes | no |
| opus_own_solver_run4 | 3051 | -0.0001 [-0.0002, +0.0000] | 1562 / 1460 | nan | no | yes | no |
| samesolver_zone_run5 | 3051 | +0.0023 [+0.0018, +0.0027] | 318 / 594 | nan | no | yes | no |
| opus_own_solver_run5 | 3051 | +0.0007 [+0.0006, +0.0008] | 1060 / 1981 | nan | no | yes | no |
| samesolver_zone_run6 | 3051 | +0.0017 [+0.0012, +0.0022] | 424 / 631 | nan | no | yes | no |
| opus_own_solver_run6 | 3051 | +0.0007 [+0.0006, +0.0008] | 1065 / 1977 | nan | no | yes | no |

Non-inferiority (PRIMARY, margin +0.002, cost ratio ≥ 100×; reported only, since the primary win holds):
- CI upper end: all routes -0.0005, excluding no-history station -0.0005 -> within the margin in both views
- Cost per route: b 0.5110593908 USD (CLI cost field, answering call only, 500 calls; 0.5223399667999999 USD including retried and refused attempts; 0 routes refused; wall 210.6575426351961 s); a 2.3082632085658585 CPU-s = 6.411842246016274e-05 USD at 0.1 USD per CPU-hour; ratio 7970.554657946006 -> ok
- **Non-inferiority HOLDS**

Opus fair-fight systems, the two sandbox repeats (solver noise):

| System | Repeat 1 | Repeat 2 | rep1 − rep2 [95% CI] | Mean abs route diff | Identical routes |
|---|---|---|---|---|---|
| opus_run2 | 0.0279 | 0.0272 | +0.0008 [+0.0005, +0.0011] | 0.0010 | 289 |
| opus_run3 | 0.0242 | 0.0243 | -0.0001 [-0.0003, +0.0000] | 0.0001 | 489 |
| opus_run4 | 0.0243 | 0.0243 | +0.0000 [-0.0000, +0.0000] | 0.0000 | 493 |
| opus_run5 | 0.0247 | 0.0247 | -0.0000 [-0.0000, +0.0000] | 0.0000 | 484 |
| opus_run6 | 0.0254 | 0.0254 | -0.0001 [-0.0002, +0.0000] | 0.0001 | 479 |

Claims:
- **PRIMARY_beats_opus_as_model** (all of PRIMARY): HOLDS
- **DESIGN_B_beats_engineer** (all of B_vs_run5, B_vs_run6): does not hold
- **DESIGN_A_beats_engineer** (all of A_vs_run2, A_vs_run3, A_vs_run4): does not hold
