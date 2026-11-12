# Machine-load check of the time-limited arms (PREREGISTRATION.md section 8)

Rule: flag if load1 > 8.0 in > 20% of minutes, an Opus batch > 1.5x its median s/route, or mean clock < 2000 MHz. Made by `scripts/final_load_check.py` from inputs only.

| Step | Minutes logged | Load at start | Share load > 8 | Max load | Mean MHz | Max temp C | Batch s/route median / max | FLAGGED |
|---|---|---|---|---|---|---|---|---|
| opus_run2_rep1 | 15 | 5.74 | 0% | 5.13 | 2388 | 88.0 | 0.467 / 0.471 | no |
| opus_run3_rep1 | 23 | 3.65 | 0% | 3.68 | 2787 | 88.0 | 0.406 / 0.431 | no |
| opus_run4_rep1 | 71 | 2.94 | 0% | 4.89 | 2601 | 88.0 | 1.353 / 1.513 | no |
| opus_run5_rep1 | 36 | 2.72 | 0% | 3.35 | 2693 | 88.0 | 0.679 / 0.683 | no |
| opus_run6_rep1 | 37 | 2.87 | 0% | 3.95 | 2715 | 88.0 | 0.702 / 0.705 | no |
| opus_run2_rep2_sample | 4 | 2.98 | 0% | 3.0 | 2790 | 89.0 | 0.465 / 0.465 | no |
| opus_run3_rep2_sample | 4 | 2.83 | 0% | 3.34 | 2713 | 88.0 | 0.416 / 0.416 | no |
| opus_run4_rep2_sample | 12 | 3.37 | 0% | 3.58 | 2699 | 88.0 | 1.397 / 1.397 | no |
| opus_run5_rep2_sample | 6 | 3.33 | 0% | 3.34 | 2666 | 88.0 | 0.679 / 0.679 | no |
| opus_run6_rep2_sample | 6 | 2.99 | 0% | 3.2 | 2734 | 88.0 | 0.702 / 0.702 | no |
| jpt_prepare | 1 | 2.87 | 0% | 2.13 | 3747 | 87.0 | — | no |
| jpt_apply | 38 | 2.02 | 0% | 3.85 | 2886 | 88.0 | — | no |
| rules | 101 | 4.07 | 0% | 7.09 | 2530 | 88.0 | — | no |

Descriptive, not deciding (check-in 1.1): 1-minute load during each final Opus-built repeat next to the same program's VALIDATION run (`runs/opus_fight_run<N>/load_log.txt`; runs 2-3 had no load log).

| Step | Final median / p90 load | Val median / p90 / max load |
|---|---|---|
| opus_run2_rep1 | 4.70 / 4.93 | no val load log |
| opus_run3_rep1 | 2.98 / 3.30 | no val load log |
| opus_run4_rep1 | 3.19 / 4.46 | 12.19 / 15.07 / 17.59 |
| opus_run5_rep1 | 3.06 / 3.22 | 4.57 / 5.63 / 9.28 |
| opus_run6_rep1 | 3.27 / 3.56 | 4.56 / 5.07 / 8.02 |
| opus_run2_rep2_sample | 2.98 / 3.00 | no val load log |
| opus_run3_rep2_sample | 3.04 / 3.34 | no val load log |
| opus_run4_rep2_sample | 3.07 / 3.43 | 12.19 / 15.07 / 17.59 |
| opus_run5_rep2_sample | 3.23 / 3.27 | 4.57 / 5.63 / 9.28 |
| opus_run6_rep2_sample | 3.04 / 3.17 | 4.56 / 5.07 / 8.02 |

Caveat: the validation load logs have no CPU-clock column, so rule (c) (mean clock < 2,000 MHz) has no validation anchor.

Descriptive, not deciding (check-in 1.3): the same-solver build (`scripts/final_samesolver.py`, section 6.3). For comparison, FINAL (v1_exact) used 4 workers and 2.31 CPU-s per route (that includes its own zone decoding). The logger restarted about 7 minutes after this build began.

| Arm | Minutes logged | Workers | CPU-s/route | Share load > 8 | Median / max load | Mean MHz | Max temp C |
|---|---|---|---|---|---|---|---|
| samesolver_run2 | 26 | 4 | 2.06 | 0% | 5.70 / 6.30 | 2251 | 88.0 |
| samesolver_run3 | 26 | 4 | 2.07 | 0% | 6.13 / 6.91 | 2158 | 88.0 |
| samesolver_run4 | 28 | 4 | 2.13 | 71% | 12.18 / 16.19 | 2028 | 88.0 |
| samesolver_run5 | 23 | 4 | 2.06 | 4% | 5.93 / 8.96 | 2277 | 88.0 |
| samesolver_run6 | 28 | 4 | 2.11 | 61% | 10.31 / 13.43 | 2166 | 87.0 |
