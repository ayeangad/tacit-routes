# Pre-registration: tacit-routes final test

**Status: DRAFT, not locked.** It is locked only after a final read-through. Locking means setting `"locked": true` and this file's sha256 in `runs/final/PREREG_CONFIG.json`; `scripts/final_score.py` refuses to run otherwise. The sealed answers (`data/eval_frozen/raw/score/`) are opened once, by that script, after locking. The two [TBD] items were filled on 2026-11-12 ~01:45 IST from the gate result and the frozen hashes (previous text: `runs/final/PREREGISTRATION_before_TBD_fill.md`). Amended 2026-11-12 ~11:50 IST after a pre-lock review, still before any final-test answer was opened: same-solver procedure written down and built, the after-opening rule, the crash rule, the lock record, and the history wording (previous text: `runs/final/PREREGISTRATION_before_prelock_review.md`).

Written 2026-11-11 from the plan of record. The reasoning behind each choice is in `DECISION_LOG.md`.

## 1. Question

Does a small specialist model, trained on a laptop from past routes, predict the order real delivery drivers visit their stops better than Opus 5.5 used as the model, given history drawn from the same training routes (summarised in Opus's prompt: first-zone counts, top next zones, 8 most similar past routes; FINAL uses the full count tables) and the same inside-zone solver?

Secondary question: does it beat the small systems Opus 5.5 builds itself when given the same data and tools?

## 2. Data

- **Final test:** the 3,052 official evaluation routes of the 2021 Amazon Last Mile Routing Research Challenge (`data/eval_frozen/`). All are High quality. Dates 2018-06-01 to 2018-10-02; training covers 2018-07-19 to 2018-08-26 (1,524 routes before, 36 inside, 1,492 after). 18 stations; DBO6 (1 route) has no training routes.
- **History and training data:** the 5,762 training routes only. Validation (300) and pilot (50) routes are never used as history.
- **History rule (headline): the competition rule.** Any training route may be used as history, whatever its date. This applies to every arm. Past-only variants are reported as a robustness check.
- **Coverage of the final test** (`results/eval_coverage.md`): 81% of zones seen in training (val 90%).

## 3. Metric

- The official challenge score (`scripts/official_rc_cli/score.py`, unmodified), computed through the same path as `results/val_comparison.md` (travel times rounded to 0.1 s). Lower is better.
- An invalid or missing route gets the challenge's invalid score ("strict").
- **OPUS-MODEL only:** an invalid zone list is also scored "repaired", using the fallback in `scripts/zone_rich.py`. The **primary comparison uses the repaired view**, which is conservative toward Opus. Strict is reported beside it. (Review 5, point 1.)
- **OPUS-MODEL calls refused by the API safeguard** (added 2026-11-12 ~02:40 IST after one false positive, before any answer was opened): a refused call is retried at the end of the run, at most 3 attempts per route. If every attempt is refused, the route has no Opus answer: STRICT scores it invalid; REPAIRED treats it as an empty zone list, so the fallback answers for it. The number of refused routes is reported. (`runs/final/incidents.log`.)
- A route's arm score is the plain mean over the routes in that comparison.

## 4. Arms

| Arm | What it is | Routes |
|---|---|---|
| **FINAL** (ours, design B) | v1: 5-seed ensemble, about 147k weights plus a history table, beam 5 + λ = 0.5. Inside-zone solver: **exact** (Held–Karp up to 12 stops, else OR-Tools guided local search 1 s). The gate (`results/exact_solver_gate.md`, rule fixed in advance) said ADOPT on 234 held-out train routes: exact − greedy −0.00023 [−0.00045, −0.000005], 134/95/5. The upper end only just clears 0; disclosed | all 3,052 |
| FINAL-greedy | v1 with the greedy inside-zone solver. Always reported, so the solver's effect is visible | all 3,052 |
| **v0-hist** (ours, design A) | Frozen before any out-of-time work: 28,481 weights, beam 5 + λ = 0.5, greedy solver | all 3,052 |
| v0-hist-past, v1-past | Past-only history variants (robustness) | all 3,052 |
| **OPUS-MODEL** (frontier as the model) | Opus 5.5 picks the zone order from the frozen richer-history prompt (`runs/zone_rich/FROZEN_PROMPT.md`, file sha256 bffd506e…; system prompt 8130bb2d…, template fa48f8bd…, `scripts/zone_rich.py` e6d2400a…, all in `runs/zone_rich/FROZEN.json`). Stops are filled by the SAME solver as FINAL | 500-route sample |
| OPUS-ENG runs 2, 3, 4 (blind) | Opus-built systems from fair-fight runs 2–4, run unchanged in a no-network sandbox: once on all 3,052 (the score), again on the 500-route sample (solver noise; option B) | all 3,052 |
| OPUS-ENG runs 5, 6 (informed + literature) | Opus-built systems from runs 5–6, the same way | all 3,052 |
| References | jpt-amz (2021 winner, local 1 s + 1 s settings), R2 zone-ID rule, R4-hist counting, shortest route | all 3,052 |

**The 500-route sample:**
- Fixed seed 20261112, stratified by station and by before/after the training dates.
- The ID list (`runs/final/primary_sample_500.json`, content sha256 ee719fc8…) was saved before any eval Opus call.
- FINAL is also scored on exactly these routes for the primary comparison.

**Opus-built systems** (review 5, point 2):
- **Option B** (chosen 2026-11-11 ~23:55 IST): each system is run once on all 3,052 routes, and that run is its score. A second run on the 500-route sample measures solver noise; its spread is reported.
- They run in batches of 300 routes (the scale they were built for), inside the no-network sandbox, unedited. DBO6 runs as its own one-route batch.
- **Harness-side failures** (paths, scale, memory, batching, interface) are fixed and re-run, and every incident is logged in `runs/final/incidents.log`.
- A failure that persists on a single route invalidates only that route.
- Opus's algorithm code is never edited.

**Opus effort:** the fair-fight runs used Opus's default effort. For OPUS-MODEL, default and `--effort max` were compared on the 40 development routes. Max: 0.0310 vs default 0.0319, difference −0.0009 [−0.0046, +0.0026], 9/2/29. **Max is frozen** under the "better for Opus" rule (`runs/zone_rich/FROZEN.json`; prompt sha256 8130bb2d…, template fa48f8bd…, script e6d2400a…). Max costs about $0.45 and 196 s per call, against $0.04 and 5 s.

**OPUS-MODEL-default (descriptive):** the same frozen prompt at default effort, on the same 500 routes. It is reported for the cost/accuracy picture only; it is not part of any claim.

## 5. Primary comparison (one test, no correction)

- **FINAL − OPUS-MODEL** on the 500-route sample, paired by route.
- **Win:** the mean difference is below 0 AND the paired bootstrap 95% CI (10,000 route resamples, seed 20261111) lies entirely below 0.
- Also reported: route wins/losses/ties with an exact sign test, plus cost per route (USD, CLI cost field) and latency per route (wall seconds) for both arms.

## 6. Secondary comparisons (Holm-corrected together, using sign-test p-values; intervals reported)

1. **Design B (informed):** FINAL − OPUS-ENG run 5, and FINAL − OPUS-ENG run 6, on all 3,052 routes. "We beat Opus as an engineer (informed)" is claimed ONLY if both are wins (CI below 0, Holm-adjusted p < 0.05).
2. **Design A (blind):** v0-hist − OPUS-ENG run 2, run 3 and run 4. "We beat Opus as an engineer (blind)" is claimed ONLY if all three are wins. Run 1 (cut short by our memory bug) is reported but excluded.
3. **Same-solver engineer comparison (P1, analysis):** each Opus run's zone order, filled in by FINAL's solver, against FINAL. This isolates whose zone model is better. It is descriptive, not a claim. **Procedure** (written and built 2026-11-12, before the lock, from inputs only, by `scripts/final_samesolver.py`): for Opus systems 2–6, the zone order is the first visit to each zone in that system's repeat-1 prediction (zones as FINAL defines them); the stops are filled by FINAL's exact in-zone solver with the same worker count as FINAL (4). Arms `samesolver_run2` … `samesolver_run6`, all 3,052 routes. Scored by `final_score.py` as reference comparisons: FINAL − samesolver_runN (whose zone order is better), and opus_runN_rep1 − samesolver_runN (what Opus's own in-zone solving adds or loses), with the registered paired bootstrap.

**Both views must agree** (review 5, point 3). Every comparison is computed on all 3,052 routes AND excluding DBO6. A win (primary or secondary) is claimed only if it holds in both views; for the secondaries, each view gets its own Holm adjustment. If they disagree, the result is reported as not established. (Inputs-only precheck, 2026-11-12: exactly one route has no station history, at DBO6, and it is not in the 500-route sample, so the primary's two views are identical; the views can differ only for the all-routes comparisons.) (`scripts/final_score.py` was fixed on 2026-11-12 to enforce this; before the fix it decided wins from the all-routes view only. Logged in `runs/final/incidents.log`.)

**Non-inferiority claim for a draw** (review 5, point 5):
- If the primary win fails, a second, pre-stated claim is tested: FINAL is no worse than OPUS-MODEL by more than 0.002 (upper end of the paired 95% CI below +0.002), in both views, AND costs at least 100× less per route. Cost per route: OPUS-MODEL = mean of the CLI cost field over its calls; FINAL = its measured CPU-seconds per route (about 2 s with the exact solver) priced at $0.10 per CPU-hour, a deliberately high rate (its marginal cost on the laptop is $0). Wall time per route is reported for both. Computed by `scripts/final_score.py` from `non_inferiority` in `runs/final/PREREG_CONFIG.json`.
- If it holds, the claim is "as accurate as Opus 5.5 within 0.002, at more than 100× lower cost". Otherwise no accuracy claim is made.

## 7. Reported regardless of outcome

- All arms' means, medians and 95% CIs (means and the comparisons from `final_score.py`; medians and per-arm CIs from `scripts/final_descriptive.py`).
- **Placement against published eval-set scores:** 0.0248 (2021 winner), 0.0294, 0.0300, 0.0302, 0.0353, 0.0374. Our local jpt run uses short time limits, so it is a reference, not a replication.
- Score by before/inside/after the training dates, and by gap in days (`scripts/final_descriptive.py`).
- Direction (Kendall tau of zone order) for every arm (`scripts/final_descriptive.py`; zone order = first visit to each zone, for the arm and for the driver).
- Error analysis (`scripts/final_descriptive.py`): FINAL against OPUS-MODEL and against each same-solver Opus run: per-route score difference against tau difference, wins and losses by route size, FINAL's largest losses.
- v0-hist on the 50 pilot routes, next to the stored pilot Opus arms (n = 50, descriptive; see the last item).
- Our models on the 50 pilot routes (v0-hist, plus v1 and FINAL for context), next to the stored pilot Opus arms: `scripts/final_pilot_score.py`, which also runs only after the lock, once.
- Every deviation from this document, however small.

## 8. Fixed rules

- **Before locking: dress rehearsal.** `final_predict` and `final_score` run end to end on the 300 validation routes, laid out like the eval folder with a mock sealed folder. They must reproduce the stored validation numbers exactly (v0-hist, jpt, R2; Opus code within solver noise).
- **After predictions, before the answers are opened:** an inputs-only validity check of every arm's predictions. Results are logged.
- **Shortest-route reference:** 3 s search per route (val used 10 s), disclosed.
- **Machine-load rule for time-limited arms** (check-in 1; fixed 2026-11-12 ~03:00 IST, before any final-test score; D17). Time-limited arms: the Opus-built systems (runs 2–6, both repeats), jpt, and the rules step (which includes the 3 s shortest route). `runs/final/load_log.txt` records load, CPU clock and package temperature every minute. An arm repeat is FLAGGED if, inside its step's time window: (a) the 1-minute load average is above 8.0 (the laptop's 8 hardware threads) in more than 20% of the logged minutes; or (b) for an Opus-built system, any 300-route batch takes more than 1.5× that repeat's median seconds per route over its 300-route batches; or (c) the mean CPU clock over the window is below 2,000 MHz (throttled well below the 2.4 GHz base). A flagged arm repeat is re-run in full, unchanged, after the OPUS-MODEL arm has finished, as a harness failure; the re-run is the one scored, and both runs' load numbers are reported. Steps that ran before the logger started (Opus run 2 repeat 1, batches 1–3) are judged by (b) and by the load at step start. Checked by `scripts/final_load_check.py` before any answer is opened.

- **After the answers are opened** (pre-lock review). The answers count as opened once `final_score.py` has written `results/final_test.json` and its sha256 is logged in `runs/final/LOCK_RECORD.json`. After that, they may be read only by named descriptive scripts: `scripts/final_descriptive.py` (written and self-tested before the lock), `scripts/final_pilot_score.py` (pilot routes only), and any later analysis script, which must be named in `PROJECT.md` and labelled descriptive/exploratory. None of them can change a claim.
- **Crash rule** (pre-lock review). `final_score.py` prints each arm's mean as it goes. If it crashes before `results/final_test.json` exists, that is a harness incident: the cause is fixed, logged in `runs/final/incidents.log` with the diff, and the scorer re-run. Nothing in this document, the config, the predictions or the comparison list may change in that fix.
- **Lock record** (pre-lock review). At the lock, `runs/final/LOCK_RECORD.json` logs the sha256 of this document, `runs/final/PREREG_CONFIG.json` (filled), `scripts/final_score.py`, `scripts/final_common.py`, `scripts/final_descriptive.py`, `scripts/final_pilot_score.py` and `runs/final/predictions/MANIFEST.json`. After scoring, the sha256 of `results/final_test.json` is added.

- **DBO6** (no station history): our models use pooled history only; Opus systems do whatever their code does. Included in all means.
- **No model, decoder, solver or prompt changes after locking.** The FINAL definition is whatever the gates selected before locking.
- **Losses and draws are reported as prominently as wins.**
- **The validation set was used heavily during development** (many variants, Opus runs, rehearsals). Validation numbers are optimistic and are never quoted as the result.

## 9. What would count as each outcome (written before seeing results)

- **Primary win:** "A small model trained on a laptop beat Opus 5.5 used as the model, given history from the same training routes and the same solver, on the official evaluation routes, at a small fraction of the cost per route."
- **Primary draw or loss:** reported as such. The headline becomes the cost/accuracy trade-off, with no claim of beating Opus.
- **Primary fails, non-inferiority holds:** "as accurate as Opus 5.5 used as the model (within 0.002), at more than 100× lower cost per route."
- **Engineer fight:** claimed only under the rules in section 6. A loss there is reported as "Opus 5.5, given about 1.5 hours and the data, built a small model as good as or better than ours", which itself supports the case for small specialist models.

## 10. Planned after the final test (exploratory, not part of the claims)

A data-efficiency curve (review 5). The same v1 recipe trained on 100, 300, 1,000, 3,000 and all training routes, with OPUS-MODEL given the same reduced history. It shows how much data the specialist needs before it overtakes the frontier model. Labelled exploratory.
