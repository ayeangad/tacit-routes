# Final-test plan and amendments (2026-11-11 evening)

The plan of record for the final test, written before any final-test number. PREREGISTRATION.md is the locked version.

Status: agreed (review 2c added four tightenings, all adopted below; see 'Final tightenings').

## The design principle (review 2b, point 1, adopted)

**Anything we know about the final test and use in design also goes into Opus's brief.**

- Opus runs 1–4 were told only about the 300 validation routes. They were blind to the facts that the final test runs June–October, falls mostly outside the training dates, and is all High quality.
- If we harden our model against that shift and Opus is not told, the comparison is unfair. A reviewer would catch it.
- So there are two consistent designs, and we report both:
  - **A, both blind:** v0-hist exactly as frozen today, against blind Opus fair-fight runs 2–4.
  - **B, both informed:** our final model (v0-hist or v1, by the rule in item 3), against new "informed" Opus fair-fight runs whose README states the same facts about the test that we know.
- The mixed case (a hardened model against blind Opus) is never a headline.

## 1. Primary comparison: the specialist-model claim

**Specialist as the model against frontier as the model.**
- **Arms:** our final model against Opus 5.5 picking the zone order, with the same inside-zone solver ("swap the brain").
- **Opus's history:** richer than in the pilot version: per zone, its most common next zones at this station with counts; the usual first zones; and 5–10 reference routes. Retrieval follows the competition history rule, the same as our model.
- **Prompt:** developed on validation routes only, then frozen before any evaluation call.
- **Routes:** a 500-route sample of the evaluation set, drawn with a fixed seed and stratified by station and by before/after the training dates.
- **Win:** the paired mean difference (ours minus Opus) is below 0, with a 95% bootstrap interval that excludes 0.
- **Also reported:** cost and latency per route for both arms. Accuracy, cost and latency are reported together.

## 2. Secondary comparisons (Holm-corrected together; descriptive beyond that)

1. **Engineer fight, informed (design B):** our final model against EACH "informed + literature" Opus run separately, on all 3,052 routes, Holm-corrected. We claim to beat the engineer arm only if we beat BOTH runs. No opponent is picked by validation score (validation is in-time, the test is not).
2. **Engineer fight, blind (design A):** v0-hist exactly as frozen today, against EACH blind run 2, 3 and 4 separately, on all 3,052 routes. Same rule: a win only if we beat all three. Run 1 (cut short by our memory bug) is reported but excluded from the rule.
3. **History rule:** the competition rule (any training route) is the headline. Past-only versions are reported as a robustness check.
- **References, not win criteria:** jpt-amz (2021 winner), R2, R4-hist and shortest route. Published evaluation-set scores (0.0248 / 0.0294 / 0.0300 / 0.0302) are placed beside ours with their sources.

## 3. Freeze rule, written before any rehearsal number exists (review 2b, point 2, adopted)

- **The single candidate replacement, v1:** v0-hist plus history counts at the two coarser zone-ID levels (`A-2.2`, `A-2`), plus history dropout in training, as **a 5-seed average** (log-probabilities averaged; about 145k parameters). No fork left.
- **Adoption rule, fully specified:**
  - The test routes are the 234 held-out High TRAIN routes. These are the same routes v0's λ was tuned on, which is disclosed. Both models are trained without them.
  - **Thinned history:** for each target, every train route within ±14 days of its date is removed from its history. Adopt only if v1 − v0-hist is below 0, with a paired 95% bootstrap interval that excludes 0.
  - **Full history:** the upper end of the paired 95% interval for v1 − v0-hist must be below +0.002. That is the non-inferiority margin, about half v0-hist's validation margin over Opus run 2.
  - Both conditions must hold, otherwise keep v0-hist.
- **Decoding changes** (wider or stratified beam) only if the search-error check on held-out train shows the search is losing driver orders the model rates higher.
- **The validation rehearsal is diagnostic, not a decision tool.** It has about 100 targets, so its interval is about ±0.004 (review 2b, point 3). It is read as each method's degradation slope, not as a significance test between methods.
- **Cut-off:** no other model changes. After this, the model is frozen and the pre-registration is locked.

## 4. Informed Opus fair-fight runs (design B)

- **Brief (README):** the same task as before, plus every fact about the final test that we use:
  - its dates, June–October 2018, mostly outside the training dates (up to about 48 days before and 37 after);
  - all routes High quality;
  - about 3,000 routes;
  - one station with no training routes;
  - anything the coverage report shows.
- **Deliverable:** a `predict.py` that takes an input folder and an output path, so nobody edits Opus's code to run it on the test.
- **Arena contents:** no evaluation data, as before; its code is applied to the evaluation inputs afterwards by us, inside the same sandbox.
- **Change from the review's proposal:**
  - The review proposed the reading list (the literature review) as a *second* informed run.
  - Adopted instead: **both informed runs get the same brief, reading list included.**
  - Reason: v0-hist's history idea came from that literature review. An "informed" opponent without it still knows less than we did when we designed v0-hist.
  - Two identical runs also give a second measurement of Opus's run-to-run spread.
  - **Reading list (review 2c):** sections 1–3 of the literature review, plus its sources, plus the published eval-score table and its sources. Sections 4–5 (our plan and framing) and every mention of v0, our features, our results or Opus scores are left out. These runs are labelled **"informed + literature"** everywhere.
  - **The README is word for word the blind one,** except for three additions: the test facts, the reading list, and the `predict.py <input_dir> <output_path>` interface. Any blind-against-informed difference is then attributable to those additions.
- **What we cannot equalise, stated in the write-up:** our side had a night of iteration and many validation looks. Opus gets one 3-hour session per run.

## 5. Coverage report on the evaluation inputs

- **Decision:** yes.
- **What it reads:** zone IDs and stations from `data/eval_frozen/routes/` only. Never `raw/score/`.
- **What it reports:** share of each route's zones seen in training at the three zone-ID levels, the best reference share, new or redrawn zones, and DBO6.
- **Disclosure:** stated in the pre-registration. Its findings go into the informed Opus README (the item 4 principle).

## 6. Smaller rules (all agreed)

- **Re-running Opus-written code:** always inside its sandbox and memory scope, and twice, to measure solver noise.
- **DBO6 fallback** (station with no history), written before the test: the model runs with empty history features (pooled counts only). Opus's code does whatever it does.
- **v0-hist has never been scored on the pilot 50.** It is scored on them only in the final run, beside the Opus pilot arms (descriptive, n = 50).

## Order of work

1. **Now (light, no heavy CPU while runs 3–4 are going):**
   - write the v1 code and the rehearsal scripts;
   - write the cleaned reading list and the informed README;
   - draft the pre-registration.
2. **After runs 3–4 finish (~20:00 IST):**
   - score and audit them on validation;
   - run the coverage report;
   - run the search-error check and train v1 under the rule above;
   - run the rehearsal (diagnostic).
3. **Informed Opus runs 5 and 6** (one after the other; RAM allows only one): offline containment test first, as before.
4. **Develop the richer-history Opus-as-the-model prompt on validation routes;** freeze it.
5. **Lock the pre-registration.** Angad reads it and can veto.
6. **Final run, once:** all arms on the evaluation set; open `raw/score/` once.

## Final tightenings (review 2c, all adopted)

1. **Reading list content:** fixed as in item 4.
2. **Informed README:** identical to the blind README except for the three additions.
3. **No selecting the opponent by validation score:** a win must hold against every run separately (Holm). The same rule is extended to design A.
4. **v1 is the 5-seed average,** decided now. Adoption is tested on the 234 held-out High train routes, with ±14-day removal. A numeric non-inferiority margin is added (+0.002), so "not worse" cannot be argued after the fact.

## Implementation notes (v1 code written 2026-11-11 evening; fixed before any full run)

- **Dropout settings.** History dropout hides train routes within a random 0–28 days, on half of the training routes. These values were chosen when the code was written, before any number existed, and are now fixed.
- **v1 size.** 65 features and 29,377 parameters per seed, about 147k for the 5 seeds. The decoder is fixed at beam 5 + λ = 0.5.
- **Disclosure: the 234 adoption routes are not untouched.** Both v0-hist and v1 use them for early stopping, and v0's λ was tuned on them. This affects both arms alike.
- **The ±14-day thinning is harsh.** It removes 3,300–4,500 of the 5,762 train routes per target. That is intended: it approximates the final test's gaps.
- **Rehearsal.** 76 val targets (55 early, 21 late), using group cut-offs of 08-05 and 08-09, so every target is at least 12 days from its history. The volume-matched control removes the same total number of routes, not per station. It is diagnostic only.
- **Self-checks passed.** v1's history code reproduces v0's leave-one-out and past-only tables exactly, and v0-hist with full history reproduces its cached held-out scores exactly.

## Amendment after Opus run 3 (review 3; decided 2026-11-11 ~19:40 IST; Angad may veto)

Re-ran the analysis independently and got the same numbers.
- **Same zone order (220 routes):** run 3 is better by 0.0006 [0.0003, 0.0010] and wins 129 to 84. That edge can only come from stop order inside zones.
- **Different zone order (80 routes):** +0.0023 [−0.0039, +0.0081]. This is undetermined, not "tied". These routes carry 59% of the mean gap.
- **Where exactly one method has the driver's zone order:** run 3 on 12 routes, v0-hist on 9.

So the in-zone edge is established. The zone-order comparison is unresolved, not shown to be equal.

- **P1, adopted (analysis only, pre-registered):** a "same solver" version of the engineer fight. Each Opus run's zone order is filled in by our common in-zone solver and compared with our model. This isolates whose learned zone model is better. The as-built comparison stays the stricter test.
- **P2, adopted with conditions:** an exact in-zone solver.
  - **Method, fixed now with no knobs:** dynamic programming (Held–Karp) for zones of up to 12 stops; above that, OR-Tools guided local search with a fixed 1 s.
  - **Gate:** on the 234 held-out High train routes, using the DRIVER's zone order (so only this step varies), the paired 95% interval of new − old must be below 0. Otherwise keep greedy descent.
  - **Where it applies:** the design B final model and BOTH arms of the primary comparison (the Opus-as-the-model zone arm uses the same solver). Never design A, which stays exactly as frozen.
  - **Disclosure:** the design B model is reported both with and without the exact solver. The write-up states that the change was prompted by Opus run 3 and credits run 3.
  - **Why it is accepted despite the "no new fixes" rule:** this is a correctness fix (v0.md says the step finds the shortest path, and greedy descent does not guarantee that), not tuning. It is gated by a test that isolates it, and it helps both primary arms equally.
- **P3, adopted:**
  - **Load is logged into each run folder.** Run 4 is logged from 19:38 IST, late (`runs/opus_fight_run4/load_log.txt`).
  - **No heavy jobs of ours during runs 5 and 6.**
  - **Run 4 caveat:** at 19:38 the load average was 12.6 on 8 threads. Another session's manim render and piper TTS were using about 4–5 cores, while run 4 ran its evaluation. Opus's time-limited searches get less CPU under load, so run 4 may be handicapped. This is reported with its result.

## Amendment 2: the v1 gate is matched to the real test (review 4; decided 2026-11-11T15:34:57Z, BEFORE any adoption number existed: results/v1_adoption.* did not exist and the adoption step had not started)

**Calibration, re-run independently.** Each row shows the share of zones seen and the best single reference.

| History | Zones seen | Best reference |
|---|---|---|
| Hide ±3 days | 0.863 | 0.781 |
| Hide ±5 days | 0.830 | 0.747 |
| Hide ±7 days | 0.784 | 0.704 |
| Hide ±9–10 days | about 0.71–0.73 | about 0.64 |
| Hide ±14 days | 0.530 | 0.480 |
| Final test, all | 0.812 | 0.718 |
| Final test, before half | 0.878 | — |
| Final test, after half | 0.743 | — |

So the ±14-day gate was far harsher than the test. It was proposed in review 2c, before the coverage report existed.

**Amended rule:**
- **(a) Matched condition (decides):** for each held-out route, the mean of (v1 − v0-hist) with ±3 days hidden and with ±9 days hidden. Those match the test's before and after halves, which are about 50/50 of the test. Adopt only if the paired 95% interval's upper end is below 0.
  - The 3/9 mixture was chosen over a single 6-day window, because the test is bimodal and coverage is not linear in the window. ±6 days is reported.
- **(b) Full history:** unchanged, upper end below +0.002.
- **Reported only:** ±6 days, and ±14 days as a labelled stress test.

**Fallback, accepted and fixed now:**
- If v1 fails (a) or (b), there is ONE retrain with the dropout window set from the coverage report: D uniform 0–10 days, p = 0.5, everything else identical. It faces the same gate.
- Both attempts are reported. Nothing is tried after that.
- Rationale: the 0–28 window was set before any coverage data existed. A quarter of training routes saw thinner history than anything in the test.

**Rehearsal:** GAP 5 added as the matched read (`results/rehearsal_oot_gap5.*`). GAP 11 is kept as the stress read. Both are diagnostic only.

**Implementation:**
- `scripts/v1_adoption_test.py` was amended; the original is kept at `runs/evening/v1_adoption_test_ORIGINAL.py`.
- It was smoke-tested on 3 routes with seed 0, with all output discarded, so no adoption number was seen.
