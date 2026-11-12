# Decision log: problems we hit, how we handled them, and why

A running record of every important choice in tacit-routes. For each one: what the problem was, the options we saw, what we picked and why, and what happened. Numbers trace to files in `results/`. This file is the "why" behind `docs/plan.md` (the agreed final-test plan) and `PREREGISTRATION.md` (the locked version).


**Status at the time of writing (2026-11-11, 22:30 IST):**
- v1 adopted; Opus informed runs 5–6 running overnight.
- Pre-registration not yet locked; final test not yet run.

---

## Part 1: framing the project

### D1. What is the project trying to prove?
- **Problem:** Hyde's pitch is that small specialist models trained on a company's own data beat frontier models on narrow tasks. Angad: "my absolute main goal is to beat a frontier model at a small task, that is what Hyde is all about."
- **Task chosen:** predict the order real Amazon drivers visit their stops (2021 Last Mile Routing Challenge, public, CC-BY-NC). Drivers deviate from the shortest route because of habits, unwritten "tacit" knowledge that a model can learn from history.
- **Alternatives considered:** Catan (felt like copying), wildfire, PII redaction (in `../reports/`).
- **Why this one:** real data, an official metric, a published leaderboard to calibrate against, and a clean story: learned habits versus a general reasoner.

### D2. The headline: frontier as the model, or frontier as an engineer?
- **Problem:** "Beat Opus" can mean two very different things.
  - (a) Our small model against Opus *answering directly* (Opus is the model).
  - (b) Our model against a system *Opus builds* when given the data (Opus is the engineer).
- **Alternatives:**
  - Lead with (b) as the "fair fight".
  - Lead with the reframe from review 1: improve on the 2021 winner.
  - Lead with (a).
- **Decision:** (a) is the primary claim, because it is literally Hyde's thesis. (b) is a key secondary. The 2021-winner comparison is a reference point, not the goal. Angad rejected the reframe on 2026-11-11.
- **Why:** a small model beating the frontier model used as the model is what Hyde sells. The engineer fight is a robustness story: Opus building a small model is itself evidence *for* "use specialists".

### D3. What does "fair" mean when Opus is the model?
- **Problem:** Opus ordering 150 stops directly scored 0.069, barely better than the shortest route (0.078, p = 0.20). Critics would say "Opus never saw the history" and "LLMs can't hand-solve a 150-stop TSP".
- **Approach:** remove both excuses.
  - Give Opus 3 similar past routes. Score: 0.0355, beating its direct answer on 44/50 routes.
  - Build a "swap the brain" arm: Opus picks only the zone order (about 20 zones), and our own solver fills in the stops. Score: 0.0355 at a quarter of the cost.
- **Later refinement (review 2):** Opus saw 3 routes, while our model's features count all 5,762. So the primary arm will give Opus richer history: counted next-zone statistics, usual first zones, and 5–10 reference routes. The prompt is developed on val, then frozen.

---

## Part 2: rigour and leakage

### D4. Test sets and the "never peek" rule
- **Problem:** with many variants tried on validation, any val win is optimistic.
- **Decision:**
  - The 50 pilot routes are used only for frontier comparisons and are never used to tune our models.
  - The 3,052 official evaluation routes are a frozen final test (Angad approved). Their answers stay sealed (chmod 444; only hashes recorded) until one pre-registered run.
- **Alternatives:** use val as the final test (rejected: looked at too many times); split off part of train (rejected: the eval set puts our number on the 2021 leaderboard's scale).

### D5. Privacy constraints shape the tooling
- **Problem:** Angad wants nothing visible on other devices or to anyone logged into his account.
- **Decision:**
  - No Artifacts, Docs, Drive or cloud sessions.
  - Opus calls use `claude -p --no-session-persistence` from empty folders.
  - We verified that one test call did not appear in his session list.
  - Everything is local files (the guide is a local HTML file served on 127.0.0.1).

### D6. Opus fair fight: how to let a frontier model write code safely
- **Problem:** Opus needs tools and the training data, but must not reach val answers, the eval set, the internet or Angad's files.
- **Approach:**
  - Claude Code's bubblewrap sandbox: reads and writes confined to an "arena" folder, no network, no unsandboxed escape, file tools restricted.
  - systemd memory and CPU caps.
  - Before every run, an offline containment test against a *fake* local API that scripts escape attempts. Every attempt must fail and normal work must succeed.
  - After every run, a transcript audit.
- **Problems we hit:**
  - **Run 1 was killed at 67 minutes by our memory cap.** The whole scope died when one process exceeded 3 GB. Fix: `OOMPolicy=continue`, plus a note in the prompt. Run 1 is reported but excluded from win rules.
  - **The fake API server from run 2 had not been saved.** It was rebuilt as `scripts/fake_anthropic_api.py` for runs 3–6.
  - **The audit script only knew run 1 and run 2's folder names,** so it flagged run 3's own folder as "outside". Fixed to derive each run's own arena; run 2's audit is unchanged under the fix.
  - **A cleanup command on the test leftovers was blocked by a safety check.** It was unnecessary (runs 3–4 had kept the same internal folder), so we dropped it.

---

## Part 3: the model

### D7. Why model zones, not stops
- **Evidence:**
  - Drivers almost never revisit a zone (0.96 per route, against 14.5 for the shortest route).
  - With the driver's own zone order, our inside-zone solver scores 0.0072.
- **Decision:** the model predicts the zone order; a solver orders stops inside each zone. Nearly all the error is in the zone order, so that is where learning goes.

### D8. Station history is the weapon (review 1)
- **Problem:** v0 (geometry and zone-ID features only) scored 0.0356, losing to a hand rule (0.0305). It also guessed route direction at about chance.
- **Source:** the 2021 winner's paper. Copying the precedence of one past route at the same station cut its score by 35%.
- **Decision:** v0-hist adds 18 history features from train routes, leave-one-out so a route never sees itself. Score: 0.0229, and direction mostly fixed.
- **Controls added so the claim is "the model learns", not "counting works":**
  - R4-hist, history counting with no network: 0.0331.
  - The R2 zone-ID rule: 0.0305.
  - v0-hist beats both.

### D9. Report the history table's size honestly
- The model is 28,481 weights, but it also carries a history table built from 5,762 routes (115,518 entries, 1.9 MB).
- Every size claim states both. The same applies to Opus's systems and the 2021 winner.

### D10. GPU or CPU?
- **Decision:** CPU.
- **Why:**
  - At about 28k parameters, GPU overhead outweighs any gain.
  - Most of the time goes on feature building, OR-Tools and the pure-Python scorer, all CPU-only.
  - RAM (7 GB) is the binding limit, not compute.
  - Opus's arenas were CPU-only too, so the comparison is on equal hardware.
  - "Runs on a laptop CPU" is part of the pitch.

---

## Part 4: the opponents moved, and how we responded

### D11. Opus runs 3 and 4 tie or beat v0-hist on val
- **What happened:** Opus fair-fight runs scored 0.0349 (run 1, cut short), 0.0268, then **0.0219 and 0.0215**. v0-hist is 0.0229. Run 3 against v0-hist: +0.0010 [−0.0005, +0.0027], with Opus winning 175 routes to 118.
- **Diagnosis (review 3, verified):**
  - On about 220 routes, both chose the same zone order, and Opus was better there by 0.0006. That can only come from its **exact inside-zone shortest path**; we use a greedy local search.
  - On the about 80 routes where the zone orders differ, the result is undetermined: the interval is wide and those routes carry 59% of the gap.
- **What we did NOT do:** invent new fixes to beat Opus after seeing its number. That would tune the model to the opponent.
- **What we did:** see D13 (pre-planned v1) and D15 (exact solver, a disclosed correctness fix).

### D12. One "beat Opus" rule that can't be cherry-picked (review 2c)
- **Problem:** with several Opus runs, picking the one to compare against invites cherry-picking.
- **Decision:** a win requires beating **every** run separately, Holm-corrected. This applies to the blind runs 2–4 (run 1 excluded, cut short by our bug) and to the informed runs 5–6. No opponent is picked by validation score.

### D13. The out-of-time risk (review 2), and v1
- **Problem:**
  - Validation is a random split inside the training dates; the final test runs June–October, mostly outside them.
  - v0-hist had only ever seen dense history.
  - The validation win could vanish when history is thin.
- **Measured:** exact-zone history coverage falls by about a quarter as the date gap grows to 4 weeks. Opus's systems count history at 3 zone-ID levels; v0-hist only at exact IDs.
- **Options:**
  - Freeze v0-hist now and accept the risk.
  - Open-ended improvement (rejected: endless tinkering).
  - **One pre-specified replacement with a written adoption rule** (chosen).
- **v1, specified before any number:** v0-hist plus history at the coarser zone-ID levels, plus history dropout during training, as a 5-seed average (about 147k parameters).
- **Adoption rule:** on 234 held-out train routes, v1 must beat v0-hist with history thinned, and must not be worse than +0.002 with full history.

### D14. The thinning gate was miscalibrated, and we fixed it before seeing results (review 4)
- **Problem:** the gate hid ±14 days of history, which left 53% of zones seen. The real final test, measured by the coverage report, has 81% seen, which corresponds to hiding about ±5–7 days. A harsh gate could pass v1 for winning a scenario the test doesn't contain.
- **Decision (timestamped before any adoption number existed):**
  - The deciding condition became the per-route mean at ±3 and ±9 days, matching the test's before and after halves.
  - Chosen over a single 6-day window, because the test is bimodal and coverage is not linear in the window.
  - ±6 and ±14 days are reported only. A one-shot fallback retrain was fixed in advance.
- **Result:** **v1 ADOPTED.**
  - Matched condition: −0.0015 [−0.0025, −0.0006].
  - Full history: −0.0018.
  - Better under every condition. Fallback not needed.
- **Supporting diagnostics (76 val routes, wide intervals):** with test-like thinning, v0-hist got +0.0024 worse, while v1 got only +0.0007 worse.

### D15. Exact inside-zone solver (P2), accepted as a correctness fix
- **Problem:** our docs say "shortest path inside each zone", but greedy descent doesn't guarantee it. Opus run 3 solves it exactly.
- **Tension:** this fix was prompted by seeing an opponent's result.
- **Resolution:**
  - **Fixed method:** Held–Karp dynamic programming for ≤12 stops; OR-Tools with 1 s above that.
  - **Gated in isolation** using the driver's own zone order. Adopted only if it clearly improves.
  - **Scope:** applied to the informed-design model and both arms of the primary comparison (the Opus-as-model arm uses the same solver); never to the blind design.
  - **Disclosure:** reported with and without it, and credited to Opus run 3.
- **Status (2026-11-12 01:35 IST):** built (`scripts/exact_zone.py`; matched brute force on 236 zones) and gated. **ADOPT**: exact − greedy −0.00023 [−0.00045, −0.000005], 134/95/5, sign p 0.012, n = 234 held-out train routes (`results/exact_solver_gate.md`). The upper end clears 0 by a hair, so the honest reading is "a small gain that only just passes the bar we set". Other way: had it failed, FINAL would have kept greedy descent and OPUS-MODEL with it. On val (descriptive) v1 exact − v1 greedy = −0.0004 [−0.0006, −0.0002]. CPU 1.8 s vs 0.09 s per route.

### D16. Blind versus informed: what Opus is told (review 2b; the most important catch)
- **Problem:** Opus runs 1–4 were told only about the 300 val routes. They never knew the final test is out of time. If we harden our model for the shift and Opus wasn't told, the comparison is unfair.
- **Decision:** two consistent designs, both reported.
  - **A, both blind:** v0-hist as frozen before the shift work, against runs 2–4.
  - **B, both informed:** our final model (v1) against runs 5–6. Their README is the blind one word for word plus three additions: the test facts with coverage numbers, a reading list (the published literature, cleaned of all our own results), and an input/output interface.
  - **The mixed case is never a headline.**
- **Why give Opus the reading list:** v0-hist's history idea came from that literature, so an informed opponent without it would know less than we did.

### D17. Machine load as a fairness issue (P3)
- **Problem:** during run 4, another session's manim render and text-to-speech pushed the load to 12–17 on 8 threads. Opus's code uses time-limited searches, so load can handicap it. CPU temperature reached 85 °C (limit 100 °C; no damage risk, but it throttles).
- **Decision:**
  - Run 4's load is logged and reported as a caveat.
  - Runs 5–6 log load every minute.
  - None of our heavy jobs run during them; they start only after all evening work finishes.
- **Recommended to Angad:** pause, rather than lower the priority of, other heavy jobs during Opus runs.

---

## Part 5: facts we got wrong and corrected

| Claim | Correction | How we found out |
|---|---|---|
| "Top of the field is 0.0374 (Wu et al.)" | The 2021 winner scored 0.0248, with no learned model | review 1; checked on the team page |
| "The final test has Low-quality routes mixed in" | The eval set is all High quality | review 2; verified in Wu et al. v2, Sections 1 and 3 |
| "Any change now weakens the claim" | Overstated. Changes chosen on train-only evidence don't touch the sealed test; the risk is tinkering, which written rules control | review 2 |
| "Zone-order models are tied"  | Undetermined, not tied: the differing routes carry 59% of the gap with a wide interval | Re-check of review 3 |
| The ±14-day gate | Far harsher than the real test; recalibrated | review 4, against the coverage report |

---

## Part 6: how we work (process decisions)

- **Pre-register before results.** Every gate, margin and fallback is written, with a timestamp, before the number it governs exists. When a rule had to change (D14), it changed before any result and the original script was kept.
- **Outside review.** Each review point was checked by re-running the scripts and reading the sources before it was accepted, amended or rejected.
- **Report losses.** v0 without history lost to a hand rule; Opus runs 3–4 edge out v0-hist on val; the 2021 winner beats everything. All of this stays in.
- **One-shot final test.** The answers are opened once, by a pre-registered script.

## Part 7: still open

1. ~~Score v1 on all 300 val routes, and Opus runs 5–6.~~ Done 2026-11-12 01:38 IST (`results/v1_val.md`, `results/opus_fight_run{5,6}.json`).
2. ~~Build and gate the exact inside-zone solver (D15).~~ Done: ADOPT, narrowly.
3. ~~Develop and freeze the richer-history Opus-as-the-model prompt on val (D3).~~ Done: prompt A, max effort (`runs/zone_rich/FROZEN.json`).
4. ~~Lock the pre-registration.~~ Done 2026-11-12 09:45Z (`runs/final/LOCK_RECORD.json`).
5. ~~The final run on the evaluation set, once.~~ Done (`results/final_test.md`).

## Where the evidence lives

| Topic | File |
|---|---|
| Agreed final-test plan and amendments | `docs/plan.md` |
| Val comparisons | `results/val_comparison.md`, `results/opus_fight_run{2,3,4}.*` |
| v1 | `results/v1_adoption.md`, `results/rehearsal_oot.md`, `results/rehearsal_oot_gap5.md` |
| Final-test coverage | `results/eval_coverage.md` |
| Opus brief additions | `runs/opus_informed_brief/` |

---

## Update 2026-11-11 22:45 IST

### D18. Starting the final stretch without breaking our own rules
- **Request:** Angad asked to start all remaining steps at once.
- **Tension:**
  - Heavy CPU work during Opus runs 5–6 would handicap them (D17).
  - Opening the sealed answers is irreversible.
- **Decision:**
  - Write and smoke-test all code now at low CPU. Opus prompt development is fine now, because the load is on Anthropic's side.
  - Heavy scoring runs automatically after run 6 finishes.
  - The final scoring script is hard-locked behind `PREREGISTRATION.md` + `runs/final/PREREG_CONFIG.json`. It is locked only after Angad has read the draft. That is the one human checkpoint kept, because it cannot be undone.
- **Alternatives:** run everything immediately (rejected: unfair to Opus runs 5–6 and risks an unreviewed irreversible step); wait until morning to start anything (rejected: wastes the night).
- **Also decided:** DeepSeek is not added to the final test (Angad: "skip it"). Beating the strongest frontier model covers the weaker one, and adding it would need new spend.

### D19. Closing loopholes that could create a hollow win (review 5)
- **Problem:** at n = 3,052, one invalid route moves a mean by 0.0003; at n = 500, by 0.002. That is the size of the effects we are measuring. So scoring and harness choices could manufacture a win.
- **Decisions:**
  - The primary uses the *repaired* view of Opus's answers (favours Opus).
  - Opus-built code runs in 300-route batches, and harness failures are fixed rather than scored as invalid.
  - Every claim must hold both with and without DBO6.
  - An end-to-end dress rehearsal on validation must reproduce known numbers before the lock.
  - A pre-stated non-inferiority claim gives a draw a meaning: within 0.002 at ≥100× lower cost.
  - Opus-as-the-model gets whichever effort level (default or max) scores better for it on the development routes.
- **Principle:** every ambiguity is resolved in the opponent's favour, so any win we report survives the harshest reading.

---

## Part 8: defending each decision: the number, its sample size, and "what if it had gone the other way"

This follows Hyde's own published grading standard: their Catan hackathon rubric (`../research_notes/Hyde final project ideas/hyde_signals.md`, around line 213), checked against the official problem statement. To defend a decision they pick, it asks for "the number behind it, its sample size, and what you would have concluded if that number had come out the other way".

| Decision | The number (n) | If it had come out the other way |
|---|---|---|
| D3: Opus needs history to be a fair opponent | Opus + 3 past routes 0.0355 vs Opus direct 0.0693; better on 44/50 (n = 50 pilot) | If history hadn't helped Opus, "Opus never saw the history" would have been a weak objection, and Opus direct would have been a fair primary opponent |
| D7: model zones, not stops | Driver zone revisits 0.96 per route vs 14.5 for the shortest route (n = 50); driver-zone-order oracle 0.0072 (n = 300 val) | If drivers revisited zones often, zone order wouldn't define the route, and we'd need a stop-level model (much harder; published stop-level pointer nets reach ~0.037) |
| D8: station history is the weapon | v0 0.0356 → v0-hist 0.0229; history better on 170/221 non-tied routes (n = 300 val) | If history hadn't helped, the tacit-knowledge thesis would be in doubt; the model would only be learning geometry, which a hand rule (0.0305) already does better |
| D8: the network adds more than counting | v0-hist − R4-hist −0.0101 [−0.0131, −0.0074] (n = 300) | If counting had matched the network, the honest claim would be "a lookup table beats Opus", with no need for a model |
| D11: Opus as engineer is strong | Runs 3/4: 0.0219 / 0.0215 vs v0-hist 0.0229 (n = 300 val) | If Opus had stayed at run 2's 0.0268, the engineer fight would have looked won on val, but the run-to-run spread (0.005) would still have made one run insufficient |
| D13/D14: adopt v1 | Matched thinning: −0.0015 [−0.0025, −0.0006], 48/25 (n = 234 held-out train) | If v1 had failed, we'd have done one calibrated retrain, then kept v0-hist and reported the out-of-time risk as unmitigated |
| Rehearsal (diagnostic) | Test-like thinning: v1 +0.0007 vs v0-hist +0.0024 (n = 76 val; wide intervals) | If v1 had degraded as much as v0-hist, dropout would have been doing nothing useful; this was not a decision gate |
| D15: exact solver | Gate result **[pending]** (n = 234, driver zone order) | If it fails, greedy stays; the in-zone gap to Opus then remains, and is disclosed as a known loss |
| Search-error check | Beam loses a higher-rated driver order on 2/234 routes | If search errors were common, a wider beam would have been the cheap fix; since they're rare, remaining errors are model errors |
| Primary-arm effort | **[pending]**: default vs max on 40 dev routes | Whichever is better for Opus is used; either way Opus gets its best setting |

## Part 9: where our model loses (diagnosis with evidence)

**On validation** (final-test version pending):
- **Wrong zone order on about a fifth of routes, carrying over half the error.**
  - On 60 of 300 routes, v0-hist's zone order is scrambled or reversed (Kendall tau ≤ 0.5). Those routes carry 53% of its total error, and the 2021 winner is better on them (0.046 vs 0.061).
  - On the 158 routes where our zone order is right, we beat the winner (0.0068 vs 0.0081).
  - Source: `a one-off analysis script`.
- **Inside zones.** On routes where we and Opus pick the same zone order, Opus is better by 0.0006, because it solves paths exactly. Addressed by D15, if the gate passes.
- **Without history.** v0 without history gets route direction about at chance (tau +0.07) and loses to a hand rule.

**What we tried for the zone-order tail:**
- Beam 5 vs greedy (helps).
- A search-error check, which showed the beam isn't the problem.
- Multi-level history and dropout (v1).

**What we did not try, and what it might cost:**
- Learned choice between candidate zone orders (minimum Bayes risk decoding).
- jpt-style hard precedence constraints from a reference route.
- A seed ensemble beyond v1's 5.
- A stop-level model.
- Training on Low-quality routes (excluded; 102 routes).

## Part 10: things we did not measure

- **The fair-fight Opus runs ran at default effort.** We never measured whether higher effort would make Opus a better engineer. (For Opus-as-the-model, effort is being tested.)
- **jpt-amz at contest settings** (about 42 s + 20 s per route). We ran it at 1 s + 1 s, so its final-test number is a reference, not a replication of 0.0248.
- **Time windows, package volume and service time** as features beyond v0's basic ones. Published evidence says time windows hurt.
- **Variance across our own training seeds** for v0-hist (single seed). v1 averages 5 seeds but doesn't report their spread yet.
- **How Opus-as-the-model degrades with less history** (the planned post-test data-efficiency curve).
- **Real-world value.** Whether better stop-order prediction helps any operational decision. This is a prediction benchmark only.
- **Cost of the Opus fair-fight sessions in real money.** The CLI cost field is list price; the calls ran on a Max subscription.

## Update 2026-11-12 01:45 IST: the last validation numbers before the final test

**Opus informed runs 5 and 6** (told the final-test facts, given the reading list): run 5 took 98 min, cost $2.74 and scored 0.0214 [0.0185, 0.0244] on val; run 6 took 90 min, cost $3.47 and scored 0.0216 [0.0187, 0.0247]. Audits clean (run 6's four "answer hunting" flags are one command using `../` inside its own arena; reviewed in `runs/opus_fight_run6/audit_manual_review.txt`). Both hardened themselves against out-of-time history.

**FINAL = v1 + exact solver scores 0.0224 on val**, against 0.0214–0.0219 for Opus runs 3–6 and 0.0213 for jpt. Every v1-exact − Opus gap is +0.0005 to +0.0011 with a CI that includes 0, and v1 wins MORE routes than it loses against runs 5 and 6 (168/130, 169/129): Opus's programs lose more often but by less; ours has a tail of bad routes (Part 9). Val is in-time, which removes the case v1 was built for (thin history), so these numbers can't settle the engineer fight. They do lower the prior on the design-B claim: expect a draw or a narrow loss, and the pre-registration already says how that is reported.

What would change this reading: if the final test (mostly out of time) shows v1 ahead of runs 5–6, the out-of-time hardening is the reason to look at; if it shows v1 behind, the tail routes are (Part 9).
