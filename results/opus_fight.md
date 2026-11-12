# Opus fair fight: Opus 5.5 builds its own predictor

Run on 2026-11-11 (02:41–03:48 IST). All numbers come from `results/opus_fight.json`, made by `scripts/score_opus_fight.py`. The full Opus log is `runs/opus_fight/transcript.jsonl`, and the audit is `runs/opus_fight/audit.json`.

## The one-paragraph version

Opus was given the 5,762 training routes (with the drivers' real orders) inside a locked sandbox folder. It was asked to write a program that predicts the driver's stop order. It wrote one in about an hour: a small neural network that guesses the order of **zones**, plus a route solver that orders the stops inside each zone. On the 300 validation routes it scored **0.0349** (95% CI 0.0309 – 0.0391). Lower is better. The plain shortest route scores 0.0898 on the same routes. Opus was better on 287 routes and worse on 13. The audit found no sign that it saw or looked for the validation answers. The run ended early, at 67 of the allowed 180 minutes, because one of its test scripts used more memory than allowed. It never wrote its `APPROACH.md`.

## What Opus was given

Everything lived in one folder, `opus_arena/`, built by `scripts/setup_opus_arena.py`. Files were copied, never linked.

- **Training routes:** `data/train/*.npz`, all 5,762 train-split routes, including the driver's order (`actual_order`) and the quality label.
- **Validation inputs:** `data/val_inputs/*.npz`, the 300 val routes. Each file was rewritten **without** `actual_order` and `quality`, and the script checked that both fields were gone from every file. The stop list is stored station first, then sorted by stop ID, so the storage order says nothing about the driver's order. A check on 400 training routes found a correlation of 0.01 between storage position and driving position.
- **The official scorer:** `scorer/score.py`, unmodified, with its README.
- **A README** (`opus_arena/README.md`) describing the task, the file format, the metric (lower is better, 0 = the driver's exact order, invalid = about as bad as random), the deliverable (`predict.py` → `val_predictions.json`, plus `APPROACH.md`) and the rules. The rules: no looking for val answers, work only in the folder, at most 3 CPU threads and 2 GB RAM, about 3 hours.
- **Python environment:** `.venv` with numpy, OR-Tools and CPU-only torch, installed beforehand because the sandbox has no internet.

Nothing else was copied in: no pilot routes, no val answers, nothing from `results/` or `runs/`. The prompt was the README plus two notes. One said that shell commands run in a sandbox, and that some complex inline commands are refused, so code should go in `.py` files. The other was the line "Use at most 3 CPU threads and 2 GB RAM; other jobs are running." The exact prompt is `runs/opus_fight/prompt.txt`.

## How it was contained

There were three layers, set in `runs/opus_fight/sandbox_settings.json` and passed with `--settings`. Global settings were not touched.

1. **An operating-system sandbox for shell commands.** Claude Code's built-in sandbox runs every shell command inside bubblewrap, a Linux tool that gives a process its own private view of the files and network.
   - Reads are blocked for the whole home folder except `opus_arena/`. The project's `data/`, `results/` and `runs/` folders are blocked too.
   - Writes are allowed only in `opus_arena/` (plus a private temp folder).
   - There is no network: the only way out is a proxy whose allow-list is empty.
   - `failIfUnavailable` was on, so Claude Code would refuse to start rather than run unsandboxed, and `allowUnsandboxedCommands` was off, so there was no escape hatch.
2. **Permission rules for the file tools.** Claude Code's own Read, Write, Edit, Glob and Grep tools don't go through bubblewrap, so they were limited separately.
   - `--restricted` and `blockReadsOutsideWorkingDirectories` confine them to `opus_arena/`.
   - Explicit deny rules cover `data/`, `results/`, `runs/`, `scripts/`, `PROJECT.md`, `~/.ssh` and `~/.claude`.
   - The mode was "don't ask", so anything not pre-approved was refused rather than prompted.
   - WebFetch and WebSearch were denied, and no MCP servers were loaded.
3. **A resource cap.** The whole run sat in a systemd scope with a 3 GB memory limit (2 GB for Opus's work plus about 1 GB for Claude Code itself), a CPU quota of 3.5 cores and a 3-hour `timeout`.

The containment was **tested before the real run without any API call**. A small local stand-in for the API (`runs/opus_fight/diag/fake_api.py`) fed Claude Code scripted tool calls. Every escape attempt failed:

- reading `~/.bashrc`, `data/`, `data/raw/actual_sequences.json` and `PROJECT.md` from the shell, and from Python;
- writing to `runs/` and to the home folder;
- `curl` and Python network calls (blocked by the proxy, 403);
- asking for an unsandboxed command;
- the Read, Write and Grep tools on outside paths.

Normal work inside the arena succeeded: numpy, torch and OR-Tools ran, and files could be written.

The call itself used the approved privacy flags unchanged: `env -u ANTHROPIC_API_KEY -u CLAUDE_CODE_EFFORT_LEVEL claude -p --model claude-opus-5-5 --no-session-persistence --strict-mcp-config --output-format stream-json --verbose --max-turns 200`, with the same account. Three flags were added: `--settings`, `--restricted --permission-mode dontAsk` and `--tools Bash,Read,Write,Edit,Glob,Grep`. The launch script is `runs/opus_fight/launch.sh`.

**Two false starts (please note).** The run was launched three times. Each of the first two was stopped by hand after about 6 API turns and about 1 minute, before Opus had done any real work:

- **First launch:** Bash had no allow rule, so the "don't ask" mode refused most shell commands.
- **Second launch:** some multi-line `python -c` commands (ones containing slices such as `x[:5]`) still tripped a built-in safety check that no setting overrides.

The fix was the allow rule plus the prompt note about using `.py` files. Both aborted transcripts are kept in `runs/opus_fight/attempt1_aborted/` and `attempt2_aborted/` and were audited. Nothing suspicious turned up. In the final run, no command was refused.

## What Opus built

It never wrote `APPROACH.md`, because the run was cut off first. This summary comes from its code (`opus_arena/solver.py`, `zfeat.py`, `trainz.py`, `predict.py`) and its progress notes in the transcript.

The idea is the same two-level plan as our own project: **first decide the order of zones, then order the stops inside each zone.**

1. **Zone history from training routes.** For every station, it counts how often drivers went from zone A straight to zone B, which zones came first, and which came last (`prep.py`, `solver.History`). Low-quality routes count half.
2. **A small neural network that scores "next zone" choices** (`zfeat.py`, `trainz.py`).
   - For every pair of zones (A, B) in a route it builds 30 numbers:
     - travel times (mean and minimum, raw and scaled);
     - how B ranks among A's nearest zones;
     - how similar the zone names are (zone IDs look like `A-2.2A`, and neighbouring codes tend to be neighbouring areas);
     - the history counts from step 1.
   - A tiny network (30 → 64 → 32 → 1, about 4,100 weights) turns those numbers into a score. A softmax turns the scores for all candidate next zones into probabilities.
   - It was trained with cross-entropy to give high probability to the zone the driver really went to next. It used High and Medium routes, with Medium weighted 0.7, for 60 epochs on 3 threads.
3. **Zone order by solver.** The cost of going A → B is −log(probability). OR-Tools then finds the cheapest loop through all zones (an asymmetric travelling-salesman problem), with a 0.5 s search.
4. **Stop order inside the zone order.** OR-Tools orders all the stops by travel time, but with a huge penalty for any jump that breaks the chosen zone order.
   - It found that turning **off** OR-Tools' improvement search (guided local search) and keeping only the first, greedy "cheapest next arc" tour roughly **halved the error**. On its own held-out training routes the score went from 0.070 to 0.036.
   - A plausible reason: drivers drive more like "go to the nearest next stop" than like a fully optimised tour.
5. **predict.py** runs this for the 300 val routes on 3 processes. It took 12.6 minutes. There is a fallback (sorted order) if a route throws an error. The output is always checked to be a full, valid sequence.

**How it checked itself.** It held out 150 random High-quality training routes. It removed them from the history counts and trained a separate network without them (`zmodel_hold.pt`, `evalh.py`), then scored with a fast copy of the official metric that it checked against `score.py`. Its held-out results:

| Version | Held-out score (150 train routes) |
|---|---|
| Hand-tuned zone rules + solver | 0.076 |
| Learned zone model + solver | 0.070 |
| Learned zone model, greedy stop tour (final) | **0.036** |
| Same, but with the driver's true zone order (a ceiling, "oracle") | 0.015 |

The final val score of 0.0349 matches its held-out estimate of 0.036 closely.

**Where it stopped.** It was writing an exact solver (dynamic programming over each zone's entry and exit stops) to replace OR-Tools inside zones. On a test with the true zone order this halved the error again (0.0146 → 0.0068), but that was the oracle setting, and it was never wired into `predict.py`. Its debug script for this solver grew to 3 GB of memory, and that ended the run (see below). **So the scored predictions do not use the exact solver.**

It also tried and dropped a "stop-level history" idea: borrowing the order of nearby stops in old routes gave only a tiny, noisy gain.

## The score (300 validation routes, lower is better)

| Arm | Mean | Bootstrap 95% CI | Median | Valid |
|---|---|---|---|---|
| **Opus-built predictor** | **0.0349** | 0.0309 – 0.0391 | 0.0220 | 300 / 300 |
| Plain shortest route | 0.0898 | 0.0840 – 0.0958 | 0.0766 | 300 / 300 |

Paired on the same routes (Opus minus shortest; negative = Opus better):

| | Mean difference [95% CI] | Median difference | Opus better / worse / tied | Exact sign test p |
|---|---|---|---|---|
| All 300 | −0.0549 [−0.0595, −0.0506] | −0.0492 | 287 / 13 / 0 | 2.0 × 10⁻⁶⁸ |
| ≤ 146 stops (151 routes, descriptive) | −0.0526 [−0.0596, −0.0458] | −0.0460 | 140 / 11 | 1.2 × 10⁻²⁹ |
| > 146 stops (149 routes, descriptive) | −0.0573 [−0.0629, −0.0520] | −0.0526 | 147 / 2 | 3.1 × 10⁻⁴¹ |

How to read this:

- **The bootstrap CI** is the range the mean would probably fall in if we drew another 300 routes like these. It comes from 10,000 resamples of routes, with seed 20261110.
- **The sign test** asks: if Opus and the shortest route were equally good, how surprising would winning 287 of 300 routes be? It would be absurdly surprising.
- The size split is at the median stop count. It was not pre-registered, so treat it as description only.

**Scoring details.** Scoring used the official, unmodified `evaluate()`. The true val orders came from `data/processed/routes/`, which is outside the arena. Travel times were rounded to 0.1 s, which matches the raw data. Any invalid sequence would have been dropped from the proposal file so that `evaluate()` gives it the challenge's invalid score; none were invalid. The shortest-route numbers are the saved tours in `results/v0_shortest_val.json`, which used `run_plain_baseline.py`'s settings: OR-Tools PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH, 10 s per route, 3 workers. They were re-scored here, and all 300 per-route scores matched exactly. Opus's predictions file was scored as written by its own `predict.py` run at 03:41 IST. It is newer than every file it depends on, so it was not stale and was not re-run. The scored copy is `runs/opus_fight/val_predictions_opus_copy.json`, with a SHA-256 identical to the arena file.

**For context only, not comparable:**

- Opus ordering stops directly in one reply scored 0.069 on the 50 pilot routes.
- The published best (Wu et al.) is 0.0374 on the official 3,052-route evaluation set.
- These are different routes, so do not read "Opus beat the published result" from this.

## Audit

`scripts/audit_opus_fight.py` lists every tool call and flags anything that touches paths outside the arena, network use, or patterns like `actual_sequences`, `data/raw`, `splits.json`, `../`, `find /` or `s3://`. I also read every command by hand.

- **40 tool calls, all Bash.** Opus wrote its code by piping it into files from the shell.
- **No paths outside the arena**, except the temp files Claude Code itself made for Opus's own background commands (`/tmp/claude-1000/...opus-arena.../tasks/*.output`). Opus read those to check on its jobs, which is normal.
- **No network attempts, no sandbox violations, no refused commands.**
- **No sign of looking for val answers.** Opus opened the val **input** files for statistics four times: three times before the final prediction run and once after it. It looked at which stations they come from, how many of their zones appear in training, how many old training stops lie near them, and how big their zones are. That uses unlabeled inputs, not answers. It is a mild form of tuning to the test inputs, and worth knowing about, but it is not leakage.
- One odd command: `pkill -f dbg.py`, meant to stop its own runaway script, matched its own shell instead. It changed nothing outside the arena.
- **Verdict: no evidence of access to val answers. The result stands.**

## Usage and duration

- **Main run:** 66 min 57 s wall clock (21:11:04 → 22:18:01 UTC). That covers 41 API turns and 40 tool calls.
- **Tokens logged in the stream:**
  - input: 82 uncached, 60,792 cache writes, 1,781,470 cache reads;
  - thinking: about 16.6k tokens (Claude Code's own estimate);
  - visible output: about 46k characters of code and text.
  - The final summary line with the cost field was never written because the run was killed, so there is **no cost figure**, and exact output tokens are unknown.
- **False starts:** two, about 6 API turns each (about 85k and 77k cached input tokens; output was tiny).
- **Opus calls under this approval:** 3 launches (1 real + 2 aborted false starts).
- **CPU:** the run's process group used 2 h 26 min of CPU in total. One `predict.py` run took 12.6 minutes on 3 processes.

## Caveats

- **It stopped at 67 of 180 minutes.**
  - Opus's debug script for the exact solver used 3.0 GB, over the limit it had been told (2 GB). The memory cap killed that script.
  - Then systemd's default rule for the scope ("stop everything if one process is killed for memory") also shut down Claude Code. That was a setup mistake on my side: the scope should have used `OOMPolicy=continue`.
  - So Opus lost about 113 minutes it could have used. Its own notes say it was about to wire in a faster, exact inside-zone solver that had halved the error in the oracle setting. **A full-length run could plausibly score lower than 0.0349.**
- **No `APPROACH.md`.** The method summary above is mine, reconstructed from its code and notes.
- **One run, one seed.** The zone solver uses a 0.5 s time-limited search, so re-running `predict.py` on a busy machine may give slightly different predictions. There is no measure of run-to-run spread.
- **Compute was tight.** The machine's load average was about 11 during the run because other jobs were running. Opus had 3 threads and a 2 GB budget, and it worked fast, with small models and short training. With more time or memory it might do better.
- **Different conditions from our own models.**
  - Opus worked inside a sandbox with no internet, and it could not see the project's notes, the pilot results or the papers.
  - It could see the val **inputs**, and it looked at their statistics.
  - It was not told that the val routes are all High quality, while our models know this.
  - It tuned on its own 150 held-out training routes, not on our held-out 10%.
  - A head-to-head with v0 or the rules baseline should use exactly these 300 routes and `results/opus_fight.json`'s per-route scores.
- **This is the "fair beat Opus" opponent, and it is strong.** 0.0349 is far better than Opus ordering stops directly, and better than the shortest route on 96% of routes. To claim "a small model beat Opus" honestly, our model has to beat this number on these routes, or on the frozen test set if Opus's predictor is re-run there.

## Run 1 vs run 2 (added after run 2)

Run 1 was cut off by our memory-cap setup, so Opus was given a second, clean run. Run 2 used a fresh folder (`opus_arena_run2/`), started from scratch, and could not see run 1's folder. The scope had `OOMPolicy=continue`, the containment and flags were otherwise the same, and containment was re-tested with the fake API before launch. Details are in `results/opus_fight_run2.md`.

| | Run 1 | Run 2 |
|---|---|---|
| How it ended | Killed at 67 min (memory-cap setup) | Finished on its own at 72 min |
| `APPROACH.md` | Not written | Written |
| Method | Learned zone model + greedy travel-time tour inside zones | Learned zone model + **learned stop model** inside zones |
| Its own held-out estimate | 0.036 (150 High train routes) | 0.034 mixed / 0.030 High (200 train routes) |
| Val mean [95% CI] | 0.0349 [0.0309, 0.0391] | **0.0268 [0.0235, 0.0303]** |
| Better than shortest on | 287 / 300 | 291 / 300 |
| Cost / output tokens | Unknown (killed before the summary) | $1.37 / 34,580 (10,653 thinking) |
| Audit | Clean | Clean |

Paired run 2 − run 1 on the same 300 routes: −0.0080 [−0.0105, −0.0055]. Run 2 was better on 217 routes and worse on 83, exact sign test p = 5.6 × 10⁻¹⁵. With one run each, this difference mixes the setup fix with ordinary run-to-run variation in what Opus chooses to build. **The number to beat is now run 2's 0.0268.**
