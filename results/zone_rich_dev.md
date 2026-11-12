# Opus-as-the-model, richer history: prompt development (VALIDATION, 40 routes)

**This is development, not a result.** These 40 val routes were used to choose between two prompt versions and freeze one. The primary comparison runs the frozen prompt, unchanged, on the eval sample. Numbers: `results/zone_rich_dev.json`; code: `scripts/zone_rich.py`; frozen record: `runs/zone_rich/FROZEN_PROMPT.md`.

**Setup:**
- Opus 5.5 picks the zone order; our greedy in-zone solver fills in the stops.
- History follows the competition rule: all same-station train routes, any date.
- Evidence per route:
  - the usual first zones;
  - the top-3 next zones per zone, with counts;
  - coarse-level counts for zones never seen exactly;
  - the 8 most similar past routes, reduced to this route's zones.
- Version B adds a guidance paragraph (habits, zone-ID sweep, history first).

| Version | Mean score (40 val routes) | Valid | Cost per call | Wall time per call |
|---|---|---|---|---|
| **A, frozen** | **0.0319** | 40/40 | $0.041 | 5.2 s |
| B | 0.0335 | 40/40 | $0.043 | 5.6 s |

B − A: +0.0016 [+0.0002, +0.0033]. B was better on 3 routes and worse on 8, so the guidance paragraph hurt.

**For orientation only (descriptive, same 40 routes, stored val scores):** v0-hist 0.0255, Opus fair-fight run 2 0.0293, jpt 0.0247. These came from separate systems and val was looked at many times; they are not a test.

**Calls:** 120 (40 per version, plus 40 for version A at max effort), all valid. Total cost by the CLI field: A $1.629, B $1.708. Before any API call, 3 local launches failed because the system-prompt path was relative; that was fixed, and the log is kept.

**Frozen sha256:**
- system prompt `8130bb2daca43586c656381c89ba1d65e7dbd9fb7ad03e04548228000e8c9c82`
- template `fa48f8bdc3001f81a0053cc1fd75b258e1f253cbfe84c457e726b80a605d616e`
- script `a43794560f4bec36b7b484b7d24e01eee4f442acf5ff7f3536ce9cc84dd4d2ab`

## Effort (added after review 5)

Frozen version A, identical prompts, rerun with `--effort max` on the same 40 routes:
- **Score:** 0.0310, against 0.0319 at default effort. max − default = −0.0009 [−0.0046, +0.0026]; max better on 9 routes, worse on 2, tied on 29.
- **Cost:** $0.451 and 196 s per call, against $0.041 and 5.2 s.
- **Frozen: max effort,** under the rule "better for Opus; ties go to max".

Per-route STRICT and REPAIRED views are both written; the primary comparison uses REPAIRED. `--solver exact` is wired to `scripts/exact_zone.py`.

Updated script sha256: `e6d2400ac9eb8a18343701e5382d69b2268885bebbe60e18ddb31c51237cbe37`.
