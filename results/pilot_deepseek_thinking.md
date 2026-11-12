# Pilot rerun: DeepSeek with thinking on

**Status: complete, all 50 routes, one call each.** The run finished on 2026-11-10 after several stops and resumes:

1. Stopped after 3 calls because they projected to $3.81 against a $2.00 cap. Angad then approved $5.00.
2. Crashed on a dropped connection (`IncompleteRead`). The runner was fixed to treat this as a network error.
3. Stopped at 23 routes on HTTP 402 "Insufficient Balance". The failed route (`7f5d7cee`) was logged as an error and retried successfully later.
4. After Angad's $2 top-up, stopped at 44 routes on the hard spend guard (cap $4.20).
5. The last 6 routes ran at cap $5.00, 2 calls at a time, with no errors.

Up to 6 calls may have been billed without being logged: 4 lost in the crash and 2 in flight at the 402 stop. That is at most about $0.60 at peak prices, not included in the spend below. All 6 of those routes were later called again, and each route has exactly one logged answer.

All numbers come from `results/pilot_deepseek_thinking.json`, built by `scripts/score_deepseek_thinking.py` from `runs/pilot/deepseek_thinking.jsonl` (calls made by `scripts/run_deepseek_thinking.py`).

## What ran

- **Routes:** 50 of the 50 pilot routes, the same prompts (`runs/pilot/prompts/`) and system prompt (`runs/pilot/system_prompt.txt`), unchanged.
- **Model:** `deepseek-flash` (returned: deepseek-flash), the same model as the thinking-off run.
- **Settings:** `{"thinking": {"type": "enabled"}, "reasoning_effort": "high", "max_tokens": 393216, "stream": false}`. Thinking enabled with effort `high` (DeepSeek's documented default). Temperature not sent: DeepSeek's docs say it has no effect in thinking mode. top_p left at its default. max_tokens 393,216 is the documented maximum (384K). The thinking-off run used temperature 0, max_tokens 4,096, thinking disabled.
- **Parsing and validity:** `run_pilot.py`'s `parse_sequence`, unchanged, applied to the answer (`content`), not the reasoning. Our validity flag matched the official feasibility flag on every route.
- **One call per route**, up to 4 in parallel, no retries for wrong answers, at most 2 retries for network errors (calls that needed a retry: 1).

## Results (lower is better)

| Arm | Routes | Mean score | Bootstrap 95% CI | Median | Valid |
|---|---|---|---|---|---|
| DeepSeek, thinking on | 50 | **0.1166** | 0.0841 – 0.1623 | 0.0823 | 49 / 50 |
| DeepSeek, thinking off, same routes | 50 | 0.8864 | 0.7945 – 0.9806 | 0.8771 | 1 / 50 |
| Shortest route, same routes | 50 | 0.0784 | 0.0690 – 0.0882 | 0.0750 | 50 / 50 |
| Opus | 50 | 0.0693 | 0.0581 – 0.0814 | 0.0672 | 50 / 50 |
| DeepSeek thinking on, Opus's routes | 50 | 0.1166 | 0.0841 – 0.1623 | 0.0823 | 49 / 50 |
| Shortest route, Opus's routes | 50 | 0.0784 | 0.0690 – 0.0882 | 0.0750 | 50 / 50 |

Mean score over valid answers only: 0.0990.

Paired differences (first minus second, negative = first better):

| Comparison | Routes | Mean difference | Bootstrap 95% CI | First better / worse / ties | Sign test p |
|---|---|---|---|---|---|
| Thinking on − shortest | 50 | +0.0382 | +0.0088 – +0.0821 | 20 / 30 / 0 | 0.203 |
| Thinking on − thinking off | 50 | -0.7698 | -0.8598 – -0.6776 | 48 / 1 / 1 | 1.78e-13 |
| Opus − thinking on (Opus's routes) | 50 | -0.0472 | -0.0910 – -0.0189 | 38 / 12 / 0 | 0.000306 |
| Opus − thinking on (both valid) | 49 | -0.0299 | -0.0470 – -0.0154 | 37 / 12 / 0 | 0.00047 |

Opus comparison: `runs/pilot/opus.jsonl` had 50 completed routes when this was scored; n = 50. Opus scores here are computed by this script with the same official scorer, not read from `results/pilot.json`.

## Validity

49 of 50 valid. Finish reasons: `stop` 50.

| Invalid reason | Routes |
|---|---|
| missing 1 stops | 1 |

0 of 50 answers repeat at least one stop ID (thinking off: 46 of 50).
Thinking-off run, same routes: 1 valid. Routes valid in both runs: 1.

## Tokens, latency and spend

- **Spend:** $3.6337 at DeepSeek's peak list prices ($0.006 / $0.30 / $1.20 per 1M cache-hit input / cache-miss input / output tokens; reasoning billed as output). Caps in order: $2.00; then $5.00, approved by Angad after the 3-call projection of $3.81; $4.20 for one resume after a $2 top-up; then $5.00 again for the last 6 routes. From the crash onward, the guard also counted $0.60 of suspected unlogged spend. Off-peak prices are half; failed network attempts, if any, are not counted.
- **Input tokens:** 266,644 total (mean 5,333), 36,864 of them cache hits.
- **Output tokens:** 2,970,449 total, mean 59,409, range 33,312 – 113,227. Reasoning tokens 2,954,678 total, mean 59,094.
- **Answer length:** mean 730 characters (reasoning text 7,976,769 characters in total).
- **Wall time per call:** mean 316 s, median 314 s, range 142 – 603 s (up to 4 calls ran at once; 2 for the last 6 routes).

## Per route

| Route | Stops | Valid | Score | Shortest | Thinking off | Opus | Output tok | Reasoning tok | Wall s |
|---|---|---|---|---|---|---|---|---|---|
| 13ad9c2a | 63 | yes | 0.0160 | 0.0382 | 0.3130 | 0.0041 | 69,528 | 69391 | 427 |
| 1a4903de | 150 | yes | 0.0325 | 0.0273 | 0.8745 | 0.0388 | 53,126 | 52797 | 263 |
| 1d2aa8bd | 100 | yes | 0.0821 | 0.0982 | 0.4735 | 0.0726 | 64,757 | 64545 | 356 |
| 21662e0b | 131 | yes | 0.0292 | 0.0483 | 1.1569 | 0.0247 | 74,175 | 73897 | 406 |
| 221006e8 | 146 | yes | 0.2866 | 0.1368 | 1.2177 | 0.1217 | 62,929 | 62616 | 309 |
| 245f1b63 | 190 | yes | 0.0208 | 0.0444 | 1.1326 | 0.0123 | 72,263 | 71858 | 382 |
| 35f217ab | 146 | yes | 0.1393 | 0.1518 | 1.5223 | 0.0326 | 76,254 | 75944 | 387 |
| 39f16101 | 104 | yes | 0.1082 | 0.0502 | 0.4320 | 0.0267 | 36,527 | 36308 | 176 |
| 3d4c9fc0 | 118 | yes | 0.0640 | 0.0423 | 0.5985 | 0.0527 | 41,056 | 40804 | 218 |
| 41e3914a | 165 | yes | 0.1726 | 0.1119 | 0.8028 | 0.0694 | 46,385 | 46031 | 217 |
| 42017eb9 | 192 | yes | 0.1517 | 0.0912 | 1.0199 | 0.1076 | 33,868 | 33451 | 142 |
| 43dffcb4 | 190 | yes | 0.1188 | 0.0854 | 1.3667 | 0.0964 | 68,596 | 68185 | 355 |
| 4437d348 | 123 | yes | 0.0473 | 0.0778 | 0.9554 | 0.0508 | 55,391 | 55133 | 305 |
| 46bd777c | 152 | yes | 0.0899 | 0.0596 | 0.7721 | 0.0829 | 54,230 | 53903 | 273 |
| 4e6a1048 | 103 | yes | 0.0961 | 0.0899 | 1.2017 | 0.0745 | 40,483 | 40259 | 226 |
| 4ea6084c | 182 | yes | 0.0825 | 0.1266 | 1.2979 | 0.0616 | 54,134 | 53747 | 265 |
| 538b2dc3 | 173 | yes | 0.0420 | 0.0394 | 0.7660 | 0.0551 | 67,887 | 67523 | 355 |
| 55f39e2d | 137 | yes | 0.4116 | 0.0567 | 1.7032 | 0.1060 | 54,078 | 53791 | 227 |
| 58914695 | 153 | yes | 0.1021 | 0.0825 | 0.7189 | 0.0550 | 79,153 | 78823 | 393 |
| 5c999be3 | 143 | yes | 0.0685 | 0.0814 | 1.1741 | 0.0257 | 74,230 | 73922 | 398 |
| 657b0dfd | 159 | no | 0.9767 | 0.0968 | 0.9767 | 0.0803 | 73,027 | 72531 | 357 |
| 738c4cd7 | 147 | yes | 0.0706 | 0.0964 | 1.4360 | 0.0731 | 68,354 | 68040 | 385 |
| 75c3df34 | 142 | yes | 0.0911 | 0.0719 | 0.9144 | 0.0872 | 52,732 | 52433 | 261 |
| 7af29c8f | 103 | yes | 0.0617 | 0.0864 | 0.4593 | 0.0400 | 48,514 | 48296 | 254 |
| 7ee64ab0 | 145 | yes | 0.0915 | 0.0831 | 0.7677 | 0.0939 | 56,515 | 56210 | 293 |
| 7f5d7cee | 167 | yes | 0.0789 | 0.0842 | 0.6985 | 0.0708 | 64,273 | 63921 | 363 |
| 8961343a | 115 | yes | 0.2063 | 0.1245 | 0.8030 | 0.1631 | 55,421 | 55174 | 311 |
| 91bdcdc0 | 151 | yes | 0.0944 | 0.0926 | 0.8766 | 0.1080 | 51,573 | 51239 | 271 |
| 974820df | 127 | yes | 0.1017 | 0.0891 | 0.0837 | 0.0851 | 60,102 | 59824 | 343 |
| 9bfdaaf8 | 129 | yes | 0.0328 | 0.0710 | 1.1968 | 0.0272 | 59,251 | 58978 | 323 |
| 9cfaa79e | 146 | yes | 0.0947 | 0.1103 | 1.0596 | 0.0853 | 53,048 | 52730 | 306 |
| a9089a2d | 162 | yes | 0.0527 | 0.0834 | 1.1919 | 0.0823 | 54,353 | 54009 | 278 |
| ae7545f8 | 130 | yes | 0.0805 | 0.0720 | 0.8903 | 0.0598 | 63,787 | 63508 | 358 |
| b602af6e | 152 | yes | 0.1392 | 0.0577 | 0.9628 | 0.0737 | 67,168 | 66848 | 330 |
| b738fd5a | 151 | yes | 0.0586 | 0.0714 | 0.5561 | 0.0628 | 50,375 | 50048 | 245 |
| b8f7dfa5 | 123 | yes | 0.0474 | 0.0350 | 0.5282 | 0.0511 | 52,036 | 51774 | 294 |
| baa60580 | 129 | yes | 0.0230 | 0.0221 | 0.4890 | 0.0227 | 48,596 | 48318 | 287 |
| c16cda48 | 158 | yes | 0.0713 | 0.0736 | 0.8468 | 0.0213 | 62,158 | 61822 | 326 |
| cd66a2c5 | 179 | yes | 0.1673 | 0.0970 | 1.1416 | 0.1126 | 62,647 | 62268 | 317 |
| ce9be28a | 155 | yes | 0.1193 | 0.0676 | 1.1004 | 0.0684 | 34,359 | 34031 | 178 |
| d22b3e06 | 111 | yes | 0.1392 | 0.1679 | 0.4117 | 0.1051 | 56,974 | 56732 | 320 |
| d3e6f720 | 137 | yes | 0.0612 | 0.0500 | 0.5780 | 0.0331 | 40,694 | 40394 | 215 |
| d400f78e | 150 | yes | 0.1589 | 0.0561 | 0.9298 | 0.2038 | 55,726 | 55409 | 322 |
| d85ecd2b | 160 | yes | 0.0554 | 0.0763 | 0.8776 | 0.1057 | 113,227 | 112884 | 603 |
| da4b54de | 167 | yes | 0.0528 | 0.0659 | 0.6693 | 0.0334 | 68,612 | 68251 | 347 |
| dab18af0 | 129 | yes | 0.0240 | 0.0138 | 0.5841 | 0.0113 | 46,991 | 46716 | 249 |
| e4d411df | 152 | yes | 0.1069 | 0.0556 | 0.4694 | 0.0208 | 33,312 | 32836 | 149 |
| f9148f94 | 159 | yes | 0.0730 | 0.0710 | 0.7671 | 0.0544 | 94,278 | 93939 | 452 |
| fbc3bd0d | 148 | yes | 0.3114 | 0.1826 | 1.5248 | 0.1929 | 87,602 | 87288 | 470 |
| fee60072 | 185 | yes | 0.0248 | 0.0526 | 1.0295 | 0.0659 | 55,694 | 55299 | 509 |

## Caveats

- One call per route; thinking mode samples (temperature has no effect), so a rerun could differ.
- Reasoning effort `high` is DeepSeek's default, not `max`. Opus's effort may have been medium (see `pilot.md`), so the arms are closer on reasoning but still not identical.
- The Opus comparison covers only the routes Opus had finished when this was scored.

## How to rerun

```
.venv/bin/python -I scripts/run_deepseek_thinking.py --pilot-dir runs/pilot --sample-dir data/raw/sample --cap-usd 5.00 --workers 4
.venv/bin/python -I scripts/score_deepseek_thinking.py --sample-dir data/raw/sample --pilot-dir runs/pilot --results-dir results
```
