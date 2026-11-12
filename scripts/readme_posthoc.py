#!/usr/bin/env python3
"""DESCRIPTIVE, post-hoc (after-opening rule, PREREGISTRATION section 8): extra numbers for the README.
Reads only results/final_test.json per-route SCORES (no answer files). Changes no claim.
  1. Opus-as-the-model against rule baselines and v0hist on the 500-route sample.
  2. Robustness of the primary: ties, trimmed means, how much of the mean sits in the largest differences."""
import json
import numpy as np

d = json.load(open("results/final_test.json"))
pr = d["per_route"]
ids = sorted(pr["opus_model_repaired_exact"])
rng = np.random.default_rng(20261111)
B = 10000


def ci(x):
    x = np.asarray(x)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def diff(a, b):
    return np.array([pr[a][r] - pr[b][r] for r in ids])


out = {}
print("arms:", sorted(pr))
for other in ["rules_R2", "rules_R4", "v0hist", "jpt"]:
    if other in pr:
        x = diff("opus_model_repaired_exact", other)
        m, lo, hi = ci(x)
        out[f"opus_model - {other}"] = (m, lo, hi, int((x < 0).sum()), int((x > 0).sum()))
        print(f"opus_model - {other}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  opus better {int((x<0).sum())} / worse {int((x>0).sum())}")
x = diff("v1_exact", "opus_model_repaired_exact")
print(f"PRIMARY: {ci(x)[0]:+.4f}, better {int((x<0).sum())}, worse {int((x>0).sum())}, tied {int((x==0).sum())}")
for frac in [0.05]:
    k = int(round(frac * len(x)))
    s = np.sort(x)
    t = s[k:len(s) - k]
    m, lo, hi = ci(t)
    print(f"drop {k} most negative and {k} most positive: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n {len(t)})")
    a = x[np.argsort(-np.abs(x))[k:]]
    m, lo, hi = ci(a)
    print(f"drop {k} largest |diff|: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n {len(a)})")
order = np.argsort(x)
for n in [5, 10]:
    print(f"share of total gap in our {n} biggest wins: {x[order[:n]].sum() / x.sum():.2f}; "
          f"mean without them {np.delete(x, order[:n]).mean():+.4f}")
