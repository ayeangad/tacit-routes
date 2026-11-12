#!/usr/bin/env python3
"""Hand-made checks that we are calling the official rc-cli scorer correctly.

  .venv/bin/python -I scripts/test_scorer.py

The expected values below were worked out by hand, not by running the scorer.
"""
import copy
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "official_score", Path(__file__).resolve().parent / "official_rc_cli" / "score.py")
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)

# Four stops, so 16 matrix cells: eight 0s and eight 2s. Mean 1, std 1, so normalising
# gives z = -1 or +1, and shifting by the minimum gives back exactly 0 and 2.
STOPS = ["SS", "AA", "BB", "CC"]
MATRIX = {
    "SS": {"SS": 0, "AA": 0, "BB": 2, "CC": 2},
    "AA": {"SS": 2, "AA": 0, "BB": 2, "CC": 0},
    "BB": {"SS": 2, "AA": 0, "BB": 0, "CC": 2},
    "CC": {"SS": 0, "AA": 2, "BB": 2, "CC": 0},
}
ACTUAL = ["SS", "AA", "BB", "CC", "SS"]


def score(sub):
    return official.score(list(ACTUAL), list(sub), copy.deepcopy(MATRIX))


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def main():
    # 1. Perfect match scores exactly 0, so lower is better.
    assert score(ACTUAL) == 0

    # 2. Normalisation matches the hand calculation (all values stay 0 or 2).
    norm = official.normalize_matrix(copy.deepcopy(MATRIX))
    assert all(close(norm[o][d], MATRIX[o][d]) for o in STOPS for d in STOPS)

    # 3. Swap the first two stops: S B A C S against S A B C S.
    #    SD: actual interior [A,B,C], proposed [B,A,C] gives positions [1,0,2].
    #        sum(|step|-1) = 0 + 1 = 1, and 2/(3*2) * 1 = 1/3.
    #    ERP: gaps cost 1000, so only substitutions happen: A->B = 2, B->A = 0.
    #        Total 2 over 2 edits, so ERP per edit = 1.
    #    Score = 1/3 * 1 = 1/3.
    swapped = ["SS", "BB", "AA", "CC", "SS"]
    assert close(official.seq_dev(ACTUAL, swapped), 1 / 3)
    assert close(official.erp_per_edit(ACTUAL, swapped, norm), 1.0)
    assert close(score(swapped), 1 / 3)

    # 4. A metric quirk worth knowing: the exact reverse of the driver's route has
    #    sequence deviation 0, so it scores 0 despite a large edit distance.
    #    Interior positions [2,1,0] step by exactly 1 each time.
    reverse = ["SS", "CC", "BB", "AA", "SS"]
    assert official.seq_dev(ACTUAL, reverse) == 0
    assert score(reverse) == 0
    assert official.erp_per_edit(ACTUAL, reverse, norm) > 0

    # 5. evaluate(): a route missing a stop takes the supplied invalid score.
    rid = "RouteID_test"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.json").write_text(json.dumps({rid: {"actual": {"SS": 0, "AA": 1, "BB": 2, "CC": 3}}}))
        (d / "p.json").write_text(json.dumps({rid: {"proposed": {"SS": 0, "AA": 1, "BB": 2}}}))
        (d / "c.json").write_text(json.dumps({rid: MATRIX}))
        (d / "i.json").write_text(json.dumps({rid: 0.75}))
        out = official.evaluate(str(d / "a.json"), str(d / "p.json"), str(d / "c.json"), str(d / "i.json"))
    assert out["route_feasibility"][rid] is False
    assert out["route_scores"][rid] == 0.75

    print("all scorer checks passed")


if __name__ == "__main__":
    sys.exit(main())
