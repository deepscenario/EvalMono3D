"""Reproducibility guard: the example must always produce the canonical numbers.

These are the exact values published with the repository. If a change moves
them, evaluation is no longer reproducible and the change must be reverted.

Run with::

    pytest -q
"""

import math
import os

from evalmono3d import evaluate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(HERE, "example", "cdrone_test_example.json")
PRED = os.path.join(HERE, "example", "cdrone_test_example_pred.pth")

# Canonical results (car category) -- do not change without a very good reason.
EXPECTED_AP2D = 75.04950495049505
EXPECTED_AP3D = 25.247524752475247


def test_example_results_are_bit_identical():
    results = evaluate("cdrone_test_example", GT, PRED)
    assert math.isclose(results["per_category"]["car"]["AP2D"], EXPECTED_AP2D, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(results["per_category"]["car"]["AP3D"], EXPECTED_AP3D, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(results["AP2D"], EXPECTED_AP2D, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(results["AP3D"], EXPECTED_AP3D, rel_tol=0, abs_tol=1e-9)
