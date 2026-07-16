"""Smoke test: proves tests_p2/ is collected by the Stop-hook gate from day one
(chore 0, NIGHT-ORDERS 2026-07-16). Real sandbox/estimator tests join this tree
tests-first as phase-2 code lands.
"""
import numpy as np


def test_tree_is_gated():
    # trivially true, but a collected+passing test is what registration means
    assert np.allclose(np.eye(2) @ np.eye(2), np.eye(2))
