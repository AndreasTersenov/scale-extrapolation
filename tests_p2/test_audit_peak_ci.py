"""Validation for the takeover-audit peak-CI instrument (scripts_p2/audit_peak_ci.py).

Tests-first per CLAUDE.md: the bootstrap-CI wrapper around the frozen stage-D
peaks_count is validated before its numbers enter any document.
1. peaks_count agrees with a hand-constructed field (known local maxima).
2. Truth-vs-truth null: split-half excess on i.i.d. synthetic fields is
   consistent with zero within the bootstrap CI (the GRF-null discipline).
"""
import numpy as np
import pytest

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from audit_peak_ci import bootstrap_excess, peaks_count


def test_peaks_count_hand_case():
    f = np.zeros((8, 8))
    f[2, 2] = 5.0   # isolated bright peak
    f[5, 5] = 5.0   # second isolated peak
    f[5, 6] = 4.9   # shoulder, not a local max
    # standardized field: two strict local maxima above any nu <= ~4 sigma
    assert peaks_count(f, 1.0) == 2
    # plateau tie must NOT count (strict inequality)
    g = np.zeros((8, 8))
    g[3, 3] = g[3, 4] = 3.0
    assert peaks_count(g, 1.0) == 0


def test_bootstrap_null_covers_zero():
    rng = np.random.default_rng(7)
    fields = rng.standard_normal((48, 32, 32))
    a, b = fields[:24], fields[24:]
    res = bootstrap_excess(a, b, nu=1.0, n_boot=400, seed=0)
    lo95, hi95 = res["ci95"]
    assert lo95 < 0.0 < hi95, f"null excess CI excludes zero: {res}"
    # and the point estimate is small compared to the CI half-width
    assert abs(res["excess"]) < 3 * res["se"]
