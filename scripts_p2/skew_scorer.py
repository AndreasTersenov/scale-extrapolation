"""T3 skewness scorer (phase-3b diagnosis; tests-first in
tests/test_skew_scorer.py before touching any real data).

detail_skew(field, octave): channel-pooled Fisher skewness of the octave's
Haar detail coefficients (the coef_stats_profile octave convention).
map_skew(field): Fisher skewness of the (optionally smoothed) map pixels.
stack_*: per-map values -> stack mean and tile-bootstrap SE.
"""
from __future__ import annotations

import numpy as np

from parity_localization import dwt_levels


def _fisher_skew(x):
    x = np.asarray(x, np.float64).ravel()
    m = x.mean()
    s = x.std()
    return float(((x - m) ** 3).mean() / (s**3 + 1e-30))


def detail_skew(field, octave):
    coeffs = dwt_levels(np.asarray(field, np.float64), octave)[1]
    return _fisher_skew(np.concatenate([np.asarray(c, np.float64).ravel()
                                        for c in coeffs]))


def map_skew(field):
    return _fisher_skew(field)


def _stack(vals, n_boot=5000, seed=0):
    vals = np.asarray(vals, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    boot = vals[idx].mean(axis=1)
    return float(vals.mean()), float(boot.std(ddof=1))


def stack_detail_skew(fields, octave, n_boot=5000, seed=0):
    return _stack([detail_skew(f, octave) for f in fields], n_boot, seed)


def stack_map_skew(fields, n_boot=5000, seed=0):
    return _stack([map_skew(f) for f in fields], n_boot, seed)
