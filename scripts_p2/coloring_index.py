"""Coloring index C for detail-coefficient planes (N1 gate probe; prereg
log/2026-07-29-prereg-n1-mechanism.md).

C = (mean annular FFT power, 1 <= |k| <= N/8) / (mean annular power,
N/4 < |k| <= N/2), detail channels summed, N = detail-plane size. White
details -> C ~= 1; spatially colored (red) details -> C > 1. Bands are FIXED
here and validated in tests/test_coloring_index.py on synthetic fields
BEFORE any real data is touched (the prereg's band rule). Stack estimate is
ratio-of-means (stabler than mean-of-ratios); SE by tile bootstrap.
"""
from __future__ import annotations

import numpy as np

from parity_localization import dwt_levels


def _band_masks(N):
    k = np.fft.fftfreq(N) * N
    kk = np.hypot(k[:, None], k[None, :])
    low = (kk >= 1) & (kk <= N / 8)
    high = (kk > N / 4) & (kk <= N / 2)
    return low, high


def band_powers(plane):
    """(low-band mean power, high-band mean power) of one 2-D plane."""
    N = plane.shape[0]
    low, high = _band_masks(N)
    P = np.abs(np.fft.fft2(np.asarray(plane, np.float64))) ** 2 / N**2
    return float(P[low].mean()), float(P[high].mean())


def field_band_powers(field, octave):
    """Summed-over-channels (low, high) band powers of `field`'s detail
    coefficients at `octave` (the coef_stats_profile octave convention)."""
    lo = hi = 0.0
    for c in dwt_levels(field, octave)[1]:
        l, h = band_powers(c)
        lo += l
        hi += h
    return lo, hi


def stack_coloring(fields, octave, n_boot=5000, seed=0):
    """C (ratio of stack-mean band powers) and its tile-bootstrap SE."""
    pairs = np.array([field_band_powers(np.asarray(f, np.float64), octave)
                      for f in fields])
    los, his = pairs[:, 0], pairs[:, 1]
    C = float(los.mean() / his.mean())
    rng = np.random.default_rng(seed)
    n = len(fields)
    idx = rng.integers(0, n, (n_boot, n))
    boot = los[idx].mean(axis=1) / his[idx].mean(axis=1)
    return C, float(boot.std(ddof=1))
