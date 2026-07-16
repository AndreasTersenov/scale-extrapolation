"""Independent Haar (periodization) analysis for the truth computation.

Matches pywt.wavedec2(., 'haar', mode='periodization') EXACTLY (machine precision;
sign conventions determined empirically 2026-07-16 and locked by the cross-check test
in tests_p2/test_sandbox_conditional.py). scaledrift is deliberately NOT imported:
the truth side of Gate A must not share code with the instrument under test.
"""
from __future__ import annotations

import numpy as np


def haar_level(f):
    """One analysis level: (cA, (cH, cV, cD)), pywt-haar-periodization conventions."""
    f = np.asarray(f, dtype=np.float64)
    a = f[0::2, 0::2]
    b = f[0::2, 1::2]
    c = f[1::2, 0::2]
    d = f[1::2, 1::2]
    cA = (a + b + c + d) / 2.0
    cH = (a + b - c - d) / 2.0     # top minus bottom rows
    cV = (a + c - b - d) / 2.0     # left minus right columns
    cD = (a + d - b - c) / 2.0
    return cA, (cH, cV, cD)


def haar_synth_level(cA, bands):
    """Inverse of haar_level (orthonormal synthesis; roundtrip tested)."""
    cH, cV, cD = bands
    H, W = cA.shape
    f = np.empty((2 * H, 2 * W), dtype=np.float64)
    f[0::2, 0::2] = (cA + cH + cV + cD) / 2.0
    f[0::2, 1::2] = (cA + cH - cV - cD) / 2.0
    f[1::2, 0::2] = (cA - cH + cV - cD) / 2.0
    f[1::2, 1::2] = (cA - cH - cV + cD) / 2.0
    return f


def haar_atom(shape, j, kind, band, pos):
    """Field-space atom of one level-j Haar coefficient (orthonormal basis vector).

    kind: 'w' (detail; band in 0..2 for H,V,D) or 'c' (approximation; band ignored).
    pos: (row, col) on the level-j grid. Used for analytic conditional-variance
    truth: Cov(coef_a, coef_b) = <atom_a, Sigma atom_b> for any stationary Sigma.
    """
    Hj, Wj = shape[0] // 2 ** j, shape[1] // 2 ** j
    cA = np.zeros((Hj, Wj))
    bands = [np.zeros((Hj, Wj)) for _ in range(3)]
    if kind == "c":
        cA[pos] = 1.0
    else:
        bands[band][pos] = 1.0
    f = haar_synth_level(cA, tuple(bands))
    for _ in range(j - 1):
        h = f.shape[0]
        z = (np.zeros((h, h)), np.zeros((h, h)), np.zeros((h, h)))
        f = haar_synth_level(f, z)
    return f


def octave_wc_pooled(f, j):
    """(w, c) at octave j for one field: orientation-pooled detail + tiled coarse.

    Mirrors scaledrift.octave_wc's OUTPUT definition (independent implementation):
    w stacks (cH, cV, cD) flattened in that order; c is cA at the same level tiled x3.
    """
    cA = np.asarray(f, dtype=np.float64)
    for _ in range(j):
        cA, (cH, cV, cD) = haar_level(cA)
    w = np.concatenate([cH.reshape(-1), cV.reshape(-1), cD.reshape(-1)])
    c = np.tile(cA.reshape(-1), 3)
    return w, c
