"""I — the oct-1 phase-texture instrument (R47 order 3; prereg
2026-08-05-oct1fix; tests-first in tests/test_oct1_texture.py on synthetics
before any real data).

Operates on the oct-1 CONTRIBUTION field u = f − bandlimit(f) (the map-space
layer carried by the finest-octave details), standardized per map.

fragmentation_profile(f, nus): per threshold ν_u — component count, hole
    count, χ, small-component fraction (area ≤ 4 px) of {u > ν_u·σ_u}
    (4-connectivity, the judge's family). D4-invariant.
alignment_stat(f): Pearson corr( u², |∇ bandlimit(f)|² ) — does the fine
    texture organize on coarse structure? D4-invariant (gradient magnitude).
stack_*: per-map values → stack mean ± tile-bootstrap SE.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from parity_localization import dwt_levels, idwt_levels

FRAG_NUS = (0.5, 1.0, 1.5)
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def bandlimit(field):
    c = list(dwt_levels(np.asarray(field, np.float64), 1))
    c[1] = tuple(np.zeros_like(np.asarray(x)) for x in c[1])
    return idwt_levels(c)


def contribution(field):
    f = np.asarray(field, np.float64)
    u = f - bandlimit(f)
    return u / u.std()


def _comp_holes(mask):
    lab, ncomp = label(mask, structure=S4)
    labc, ncompl = label(~mask, structure=S4)
    border = set(np.concatenate([labc[0], labc[-1], labc[:, 0],
                                 labc[:, -1]]))
    border.discard(0)
    holes = ncompl - len(border)
    sizes = np.bincount(lab.ravel())[1:]
    small = float((sizes <= 4).mean()) if ncomp else 0.0
    return ncomp, holes, small


def fragmentation_profile(field, nus=FRAG_NUS):
    """[ncomp, holes, chi, small_frac] per threshold, concatenated."""
    u = contribution(field)
    out = []
    for nu in nus:
        ncomp, holes, small = _comp_holes(u > nu)
        out += [ncomp, holes, ncomp - holes, small]
    return np.array(out, np.float64)


def alignment_stat(field):
    f = np.asarray(field, np.float64)
    bl = bandlimit(f)
    u = f - bl
    gy, gx = np.gradient(bl)
    g2 = gy * gy + gx * gx
    a, b = (u * u).ravel(), g2.ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-30))


def _stack(vals, n_boot=5000, seed=0):
    vals = np.asarray(vals, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), (n_boot, len(vals)))
    boot = vals[idx].mean(axis=0 if vals.ndim == 1 else 1)
    boot = vals[idx].mean(axis=1)
    return vals.mean(axis=0), boot.std(axis=0, ddof=1)


def stack_fragmentation(fields, nus=FRAG_NUS, n_boot=5000, seed=0):
    return _stack([fragmentation_profile(f, nus) for f in fields],
                  n_boot, seed)


def stack_alignment(fields, n_boot=5000, seed=0):
    m, s = _stack([[alignment_stat(f)] for f in fields], n_boot, seed)
    return float(m[0]), float(s[0])
