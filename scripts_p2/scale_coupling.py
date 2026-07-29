"""Cross-scale detail-energy coupling K (N1 diagnostic D2; descriptive).

K_(j,j+1) = mean over maps of Pearson corr( E_j, upsample(E_{j+1}) ), where
E_j is the per-pixel detail energy (sum over the 3 channels of the squared
octave-j detail coefficients) and upsample is 2x nearest-neighbour. The
cascade signature: real fields concentrate detail energy where coarser-scale
energy is high (intermittency); independently textured octaves give K ~= 0.
Tile-bootstrap SEs. Validated tests-first in tests/test_scale_coupling.py
(discrimination coupled-vs-independent, D4 invariance) before touching data.
"""
from __future__ import annotations

import numpy as np

from parity_localization import dwt_levels


def detail_energy(field, octave):
    """Per-pixel channel-summed squared detail coefficients at `octave`."""
    H, V, D = (np.asarray(c, np.float64)
               for c in dwt_levels(field, octave)[1])
    return H * H + V * V + D * D


def coupling(field, j):
    """Pearson corr(E_j, 2x-upsampled E_{j+1}) for one field."""
    e_fine = detail_energy(field, j)
    e_coarse = detail_energy(field, j + 1)
    up = np.repeat(np.repeat(e_coarse, 2, axis=0), 2, axis=1)
    a, b = e_fine.ravel(), up.ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-30))


def stack_coupling(fields, j, n_boot=5000, seed=0):
    """Mean coupling over a stack and its tile-bootstrap SE."""
    ks = np.array([coupling(np.asarray(f, np.float64), j) for f in fields])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ks), (n_boot, len(ks)))
    boot = ks[idx].mean(axis=1)
    return float(ks.mean()), float(boot.std(ddof=1))
