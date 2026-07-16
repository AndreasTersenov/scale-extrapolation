"""Held-out statistics for the phase-1d validation pilot (step 2).

These are NEW estimators (scaledrift stays frozen): wavelet-L1 per octave and 2-D
scattering log-coefficients (kymatio) — statistics never used in the generator's
design or repair loops. Each returns PER-FIELD values so callers bootstrap over
fields; `z_*` helpers implement the pre-registered comparisons. Validation gates in
tests_wfm/test_heldout_stats.py (identical-stack zero, GRF-vs-GRF null, gross-
difference detection) per the CLAUDE.md backpressure rules.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from wfm import haar


def wavelet_l1(fields, j):
    """Per-field mean |detail coefficient| at octave j. fields: (N,H,W) array."""
    f = jnp.asarray(np.asarray(fields, np.float32))[..., None]
    det, _ = haar.octave_pair(f, j)
    return np.asarray(jnp.mean(jnp.abs(det), axis=(1, 2, 3)))


_SCT = {}


def scattering_logmeans(fields, J=4, L=4, order2_only=True):
    """Per-field mean over positions of log(coeff+eps), per scattering channel.

    Returns (N, C). Order-2 channels are the non-Gaussian-texture-sensitive ones
    (pre-registered summary uses them).
    """
    from kymatio.numpy import Scattering2D
    key = (J, L, fields.shape[-2], fields.shape[-1])
    if key not in _SCT:
        _SCT[key] = Scattering2D(J=J, shape=fields.shape[-2:], L=L)
    S = _SCT[key]
    out = S(np.asarray(fields, np.float32))          # (N, C, h, w)
    lm = np.log(out + 1e-12).mean(axis=(2, 3))
    if order2_only:
        n0 = 1
        n1 = J * L
        lm = lm[:, n0 + n1:]                          # order-2 channels only
    return lm


def peak_counts(fields, nu):
    """Per-field count of local maxima above nu (fields are unit-variance maps, so nu
    is in sigma units). Local maximum = strictly greater than all 8 neighbours
    (interior pixels only)."""
    f = np.asarray(fields, np.float32)
    c = f[:, 1:-1, 1:-1]
    is_max = np.ones(c.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            is_max &= c > f[:, 1 + dy:f.shape[1] - 1 + dy,
                            1 + dx:f.shape[2] - 1 + dx]
    return (is_max & (c > nu)).sum(axis=(1, 2)).astype(float)


def z_stack(gen_vals, real_vals, n_boot=200, seed=0):
    """z of the mean difference between two per-field stacks (1-D or (N,C)),
    bootstrap-over-fields SEs. Returns scalar (1-D input) or per-channel array."""
    g = np.atleast_2d(np.asarray(gen_vals, float).T).T
    r = np.atleast_2d(np.asarray(real_vals, float).T).T
    rng = np.random.default_rng(seed)

    def boot_se(a):
        n = a.shape[0]
        ms = np.stack([a[rng.integers(0, n, n)].mean(0) for _ in range(n_boot)])
        return ms.std(0, ddof=1)

    diff = g.mean(0) - r.mean(0)
    se = np.hypot(boot_se(g), boot_se(r))
    z = np.where(se > 0, diff / np.maximum(se, 1e-30), 0.0)
    return z if z.size > 1 else float(z.ravel()[0])


def scattering_summary(z_channels, thresh=3.0):
    az = np.abs(np.asarray(z_channels))
    return {"frac_flagged": float((az >= thresh).mean()),
            "median_absz": float(np.median(az)), "n_channels": int(az.size)}
