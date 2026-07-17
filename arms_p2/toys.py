"""Shared field-toy generators for the C3 gate suite and the R13 objective bake-off.

Promoted from tests_p2/test_energy_score.py (2026-07-17) so the bake-off runner and
the tests use ONE implementation: smooth conditioning fields, known conditional
laws (Gaussian / one-sided exp / symmetric t(5); flat or modulated sigma), and the
held-out residual shape statistics the gates adjudicate.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp


def smooth_coarse(key, n, hw=16):
    """Smooth random 'coarse' fields in ~[-1.5, 1.5] (low-pass filtered noise)."""
    x = jax.random.normal(key, (n, hw, hw, 1))
    k = jnp.ones((5, 5, 1, 1)) / 25.0
    for _ in range(3):
        x = jax.lax.conv_general_dilated(x, k, (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return 1.5 * x / jnp.std(x)


def true_mean(c):
    return 0.5 * jnp.tanh(c)


def true_sigma(c, flat=False):
    if flat:
        return 0.85 * jnp.ones_like(c)
    return jnp.maximum(0.85 + 0.12 * c, 0.2)


def make_data(key, noise="gauss", flat_sigma=True, n=192, hw=16):
    """detail = true_mean(coarse) + true_sigma(coarse)*eps with the named noise law.

    exp: standardized exponential (mean 0, var 1, skewness 2, excess kurt 6).
    t5:  unit-variance Student-t(5) (symmetric, excess kurt 6).
    """
    kc, kz = jax.random.split(key)
    coarse = smooth_coarse(kc, n, hw)
    shape = (n, hw, hw, 3)
    if noise == "gauss":
        eps = jax.random.normal(kz, shape)
    elif noise == "exp":
        eps = jax.random.exponential(kz, shape) - 1.0
    elif noise == "t5":
        eps = jax.random.t(kz, 5.0, shape) / np.sqrt(5.0 / 3.0)
    else:
        raise ValueError(noise)
    detail = true_mean(coarse) + true_sigma(coarse, flat_sigma) * eps
    return detail, coarse


def resid_stats(gen, coarse_bc):
    """Pooled standardized residual (gen - true_mean) shape statistics."""
    r = np.asarray(gen - true_mean(coarse_bc)).ravel().astype(np.float64)
    r = (r - r.mean()) / r.std()
    return {"skew": float(np.mean(r ** 3)),
            "kurt": float(np.mean(r ** 4) - 3.0),
            "q999": float(np.quantile(np.abs(r), 0.999))}
