"""Shared helpers for the validation-gate tests.

All fields are synthetic and generated in-test (the GRF null gate is executable, per
CLAUDE.md): no dependence on the cluster data directory, so the gate runs anywhere.
"""
import numpy as np

from scaledrift import lognormal_field, powerlaw_grf

# Small, fixed sizes chosen so the whole suite runs well under the 300 s Stop-hook
# timeout while keeping estimator noise low enough for the gates to be decisive.
GRF_ALPHA = 1.5      # power-law index P(k) ~ k^-alpha (self-similar => scale-invariant)
LN_ALPHA = 1.2
LN_SIGMA = 0.8       # lognormal non-Gaussianity strength


def make_grf(n, seed, shape=(128, 128), alpha=GRF_ALPHA):
    rng = np.random.default_rng(seed)
    return [powerlaw_grf(shape, alpha, rng) for _ in range(n)]


def make_lognormal(n, seed, shape=(128, 128), alpha=LN_ALPHA, sigma=LN_SIGMA):
    rng = np.random.default_rng(seed)
    return [lognormal_field(shape, alpha, sigma, rng, normalize=True) for _ in range(n)]
