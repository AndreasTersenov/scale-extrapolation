"""Shared helpers for the validation-gate tests.

All fields are synthetic and generated in-test (the GRF null gate is executable, per
CLAUDE.md): no dependence on the cluster data directory, so the gate runs anywhere.
"""
import os

# MUST precede numpy: on the shared login node the cgroup thread limit can make
# OpenBLAS's per-core thread spawn die at pthread_create, and the C library then
# exits(1) SILENTLY mid-test (pytest prints dots, no summary, exit 1 -- seen 2026-07-11
# as a Stop-hook failure). These tests are small-matrix; BLAS threads buy nothing here.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
