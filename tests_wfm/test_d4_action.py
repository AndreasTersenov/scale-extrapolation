"""Tests-first validation for arm F2's group-averaged sampler
(prereg log/2026-07-25-prereg-f2-groupavg.md, R33).

Gates encoded:
 1. Haar/D4 commutation: dwt2(g.x).coarse == g.(dwt2(x).coarse) exactly for
    all 8 group elements (the synthesis-transform-analysis route's license).
 2. Identity equivalence: the group-averaged assembly with g=identity for
    every field reproduces plain assembly exactly.
 3. Symmetrization: a deterministic pseudo-model with a constant channel-mean
    defect, averaged over the 8 elements, yields exactly zero ensemble
    channel means (the group average nulls every sign-moved statistic).
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

import jax.numpy as jnp

from f2_group import D4_ELEMENTS, apply_g, apply_g_inv, assemble_in_frame
from wfm import haar

RNG = np.random.default_rng(20260727)


def test_commutation_all_elements():
    x = jnp.asarray(RNG.standard_normal((2, 16, 16, 1)))
    cA, _ = haar.dwt2(x)
    for g in D4_ELEMENTS:
        cA_g, _ = haar.dwt2(apply_g(x, g))
        assert np.allclose(np.asarray(cA_g), np.asarray(apply_g(cA, g)),
                           atol=1e-6), g


def test_g_inverse_roundtrip():
    x = jnp.asarray(RNG.standard_normal((3, 8, 8, 2)))
    for g in D4_ELEMENTS:
        back = apply_g_inv(apply_g(x, g), g)
        assert np.allclose(np.asarray(back), np.asarray(x)), g


def test_identity_equivalence():
    coarse = jnp.asarray(RNG.standard_normal((4, 8, 8, 1)))
    det = jnp.asarray(RNG.standard_normal((4, 8, 8, 3)))
    plain = haar.idwt2(coarse, (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
    framed = assemble_in_frame(coarse, lambda c: det, D4_ELEMENTS[0])
    assert np.allclose(np.asarray(framed), np.asarray(plain))


def test_group_average_nulls_channel_means():
    coarse = jnp.asarray(RNG.standard_normal((4, 8, 8, 1)))
    bias = jnp.asarray(np.array([0.3, -0.2, 0.1], np.float32))

    def biased_model(c):
        # deterministic defect: constant per-channel offset (the mean layer)
        return jnp.broadcast_to(bias, c.shape[:3] + (3,))

    fields = [assemble_in_frame(coarse, biased_model, g) for g in D4_ELEMENTS]
    avg = jnp.mean(jnp.stack(fields), axis=0)
    _, (h, v, d) = haar.dwt2(avg)
    for c in (h, v, d):
        assert abs(float(jnp.mean(c))) < 1e-6
