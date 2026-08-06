"""Tests-first validation for the TIDAL eigenframe-conditioning machinery
(prereg log/2026-08-05-prereg-night3.md "TIDAL — design").

Gates encoded:
 1. D4 covariance of hessian_features at machine precision, all 8 elements,
    matching f2_group.apply_g's convention (rot90^k over axes (1,2), then
    W-flip^f): tr invariant; sign(a1) = (-1)^k; sign(a2) = (-1)^(k+f)
    (spin-2: 90-degree rotation negates both; W-mirror fixes a1, negates a2).
 2. F2-form exactness on the ASSEMBLED tidal sampler: with fixed init params
    and a fixed key, per-element assembly on the original coarse is
    bit-identical to g^{-1} · (all-identity assembly on the transformed
    coarse) — features are computed inside the frame, so
    d = g^{-1}·model(g·c, H(g·c)) is exactly the F2 form.
 3. Trainer smoke: 16 steps on tiny synthetic tiles — finite loss, feat
    stats present in meta ((3,) positive per octave).
 4. Sampling smoke: gen_groupavg_tidal from octave 2 on an 8-map batch
    returns finite fields of the right shape; feat_std_ladder's octave-1
    extrapolation is exercised on the way.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

import jax
import jax.numpy as jnp

from arms_p2.tidal.features import hessian_features
from arms_p2.tidal.sample import (feat_std_ladder, gen_groupavg_tidal,
                                  tidal_model_fn)
from arms_p2.tidal.train import train_tidal_generator
from f2_group import D4_ELEMENTS, apply_g, apply_g_inv, assemble_group_assigned
from wfm import haar
from wfm.model import ConditionalUNet

RNG = np.random.default_rng(20260805)

# channel action derived in arms_p2/tidal/features.py's module doc
SIGNS = {(k, f): (1.0, (-1.0) ** k, (-1.0) ** (k + f))
         for f in (0, 1) for k in (0, 1, 2, 3)}


def test_d4_covariance_all_elements():
    x = jnp.asarray(RNG.standard_normal((2, 16, 16, 1)).astype(np.float32))
    feats = hessian_features(x)
    assert float(jnp.abs(feats).max()) > 1e-3   # non-vacuous
    for g in D4_ELEMENTS:
        got = hessian_features(apply_g(x, g))
        want = apply_g(feats * jnp.asarray(SIGNS[g]), g)
        assert np.allclose(np.asarray(got), np.asarray(want), atol=1e-6), g


def _toy(hw=8):
    """Tiny ConditionalUNet initialized on the 4-channel tidal conditioning."""
    model = ConditionalUNet(out_channels=3, channels=(4, 8), bottleneck=16,
                            cond_dim=0, cond_mode="film", variance_head=False)
    coarse = jax.random.normal(jax.random.PRNGKey(0), (2, hw, hw, 1))
    cond4 = jnp.concatenate([coarse, hessian_features(coarse)], axis=-1)
    detail = jax.random.normal(jax.random.PRNGKey(2), (2, hw, hw, 3))
    params = model.init(jax.random.PRNGKey(1), detail,
                        np.zeros(2, np.float32), cond4, None)["params"]
    return model, params, coarse


def test_f2_form_exactness_assembled_sampler():
    model, params, coarse = _toy()
    fstat = np.array([0.9, 1.1, 1.3], np.float32)
    model_fn = tidal_model_fn(model.apply, params, fstat, 0.7,
                              jax.random.PRNGKey(7), n_steps=8)
    B = coarse.shape[0]
    for gi, g in enumerate(D4_ELEMENTS):
        direct = assemble_group_assigned(coarse, model_fn,
                                         np.full(B, gi, int))
        ident = assemble_group_assigned(apply_g(coarse, g), model_fn,
                                        np.zeros(B, int))
        f2 = apply_g_inv(ident, g)
        assert np.array_equal(np.asarray(direct), np.asarray(f2)), g


def test_trainer_smoke():
    tiles = RNG.standard_normal((12, 16, 16)).astype(np.float32)
    state, meta = train_tidal_generator(tiles, [1, 2], arm="A",
                                        channels=(4, 8), steps=16, batch=4,
                                        lr=1e-3, seed=0, augment=True)
    assert np.isfinite(meta["loss0"]) and np.isfinite(meta["lossN"])
    assert set(meta["feat_std_by_j"]) == {1, 2}
    for j in (1, 2):
        fs = np.asarray(meta["feat_std_by_j"][j])
        assert fs.shape == (3,)
        assert np.all(np.isfinite(fs)) and np.all(fs > 0)


def test_sampling_smoke_from_octave2():
    model, params, _ = _toy()
    fields = jnp.asarray(RNG.standard_normal((8, 16, 16, 1))
                         .astype(np.float32))
    coarse2 = haar.octave_pair(fields, 2)[1]                  # (8,4,4,1)
    fstd = feat_std_ladder({2: np.array([0.8, 0.9, 1.0], np.float32),
                            3: np.array([1.6, 1.8, 2.0], np.float32)})
    assert 1 in fstd and fstd[1].shape == (3,)                # ladder reaches 1
    assert np.all(fstd[1] > 0)
    gen = gen_groupavg_tidal(model.apply, params, coarse2, 2,
                             jax.random.PRNGKey(3), {1: 0.5, 2: 0.7}, fstd,
                             np.random.default_rng(11))
    gen = np.asarray(gen)
    assert gen.shape == (8, 16, 16, 1)
    assert np.all(np.isfinite(gen))
