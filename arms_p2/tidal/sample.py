"""TIDAL sampler plumbing (prereg log/2026-08-05-prereg-night3.md).

The F2 group-averaged sampler with eigenframe features computed INSIDE the
group frame: the model_fn handed to assemble_group_assigned receives the
ALREADY-TRANSFORMED coarse c_g and computes H(c_g) itself, so the sampled
detail is d = g^{-1} · model'(g·c) with model'(x) = model(x, H(x)) — exactly
the F2 form. Proved on the assembled sampler in
tests_p2/test_tidal_features.py (per-element assembly == g^{-1} · identity
assembly on the transformed coarse, bit-exact).

Mirrors l1p_lib.sample_base_fn + gen_groupavg_base including base_by_j
support (the oct-1 colored base still works: only x0 changes, the 4-channel
conditioning path is identical).
"""
from __future__ import annotations

import os
import sys

import numpy as np

import jax
import jax.numpy as jnp

# Same shim as arms_p2/c1t/flow.py's jax_flows one, for scripts_p2 flat
# imports (l1p_lib, f2_group): kept HERE so this module is import-order
# independent of which script/test pulls it first.
_SP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "scripts_p2")
if _SP not in sys.path:
    sys.path.append(_SP)

from f2_group import D4_ELEMENTS, assemble_group_assigned  # noqa: E402
from l1p_lib import sample_base_fn, white_base  # noqa: E402

from .features import SIGMA_H, hessian_features  # noqa: E402


def tidal_model_fn(apply_fn, params, feat_stats, std_j, key, n_steps=80,
                   base_fn=None, sigma_h=SIGMA_H):
    """model_fn(c_g) closure for assemble_group_assigned at one octave.

    Computes H(c_g) from the (already-transformed) coarse, standardizes with
    the TRAINING constants feat_stats ((3,) per-channel std), concatenates,
    and samples through sample_base_fn. Returns the detail in FIELD units."""
    bf = white_base if base_fn is None else base_fn
    fs = jnp.asarray(feat_stats)

    def model_fn(c_g):
        feats = hessian_features(c_g, sigma=sigma_h) / fs
        cond4 = jnp.concatenate([c_g, feats], axis=-1)
        det_n = sample_base_fn(apply_fn, params, key, cond4, bf,
                               n_steps=n_steps, cond_vec=None)
        return det_n * std_j

    return model_fn


def gen_groupavg_tidal(apply_fn, params, coarse, j_start, key, std, feat_std,
                       grng, base_by_j=None):
    """l1p_lib.gen_groupavg_base through the tidal (4-channel) conditioning.

    feat_std: {octave: (3,)} training feature stds (feat_std_ladder output);
    base_by_j: {octave: base_fn}; octaves absent use the white t base."""
    base_by_j = base_by_j or {}
    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)
        model_fn = tidal_model_fn(apply_fn, params, feat_std[j], std[j], k,
                                  n_steps=80, base_fn=base_by_j.get(j))
        assign = grng.integers(0, len(D4_ELEMENTS), coarse.shape[0])
        coarse = assemble_group_assigned(coarse, model_fn, assign)
    return coarse


def feat_std_ladder(feat_std_by_j):
    """Extend feat_std_by_j down to octave 1, per channel.

    l1p_lib.std_from's rule applied per feature channel: fit log std vs j
    over the trained octaves, extrapolate below. Trained octaves keep their
    stored constants verbatim."""
    js = np.array(sorted(int(j) for j in feat_std_by_j))
    fs = np.stack([np.asarray(feat_std_by_j[j], np.float64) for j in js])
    out = {int(j): np.asarray(feat_std_by_j[j], np.float32) for j in js}
    coeffs = [np.polyfit(js, np.log(fs[:, c]), 1) for c in range(fs.shape[1])]
    for j in range(1, js.min()):
        out[j] = np.array([np.exp(a_ * j + b_) for a_, b_ in coeffs],
                          np.float32)
    return out
