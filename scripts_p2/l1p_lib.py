"""L1' sampler plumbing (R39): the F2 group-averaged sampler with a
per-octave BASE function — white t(5) everywhere except where a colored
base is supplied (octave 1 in L1').

sample_base_fn with base_fn = t_base is EXACTLY sample_tbase (same key
usage — x0 = base_fn(key, shape), no extra split), asserted in
tests_p2/test_l1p_sampler.py BEFORE any job, so the runtime replay gates
test the committed chain, not this code.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from arms_p2.c1t.flow import NU, t_base
from f2_group import D4_ELEMENTS, assemble_group_assigned


def white_base(key, shape):
    return t_base(key, shape, NU)


def make_colored_base(filt, z_grid, x_grid):
    from colored_base import colored_t_base

    def base_fn(key, shape):
        return colored_t_base(key, shape, filt, z_grid, x_grid)
    return base_fn


def sample_base_fn(apply_fn, params, key, coarse, base_fn, out_channels=3,
                   n_steps=80, cond_vec=None):
    """sample_tbase verbatim with x0 = base_fn(key, shape)."""
    B, H, W, _ = coarse.shape
    x0 = base_fn(key, (B, H, W, out_channels))
    dt = 1.0 / n_steps
    ts = jnp.arange(n_steps) * dt

    def vf(t, x):
        return apply_fn({"params": params}, x, jnp.full((B,), t), coarse,
                        cond_vec)

    def heun(x, t):
        v1 = vf(t, x)
        v2 = vf(t + dt, x + dt * v1)
        return x + 0.5 * dt * (v1 + v2), None

    x1, _ = jax.lax.scan(heun, x0, ts)
    return x1


def gen_groupavg_base(apply_fn, params, coarse, j_start, key, std, grng,
                      base_by_j=None):
    """f2_sample.gen_groupavg (arm A) with a per-octave base override.

    base_by_j: {octave: base_fn}; octaves absent use the white t base."""
    base_by_j = base_by_j or {}
    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)
        bf = base_by_j.get(j, white_base)

        def model_fn(c_g, _k=k, _s=std[j], _bf=bf):
            det_n = sample_base_fn(apply_fn, params, _k, c_g, _bf,
                                   n_steps=80, cond_vec=None)
            return det_n * _s

        assign = grng.integers(0, len(D4_ELEMENTS), coarse.shape[0])
        coarse = assemble_group_assigned(coarse, model_fn, assign)
    return coarse


def std_from(train, octs):
    # verbatim from f2_sample.py (== training-side computation)
    from wfm.dataset import d4_augment, field_to_octaves
    _, std_by_j = field_to_octaves(d4_augment(train), octs)
    js = np.array(sorted(std_by_j))
    a_, b_ = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
    std = dict(std_by_j)
    for j in range(1, min(octs)):
        std[j] = float(np.exp(a_ * j + b_))
    return std
