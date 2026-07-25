"""Corrected coarse-to-fine recursion for arm F (JAX side; prereg R32).

Identical to arms_p2.c1t.train.generate_recursive_tbase except that a constant
per-octave, per-channel offset (WFM channel order H,V,D — the recursion's own
det channel order; field units, i.e. AFTER the std multiplication) is
subtracted from each sampled detail triple before synthesis. With all-zero
offsets this must reproduce generate_recursive_tbase EXACTLY (same PRNG key)
— asserted at runtime by the sample phase (the equivalence gate).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from arms_p2.c1t.flow import sample_tbase
from wfm import haar


def generate_recursive_corrected(apply_fn, params, coarse, j_start, key,
                                 detail_std, offsets_wfm, cond_fn=None,
                                 n_steps=80):
    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)
        cond = None if cond_fn is None else cond_fn(j)
        det_n = sample_tbase(apply_fn, params, k, coarse, 3, n_steps=n_steps,
                             cond_vec=cond)
        s = detail_std(j) if callable(detail_std) else detail_std[j]
        det = det_n * s
        if j in offsets_wfm:
            det = det - jnp.asarray(offsets_wfm[j],
                                    jnp.float32)[None, None, None, :]
        coarse = haar.idwt2(coarse, (det[..., 0:1], det[..., 1:2],
                                     det[..., 2:3]))
    return coarse
