"""Coarse-to-fine recursive generation for the wavelet flow-matching generator.

Given a coarse field at some octave, repeatedly: sample a detail triple from the shared
conditional model, un-standardize it by that octave's detail amplitude, and invert one
Haar level to obtain the next-finer coarse field. Recurse to octave 0 (the full field).

The per-octave detail amplitude (``detail_std``) is the trivial variance scaling the model
does not carry (it works in standardized detail space so the weights transfer across
octaves); in the toy overfit it is measured from the field, in the full experiment it is a
fixed function of octave (the power-spectrum amplitude, the cheap P4 quantity).
"""
from __future__ import annotations

import jax

from . import haar
from .cfm import sample


def generate_recursive(apply_fn, params, coarse, j_start, key, detail_std,
                       cond_fn=None, n_steps=80, solver="heun"):
    """Recursively refine ``coarse`` (at octave ``j_start``) down to the full field.

    ``detail_std`` maps octave j -> amplitude (dict or callable). ``cond_fn`` maps octave
    j -> scale-coordinate vector (B, cond_dim) for arm B, or None for arm A. Deterministic
    given ``key``.
    """
    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)
        cond = None if cond_fn is None else cond_fn(j)
        det_n = sample(apply_fn, params, k, coarse, 3, n_steps=n_steps,
                       cond_vec=cond, solver=solver)
        s = detail_std(j) if callable(detail_std) else detail_std[j]
        det = det_n * s
        coarse = haar.idwt2(coarse, (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
    return coarse
