"""Direct noise-conditioned sampling for the C3 arm.

The frozen ConditionalUNet is reused with minimal surgery (prereg): Gaussian noise z
enters where the flow state x_t entered, and the t-embedding is fed the constant 0.
The network output IS the standardized detail sample — one forward = one sample. No
ODE integration, no NLL head (retired, R3).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from wfm import haar


def sample_direct(apply_fn, params, key, coarse, out_channels=3, cond_vec=None):
    """One detail sample per conditioning: model(z, t=0 | coarse, cond), z ~ N(0, I)."""
    B, H, W, _ = coarse.shape
    z = jax.random.normal(key, (B, H, W, out_channels))
    return apply_fn({"params": params}, z, jnp.zeros((B,)), coarse, cond_vec)


def generate_recursive_direct(apply_fn, params, coarse, j_start, key, detail_std,
                              cond_fn=None):
    """Coarse-to-fine recursion (wfm.generate.generate_recursive with the direct
    sampler): sample standardized detail, un-standardize by the octave amplitude,
    invert one Haar level; recurse to octave 0. Deterministic given ``key``."""
    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)
        cond = None if cond_fn is None else cond_fn(j)
        det_n = sample_direct(apply_fn, params, k, coarse, 3, cond_vec=cond)
        s = detail_std(j) if callable(detail_std) else detail_std[j]
        det = det_n * s
        coarse = haar.idwt2(coarse, (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
    return coarse
