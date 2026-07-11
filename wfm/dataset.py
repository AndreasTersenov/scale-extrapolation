"""Turn a stack of field tiles into per-octave (detail, coarse) training pools.

Each tile is normalized to unit variance, then Haar-decomposed. For each octave j we pool
the detail triples and coarse maps over all tiles and standardize the detail by a single
per-octave amplitude ``std_j`` (measured over the pool) — the model trains in standardized
detail space so its weights transfer across octaves; generation multiplies back by std_j.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from . import haar


def normalize_tiles(tiles):
    """(N,H,W) -> (N,H,W,1) each normalized to zero mean, unit variance."""
    t = jnp.asarray(tiles)[..., None]
    m = t.mean(axis=(1, 2, 3), keepdims=True)
    s = t.std(axis=(1, 2, 3), keepdims=True)
    return (t - m) / s


def d4_augment(tiles):
    """All 8 D4 orientations (4 rotations x optional mirror) of a (N,H,W) tile stack.

    FIELD-level augmentation (attempt 4a): tiles are transformed BEFORE wavelet
    decomposition, so the (detail, coarse) conditional law of each block is the data's
    law under that orientation exactly -- no subband permutation/sign bookkeeping.
    Returns (8N,H,W); block k*N:(k+1)*N is one orientation.
    """
    t = np.asarray(tiles)
    out = []
    for base in (t, t[:, :, ::-1]):                     # identity, mirror
        for k in range(4):                              # 4 rotations
            out.append(np.rot90(base, k, axes=(1, 2)))
    return np.ascontiguousarray(np.concatenate(out, axis=0))


def field_to_octaves(tiles, octaves):
    """Return ``pools[j] = (detail_n, coarse)`` and ``std_by_j[j]`` over a tile stack.

    detail_n: (N,Hj,Wj,3) standardized by std_j; coarse: (N,Hj,Wj,1); std_j: float.
    """
    fields = normalize_tiles(tiles)
    pools, std_by_j = {}, {}
    for j in octaves:
        detail, coarse = haar.octave_pair(fields, j)
        s = float(detail.std())
        pools[j] = (detail / s, coarse)
        std_by_j[j] = s
    return pools, std_by_j
