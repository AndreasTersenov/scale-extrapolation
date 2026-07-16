#!/usr/bin/env python
"""Step 4: synthesize the exactly scale-invariant control field (prereg
log/2026-07-16-prereg-step4-selfsim.md; deterministic seed).

detail_j = std_j * (a + b*standardized(coarse_j)) * z at EVERY octave (same law in
standardized coordinates: no drift, and exactly the Gaussian-NLL head's model class).
Writes data_cache/tiles_selfsim.npz (386 tiles, 128x128).
"""
import os
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

from wfm import haar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, A, B = 386, 0.9, 0.25
key = jax.random.PRNGKey(20260716)

k1, key = jax.random.split(key)
coarse = jax.random.normal(k1, (N, 8, 8, 1))
ker = jnp.ones((3, 3, 1, 1)) / 9.0
coarse = jax.lax.conv_general_dilated(coarse, ker, (1, 1), "SAME",
                                      dimension_numbers=("NHWC", "HWIO", "NHWC"))
for j in range(4, 0, -1):
    std_j = 0.5 * (1.5 ** j)
    chat = (coarse - coarse.mean(axis=(1, 2, 3), keepdims=True)) / \
        coarse.std(axis=(1, 2, 3), keepdims=True)
    key, kz = jax.random.split(key)
    z = jax.random.normal(kz, coarse.shape[:3] + (3,))
    det = std_j * (A + B * chat) * z
    coarse = haar.idwt2(coarse, (det[..., 0:1], det[..., 1:2], det[..., 2:3]))

tiles = np.asarray(coarse[..., 0], np.float32)
np.savez(os.path.join(REPO, "data_cache", "tiles_selfsim.npz"), selfsim=tiles)
print("wrote data_cache/tiles_selfsim.npz", tiles.shape,
      "std", float(tiles.std()))
