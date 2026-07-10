"""Rung (iii) scaffolding gate — the multi-field, two-arm generator trains and generates.

Not the P-null verdict (that needs the full GPU run + scaledrift measurement); this is the
backpressure that the multi-field trainer (arm A and arm B) and eval-generation from a
held-out real coarse run end-to-end, produce finite fields of the right shape, and are
deterministic under a fixed seed. Small budget so it runs on CPU.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from wfm.dataset import field_to_octaves
from wfm.generate import generate_recursive
from wfm.train import train_generator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tiles():
    return np.load(os.path.join(REPO, "data_cache", "tiles_128.npz"))["grf"]


def test_arm_a_trains_and_generates_deterministically():
    tiles = _tiles()
    state, meta = train_generator(tiles[:12], [1, 2], arm="A", channels=(16, 32),
                                  steps=120, batch=6, seed=0)
    assert meta["lossN"] < meta["loss0"]
    assert set(meta["std_by_j"]) == {1, 2}
    pools, _ = field_to_octaves(tiles[12:16], [2])
    coarse = pools[2][1]
    key = jax.random.PRNGKey(3)
    g1 = generate_recursive(state.apply_fn, state.params, coarse, 2, key,
                            meta["std_by_j"], n_steps=30)
    g2 = generate_recursive(state.apply_fn, state.params, coarse, 2, key,
                            meta["std_by_j"], n_steps=30)
    assert g1.shape == (4, 128, 128, 1)
    assert bool(jnp.all(jnp.isfinite(g1)))
    assert jnp.array_equal(g1, g2)


def test_arm_b_accepts_scale_coordinate():
    tiles = _tiles()
    cond = {1: [0.0, 0.0], 2: [1.0, 0.0]}
    state, meta = train_generator(tiles[:12], [1, 2], arm="B", cond_by_octave=cond,
                                  channels=(16, 32), steps=120, batch=6, seed=0)
    assert meta["cond_dim"] == 2
    pools, _ = field_to_octaves(tiles[12:16], [2])
    coarse = pools[2][1]
    cond_fn = lambda j: jnp.broadcast_to(jnp.asarray(cond[j], jnp.float32), (4, 2))
    g = generate_recursive(state.apply_fn, state.params, coarse, 2, jax.random.PRNGKey(1),
                           meta["std_by_j"], cond_fn=cond_fn, n_steps=30)
    assert g.shape == (4, 128, 128, 1) and bool(jnp.all(jnp.isfinite(g)))
