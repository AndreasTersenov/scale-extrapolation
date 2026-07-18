"""Field-scale plumbing gates for arm C1-t (R17: the bake-off validated the
t-base ODE at toy scale only; these gates cover the production-shape pipeline).

Light by design (the R17 wording): trainer pipeline identity with C1 (D4 + std
convention + hooks + arm-B conditioning), multi-octave recursion producing
finite fields at the right shapes with correct std un-scaling, and sampler
determinism — the statistical adjudication is the pre-registered arm itself.
"""
import numpy as np

import jax
import jax.numpy as jnp

from arms_p2.c1t.train import generate_recursive_tbase, train_c1t_generator
from wfm.dataset import d4_augment, field_to_octaves


def test_c1t_trainer_pipeline_matches_c1():
    rng = np.random.default_rng(0)
    tiles = rng.normal(size=(6, 64, 64)).astype(np.float32)
    _, std_ref = field_to_octaves(d4_augment(tiles), [2, 3])
    seen = []
    state, meta = train_c1t_generator(
        tiles, [2, 3], arm="A", channels=(8, 16), steps=4, batch=8, lr=1e-3,
        seed=0, ckpt_steps=(2,), on_checkpoint=lambda s, st, l: seen.append(s),
        augment=True)
    assert seen == [2]
    assert meta["cond_dim"] == 0 and np.isfinite(meta["lossN"])
    assert meta["objective"] == "cfm_tbase" and meta["nu"] == 5.0
    for j in (2, 3):
        assert np.isclose(meta["std_by_j"][j], std_ref[j]), "std convention must be C1's"
    _, meta_b = train_c1t_generator(
        tiles, [2, 3], arm="B", cond_by_octave={2: [0.5, 0.3], 3: [0.4, 0.2]},
        channels=(8, 16), steps=2, batch=8, cond_mode="film", augment=True)
    assert meta_b["cond_dim"] == 2 and meta_b["cond_mode"] == "film"


def test_c1t_recursion_shapes_and_determinism():
    rng = np.random.default_rng(1)
    tiles = rng.normal(size=(6, 64, 64)).astype(np.float32)
    state, meta = train_c1t_generator(tiles, [2, 3], arm="A", channels=(8, 16),
                                      steps=3, batch=8, augment=True)
    std = dict(meta["std_by_j"])
    std[1] = std[2] * 2.0                       # synthetic extrapolated amplitude
    pools, _ = field_to_octaves(tiles[:2], [3])
    coarse = pools[3][1]                        # (2, 8, 8, 1)
    g1 = generate_recursive_tbase(state.apply_fn, state.params, coarse, 3,
                                  jax.random.PRNGKey(7), std, n_steps=8)
    g2 = generate_recursive_tbase(state.apply_fn, state.params, coarse, 3,
                                  jax.random.PRNGKey(7), std, n_steps=8)
    g3 = generate_recursive_tbase(state.apply_fn, state.params, coarse, 3,
                                  jax.random.PRNGKey(8), std, n_steps=8)
    assert g1.shape == (2, 64, 64, 1)
    assert bool(jnp.all(jnp.isfinite(g1)))
    assert jnp.array_equal(g1, g2) and not jnp.array_equal(g1, g3)
    # std un-scaling: doubling every octave amplitude must change the output
    std2 = {j: 2 * v for j, v in std.items()}
    g4 = generate_recursive_tbase(state.apply_fn, state.params, coarse, 3,
                                  jax.random.PRNGKey(7), std2, n_steps=8)
    assert not jnp.allclose(g1, g4)
