"""Backpressure (b) — generation is deterministic: a fixed seed gives an identical field.

Reproducibility gate for the coarse-to-fine recursion. Here we check the atoms it is built
from: the conditional sampler and one Haar-reconstruction step are bit-for-bit reproducible
under a fixed key, and different keys give different samples.
"""
import jax
import jax.numpy as jnp

from wfm import haar
from wfm.cfm import make_train_state, sample
from wfm.model import ConditionalUNet


def _tiny_state(seed=0, n=16):
    model = ConditionalUNet(out_channels=3, channels=(8, 16), bottleneck=32, cond_dim=0)
    detail_shape, coarse_shape = (1, n, n, 3), (1, n, n, 1)
    state = make_train_state(model, jax.random.PRNGKey(seed), detail_shape, coarse_shape,
                             0, 1e-3)
    coarse = jax.random.normal(jax.random.PRNGKey(seed + 1), coarse_shape)
    return state, coarse


def test_sampler_is_deterministic_under_fixed_key():
    state, coarse = _tiny_state()
    key = jax.random.PRNGKey(123)
    a = sample(state.apply_fn, state.params, key, coarse, 3, n_steps=20)
    b = sample(state.apply_fn, state.params, key, coarse, 3, n_steps=20)
    assert jnp.array_equal(a, b)


def test_different_keys_give_different_samples():
    state, coarse = _tiny_state()
    a = sample(state.apply_fn, state.params, jax.random.PRNGKey(1), coarse, 3, n_steps=20)
    b = sample(state.apply_fn, state.params, jax.random.PRNGKey(2), coarse, 3, n_steps=20)
    assert not jnp.array_equal(a, b)


def test_one_recursion_step_is_deterministic():
    """sample a detail triple, invert one Haar level -> a finer field; fixed seed identical."""
    state, coarse = _tiny_state(n=16)
    key = jax.random.PRNGKey(7)

    def one_step():
        det = sample(state.apply_fn, state.params, key, coarse, 3, n_steps=20)
        cH, cV, cD = det[..., 0:1], det[..., 1:2], det[..., 2:3]
        return haar.idwt2(coarse, (cH, cV, cD))

    f1, f2 = one_step(), one_step()
    assert f1.shape == (1, 32, 32, 1)
    assert jnp.array_equal(f1, f2)
