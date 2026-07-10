"""Conditional flow-matching loss, sampler, and train state for the wavelet generator.

The OT interpolation and the (x1 - x0) velocity target are the ``jax_flows`` FM core; here
they are made conditional on (coarse, scale_coord) and wired to euler/heun ODE sampling via
``jax.lax.scan`` (diffrax is avoided on this cluster).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from jax_flows.flow_matching import ot_interpolate  # reuse the FM core


class TrainState(train_state.TrainState):
    key: jax.Array = None


def make_train_state(model, key, detail_shape, coarse_shape, cond_dim, lr,
                     total_steps=10000, warmup=0):
    key_init, key_train = jax.random.split(key)
    B = detail_shape[0]
    dummy = (jnp.ones(detail_shape), jnp.ones((B,)), jnp.ones(coarse_shape),
             jnp.ones((B, cond_dim)) if cond_dim > 0 else None)
    params = model.init(key_init, *dummy)["params"]
    if warmup > 0:
        sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, total_steps)
    else:
        sched = lr
    tx = optax.adam(sched)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, key=key_train)


def cfm_loss(params, apply_fn, detail, coarse, key, cond_vec=None):
    """L = E_{t,x0} || v(x_t, t | coarse, cond) - (detail - x0) ||^2."""
    key_t, key_n = jax.random.split(key)
    B = detail.shape[0]
    t = jax.random.uniform(key_t, (B,))
    x0 = jax.random.normal(key_n, detail.shape)
    x_t = ot_interpolate(x0, detail, t)
    target = detail - x0
    v = apply_fn({"params": params}, x_t, t, coarse, cond_vec)
    return jnp.mean((v - target) ** 2)


def make_step(cond_vec=None):
    """Return a jitted training step closing over a fixed ``cond_vec`` (or None)."""
    @jax.jit
    def step(state, detail, coarse):
        key, new_key = jax.random.split(state.key)
        loss, grads = jax.value_and_grad(cfm_loss)(
            state.params, state.apply_fn, detail, coarse, key, cond_vec)
        state = state.apply_gradients(grads=grads).replace(key=new_key)
        return state, loss
    return step


def sample(apply_fn, params, key, coarse, out_channels, n_steps=100,
           cond_vec=None, solver="heun"):
    """Integrate dx/dt = v(x, t | coarse, cond) from t=0 (noise) to t=1 (detail)."""
    B, H, W, _ = coarse.shape
    x0 = jax.random.normal(key, (B, H, W, out_channels))
    dt = 1.0 / n_steps
    ts = jnp.arange(n_steps) * dt

    def vf(t, x):
        return apply_fn({"params": params}, x, jnp.full((B,), t), coarse, cond_vec)

    def euler(x, t):
        return x + dt * vf(t, x), None

    def heun(x, t):
        v1 = vf(t, x)
        v2 = vf(t + dt, x + dt * v1)
        return x + 0.5 * dt * (v1 + v2), None

    x1, _ = jax.lax.scan(euler if solver == "euler" else heun, x0, ts)
    return x1
