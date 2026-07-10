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


def score_from_velocity(v, x, t):
    """Score s(x,t)=grad log p_t(x) from the OT/linear-path velocity: (t*v - x)/(1-t).

    Exact for x_t=(1-t)x0 + t*x1, x0~N(0,I). ``t`` is a scalar or per-sample array.
    """
    return (t * v - x) / (1.0 - t)


def sample_sde(apply_fn, params, key, coarse, out_channels, n_steps=100,
               cond_vec=None, churn=0.0):
    """Marginal-preserving churn-SDE sampler (t: 0->1).

    dx = [v + eps(t)*s] dt + sqrt(2 eps(t)) dW with eps(t)=churn*(1-t) and
    s=(t*v - x)/(1-t), so the update is
        x += [v + churn*(t*v - x)] dt + sqrt(2*churn*(1-t)*dt) * z.
    ``churn=0`` reduces to the deterministic probability-flow (Euler) ODE; any churn>=0
    preserves the marginals (so it corrects velocity-approximation error / under-dispersion
    without changing the target distribution).
    """
    B, H, W, _ = coarse.shape
    key0, key_path = jax.random.split(key)
    x0 = jax.random.normal(key0, (B, H, W, out_channels))
    dt = 1.0 / n_steps
    ts = jnp.arange(n_steps) * dt
    keys = jax.random.split(key_path, n_steps)

    def vf(t, x):
        return apply_fn({"params": params}, x, jnp.full((B,), t), coarse, cond_vec)

    def step(x, inp):
        t, k = inp
        v = vf(t, x)
        drift = v + churn * (t * v - x)                 # v + eps(t)*score
        noise = jnp.sqrt(2.0 * churn * (1.0 - t) * dt) * jax.random.normal(k, x.shape)
        return x + drift * dt + noise, None

    x1, _ = jax.lax.scan(step, x0, (ts, keys))
    return x1
