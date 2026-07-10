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


def _per_bin_std(w, onehot, cnt):
    """Std of ``w`` within each coarse quantile bin (differentiable in w)."""
    m = (onehot.T @ w) / cnt
    msq = (onehot.T @ (w * w)) / cnt
    return jnp.sqrt(jnp.maximum(msq - m * m, 1e-8))


def cfm_loss_dispersion(params, apply_fn, detail, coarse, key, cond_vec=None,
                        lam=0.3, n_bins=8):
    """CFM loss + a conditional-DISPERSION matching regularizer (step-(c) objective).

    L = L_cfm + lam * mean_bin ( sd_pred(bin) - sd_data(bin) )^2, where per coarse quantile
    bin (over the minibatch at this octave): sd_data = std of the target detail, and
    sd_pred = std of the model's one-step (Tweedie) data estimate x1_hat = x_t + (1-t)*v.
    L2 flow matching mean-collapses => sd_pred(bin) shrinks below sd_data(bin); this penalty
    opposes the collapse directly, with no sampling in the training loop. The coarse-bin
    assignment is data (constant in params); gradients flow only through sd_pred.
    """
    key_t, key_n = jax.random.split(key)
    B = detail.shape[0]
    t = jax.random.uniform(key_t, (B,))
    x0 = jax.random.normal(key_n, detail.shape)
    x_t = ot_interpolate(x0, detail, t)
    v = apply_fn({"params": params}, x_t, t, coarse, cond_vec)
    cfm = jnp.mean((v - (detail - x0)) ** 2)

    t_bc = t.reshape(B, *([1] * (detail.ndim - 1)))
    x1_hat = x_t + (1.0 - t_bc) * v                       # Tweedie mean E[x1|x_t]
    w_pred = x1_hat.reshape(-1)
    w_data = detail.reshape(-1)
    c = jnp.broadcast_to(coarse, detail.shape).reshape(-1)   # coarse per detail location
    edges = jnp.quantile(c, jnp.linspace(0.0, 1.0, n_bins + 1))
    idx = jnp.clip(jnp.digitize(c, edges[1:-1]), 0, n_bins - 1)
    onehot = jax.nn.one_hot(idx, n_bins)
    cnt = onehot.sum(0) + 1e-6
    reg = jnp.mean((_per_bin_std(w_pred, onehot, cnt)
                    - _per_bin_std(w_data, onehot, cnt)) ** 2)
    return cfm + lam * reg


def make_step(cond_vec=None, lam=0.0, n_bins=8):
    """Return a jitted training step. ``lam`` > 0 adds the dispersion regularizer."""
    @jax.jit
    def step(state, detail, coarse):
        key, new_key = jax.random.split(state.key)

        def loss_fn(p):
            if lam > 0:
                return cfm_loss_dispersion(p, state.apply_fn, detail, coarse, key,
                                           cond_vec, lam, n_bins)
            return cfm_loss(p, state.apply_fn, detail, coarse, key, cond_vec)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
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
