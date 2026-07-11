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


def _dispersion_penalty(detail, coarse, x1_hat, n_bins):
    """mean over coarse quantile bins of ( std(x1_hat) - std(detail) )^2."""
    w_pred = x1_hat.reshape(-1)
    w_data = detail.reshape(-1)
    c = jnp.broadcast_to(coarse, detail.shape).reshape(-1)
    edges = jnp.quantile(c, jnp.linspace(0.0, 1.0, n_bins + 1))
    idx = jnp.clip(jnp.digitize(c, edges[1:-1]), 0, n_bins - 1)
    onehot = jax.nn.one_hot(idx, n_bins)
    cnt = onehot.sum(0) + 1e-6
    return jnp.mean((_per_bin_std(w_pred, onehot, cnt)
                     - _per_bin_std(w_data, onehot, cnt)) ** 2)


def cfm_loss_dispersion(params, apply_fn, detail, coarse, key, cond_vec=None,
                        lam=0.3, n_bins=8, t_lo=0.0):
    """CFM loss + a conditional-DISPERSION matching regularizer.

    L = L_cfm + lam * mean_bin ( std(x1_hat) - std(detail) )^2, per coarse quantile bin,
    with x1_hat = x_t + (1-t)*v the one-step (Tweedie) data estimate. ``t_lo`` selects the
    variant:
      * ``t_lo == 0`` (step-c original): the penalty shares the CFM t~U(0,1). But x1_hat is
        E[x1|x_t], structurally under-dispersed for t<1 -> mis-specified target (it failed).
      * ``t_lo > 0`` (step-c' option 1, t-consistent): a SECOND forward at t~U(t_lo,1) with
        fresh noise, where x_t already carries most of x1 so x1_hat is a faithful estimate and
        std(x1_hat)~std(detail) for a good model, while a collapsed velocity still shows a
        spread deficit. Recommended t_lo=0.6.
    Bin assignment is data (constant in params); gradients flow through x1_hat only.
    """
    key_t, key_n, key_rt, key_rn = jax.random.split(key, 4)
    B = detail.shape[0]
    t = jax.random.uniform(key_t, (B,))
    x0 = jax.random.normal(key_n, detail.shape)
    x_t = ot_interpolate(x0, detail, t)
    v = apply_fn({"params": params}, x_t, t, coarse, cond_vec)
    cfm = jnp.mean((v - (detail - x0)) ** 2)
    if lam == 0:
        return cfm

    if t_lo > 0.0:                                    # option 1: late-t second forward
        tr = t_lo + (1.0 - t_lo) * jax.random.uniform(key_rt, (B,))
        x0r = jax.random.normal(key_rn, detail.shape)
        x_tr = ot_interpolate(x0r, detail, tr)
        vr = apply_fn({"params": params}, x_tr, tr, coarse, cond_vec)
        t_bc = tr.reshape(B, *([1] * (detail.ndim - 1)))
        x1_hat = x_tr + (1.0 - t_bc) * vr
    else:                                             # original: reuse the CFM t
        t_bc = t.reshape(B, *([1] * (detail.ndim - 1)))
        x1_hat = x_t + (1.0 - t_bc) * v
    return cfm + lam * _dispersion_penalty(detail, coarse, x1_hat, n_bins)


G_CLIP = (-5.0, 3.0)     # log-sigma numerical guard (free periphery)


def cfm_loss_nll(params, apply_fn, detail, coarse, key, cond_vec=None):
    """Phase-1c option-2 loss: CFM on the velocity + Gaussian NLL for the log-sigma head.

    The model outputs [v, g] (``variance_head=True``). The velocity trains with the
    standard CFM L2 at random (x0, t) -- unchanged. The NLL is anchored at the flow's
    center point (x_t = 0, t = 0), where the one-step OT-path predictor
    mu = x_t + (1-t) v = v(0, 0 | coarse, cond) equals E[detail | coarse, cond] at the
    CFM optimum (x0 is independent of x1, so v(x, 0) = E[x1] - x). There

        NLL = mean( 0.5 * ( (detail - mu)^2 * e^{-2g} + 2 g ) ),

    i.e. the full (mean, variance) Gaussian conditional is NLL-trained at the anchor.
    Training mu through the NLL is necessary: with CFM gradients alone, v(0, 0) is
    contaminated by nearby t > 0 targets (which are smaller in magnitude) and the
    conditional mean comes out shrunk ~15% at extreme coarse values (seen in the
    validation gate). e^{2g(0,0|coarse,cond)} regresses the per-coefficient conditional
    variance Var(detail | coarse, cond) -- exactly the object var_slope measures.
    """
    key_t, key_n = jax.random.split(key)
    B = detail.shape[0]
    C = detail.shape[-1]
    t = jax.random.uniform(key_t, (B,))
    x0 = jax.random.normal(key_n, detail.shape)
    x_t = ot_interpolate(x0, detail, t)
    out = apply_fn({"params": params}, x_t, t, coarse, cond_vec)
    cfm = jnp.mean((out[..., :C] - (detail - x0)) ** 2)

    out0 = apply_fn({"params": params}, jnp.zeros_like(detail), jnp.zeros((B,)),
                    coarse, cond_vec)
    mu = out0[..., :C]
    g = jnp.clip(out0[..., C:], *G_CLIP)
    r = detail - mu
    nll = jnp.mean(0.5 * (r * r * jnp.exp(-2.0 * g) + 2.0 * g))
    return cfm + nll


def sample_nll(apply_fn, params, key, coarse, out_channels, cond_vec=None, **_):
    """Pre-registered phase-1c sampler: deterministic mean-path endpoint + explicit
    variance, no churn.

    detail = mu + e^{g} * z with (mu, g) = model(x_t=0, t=0 | coarse, cond) and
    z ~ N(0, I). mu is the flow's conditional-mean estimate (the OT mean-path endpoint
    from the distribution center); ALL conditional variance is carried explicitly by the
    learned per-coefficient e^{2g} -- no pushforward/noise double-counting.
    """
    B, H, W, _ = coarse.shape
    zeros = jnp.zeros((B, H, W, out_channels))
    out0 = apply_fn({"params": params}, zeros, jnp.zeros((B,)), coarse, cond_vec)
    mu = out0[..., :out_channels]
    g = jnp.clip(out0[..., out_channels:], *G_CLIP)
    z = jax.random.normal(key, mu.shape)
    return mu + jnp.exp(g) * z


def make_step(cond_vec=None, lam=0.0, n_bins=8, t_lo=0.0, nll=False):
    """Return a jitted training step. ``lam`` > 0 adds the dispersion regularizer (t_lo>0
    selects the option-1 late-t variant); ``nll=True`` uses the phase-1c NLL-head loss."""
    @jax.jit
    def step(state, detail, coarse):
        key, new_key = jax.random.split(state.key)

        def loss_fn(p):
            if nll:
                return cfm_loss_nll(p, state.apply_fn, detail, coarse, key, cond_vec)
            if lam > 0:
                return cfm_loss_dispersion(p, state.apply_fn, detail, coarse, key,
                                           cond_vec, lam, n_bins, t_lo)
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
