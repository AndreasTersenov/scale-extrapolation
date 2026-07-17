"""Heavy-tailed-base conditional flow matching (R13 bake-off candidate 3).

The C1 CFM recipe verbatim (OT interpolation, velocity target x1 - x0, heun ODE
pushforward) with ONE change: the base distribution x0 is unit-variance
Student-t(nu) instead of N(0, I). Tail order of an ODE pushforward is set by the
base (the learned map is Lipschitz on compacts), so heavy tails become a
FIRST-order property of the sampler rather than something the objective must
discover — the repair aimed at the measured beta=1 symmetric-tail blindness and,
upstream, at C1's own kurtosis deficit (18-21% sandbox, the Gaussian-base ODE tail
deficit in the RESHAPE taxonomy).

NOTE (R13): this is a SAMPLER/BASE change, not an objective change — if it wins
the bake-off it runs as a NEW arm (C1-t) with its own prereg, not as C3.

nu is pinned per-use: the bake-off toys use nu=5 (matching the t(5) gate target);
a C1-t prereg must pin nu for the sandbox from stated reasoning, before results.
"""
from __future__ import annotations

import os
import sys

import numpy as np

import jax
import jax.numpy as jnp

# Same shim as wfm/__init__.py (jax_flows is not pip-installed on this cluster —
# its pyproject pins a diffrax broken here; we need only the pure-JAX core). Kept
# HERE too so this module is import-order independent: the 6-job bake-off
# submission (16609141-47) failed at t=0 because the runner imported c1t before
# any wfm module had installed the path.
_JF = os.path.expanduser("~/software/jax_flows")
if _JF not in sys.path:
    sys.path.append(_JF)

from jax_flows.flow_matching import ot_interpolate  # noqa: E402

NU = 5.0


def t_base(key, shape, nu=NU):
    """Unit-variance Student-t(nu) base draws (nu > 2)."""
    return jax.random.t(key, nu, shape) / np.sqrt(nu / (nu - 2.0))


def make_tcfm_step(cond_vec=None, nu=NU):
    """Jitted CFM training step with the t(nu) base — wfm.cfm.cfm_loss otherwise."""
    @jax.jit
    def step(state, detail, coarse):
        key, new_key = jax.random.split(state.key)
        key_t, key_n = jax.random.split(key)
        B = detail.shape[0]

        def loss_fn(p):
            t = jax.random.uniform(key_t, (B,))
            x0 = t_base(key_n, detail.shape, nu)
            x_t = ot_interpolate(x0, detail, t)
            v = state.apply_fn({"params": p}, x_t, t, coarse, cond_vec)
            return jnp.mean((v - (detail - x0)) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads).replace(key=new_key)
        return state, loss
    return step


def sample_tbase(apply_fn, params, key, coarse, out_channels=3, n_steps=80,
                 cond_vec=None, nu=NU):
    """Heun ODE pushforward from the t(nu) base (wfm.cfm.sample with x0 ~ t)."""
    B, H, W, _ = coarse.shape
    x0 = t_base(key, (B, H, W, out_channels), nu)
    dt = 1.0 / n_steps
    ts = jnp.arange(n_steps) * dt

    def vf(t, x):
        return apply_fn({"params": params}, x, jnp.full((B,), t), coarse, cond_vec)

    def heun(x, t):
        v1 = vf(t, x)
        v2 = vf(t + dt, x + dt * v1)
        return x + 0.5 * dt * (v1 + v2), None

    x1, _ = jax.lax.scan(heun, x0, ts)
    return x1
