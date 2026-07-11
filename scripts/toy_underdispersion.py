#!/usr/bin/env python
"""Minimal 1-D reproduction of the generator blocker: conditional flow matching
under-disperses conditional variance, WORSE with training, and stochastic sampling helps.

Data: x1 | c ~ N(0, sigma(c)^2) with sigma growing with the condition c (the 1-D analogue of
Var(detail | coarse) rising with the coarse field). We train a tiny CFM v(x_t, t, c), then
measure the conditional std of ODE-generated samples vs the TRUE sigma(c). Everything is
known exactly, so the failure is unambiguous.
"""
import os
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, VERM, GREEN, ORANGE = "#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00"


def sigma(c):                       # true conditional std rises with the condition
    return 0.15 + 0.85 * jax.nn.sigmoid(2.0 * c)


# FIXED finite dataset (like the real ~330 tiles): finite data is what drives the
# over-fitting / mean-collapse. Note E[x1|c]=0 here, so ALL the signal is in the variance.
_DK = jax.random.PRNGKey(7)
_kc, _kx = jax.random.split(_DK)
_C = jax.random.uniform(_kc, (1500,), minval=-2.0, maxval=2.0)
_X1 = sigma(_C) * jax.random.normal(_kx, (1500,))


def data(key, n):
    idx = jax.random.randint(key, (n,), 0, _X1.shape[0])
    return _X1[idx], _C[idx]


def temb(t, d=8):
    f = 2.0 ** jnp.arange(d) * jnp.pi
    a = t[:, None] * f[None, :]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1)


def init(key, h=128):
    ks = jax.random.split(key, 4)
    d_in = 1 + 16 + 1
    def dense(k, i, o):
        return (jax.random.normal(k, (i, o)) * (1.0 / i ** 0.5), jnp.zeros(o))
    return [dense(ks[0], d_in, h), dense(ks[1], h, h), dense(ks[2], h, h), dense(ks[3], h, 1)]


def mlp(p, x, t, c):
    h = jnp.concatenate([x[:, None], temb(t), c[:, None]], -1)
    for w, b in p[:-1]:
        h = jax.nn.silu(h @ w + b)
    w, b = p[-1]
    return (h @ w + b)[:, 0]


def loss(p, key):
    kd, kt, kn = jax.random.split(key, 3)
    x1, c = data(kd, 512)
    t = jax.random.uniform(kt, (512,))
    x0 = jax.random.normal(kn, (512,))
    xt = (1 - t) * x0 + t * x1
    return jnp.mean((mlp(p, xt, t, c) - (x1 - x0)) ** 2)


def train(steps, seed=0):
    key = jax.random.PRNGKey(seed)
    p = init(key)
    opt = optax.adam(2e-3)
    st = opt.init(p)
    snaps = {}
    milestones = {500, 2000, 20000}

    @jax.jit
    def step(p, st, key):
        g = jax.grad(loss)(p, key)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st

    for i in range(1, steps + 1):
        key, k = jax.random.split(key)
        p, st = step(p, st, k)
        if i in milestones:
            snaps[i] = jax.tree_util.tree_map(lambda a: a, p)
    return snaps


def gen_std(p, cgrid, n=4000, nsteps=60, churn=0.0, seed=1):
    key = jax.random.PRNGKey(seed)
    out = []
    for cval in cgrid:
        key, k = jax.random.split(key)
        x = jax.random.normal(k, (n,))
        c = jnp.full((n,), cval)
        dt = 1.0 / nsteps
        for j in range(nsteps):
            t = j * dt
            v = mlp(p, x, jnp.full((n,), t), c)
            if churn > 0:
                key, kn = jax.random.split(key)
                s = (t * v - x) / (1 - t + 1e-6)
                x = x + (v + churn * (1 - t) * s) * dt + jnp.sqrt(2 * churn * (1 - t) * dt) * jax.random.normal(kn, (n,))
            else:
                x = x + v * dt
        out.append(float(jnp.std(x)))
    return np.array(out)


snaps = train(20000)
cgrid = np.linspace(-1.8, 1.8, 19)
true = np.array(sigma(jnp.asarray(cgrid)))

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
# Panel 1: under-dispersion + it gets WORSE with training
ax[0].plot(cgrid, true, "-", color=K, lw=2.5, label="TRUE conditional std")
for s, col, lab in [(500, "#9ecae1", "generated @ 500 steps"),
                    (2000, BLUE, "generated @ 2k steps (best)"),
                    (20000, "#08306b", "generated @ 20k steps (collapsed)")]:
    ax[0].plot(cgrid, gen_std(snaps[s], cgrid), "--o", ms=3, color=col, label=lab)
ax[0].set_xlabel("condition c  (analogue of the coarse field)")
ax[0].set_ylabel("conditional std of generated samples")
ax[0].set_title("Flow matching UNDER-disperses — and it gets WORSE with training\n"
                "(the generated spread falls below the truth, most at high c)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
# Panel 2: churn helps but uniformly (can't match the shape)
ax[1].plot(cgrid, true, "-", color=K, lw=2.5, label="TRUE conditional std")
ax[1].plot(cgrid, gen_std(snaps[2000], cgrid), "--o", ms=3, color=BLUE, label="deterministic ODE (under)")
for ch, col in [(1.0, GREEN), (4.0, ORANGE)]:
    ax[1].plot(cgrid, gen_std(snaps[2000], cgrid, churn=ch), "--s", ms=3, color=col,
               label=f"+ SDE churn ε₀={ch:g}")
ax[1].set_xlabel("condition c")
ax[1].set_ylabel("conditional std of generated samples")
ax[1].set_title("Stochastic sampling (churn) lifts the spread — but ~uniformly\n"
                "(it can't reshape the c-dependence; a LEARNED noise is needed)")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
fig.suptitle("The generator blocker in 1-D (known truth): why P6 is blocked and why a training penalty can't fix it",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(REPO, "results", "toy_underdispersion.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
