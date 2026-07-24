#!/usr/bin/env python
"""Explainer figure for phase-1c step 1: the Gaussian-NLL head fixes under-dispersion
AT UNIT SCALE, where the truth is known exactly.

Same toy law as tests_wfm/test_nll_head.py: detail = 0.5*tanh(c) + sigma(c)*z with
sigma(c) = 0.85 + 0.12*c (linear, so no estimator-side curvature bias). One network is
trained once with the phase-1c loss; then it is sampled TWO ways:
  * the OLD deterministic ODE over its velocity channels (the collapsed baseline), and
  * the NEW pre-registered sampler (deterministic mean-path + explicit learned e^g noise).
Left panel: the sigma-head's learned spread vs the true one. Right panel: the spread of
actual generated samples per coarse bin, old vs new sampler vs truth.
"""
import os
try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wfm.cfm import make_step, make_train_state, sample_nll
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K, BLUE, GREEN, ORANGE = "#000000", "#0072B2", "#009E73", "#E69F00"


def smooth_coarse(key, n, hw=16):
    x = jax.random.normal(key, (n, hw, hw, 1))
    k = jnp.ones((5, 5, 1, 1)) / 25.0
    for _ in range(3):
        x = jax.lax.conv_general_dilated(x, k, (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return 1.5 * x / jnp.std(x)


def true_mean(c):
    return 0.5 * jnp.tanh(c)


def true_sigma(c):
    return jnp.maximum(0.85 + 0.12 * c, 0.2)


def make_data(key, n=256):
    kc, kz = jax.random.split(key)
    coarse = smooth_coarse(kc, n)
    z = jax.random.normal(kz, coarse.shape[:3] + (3,))
    return true_mean(coarse) + true_sigma(coarse) * z, coarse


detail, coarse = make_data(jax.random.PRNGKey(3))
model = ConditionalUNet(out_channels=3, channels=(8, 16), bottleneck=32,
                        embed_dim=32, cond_dim=0, variance_head=True)
state = make_train_state(model, jax.random.PRNGKey(0), (16,) + detail.shape[1:],
                         (16,) + coarse.shape[1:], 0, 3e-3, total_steps=2500, warmup=100)
step = make_step(None, nll=True)
rng = np.random.default_rng(0)
for i in range(2500):
    idx = rng.integers(0, detail.shape[0], 16)
    state, loss = step(state, detail[idx], coarse[idx])
print(f"trained: final loss {float(loss):.3f}")

B = coarse.shape[0]
zeros = jnp.zeros_like(detail)
out0 = state.apply_fn({"params": state.params}, zeros, jnp.zeros((B,)), coarse, None)
sig_learned = jnp.exp(out0[..., 3:])

# OLD sampler: deterministic Euler ODE over the velocity channels of the SAME weights
def sample_ode(key, nsteps=60):
    x = jax.random.normal(key, detail.shape)
    dt = 1.0 / nsteps
    for j in range(nsteps):
        t = jnp.full((B,), j * dt)
        v = state.apply_fn({"params": state.params}, x, t, coarse, None)[..., :3]
        x = x + v * dt
    return x

gen_old = sample_ode(jax.random.PRNGKey(11))
gen_new = sample_nll(state.apply_fn, state.params, jax.random.PRNGKey(11), coarse, 3)

edges = np.linspace(-1.2, 1.2, 9)
mid = 0.5 * (edges[:-1] + edges[1:])
cflat = np.asarray(jnp.broadcast_to(coarse, detail.shape)).ravel()
bidx = np.digitize(cflat, edges) - 1
bidx[(cflat < edges[0]) | (cflat >= edges[-1])] = -1

def per_bin(arr, fn):
    a = np.asarray(arr).ravel()
    return np.array([fn(a[bidx == b]) for b in range(len(edges) - 1)])

s_true_curve = np.asarray(true_sigma(jnp.asarray(mid)))
s_head = per_bin(sig_learned, np.mean)
s_data = per_bin(detail, np.std)
s_old = per_bin(gen_old, np.std)
s_new = per_bin(gen_new, np.std)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.9))
ax[0].plot(mid, s_true_curve, "-", color=K, lw=2.5, label="TRUE spread σ(c)")
ax[0].plot(mid, s_head, "--o", ms=4, color=ORANGE, label="what the σ-head learned  (e^g)")
ax[0].set_xlabel("brightness of the coarse field at the pixel  (condition c)")
ax[0].set_ylabel("spread (std) of the detail, given c")
ax[0].set_title("The new head LEARNS the conditional spread\n"
                "(this is the quantity the old generator kept collapsing)")
ax[0].legend(fontsize=9)
ax[0].grid(alpha=0.25)

ax[1].plot(mid, s_data, "-", color=K, lw=2.5, label="real data")
ax[1].plot(mid, s_old, "--s", ms=4, color=BLUE,
           label="OLD sampler: deterministic ODE (collapsed)")
ax[1].plot(mid, s_new, "--o", ms=4, color=GREEN,
           label="NEW sampler: mean-path + learned noise")
ax[1].set_xlabel("brightness of the coarse field at the pixel  (condition c)")
ax[1].set_ylabel("spread (std) of GENERATED detail, given c")
ax[1].set_title("Same trained network, two ways of sampling\n"
                "(green = the pre-registered phase-1c sampler)")
ax[1].legend(fontsize=9)
ax[1].grid(alpha=0.25)

fig.suptitle("Phase-1c gate, unit scale (truth known exactly): the Gaussian-NLL head restores the spread the ODE loses",
             fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(REPO, "results", "figures", "readouts", "nllhead_gate.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
