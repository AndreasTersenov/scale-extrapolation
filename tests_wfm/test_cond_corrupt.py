"""Validation gates for 4b' conditioning augmentation (anti-compounding lever).

During training the coarse conditioning input is corrupted with per-example Gaussian
noise of relative level s ~ U(0, s_max), and s is EXPOSED to the model as an extra
conditioning dimension (cascaded-diffusion recipe); at generation s = 0. Gates:
  1. plumbing: the corrupted step trains, the extra cond dim is wired, meta records it;
  2. clean-eval non-inferiority: on the known-truth toy, a corruption-trained model
     evaluated with CLEAN conditioning (s=0) still recovers the conditional law about
     as well as an uncorrupted model -- corruption must not poison the s=0 conditional.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from wfm.cfm import make_step, make_train_state
from wfm.model import ConditionalUNet
from wfm.train import train_generator


def _smooth_coarse(key, n, hw=16):
    x = jax.random.normal(key, (n, hw, hw, 1))
    k = jnp.ones((5, 5, 1, 1)) / 25.0
    for _ in range(3):
        x = jax.lax.conv_general_dilated(x, k, (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return 1.5 * x / jnp.std(x)


def _make_data(key, n=192, hw=16):
    kc, kz = jax.random.split(key)
    coarse = _smooth_coarse(kc, n, hw)
    z = jax.random.normal(kz, (n, hw, hw, 3))
    sigma = jnp.maximum(0.85 + 0.12 * coarse, 0.2)
    return 0.5 * jnp.tanh(coarse) + sigma * z, coarse


def _train(detail, coarse, corrupt_smax, steps=1000, seed=0):
    cond_dim = 1 if corrupt_smax > 0 else 0
    model = ConditionalUNet(out_channels=3, channels=(8, 16), bottleneck=32,
                            embed_dim=32, cond_dim=cond_dim, variance_head=True)
    state = make_train_state(model, jax.random.PRNGKey(seed),
                             (16,) + detail.shape[1:], (16,) + coarse.shape[1:],
                             cond_dim, 3e-3, total_steps=steps, warmup=steps // 10)
    step = make_step(None, nll=True, corrupt_smax=corrupt_smax)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, detail.shape[0], 16)
        state, loss = step(state, detail[idx], coarse[idx])
    return state, cond_dim, float(loss)


def _sigma_profile(state, coarse, cond_dim):
    B = coarse.shape[0]
    cv = jnp.zeros((B, cond_dim)) if cond_dim else None       # clean eval: s = 0
    out = state.apply_fn({"params": state.params},
                         jnp.zeros(coarse.shape[:3] + (3,)), jnp.zeros((B,)),
                         coarse, cv)
    sig = np.asarray(jnp.exp(out[..., 3:]))
    c = np.asarray(jnp.broadcast_to(coarse, sig.shape)).ravel()
    s = sig.ravel()
    edges = np.linspace(-1.2, 1.2, 7)
    idx = np.digitize(c, edges) - 1
    idx[(c < edges[0]) | (c >= edges[-1])] = -1
    return np.array([s[idx == b].mean() for b in range(6)])


@pytest.fixture(scope="module")
def toy():
    return _make_data(jax.random.PRNGKey(3))


def test_plumbing_corrupt_runs():
    rng = np.random.default_rng(0)
    tiles = rng.normal(size=(6, 32, 32)).astype(np.float32)
    state, meta = train_generator(tiles, [1, 2], arm="A", channels=(8, 16), steps=6,
                                  batch=4, lr=1e-3, nll=True, corrupt_smax=0.3)
    assert meta["corrupt_smax"] == 0.3
    assert meta["cond_dim"] == 1                 # the exposed corruption level
    assert np.isfinite(meta["lossN"])


def test_clean_eval_non_inferior(toy):
    detail, coarse = toy
    true_prof = np.array([0.85 + 0.12 * c for c in
                          0.5 * (np.linspace(-1.2, 1.2, 7)[:-1]
                                 + np.linspace(-1.2, 1.2, 7)[1:])])
    st0, cd0, _ = _train(detail, coarse, corrupt_smax=0.0)
    stc, cdc, _ = _train(detail, coarse, corrupt_smax=0.3)
    dev0 = np.max(np.abs(_sigma_profile(st0, coarse, cd0) / true_prof - 1))
    devc = np.max(np.abs(_sigma_profile(stc, coarse, cdc) / true_prof - 1))
    # corruption training must not poison the clean (s=0) conditional law
    assert devc < 0.15, (devc, dev0)
    assert devc < dev0 + 0.08, (devc, dev0)
