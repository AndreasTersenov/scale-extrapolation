"""Validation gates for the phase-1c Gaussian-NLL detail head (option 2).

The head must, on synthetic data with KNOWN conditional law detail = m(c) + sigma(c)*z:
  1. recover the conditional mean m(c) via the flow's mean-path anchor v(0, t=0 | c);
  2. recover the conditional std sigma(c) via the log-sigma head e^{g(0, t=0 | c)};
  3. produce VARIANCE-FAITHFUL samples per coarse bin under the pre-registered sampler
     (deterministic mean-path endpoint + explicit e^g noise, no churn) -- the unit-scale
     G-1c gate;
  4. learn a FLAT sigma on constant-variance (GRF-like) data -- the unit-scale null gate;
  5. keep the API backward compatible (6-channel output sliced by consumers) and the
     sampler deterministic given a key.

Tests-first per CLAUDE.md: these gates exist before the implementation.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from wfm.cfm import cfm_loss_nll, make_step, make_train_state, sample_nll
from wfm.model import ConditionalUNet


def _smooth_coarse(key, n, hw=16):
    """Smooth random 'coarse' fields in ~[-1.5, 1.5] (low-pass filtered noise)."""
    x = jax.random.normal(key, (n, hw, hw, 1))
    k = jnp.ones((5, 5, 1, 1)) / 25.0
    for _ in range(3):
        x = jax.lax.conv_general_dilated(x, k, (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return 1.5 * x / jnp.std(x)


def _true_mean(c):
    return 0.5 * jnp.tanh(c)


def _true_sigma(c, flat=False):
    """Production-like scale: per-octave details are standardized to overall std ~1, so
    sigma near 1 (and the g-head's sigma=1 zeros-init) is the realistic regime. LINEAR
    in c so the conv net's neighborhood smoothing carries no Jensen (curvature) bias and
    the gate isolates the head mechanism itself."""
    if flat:
        return 0.85 * jnp.ones_like(c)
    return jnp.maximum(0.85 + 0.12 * c, 0.2)         # rises with c, ~[0.67, 1.03] at +-1.5


def _make_data(key, n=192, hw=16, flat=False):
    kc, kz = jax.random.split(key)
    coarse = _smooth_coarse(kc, n, hw)
    z = jax.random.normal(kz, (n, hw, hw, 3))
    detail = _true_mean(coarse) + _true_sigma(coarse, flat) * z   # broadcast over 3 chans
    return detail, coarse


def _train_nll(detail, coarse, steps=1000, seed=0):
    model = ConditionalUNet(out_channels=3, channels=(8, 16), bottleneck=32,
                            embed_dim=32, cond_dim=0, variance_head=True)
    key = jax.random.PRNGKey(seed)
    state = make_train_state(model, key, (16,) + detail.shape[1:],
                             (16,) + coarse.shape[1:], 0, 3e-3,
                             total_steps=steps, warmup=steps // 10)
    step = make_step(None, nll=True)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, detail.shape[0], 16)
        state, loss = step(state, detail[idx], coarse[idx])
    return state, float(loss)


@pytest.fixture(scope="module")
def trained():
    detail, coarse = _make_data(jax.random.PRNGKey(3))
    state, loss = _train_nll(detail, coarse)
    return state, detail, coarse, loss


def _head_outputs(state, coarse):
    """mu, sigma from the (x_t=0, t=0) anchor forward."""
    B = coarse.shape[0]
    zeros = jnp.zeros((B,) + coarse.shape[1:-1] + (3,))
    out = state.apply_fn({"params": state.params}, zeros, jnp.zeros((B,)), coarse, None)
    assert out.shape[-1] == 6, "variance_head model must output [v(3), g(3)]"
    return out[..., :3], jnp.exp(out[..., 3:])


def _bin_idx(c, edges):
    """Bin index per pixel, with out-of-range pixels MASKED (-1), not folded into the
    edge bins -- the unbounded tails are thin and would contaminate the edge-bin stats."""
    idx = np.digitize(c, edges) - 1
    idx[(c < edges[0]) | (c >= edges[-1])] = -1
    return idx


def _binned(values, coarse3, edges):
    """Mean of ``values`` per coarse bin (values/coarse broadcast to same shape)."""
    c = np.asarray(jnp.broadcast_to(coarse3, values.shape)).ravel()
    v = np.asarray(values).ravel()
    idx = _bin_idx(c, edges)
    return np.array([v[idx == b].mean() for b in range(len(edges) - 1)])


def test_head_recovers_mean_and_sigma(trained):
    state, detail, coarse, _ = trained
    mu, sig = _head_outputs(state, coarse)
    edges = np.linspace(-1.2, 1.2, 7)
    m_hat = _binned(mu, coarse, edges)
    m_true = _binned(np.asarray(_true_mean(jnp.broadcast_to(coarse, mu.shape))),
                     coarse, edges)
    s_hat = _binned(sig, coarse, edges)
    s_true = _binned(np.asarray(_true_sigma(jnp.broadcast_to(coarse, mu.shape))),
                     coarse, edges)
    # mean: within 0.1 absolute per bin (true mean spans ~[-0.7, 0.7])
    assert np.max(np.abs(m_hat - m_true)) < 0.12, (m_hat, m_true)
    # sigma: within 12% relative per bin, and MONOTONE trend captured
    assert np.max(np.abs(s_hat / s_true - 1)) < 0.12, (s_hat, s_true)
    assert s_hat[-1] > s_hat[0] + 0.15, "learned sigma must rise with c (true gap 0.24)"


def test_sampler_variance_faithful(trained):
    """Unit-scale G-1c: per-coarse-bin std of generated details matches truth <=12%."""
    state, detail, coarse, _ = trained
    gen = sample_nll(state.apply_fn, state.params, jax.random.PRNGKey(11), coarse, 3)
    assert gen.shape == detail.shape
    edges = np.linspace(-1.2, 1.2, 7)
    c = np.asarray(jnp.broadcast_to(coarse, gen.shape)).ravel()
    idx = _bin_idx(c, edges)
    g = np.asarray(gen).ravel()
    d = np.asarray(detail).ravel()
    s_gen = np.array([g[idx == b].std() for b in range(len(edges) - 1)])
    s_real = np.array([d[idx == b].std() for b in range(len(edges) - 1)])
    assert np.max(np.abs(s_gen / s_real - 1)) < 0.12, (s_gen, s_real)


def test_flat_sigma_null():
    """Constant-sigma (GRF-like) data: learned sigma flat within 10%, no spurious slope."""
    detail, coarse = _make_data(jax.random.PRNGKey(5), flat=True)
    state, _ = _train_nll(detail, coarse, steps=700, seed=1)
    _, sig = _head_outputs(state, coarse)
    edges = np.linspace(-1.2, 1.2, 7)
    s_hat = _binned(sig, coarse, edges)
    assert np.max(np.abs(s_hat / 0.85 - 1)) < 0.10, s_hat
    assert abs(s_hat[-1] - s_hat[0]) < 0.08, "no spurious sigma slope on null data"


def test_api_and_determinism(trained):
    state, detail, coarse, loss = trained
    # loss finite and NLL-shaped (can go below the pure-L2 floor)
    assert np.isfinite(loss)
    # sampler deterministic given key
    a = sample_nll(state.apply_fn, state.params, jax.random.PRNGKey(2), coarse[:4], 3)
    b = sample_nll(state.apply_fn, state.params, jax.random.PRNGKey(2), coarse[:4], 3)
    assert jnp.array_equal(a, b)
    # cfm_loss_nll runs standalone
    l = cfm_loss_nll(state.params, state.apply_fn, detail[:8], coarse[:8],
                     jax.random.PRNGKey(0))
    assert np.isfinite(float(l))
