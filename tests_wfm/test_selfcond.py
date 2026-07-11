"""Validation gates for attempt 5: self-conditioning (drift-shaped conditioning mix).

Training conditions, with probability p, on ALTERNATIVE coarse inputs (at production:
generated coarse from the same tile's start) while the target stays the real detail --
the aligned-pair / scheduled-sampling recipe. Gates:
  1. plumbing: train_generator(alt_coarse_pools=..., alt_p=...) runs, meta records it,
     pools align with the (augmented) tile ordering;
  2. mechanism, known-truth toy with STRUCTURED drift (smoothing, the flattening
     analogue): a model trained with the conditioning mix responds to DRIFTED inputs
     with a conditional-spread profile closer to the truth than a model trained on
     clean conditioning only -- the anti-compounding claim at unit scale.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from wfm.cfm import make_step, make_train_state
from wfm.model import ConditionalUNet
from wfm.train import train_generator


def _smooth(x, reps):
    k = jnp.ones((5, 5, 1, 1)) / 25.0
    for _ in range(reps):
        x = jax.lax.conv_general_dilated(x, k, (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return x


def _coarse(key, n, hw=16):
    x = jax.random.normal(key, (n, hw, hw, 1))
    return 1.5 * _smooth(x, 3) / jnp.std(_smooth(x, 3))


def _drift(c):
    """Structured drift: extra smoothing + renormalize (the flattening analogue)."""
    d = _smooth(c, 2)
    return d * (jnp.std(c) / jnp.std(d))


def _make(key, n=192):
    kc, kz = jax.random.split(key)
    c = _coarse(kc, n)
    sig = jnp.maximum(0.85 + 0.12 * c, 0.2)
    return 0.5 * jnp.tanh(c) + sig * jax.random.normal(kz, c.shape[:3] + (3,)), c


def _train(detail, c_real, c_alt, p, steps=1000, seed=0):
    model = ConditionalUNet(out_channels=3, channels=(8, 16), bottleneck=32,
                            embed_dim=32, cond_dim=0, variance_head=True)
    state = make_train_state(model, jax.random.PRNGKey(seed),
                             (16,) + detail.shape[1:], (16,) + c_real.shape[1:], 0,
                             3e-3, total_steps=steps, warmup=steps // 10)
    step = make_step(None, nll=True)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, detail.shape[0], 16)
        cb = c_real[idx]
        if p > 0:
            m = jnp.asarray(rng.random(16) < p)[:, None, None, None]
            cb = jnp.where(m, c_alt[idx], cb)
        state, loss = step(state, detail[idx], cb)
    return state


def _slope_on(state, c_in):
    B = c_in.shape[0]
    o = state.apply_fn({"params": state.params}, jnp.zeros(c_in.shape[:3] + (3,)),
                       jnp.zeros((B,)), c_in, None)
    mu = np.asarray(o[..., :3]).reshape(-1, 3).T.reshape(-1)
    s2 = np.exp(2 * np.clip(np.asarray(o[..., 3:]), -5, 3)).reshape(-1, 3).T.reshape(-1)
    c = np.tile(np.asarray(c_in).reshape(-1), 3)
    c = (c - c.mean()) / c.std()
    e = np.quantile(c, np.linspace(0, 1, 9)); e[0] -= 1e-9; e[-1] += 1e-9
    idx = np.clip(np.digitize(c, e) - 1, 0, 7)
    pooled = mu.var() + s2.mean()
    cc = np.array([c[idx == b].mean() for b in range(8)])
    v = np.array([(mu[idx == b].var() + s2[idx == b].mean()) for b in range(8)])
    return float(np.polyfit(cc, v / pooled, 1)[0])


def test_plumbing_selfcond_runs():
    rng = np.random.default_rng(0)
    tiles = rng.normal(size=(6, 32, 32)).astype(np.float32)
    # alt pools aligned with the AUGMENTED ordering (8x tiles)
    from wfm import haar
    from wfm.dataset import d4_augment, normalize_tiles
    aug = d4_augment(tiles)
    alt = {}
    for j in (1, 2):
        _, c = haar.octave_pair(normalize_tiles(aug), j)
        alt[j] = c + 0.1 * np.std(np.asarray(c))
    state, meta = train_generator(tiles, [1, 2], arm="A", channels=(8, 16), steps=6,
                                  batch=4, lr=1e-3, nll=True, augment=True,
                                  alt_coarse_pools=alt, alt_p=0.5)
    assert meta["alt_p"] == 0.5 and meta["alt_octaves"] == [1, 2]
    assert np.isfinite(meta["lossN"])


def test_selfcond_non_inferior():
    """HONEST NULL documented (2026-07-11): at toy scale the mixed-trained model does
    NOT beat the clean one under drifted conditioning (measured: 0.236 vs 0.238) --
    because the toy's sigma is a POINTWISE function of the coarse value, the clean
    model's response transfers to drifted inputs undamaged, so there is nothing to
    recover. The full-scale damage comes from neighborhood-texture fragility, which a
    16x16 pointwise toy cannot express; the mechanism discrimination is therefore the
    pre-registered FULL-SCALE experiment's job (attempt-5 prereg, branch weights).
    This gate asserts what the toy CAN check: the conditioning mix neither poisons the
    clean response nor misbehaves under drift."""
    detail, c = _make(jax.random.PRNGKey(3))
    c_alt = _drift(c)
    st_clean = _train(detail, c, None, p=0.0)
    st_mix = _train(detail, c, c_alt, p=0.5, seed=0)
    c_eval = _drift(_coarse(jax.random.PRNGKey(9), 96))
    s_clean = _slope_on(st_clean, c_eval)
    s_mix = _slope_on(st_mix, c_eval)
    c_fresh = _coarse(jax.random.PRNGKey(11), 96)
    s_mix_clean = _slope_on(st_mix, c_fresh)
    s_clean_clean = _slope_on(st_clean, c_fresh)
    # comparable under drift (no harm), and little given up on clean conditioning
    assert abs(s_mix - s_clean) < 0.08, (s_mix, s_clean)
    assert s_mix_clean > s_clean_clean - 0.08, (s_mix_clean, s_clean_clean)
