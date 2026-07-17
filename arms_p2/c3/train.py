"""C3 trainer: the C1 configuration with the objective+sampler package swapped.

Mirrors wfm.train.train_generator exactly (D4 field-level augmentation, per-octave
std standardization via field_to_octaves, octave cycling, fixed per-octave cond
vector, checkpoint hooks, Adam + warmup-cosine) with ONE change — the single
pre-registered variable: conditional flow matching + ODE pushforward is replaced by
the patched energy score + direct noise-conditioned sampling. No NLL head, no
dispersion penalty, no corruption/self-conditioning knobs (all retired or out of
scope for this arm).
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from wfm.cfm import make_train_state
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet

from .energy import M_SAMPLES, PATCH, STRIDE, patched_energy_score


def make_es_step(cond_vec=None, m=M_SAMPLES, patch=PATCH, stride=STRIDE):
    """Jitted ES training step: m model samples per conditioning, fair estimator.

    The m forwards are vmapped over a fresh (m, B, H, W, C) noise block; gradients
    flow through every sample (both ES terms are functions of the model).
    """
    @jax.jit
    def step(state, detail, coarse):
        key, new_key = jax.random.split(state.key)
        B = detail.shape[0]
        t0 = jnp.zeros((B,))

        def loss_fn(p):
            z = jax.random.normal(key, (m,) + detail.shape)
            samples = jax.vmap(
                lambda zi: state.apply_fn({"params": p}, zi, t0, coarse, cond_vec))(z)
            return patched_energy_score(samples, detail, patch=patch, stride=stride)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads).replace(key=new_key)
        return state, loss
    return step


def train_c3_generator(tiles, train_octaves, arm="A", cond_by_octave=None,
                       channels=(32, 64, 128), steps=20000, batch=32, lr=1e-3,
                       seed=0, cond_mode="film", ckpt_steps=(), on_checkpoint=None,
                       augment=True, m=M_SAMPLES, patch=PATCH, stride=STRIDE):
    """Train the shared direct sampler on many tiles across ``train_octaves``.

    Returns ``(state, meta)`` with the same meta contract as train_generator (plus
    the objective fields), so the run/score scripts read it identically.
    """
    if augment:
        tiles = d4_augment(tiles)
    pools, std_by_j = field_to_octaves(tiles, train_octaves)
    if arm == "A":
        cond_dim = 0
    else:
        assert cond_by_octave is not None, "arm B needs cond_by_octave"
        cond_dim = len(np.atleast_1d(cond_by_octave[train_octaves[0]]))

    model = ConditionalUNet(out_channels=3, channels=tuple(channels),
                            bottleneck=channels[-1] * 2, cond_dim=cond_dim,
                            cond_mode=cond_mode, variance_head=False)
    key = jax.random.PRNGKey(seed)
    k_init, _ = jax.random.split(key)
    j0 = min(train_octaves)
    d0, c0 = pools[j0]
    state = make_train_state(model, k_init, (batch,) + d0.shape[1:],
                             (batch,) + c0.shape[1:], cond_dim, lr,
                             total_steps=steps, warmup=max(1, steps // 10))

    step_fn = {}
    for j in train_octaves:
        cv = None if arm == "A" else jnp.broadcast_to(
            jnp.asarray(cond_by_octave[j], jnp.float32), (batch, cond_dim))
        step_fn[j] = make_es_step(cv, m=m, patch=patch, stride=stride)

    rng = np.random.default_rng(seed)
    ckpt_set = set(ckpt_steps)
    loss0 = None
    for i in range(steps):
        j = train_octaves[i % len(train_octaves)]
        detail, coarse = pools[j]
        idx = rng.integers(0, detail.shape[0], batch)
        state, loss = step_fn[j](state, detail[idx], coarse[idx])
        if i == 0:
            loss0 = float(loss)
        if on_checkpoint is not None and (i + 1) in ckpt_set:
            on_checkpoint(i + 1, state, float(loss))
    meta = {"std_by_j": std_by_j, "train_octaves": list(train_octaves), "arm": arm,
            "cond_by_octave": cond_by_octave, "cond_dim": cond_dim,
            "cond_mode": cond_mode, "augment": augment,
            "objective": "patched_energy_score_beta1",
            "m_samples": m, "patch": patch, "stride": stride,
            "loss0": loss0, "lossN": float(loss)}
    return state, meta
