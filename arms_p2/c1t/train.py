"""C1-t trainer: the C1 recipe verbatim with the t(nu)-base CFM step.

Mirrors arms_p2/c3/train.train_c3_generator (which mirrors wfm.train.
train_generator): D4 field-level augmentation, per-octave std standardization via
field_to_octaves, octave cycling, fixed per-octave cond vector (arm B), checkpoint
hooks, Adam + warmup-cosine. The ONE change vs C1 is the base distribution of the
flow (R17: arm = architecture + t-base + pre-registered checkpoint selection).
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from wfm.cfm import make_train_state
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet

from .flow import NU, make_tcfm_step, sample_tbase


def train_c1t_generator(tiles, train_octaves, arm="A", cond_by_octave=None,
                        channels=(32, 64, 128), steps=20000, batch=32, lr=1e-3,
                        seed=0, cond_mode="film", ckpt_steps=(), on_checkpoint=None,
                        augment=True, nu=NU):
    """Train the shared t-base conditional flow on many tiles across octaves.

    Returns ``(state, meta)`` with the train_generator meta contract (std_by_j,
    cond_dim, ...) so run/score scripts read it identically.
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
        step_fn[j] = make_tcfm_step(cv, nu=nu)

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
            "objective": "cfm_tbase", "nu": nu,
            "loss0": loss0, "lossN": float(loss)}
    return state, meta


def generate_recursive_tbase(apply_fn, params, coarse, j_start, key, detail_std,
                             cond_fn=None, n_steps=80, nu=NU):
    """Coarse-to-fine recursion with the t-base heun ODE sampler at every octave."""
    from wfm import haar
    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)
        cond = None if cond_fn is None else cond_fn(j)
        det_n = sample_tbase(apply_fn, params, k, coarse, 3, n_steps=n_steps,
                             cond_vec=cond, nu=nu)
        s = detail_std(j) if callable(detail_std) else detail_std[j]
        det = det_n * s
        coarse = haar.idwt2(coarse, (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
    return coarse
