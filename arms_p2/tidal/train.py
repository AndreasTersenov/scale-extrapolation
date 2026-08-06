"""TIDAL trainer: the C1-t recipe verbatim with eigenframe conditioning
(prereg log/2026-08-05-prereg-night3.md "TIDAL — design"; trains iff GATE-T).

Mirrors arms_p2/c1t/train.train_c1t_generator: D4 field-level augmentation,
per-octave std standardization via field_to_octaves, octave cycling, t-base
CFM step, checkpoint hooks, Adam + warmup-cosine. The ONE change is the
conditioning field: H(c) = (tr, a1, a2) of the Gaussian-smoothed Hessian of
the coarse (features.py), standardized PER POOL per channel, CONCATENATED to
the coarse channel — the model sees a 4-channel conditioning field.
ConditionalUNet is unchanged (nn.Conv infers input channels; only the init
dummy widens). The standardization constants are stored in meta as
feat_std_by_j and MUST be carried to sampling (arms_p2/tidal/sample.py).
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from wfm.cfm import make_train_state
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet

from arms_p2.c1t.flow import NU, make_tcfm_step, sample_tbase

from .features import SIGMA_H, hessian_features


def tidal_pools(tiles, train_octaves, augment=True, sigma_h=SIGMA_H):
    """Per-octave (detail_n, cond4) pools + std_by_j + feat_std_by_j.

    The ONE place the training-side feature convention lives: features are
    computed on the (augmented) pool coarse, standardized by the per-channel
    pool std, and concatenated as cond4 = [coarse, features_std]. Sampling
    must reuse feat_std_by_j verbatim (carried in meta and in every ckpt).
    """
    if augment:
        tiles = d4_augment(tiles)
    pools, std_by_j = field_to_octaves(tiles, train_octaves)
    cond_pools, feat_std_by_j = {}, {}
    for j in train_octaves:
        detail_n, coarse = pools[j]
        feats = hessian_features(coarse, sigma=sigma_h)
        fstd = np.asarray(feats).std(axis=(0, 1, 2)).astype(np.float32)
        feat_std_by_j[j] = fstd
        cond_pools[j] = (detail_n,
                         jnp.concatenate([coarse, feats / fstd], axis=-1))
    return cond_pools, std_by_j, feat_std_by_j


def tidal_feat_std(tiles, train_octaves, augment=True, sigma_h=SIGMA_H):
    """feat_std_by_j alone (run scripts embed it in every ckpt pickle)."""
    _, _, feat_std_by_j = tidal_pools(tiles, train_octaves, augment, sigma_h)
    return feat_std_by_j


def train_tidal_generator(tiles, train_octaves, arm="A", cond_by_octave=None,
                          channels=(32, 64, 128), steps=20000, batch=32, lr=1e-3,
                          seed=0, cond_mode="film", ckpt_steps=(), on_checkpoint=None,
                          augment=True, nu=NU, sigma_h=SIGMA_H):
    """Train the shared t-base conditional flow on 4-channel (coarse, H) cond.

    Returns ``(state, meta)`` with the train_generator meta contract
    (std_by_j, cond_dim, ...) plus feat_std_by_j and sigma_h.
    """
    pools, std_by_j, feat_std_by_j = tidal_pools(tiles, train_octaves,
                                                 augment, sigma_h)
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
        detail, cond = pools[j]
        idx = rng.integers(0, detail.shape[0], batch)
        state, loss = step_fn[j](state, detail[idx], cond[idx])
        if i == 0:
            loss0 = float(loss)
        if on_checkpoint is not None and (i + 1) in ckpt_set:
            on_checkpoint(i + 1, state, float(loss))
    meta = {"std_by_j": std_by_j, "train_octaves": list(train_octaves), "arm": arm,
            "cond_by_octave": cond_by_octave, "cond_dim": cond_dim,
            "cond_mode": cond_mode, "augment": augment,
            "objective": "cfm_tbase_tidal", "nu": nu, "sigma_h": sigma_h,
            "feat_std_by_j": feat_std_by_j,
            "loss0": loss0, "lossN": float(loss)}
    return state, meta


def generate_recursive_tidal(apply_fn, params, coarse, j_start, key, detail_std,
                             feat_std, cond_fn=None, n_steps=80, nu=NU,
                             sigma_h=SIGMA_H):
    """generate_recursive_tbase with H(coarse) concatenated at every octave.

    feat_std: {octave: (3,) per-channel std} (callable also accepted, like
    detail_std) — the training constants, extrapolated below the trained
    range by sample.feat_std_ladder."""
    from wfm import haar
    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)
        cond = None if cond_fn is None else cond_fn(j)
        fs = feat_std(j) if callable(feat_std) else feat_std[j]
        feats = hessian_features(coarse, sigma=sigma_h) / jnp.asarray(fs)
        cond4 = jnp.concatenate([coarse, feats], axis=-1)
        det_n = sample_tbase(apply_fn, params, k, cond4, 3, n_steps=n_steps,
                             cond_vec=cond, nu=nu)
        s = detail_std(j) if callable(detail_std) else detail_std[j]
        det = det_n * s
        coarse = haar.idwt2(coarse, (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
    return coarse
