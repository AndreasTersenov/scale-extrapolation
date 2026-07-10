"""Training drivers for the wavelet flow-matching generator.

``overfit_octave`` is the rung-(i) memorization driver (also the executable gate). Higher
rungs (recursion, full arms A/B) build on the same conditional CFM pieces in ``wfm.cfm``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import haar
from .cfm import make_step, make_train_state, sample
from .generate import generate_recursive
from .model import ConditionalUNet


def relative_l2(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def overfit_octave(field, j=2, channels=(48, 96), steps=2000, lr=2e-3,
                   sample_steps=80, seed=0):
    """Overfit one field's octave-j conditional p(detail_j | coarse_j).

    Returns ``(rel_error, info)`` where rel_error is the sampled-vs-true detail relative
    L2 (standardized space) and info carries loss endpoints and shapes. ``field`` is a 2-D
    array; it is normalized to unit variance before decomposition.
    """
    key = jax.random.PRNGKey(seed)
    f = (field - field.mean()) / field.std()
    f = jnp.asarray(f)[None, :, :, None]
    detail, coarse = haar.octave_pair(f, j)
    detail_n = (detail - detail.mean()) / detail.std()

    model = ConditionalUNet(out_channels=3, channels=tuple(channels),
                            bottleneck=channels[-1] * 2, cond_dim=0)
    k_init, k_sample = jax.random.split(key)
    state = make_train_state(model, k_init, detail_n.shape, coarse.shape, 0, lr,
                             total_steps=steps, warmup=max(1, steps // 10))
    step = make_step(None)
    loss0 = lossN = None
    for i in range(steps):
        state, loss = step(state, detail_n, coarse)
        if i == 0:
            loss0 = float(loss)
    lossN = float(loss)

    gen = sample(state.apply_fn, state.params, k_sample, coarse, 3,
                 n_steps=sample_steps, cond_vec=None, solver="heun")
    return relative_l2(gen, detail_n), {
        "loss0": loss0, "lossN": lossN, "steps": steps,
        "detail_shape": tuple(detail_n.shape), "state": state,
    }


def overfit_field_recursive(field, j_max=2, channels=(48, 96), steps=2500, lr=2e-3,
                            sample_steps=80, seed=0):
    """Rung (ii): overfit octaves 1..j_max of one field with a SHARED (weight-tied) model,
    then generate the full field coarse-to-fine from its true coarsest coarse.

    Returns ``(field_rel_error, info)`` where field_rel_error is the recursively-generated
    vs true field relative L2 (unit-variance space). Also generates twice under a fixed key
    and records exact determinism.
    """
    key = jax.random.PRNGKey(seed)
    f = (field - field.mean()) / field.std()
    f = jnp.asarray(f)[None, :, :, None]

    octs = list(range(1, j_max + 1))
    # standardize each octave's detail by its std ONLY (Haar detail mean ~ 0), so the
    # recursion recovers physical detail exactly by multiplying the sample back by std.
    detail_n, coarse, det_std_by_j = {}, {}, {}
    for j in octs:
        d, c = haar.octave_pair(f, j)
        s = float(d.std())
        detail_n[j] = d / s
        coarse[j] = c
        det_std_by_j[j] = s

    model = ConditionalUNet(out_channels=3, channels=tuple(channels),
                            bottleneck=channels[-1] * 2, cond_dim=0)
    k_init, k_gen = jax.random.split(key)
    # init on the finest octave shape; the conv net is shape-agnostic across octaves
    state = make_train_state(model, k_init, detail_n[1].shape, coarse[1].shape, 0, lr,
                             total_steps=steps, warmup=max(1, steps // 10))
    step = make_step(None)
    loss0 = None
    for i in range(steps):
        j = octs[i % len(octs)]
        state, loss = step(state, detail_n[j], coarse[j])
        if i == 0:
            loss0 = float(loss)
    lossN = float(loss)

    coarse_start = coarse[j_max] * 1.0                    # true coarsest coarse
    gen_field = generate_recursive(state.apply_fn, state.params, coarse_start, j_max,
                                   k_gen, det_std_by_j, cond_fn=None, n_steps=sample_steps)
    gen_field2 = generate_recursive(state.apply_fn, state.params, coarse_start, j_max,
                                    k_gen, det_std_by_j, cond_fn=None, n_steps=sample_steps)
    return relative_l2(gen_field, f), {
        "loss0": loss0, "lossN": lossN, "steps": steps, "octaves": octs,
        "field_shape": tuple(f.shape),
        "deterministic": bool(jnp.array_equal(gen_field, gen_field2)),
    }
