"""Training drivers for the wavelet flow-matching generator.

``overfit_octave`` is the rung-(i) memorization driver (also the executable gate). Higher
rungs (recursion, full arms A/B) build on the same conditional CFM pieces in ``wfm.cfm``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import haar
from .cfm import make_step, make_train_state, sample
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
