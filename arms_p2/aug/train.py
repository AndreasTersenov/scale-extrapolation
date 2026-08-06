"""AUG trainer: the C1-t recipe with multi-group (stack, octave) slot
cycling (prereg 2026-08-05-night3 §AUG). Mirrors
arms_p2/c1t/train.train_c1t_generator — D4 field-level augmentation,
per-(group, octave) std standardization via field_to_octaves, round-robin
slot cycling, checkpoint hooks, Adam + warmup-cosine, t(nu)-base CFM
step. Arm A only (no cond vector). meta std_by_j = the ORIGINAL group's
stds (group 0) so run/score scripts read it identically; aug-pool stds
recorded separately."""
from __future__ import annotations

import numpy as np

import jax

from wfm.cfm import make_train_state
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet

from arms_p2.c1t.flow import NU, make_tcfm_step


def train_aug_generator(groups, channels=(32, 64, 128), steps=48000,
                        batch=32, lr=1e-3, seed=0, cond_mode="film",
                        ckpt_steps=(), on_checkpoint=None, augment=True,
                        nu=NU):
    """groups: [(tiles, octave_list), ...]; group 0 = originals (its
    std_by_j is the meta contract). Returns (state, meta)."""
    pools, std_rec = {}, {}
    std_by_j0 = None
    for gi, (tiles, octs) in enumerate(groups):
        t = d4_augment(tiles) if augment else tiles
        p, s = field_to_octaves(t, octs)
        for j in octs:
            pools[(gi, j)] = p[j]
        std_rec[gi] = {int(j): float(s[j]) for j in octs}
        if gi == 0:
            std_by_j0 = dict(s)
    slots = sorted(pools)
    model = ConditionalUNet(out_channels=3, channels=tuple(channels),
                            bottleneck=channels[-1] * 2, cond_dim=0,
                            cond_mode=cond_mode, variance_head=False)
    key = jax.random.PRNGKey(seed)
    k_init, _ = jax.random.split(key)
    j0 = min(j for (gi, j) in slots if gi == 0)
    d0, c0 = pools[(0, j0)]
    state = make_train_state(model, k_init, (batch,) + d0.shape[1:],
                             (batch,) + c0.shape[1:], 0, lr,
                             total_steps=steps,
                             warmup=max(1, steps // 10))
    step_fn = make_tcfm_step(None, nu=nu)  # retraces per input shape

    rng = np.random.default_rng(seed)
    ckpt_set = set(ckpt_steps)
    visits = {str(s): 0 for s in slots}
    loss0 = None
    for i in range(steps):
        slot = slots[i % len(slots)]
        detail, coarse = pools[slot]
        idx = rng.integers(0, detail.shape[0], batch)
        state, loss = step_fn(state, detail[idx], coarse[idx])
        visits[str(slot)] += 1
        if i == 0:
            loss0 = float(loss)
        if on_checkpoint is not None and (i + 1) in ckpt_set:
            on_checkpoint(i + 1, state, float(loss))
    meta = {"std_by_j": std_by_j0,
            "aug_std": std_rec,
            "slots": [list(s) for s in slots],
            "slot_visits": visits,
            "train_octaves": list(groups[0][1]), "arm": "A",
            "cond_by_octave": None, "cond_dim": 0, "cond_mode": cond_mode,
            "augment": augment, "objective": "cfm_tbase_aug", "nu": nu,
            "loss0": loss0, "lossN": float(loss)}
    return state, meta
