#!/usr/bin/env python
"""Arm D post-training inference pass (JAX; runs after the four training jobs).

Per data arm (n1s1/n8/n32/cens), model arm A:
  1. val e2e at 2500-stride checkpoints (raw mode) — the T_coef_val curve for
     the joint-window indicator and the PL-SELECTION-style watch;
  2. test e2e at the caged pick in F2 GROUP-AVERAGED mode — the production
     texture (descriptive; the raw-mode test e2e at the pick is already in
     the training job's npz).
Outputs: results_p2/pcure_curves_gen.npz.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

try:
    os.sched_setaffinity(0, set(range(16)))
except Exception:
    pass
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

import jax

from arms_p2.c1t.flow import sample_tbase
from arms_p2.c1t.train import generate_recursive_tbase
from f2_group import D4_ELEMENTS, assemble_group_assigned
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")
ARMS = {
    "n1s1": os.path.join(REPO, "data_cache", "tiles_sandbox.npz"),
    "n8": os.path.join(SCRATCH, "tiles_placement_n8.npz"),
    "n32": os.path.join(SCRATCH, "tiles_placement_n32.npz"),
    "cens": os.path.join(SCRATCH, "tiles_placement_cens.npz"),
}
STEPS = list(range(2500, 20001, 2500))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def std_from(train):
    _, std_by_j = field_to_octaves(d4_augment(train), [2, 3, 4])
    js = np.array(sorted(std_by_j))
    a_, b_ = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
    std = dict(std_by_j)
    std[1] = float(np.exp(a_ * 1 + b_))
    return std


model = ConditionalUNet(out_channels=3, channels=(32, 64, 128), bottleneck=256,
                        cond_dim=0, cond_mode="film", variance_head=False)
out = {}
for arm, data in ARMS.items():
    tiles = np.load(data)["sandbox"].astype(np.float32)
    train, val = tiles[:-64], tiles[-64:-32]
    test = tiles[-32:]
    std = std_from(train)
    vpools, _ = field_to_octaves(val, [4])
    vcoarse = vpools[4][1]
    tpools, _ = field_to_octaves(test, [4])
    tcoarse = tpools[4][1]
    sel = json.load(open(os.path.join(RES, f"pcure_selection_{arm}.json")))
    pick = sel["A"]["selected_step"]
    ckdir = os.path.join(REPO, "data_cache", f"ckpt_pcure_{arm}")

    for si in sorted(set(STEPS + [pick])):
        with open(os.path.join(ckdir, f"armA_sandbox_s{si}.pkl"), "rb") as fh:
            params = pickle.load(fh)["params"]
        gen = generate_recursive_tbase(model.apply, params, vcoarse, 4,
                                       jax.random.PRNGKey(3000 + si), std,
                                       n_steps=80)
        out[f"{arm}_val_s{si}"] = np.asarray(gen[..., 0])
    log(f"{arm}: val curve done (pick={pick})")

    with open(os.path.join(ckdir, f"armA_sandbox_s{pick}.pkl"), "rb") as fh:
        params = pickle.load(fh)["params"]
    grng = np.random.default_rng(20260730)
    coarse = tcoarse
    key = jax.random.PRNGKey(3100)
    for j in range(4, 0, -1):
        key, k = jax.random.split(key)

        def model_fn(c_g, _k=k, _s=std[j]):
            return sample_tbase(model.apply, params, _k, c_g, 3, n_steps=80,
                                cond_vec=None) * _s

        assign = grng.integers(0, len(D4_ELEMENTS), coarse.shape[0])
        coarse = assemble_group_assigned(coarse, model_fn, assign)
    out[f"{arm}_test_f2_s{pick}"] = np.asarray(coarse[..., 0])
    log(f"{arm}: F2-mode test e2e at pick {pick} done")

np.savez(os.path.join(RES, "pcure_curves_gen.npz"), **out)
log("pcure curves SAMPLE done -> pcure_curves_gen.npz")
