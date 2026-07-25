#!/usr/bin/env python
"""F2 disambiguation pass (R34 order 1; zero training, inference only).

The F2 branch decider was oct-2 meanD z = +3.049 vs the 3.00 bar at 32 test
fields. Rescore on the 64-field replication stream (seed 20260720, disjoint):
F2 group-averaged e2e, sandbox arm A at the committed pick 7500, conditioned
on the 64 fresh tiles' coarse. Reading rule (R34, pre-stated): |z_meanD-oct2|
>= 3 at 64 fields -> real residual (group-action subtlety hunt); < 3 ->
consistent-with-null, recorded, NO retro-upgrade of the branch.

Output: results_p2/f2_disambig_gen.npz (SCORE phase under env.sh follows).
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
import jax.numpy as jnp

from arms_p2.c1t.flow import sample_tbase
from f2_group import D4_ELEMENTS, assemble_group_assigned
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
SEED_G = 20260729


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
train = tiles[:-64].astype(np.float32)
fresh = np.load(os.path.join(REPO, "data_cache",
                             "tiles_sandbox_repl.npz"))["sandbox"].astype(np.float32)

_, std_by_j = field_to_octaves(d4_augment(train), [2, 3, 4])
js = np.array(sorted(std_by_j))
a_, b_ = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
std = dict(std_by_j)
std[1] = float(np.exp(a_ * 1 + b_))

pools, _ = field_to_octaves(fresh, [4])
coarse = pools[4][1]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128), bottleneck=256,
                        cond_dim=0, cond_mode="film", variance_head=False)
with open(os.path.join(REPO, "data_cache", "ckpt_c1t_sandbox",
                       "armA_sandbox_s7500.pkl"), "rb") as fh:
    params = pickle.load(fh)["params"]

grng = np.random.default_rng(SEED_G)
key = jax.random.PRNGKey(2600)
for j in range(4, 0, -1):
    key, k = jax.random.split(key)

    def model_fn(c_g, _k=k, _s=std[j]):
        return sample_tbase(model.apply, params, _k, c_g, 3, n_steps=80,
                            cond_vec=None) * _s

    assign = grng.integers(0, len(D4_ELEMENTS), coarse.shape[0])
    coarse = assemble_group_assigned(coarse, model_fn, assign)
    log(f"octave {j} assembled")

np.savez(os.path.join(RES, "f2_disambig_gen.npz"),
         gen_A=np.asarray(coarse[..., 0]))
log("disambig SAMPLE done -> f2_disambig_gen.npz")
