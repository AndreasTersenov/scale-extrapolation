#!/usr/bin/env python
"""R35 order 7a SAMPLE phase: n32 joint-pick (17500) raw-mode test e2e + hc.
Zero training. Full H100 (the n32 d4_augment memory pattern).
Outputs results_p2/pcure_confirm_gen.npz + results_p2/pcure_confirm_hc.json.
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
from run_c1t_arms import pooled_wc, score_full
from wfm import haar
from wfm.dataset import d4_augment, field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")
JOINT_PICK = json.load(open(os.path.join(RES, "pcure_verdict.json")))[
    "arms"]["n32"]["joint_pick"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tiles = np.load(os.path.join(SCRATCH, "tiles_placement_n32.npz"))["sandbox"]
train = tiles[:-64].astype(np.float32)
shared = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
test = shared[-32:].astype(np.float32)
test_n = normalize_tiles(test)

_, std_by_j = field_to_octaves(d4_augment(train), [2, 3, 4])
js = np.array(sorted(std_by_j))
a_, b_ = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
std = dict(std_by_j)
std[1] = float(np.exp(a_ * 1 + b_))

model = ConditionalUNet(out_channels=3, channels=(32, 64, 128), bottleneck=256,
                        cond_dim=0, cond_mode="film", variance_head=False)
with open(os.path.join(REPO, "data_cache", "ckpt_pcure_n32",
                       f"armA_sandbox_s{JOINT_PICK}.pkl"), "rb") as fh:
    params = pickle.load(fh)["params"]

pools, _ = field_to_octaves(test, [4])
gen = generate_recursive_tbase(model.apply, params, pools[4][1], 4,
                               jax.random.PRNGKey(4000), std, n_steps=80)
key = jax.random.PRNGKey(4100)
hc = {}
for j in (2, 3, 4):
    det_real, coarse = haar.octave_pair(test_n, j)
    key, k = jax.random.split(key)
    det = sample_tbase(model.apply, params, k, coarse, 3, n_steps=80,
                       cond_vec=None)
    hc[str(j)] = score_full(pooled_wc(det, coarse), n_boot=200, seed=0)

np.savez(os.path.join(RES, "pcure_confirm_gen.npz"),
         gen_A=np.asarray(gen[..., 0]))
with open(os.path.join(RES, "pcure_confirm_hc.json"), "w") as f:
    json.dump({"joint_pick": int(JOINT_PICK), "hc": hc}, f, indent=1)
log(f"confirm SAMPLE done (n32 armA @ {JOINT_PICK})")
