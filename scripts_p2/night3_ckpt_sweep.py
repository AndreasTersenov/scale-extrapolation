#!/usr/bin/env python
"""NIGHT-3 CKPT-SWEEP probe (JAX; prereg 2026-08-05-night3, grant-2
adaptive probing; descriptive — adjudicates NO branch). e2e recursions
(plain generate_recursive_tbase, the E3-comparable convention) from dense
checkpoints of three committed dirs; separates the cage-selection /
capacity / dilution readings of O-NULL. Scoring is CPU-side
(night3_ckpt_score.py) — this script only renders maps."""
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

from arms_p2.c1t.train import generate_recursive_tbase
from l1p_lib import std_from
from wfm.dataset import field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
STEPS = list(range(2500, 20001, 2500))
DIRS = {
    "oracle": ("ckpt_oct1fix_oracle", [1, 2, 3, 4]),
    "prod": ("ckpt_c1t_gowerstreet", [2, 3, 4]),
    "blind": ("ckpt_stage3_blind", [3, 4]),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
coarse4 = field_to_octaves(gtest, [4])[0][4][1]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)

out, manifest = {}, {"steps": STEPS, "keys": {}}
t0 = time.time()
for di, (tag, (dirname, octs)) in enumerate(DIRS.items()):
    std = std_from(gtrain, octs)
    for step in STEPS:
        path = os.path.join(REPO, "data_cache", dirname,
                            f"armA_gowerstreet_s{step}.pkl")
        with open(path, "rb") as fh:
            params = pickle.load(fh)["params"]
        ki = 982000 + di * 100 + step // 500
        gen = generate_recursive_tbase(model.apply, params, coarse4, 4,
                                       jax.random.PRNGKey(ki), std)
        out[f"{tag}_{step}"] = np.asarray(gen[..., 0], np.float32)
        manifest["keys"][f"{tag}_{step}"] = ki
        log(f"{tag} s{step}: done [{time.time()-t0:.0f}s]")
np.savez(os.path.join(RES, "night3_ckpt_sweep.npz"), **out)
with open(os.path.join(RES, "night3_ckpt_sweep_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
log(f"wrote night3_ckpt_sweep.npz ({len(out)} stacks)")
