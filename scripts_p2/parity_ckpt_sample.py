#!/usr/bin/env python
"""Phase A' M5, SAMPLE phase (R31 order 1: "across the checkpoint curve").

Inference only — zero training. For sandbox arms A and B, e2e generations from
the committed dense checkpoints (data_cache/ckpt_c1t_sandbox, params only) on
the frozen 32 test tiles, identical protocol to run_c1t_arms' e2e block
(std_by_j recomputed deterministically from the training tiles, the
c1t_replication.py pattern). Steps: every 2500 from 2500 to 20000 (8 ckpts;
the committed picks A@7500 / B@2500 lie on/near this grid; B@2500 added).

Output: results_p2/parity_ckpt_gen.npz {armA_s2500: (32,128,128), ...}.
SCORE phase (env.sh stack, pywt) runs separately — the env split stands.
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

from arms_p2.c1t.train import generate_recursive_tbase
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet

COORD_NORM = np.array([1.5, 13.0])
STEPS = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
EXTRA = {"B": [2500]}          # B's committed pick is on the grid already


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
train, test = tiles[:-64].astype(np.float32), tiles[-32:].astype(np.float32)
coords_raw = json.load(open(os.path.join(
    REPO, "data_cache", "running_couplings_sandbox.json")))["sandbox"]
coords = {int(j): (np.asarray(v) / COORD_NORM) for j, v in coords_raw.items()}

_, std_by_j = field_to_octaves(d4_augment(train), [2, 3, 4])
js = np.array(sorted(std_by_j))
a_, b_ = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
std = dict(std_by_j)
std[1] = float(np.exp(a_ * 1 + b_))

pools, _ = field_to_octaves(test, [4])
coarse4 = pools[4][1]
B = coarse4.shape[0]
log(f"test coarse {coarse4.shape}; std_by_j {std}")

out = {}
for arm in ("A", "B"):
    cond_dim = 0 if arm == "A" else 2
    model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                            bottleneck=256, cond_dim=cond_dim, cond_mode="film",
                            variance_head=False)
    cond_fn = None
    if arm == "B":
        import jax.numpy as jnp

        def cond_fn(j):
            return jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32), (B, 2))
    for si in sorted(set(STEPS + EXTRA.get(arm, []))):
        t0 = time.time()
        with open(os.path.join(REPO, "data_cache", "ckpt_c1t_sandbox",
                               f"arm{arm}_sandbox_s{si}.pkl"), "rb") as fh:
            params = pickle.load(fh)["params"]
        gen = generate_recursive_tbase(model.apply, params, coarse4, 4,
                                       jax.random.PRNGKey(700 + si), std,
                                       cond_fn=cond_fn, n_steps=80)
        out[f"arm{arm}_s{si}"] = np.asarray(gen[..., 0])
        log(f"arm {arm} @{si}: e2e done [{time.time()-t0:.0f}s]")

np.savez(os.path.join(REPO, "results_p2", "parity_ckpt_gen.npz"), **out)
log("SAMPLE phase done -> results_p2/parity_ckpt_gen.npz; "
    "score with parity_ckpt_score.py under env.sh")
