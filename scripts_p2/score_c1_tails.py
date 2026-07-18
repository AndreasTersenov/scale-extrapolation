#!/usr/bin/env python
"""R17 attribution at zero cost: score C1's EXISTING sandbox checkpoints for
tails (kurtosis + q999, octave-2 head-conditional) on the same VALIDATION fields
C1-t's selection uses — so base-effect (Gaussian vs t base) and selection-effect
(early stop) separate without any new training. DESCRIPTIVE only.

CPU-friendly (wl-challenge-env); C1 ckpts: data_cache/ckpt_c1_sandbox
(steps 1k,2k,4k,6k,8k,12k,16k + final 20k), Gaussian-base heun-80 sampler.
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

from sandbox.truth_stats import estimand_scalars, tail_q999
from wfm import haar
from wfm.cfm import sample
from wfm.dataset import normalize_tiles
from wfm.model import ConditionalUNet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COORD_NORM = np.array([1.5, 13.0])
CKPT_DIR = os.path.join(REPO, "data_cache", "ckpt_c1_sandbox")
OUT = os.path.join(REPO, "results_p2", "c1_tails_val.json")

tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
heldout = tiles[-64:].astype(np.float32)
val_n = normalize_tiles(heldout[:32])            # C1-t's VALIDATION half
coords_raw = json.load(open(os.path.join(REPO, "data_cache",
                                         "running_couplings_sandbox.json")))["sandbox"]
coords = {int(j): (np.asarray(v) / COORD_NORM) for j, v in coords_raw.items()}

out = {}
for arm in ("A", "B"):
    with open(os.path.join(CKPT_DIR, f"arm{arm}_sandbox.pkl"), "rb") as fh:
        ck = pickle.load(fh)
    model = ConditionalUNet(out_channels=3, channels=tuple(ck["channels"]),
                            bottleneck=ck["channels"][-1] * 2,
                            cond_dim=ck["cond_dim"], cond_mode=ck["cond_mode"],
                            variance_head=False)
    det_real, coarse = haar.octave_pair(val_n, 2)
    cv = None if ck["cond_dim"] == 0 else jnp.broadcast_to(
        jnp.asarray(coords[2], jnp.float32), (coarse.shape[0], ck["cond_dim"]))
    res = {}
    key = jax.random.PRNGKey(300)
    for step_i in (1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000):
        if step_i == 20000:
            params = ck["params"]
        else:
            with open(os.path.join(CKPT_DIR,
                                   f"arm{arm}_sandbox_s{step_i}.pkl"), "rb") as fh:
                params = pickle.load(fh)["params"]
        ws = []
        for k in range(4):                      # K=4, matching the C1-t sel eval
            key, kk = jax.random.split(key)
            det = np.asarray(sample(model.apply, params, kk, coarse, 3,
                                    n_steps=80, cond_vec=cv, solver="heun"))
            for i in range(det.shape[0]):
                ws.append((np.concatenate([det[i, :, :, c_].reshape(-1)
                                           for c_ in range(3)]),
                           np.tile(np.asarray(coarse)[i, :, :, 0].reshape(-1), 3)))
        w = np.concatenate([p[0] for p in ws])
        c = np.concatenate([p[1] for p in ws])
        s = estimand_scalars(w, c)
        res[str(step_i)] = {"var_slope": s["var_slope"],
                            "kurtosis": s["kurtosis"], "q999": tail_q999(w)}
        print(f"[c1-tails] arm {arm} @{step_i}: vs={s['var_slope']:.3f} "
              f"kurt={s['kurtosis']:.2f} q999={tail_q999(w):.2f}", flush=True)
    out[arm] = res

with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(f"[c1-tails] wrote {OUT}", flush=True)
