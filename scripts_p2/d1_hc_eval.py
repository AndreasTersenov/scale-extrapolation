#!/usr/bin/env python
"""D1 head-conditional eval at a corrected-pick checkpoint (JAX). Mirrors
run_c1t_arms.py's TEST "final" block: K=1 head-conditional sample per
test field at octaves 1-4, pooled (w_gen, c_real), full couplings scored.
Writes results_p2/d1_<tag>_hc.json {"1":{...},...,"4":{...}} so the D1
battery reads hc-marginals at the CORRECTED pick (the bugged selection
JSONs' 'final' block was at the wrong checkpoint).
Usage: d1_hc_eval.py <tag> <ckpt_path> <train_oct_lo>."""
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
from sandbox.truth_stats import estimand_scalars, tail_q999
from wfm import haar
from wfm.dataset import normalize_tiles
from wfm.model import ConditionalUNet


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def pooled_wc(det, coarse):
    det = np.asarray(det)
    coarse = np.asarray(coarse)[..., 0]
    pf = []
    for i in range(det.shape[0]):
        w = np.concatenate([det[i, :, :, k].reshape(-1) for k in range(3)])
        c = np.tile(coarse[i].reshape(-1), 3)
        pf.append((w, c))
    return pf


def score_full(pf, n_boot=200, seed=0):
    w = np.concatenate([p[0] for p in pf])
    c = np.concatenate([p[1] for p in pf])
    s = estimand_scalars(w, c)
    rng = np.random.default_rng(seed)
    n = len(pf)
    boot = np.empty((n_boot, 3))
    for t in range(n_boot):
        idx = rng.integers(0, n, n)
        wb = np.concatenate([pf[i][0] for i in idx])
        cb = np.concatenate([pf[i][1] for i in idx])
        sb = estimand_scalars(wb, cb)
        boot[t] = (sb["var_slope"], sb["kurtosis"], tail_q999(wb))
    return {"var_slope": s["var_slope"],
            "var_slope_se": float(np.nanstd(boot[:, 0], ddof=1)),
            "kurtosis": s["kurtosis"],
            "kurtosis_se": float(np.nanstd(boot[:, 1], ddof=1)),
            "q999": tail_q999(w), "q999_se": float(np.nanstd(boot[:, 2], ddof=1))}


tag, ckpt_path, lo = sys.argv[1], sys.argv[2], int(sys.argv[3])
gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
test = gtiles[-32:]
test_n = normalize_tiles(test)
with open(ckpt_path, "rb") as fh:
    params = pickle.load(fh)["params"]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
key = jax.random.PRNGKey(20261450)
out = {}
for j in (1, 2, 3, 4):
    key, k = jax.random.split(key)
    det_real, coarse = haar.octave_pair(test_n, j)
    det = sample_tbase(model.apply, params, k, coarse, 3, n_steps=80,
                       cond_vec=None)
    out[str(j)] = score_full(pooled_wc(det, coarse), n_boot=200, seed=0)
    log(f"{tag} hc oct{j}: vs={out[str(j)]['var_slope']:.3f} "
        f"kurt={out[str(j)]['kurtosis']:.2f}")
with open(os.path.join(REPO, "results_p2", f"d1_{tag}_hc.json"), "w") as f:
    json.dump({"final": out, "ckpt": ckpt_path, "lo": lo}, f, indent=1)
log(f"wrote d1_{tag}_hc.json")
