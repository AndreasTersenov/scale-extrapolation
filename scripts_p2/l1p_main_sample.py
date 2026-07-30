#!/usr/bin/env python
"""L1' MAIN sample (pre-statement 67d7304; runs ONLY after the canary
passed its kill criterion on CPU). Zero training.

Gowerstreet c1t substrate (A@16000): replay gate (white, committed stream
2400/20260728+2, asserted) + 3 ADJUDICATING streams (oct2rescaled filter)
+ 1 labeled ABLATION stream (oct1-measured filter, descriptive only).
Stage-D substrate (A@9000): 1 edge continuity stream (oct2rescaled).

Outputs results_p2/l1p_main_gen.npz + l1p_main_sample.json.
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

from l1p_lib import gen_groupavg_base, make_colored_base, std_from
from wfm.dataset import field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
G2 = {"corr_min": 0.99, "ratio_tol": 5e-3, "maxabs_ceil": 5e-2}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_params(ckpt_dir, name):
    with open(os.path.join(REPO, "data_cache", ckpt_dir, name), "rb") as fh:
        return pickle.load(fh)["params"]


model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
F = np.load(os.path.join(RES, "l1_filter_gowerstreet.npz"))
cb_adj = make_colored_base(np.asarray(F["filt_oct2rescaled"]),
                           np.asarray(F["z_grid"]), np.asarray(F["x_grid"]))
cb_abl = make_colored_base(np.asarray(F["filt_oct1"]),
                           np.asarray(F["z_grid"]), np.asarray(F["x_grid"]))

gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
out_npz, out_json = {}, {"replay_gate": {}, "streams": {
    "adj1": [3401, 20260802], "adj2": [3402, 20260803],
    "adj3": [3403, 20260804], "ablation_oct1meas": [3405, 20260805],
    "edge": [3406, 20260806]}}

# ---- gowerstreet c1t substrate ----------------------------------------------
std = std_from(gtrain, [2, 3, 4])
coarse4 = field_to_octaves(gtest, [4])[0][4][1]
params = load_params("ckpt_c1t_gowerstreet", "armA_gowerstreet_s16000.pkl")

committed = np.asarray(
    np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_gowA_e2e"], np.float64)
rep = gen_groupavg_base(model.apply, params, coarse4, 4,
                        jax.random.PRNGKey(2400), std,
                        np.random.default_rng(20260728 + 2))
maps = np.asarray(rep[..., 0], np.float64)
corr = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1]
                 for a, b in zip(maps, committed)])
ratio = np.array([a.std() / b.std() for a, b in zip(maps, committed)])
rel = float(np.max(np.abs(maps - committed))) / float(committed.std())
gate = {"corr_min": float(corr.min()), "ratio_mean": float(ratio.mean()),
        "rel_maxabs": rel}
out_json["replay_gate"] = gate
log(f"gow replay gate: corr_min={corr.min():.8f} "
    f"ratio_mean={ratio.mean():.6f} rel={rel:.3e}")
assert corr.min() >= G2["corr_min"] and \
    abs(float(ratio.mean()) - 1) <= G2["ratio_tol"] and \
    rel <= G2["maxabs_ceil"], f"gow replay gate FAILED: {gate}"
log("gow replay gate: PASS")

for name, key_i, gseed, cb in (("adj1", 3401, 20260802, cb_adj),
                               ("adj2", 3402, 20260803, cb_adj),
                               ("adj3", 3403, 20260804, cb_adj),
                               ("ablation_oct1meas", 3405, 20260805, cb_abl)):
    gen = gen_groupavg_base(model.apply, params, coarse4, 4,
                            jax.random.PRNGKey(key_i), std,
                            np.random.default_rng(gseed), base_by_j={1: cb})
    out_npz[name] = np.asarray(gen[..., 0], np.float32)
    log(f"gow {name}: done")

# ---- stage-D edge continuity leg --------------------------------------------
std_d = std_from(gtrain, [3, 4])
params_d = load_params("ckpt_stageD", "armA_gowerstreet_s9000.pkl")
gen = gen_groupavg_base(model.apply, params_d, coarse4, 4,
                        jax.random.PRNGKey(3406), std_d,
                        np.random.default_rng(20260806), base_by_j={1: cb_adj})
out_npz["edge"] = np.asarray(gen[..., 0], np.float32)
log("edge continuity stream: done")

np.savez(os.path.join(RES, "l1p_main_gen.npz"), **out_npz)
with open(os.path.join(RES, "l1p_main_sample.json"), "w") as f:
    json.dump(out_json, f, indent=1)
log("L1' main sample done -> l1p_main_gen.npz")
