#!/usr/bin/env python
"""L1'' MAIN sample (prereg 2026-07-30-l1doubleprime; runs ONLY after
CANARY PASS). Zero training. Gowerstreet A@16000: replay gate (asserted) +
3 ADJUDICATING deconvolved streams (filt_adj) + 1 labeled ORACLE ablation
(filt_oracle). Outputs results_p2/l1pp_main_gen.npz + l1pp_main_sample.json."""
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


gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
std = std_from(gtrain, [2, 3, 4])
coarse4 = field_to_octaves(gtest, [4])[0][4][1]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
with open(os.path.join(REPO, "data_cache", "ckpt_c1t_gowerstreet",
                       "armA_gowerstreet_s16000.pkl"), "rb") as fh:
    params = pickle.load(fh)["params"]

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
log(f"gow replay gate: {gate}")
assert corr.min() >= G2["corr_min"] and \
    abs(float(ratio.mean()) - 1) <= G2["ratio_tol"] and \
    rel <= G2["maxabs_ceil"], f"gow replay gate FAILED: {gate}"
log("gow replay gate: PASS")

F = np.load(os.path.join(RES, "l1pp_filter.npz"))
Z, X = np.asarray(F["z_grid"]), np.asarray(F["x_grid"])
cb_adj = make_colored_base(np.asarray(F["filt_adj"]), Z, X)
cb_orc = make_colored_base(np.asarray(F["filt_oracle"]), Z, X)
out = {}
for name, key_i, gseed, cb in (("adj1", 3601, 20260808, cb_adj),
                               ("adj2", 3602, 20260809, cb_adj),
                               ("adj3", 3603, 20260810, cb_adj),
                               ("oracle_oct1meas", 3605, 20260811, cb_orc)):
    gen = gen_groupavg_base(model.apply, params, coarse4, 4,
                            jax.random.PRNGKey(key_i), std,
                            np.random.default_rng(gseed), base_by_j={1: cb})
    out[name] = np.asarray(gen[..., 0], np.float32)
    log(f"gow {name}: done")

np.savez(os.path.join(RES, "l1pp_main_gen.npz"), **out)
with open(os.path.join(RES, "l1pp_main_sample.json"), "w") as f:
    json.dump({"replay_gate": gate, "streams": {
        "adj1": [3601, 20260808], "adj2": [3602, 20260809],
        "adj3": [3603, 20260810], "oracle_oct1meas": [3605, 20260811]}},
        f, indent=1)
log("L1'' main done -> l1pp_main_gen.npz")
