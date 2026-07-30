#!/usr/bin/env python
"""Decision-package edge stream (prereg 2026-07-30-decision-package):
stage-D A@9000, deconvolved base (per-checkpoint deployment recipe), key
3701/grng 20260816; replay gate (white, 3101/20260729) vs committed
stage0_p3_gen gen_A, asserted. Writes results_p2/l1pp_decision_edge_gen.npz."""
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
std = std_from(gtrain, [3, 4])
coarse4 = field_to_octaves(gtest, [4])[0][4][1]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
with open(os.path.join(REPO, "data_cache", "ckpt_stageD",
                       "armA_gowerstreet_s9000.pkl"), "rb") as fh:
    params = pickle.load(fh)["params"]

committed = np.asarray(np.load(os.path.join(RES, "stage0_p3_gen.npz"),
                               allow_pickle=True)["gen_A"], np.float64)
rep = gen_groupavg_base(model.apply, params, coarse4, 4,
                        jax.random.PRNGKey(3101), std,
                        np.random.default_rng(20260729))
maps = np.asarray(rep[..., 0], np.float64)
corr = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1]
                 for a, b in zip(maps, committed)])
ratio = np.array([a.std() / b.std() for a, b in zip(maps, committed)])
rel = float(np.max(np.abs(maps - committed))) / float(committed.std())
gate = {"corr_min": float(corr.min()), "ratio_mean": float(ratio.mean()),
        "rel_maxabs": rel}
log(f"stage-D replay gate: {gate}")
assert corr.min() >= G2["corr_min"] and \
    abs(float(ratio.mean()) - 1) <= G2["ratio_tol"] and \
    rel <= G2["maxabs_ceil"], f"replay gate FAILED: {gate}"
log("stage-D replay gate: PASS")

F = np.load(os.path.join(RES, "l1pp_decision_edge_filter.npz"))
cb = make_colored_base(np.asarray(F["filt_edge"]),
                       np.asarray(F["z_grid"]), np.asarray(F["x_grid"]))
gen = gen_groupavg_base(model.apply, params, coarse4, 4,
                        jax.random.PRNGKey(3701), std,
                        np.random.default_rng(20260816), base_by_j={1: cb})
np.savez(os.path.join(RES, "l1pp_decision_edge_gen.npz"),
         gen_edge_deconv=np.asarray(gen[..., 0], np.float32))
with open(os.path.join(RES, "l1pp_decision_edge_sample.json"), "w") as f:
    json.dump({"replay_gate": gate, "streams": {
        "replay": [3101, 20260729], "edge_deconv": [3701, 20260816]}},
        f, indent=1)
log("edge deconvolved stream done")
