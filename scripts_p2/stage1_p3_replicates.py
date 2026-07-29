#!/usr/bin/env python
"""N1 probe 5 (prereg log/2026-07-29-prereg-n1-mechanism.md): trained-leg
reference replication, zero training.

Three F2 group-averaged e2e streams from the committed c1t gowerstreet
checkpoint (A@16000), Stage-D-era substrate conventions verbatim from
f2_sample.py (train octaves {2,3,4}, std_from, coarse4 from the 32 test
tiles):
  REPLAY  : key 2400, grng 20260728+2 — must reproduce the committed
            F2_gowA_e2e under the corrected-G2 criterion (corr_min >= 0.99,
            amplitude-ratio mean within 5e-3 of 1, rel max-abs <= 5e-2);
            asserted.
  REP1    : key 2501, grng 20260731 (fresh stream).
  REP2    : key 2502, grng 20260732 (fresh stream).

Outputs results_p2/stage1_p3_replicates.npz (rep1/rep2) +
results_p2/stage1_p3_replicates.json (replay-gate diagnostics, seeds).
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
from f2_group import D4_ELEMENTS, assemble_group_assigned
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
GOW_PICK_A = 16000
STREAMS = {"replay": (2400, 20260728 + 2),
           "rep1": (2501, 20260731),
           "rep2": (2502, 20260732)}
G2_CORR_MIN, G2_RATIO_TOL, G2_MAXABS_CEIL = 0.99, 5e-3, 5e-2


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def std_from(train, octs):
    # verbatim from f2_sample.py
    _, std_by_j = field_to_octaves(d4_augment(train), octs)
    js = np.array(sorted(std_by_j))
    a_, b_ = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
    std = dict(std_by_j)
    for j in range(1, min(octs)):
        std[j] = float(np.exp(a_ * j + b_))
    return std


def gen_groupavg(apply_fn, params, coarse, j_start, key, std, grng):
    # verbatim from f2_sample.py (arm A: no cond)
    import jax.numpy as jnp  # noqa: F401

    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)

        def model_fn(c_g, _k=k, _s=std[j]):
            det_n = sample_tbase(apply_fn, params, _k, c_g, 3, n_steps=80,
                                 cond_vec=None)
            return det_n * _s

        assign = grng.integers(0, len(D4_ELEMENTS), coarse.shape[0])
        coarse = assemble_group_assigned(coarse, model_fn, assign)
    return coarse


gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
gstd = std_from(gtrain, [2, 3, 4])
gpools, _ = field_to_octaves(gtest, [4])
gcoarse4 = gpools[4][1]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
with open(os.path.join(REPO, "data_cache", "ckpt_c1t_gowerstreet",
                       f"armA_gowerstreet_s{GOW_PICK_A}.pkl"), "rb") as fh:
    params = pickle.load(fh)["params"]

committed = np.asarray(
    np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_gowA_e2e"], np.float64)
out_npz, out_json = {}, {"streams": {k: v for k, v in STREAMS.items()},
                         "replay_gate": {}}
for name, (key_i, gseed) in STREAMS.items():
    gen = gen_groupavg(model.apply, params, gcoarse4, 4,
                       jax.random.PRNGKey(key_i), gstd,
                       np.random.default_rng(gseed))
    maps = np.asarray(gen[..., 0], np.float64)
    if name == "replay":
        corr = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1]
                         for a, b in zip(maps, committed)])
        ratio = np.array([a.std() / b.std() for a, b in zip(maps, committed)])
        rel = float(np.max(np.abs(maps - committed))) / float(committed.std())
        gate = {"corr_min": float(corr.min()), "ratio_mean": float(ratio.mean()),
                "rel_maxabs": rel}
        out_json["replay_gate"] = gate
        log(f"replay gate: corr_min={corr.min():.8f} "
            f"ratio_mean={ratio.mean():.6f} rel_maxabs={rel:.3e}")
        assert corr.min() >= G2_CORR_MIN, f"replay gate FAILED (corr): {gate}"
        assert abs(float(ratio.mean()) - 1) <= G2_RATIO_TOL, \
            f"replay gate FAILED (ratio): {gate}"
        assert rel <= G2_MAXABS_CEIL, f"replay gate FAILED (ceiling): {gate}"
        log("replay gate: PASS")
    else:
        out_npz[name] = maps.astype(np.float32)
        log(f"{name}: fresh F2 stream done")

np.savez(os.path.join(RES, "stage1_p3_replicates.npz"), **out_npz)
with open(os.path.join(RES, "stage1_p3_replicates.json"), "w") as f:
    json.dump(out_json, f, indent=1)
log("replicates done -> stage1_p3_replicates.npz + .json")
