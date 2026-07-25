#!/usr/bin/env python
"""Arms F/J, SAMPLE phase 2 (R32; inference only). Requires phase-1 outputs
(fj_offsets.json, fj_joint_pick.json).

TEST-side generations:
  F: corrected e2e (offsets_wfm subtracted inside the recursion) + corrected
     hc details (model-unit hc offsets subtracted) at the committed picks —
     sandbox A@7500 (adjudicating), B@2500 (descriptive); gowerstreet A@16000
     (the corrected real-field echo). hc pooled marginal stats computed here
     (JAX side), fields saved for env-side coefficient/parity/e2e scoring.
  J: UNcorrected e2e + hc at the joint-picked checkpoint (arm A).

Outputs: results_p2/fj_test_gen.npz + results_p2/fj_test_hc.json.
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
from arms_p2.c1t.train import generate_recursive_tbase
from fj_recursion import generate_recursive_corrected
from run_c1t_arms import pooled_wc, score_full
from wfm import haar
from wfm.dataset import d4_augment, field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
COORD_NORM = np.array([1.5, 13.0])
PICKS = {"A": 7500, "B": 2500}
GOW_PICK_A = 16000
OFF = json.load(open(os.path.join(RES, "fj_offsets.json")))
JP = json.load(open(os.path.join(RES, "fj_joint_pick.json")))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def std_from(train, octs):
    _, std_by_j = field_to_octaves(d4_augment(train), octs)
    js = np.array(sorted(std_by_j))
    a_, b_ = np.polyfit(js, np.log([std_by_j[j] for j in js]), 1)
    std = dict(std_by_j)
    for j in range(1, min(octs)):
        std[j] = float(np.exp(a_ * j + b_))
    return std


def load_params(ckpt_dir, arm, field, step):
    with open(os.path.join(REPO, "data_cache", ckpt_dir,
                           f"arm{arm}_{field}_s{step}.pkl"), "rb") as fh:
        return pickle.load(fh)["params"]


def model_for(arm):
    return ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                           bottleneck=256, cond_dim=(0 if arm == "A" else 2),
                           cond_mode="film", variance_head=False)


def wfm_offsets(key):
    return {int(j): np.asarray(v, np.float32)
            for j, v in OFF["e2e"][key]["mean_wfm"].items()}


out_npz, out_json = {}, {"hc": {}}

# ---------------- sandbox ----------------
tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
train, test = tiles[:-64].astype(np.float32), tiles[-32:].astype(np.float32)
test_n = normalize_tiles(test)
std = std_from(train, [2, 3, 4])
pools, _ = field_to_octaves(test, [4])
coarse4 = pools[4][1]
coords_raw = json.load(open(os.path.join(
    REPO, "data_cache", "running_couplings_sandbox.json")))["sandbox"]
coords = {int(j): (np.asarray(v) / COORD_NORM) for j, v in coords_raw.items()}
B = coarse4.shape[0]

for arm in ("A", "B"):
    model = model_for(arm)
    params = load_params("ckpt_c1t_sandbox", arm, "sandbox", PICKS[arm])
    cond_fn = None
    if arm == "B":
        def cond_fn(j):
            return jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32), (B, 2))
    offs = wfm_offsets(f"sandbox_{arm}@{PICKS[arm]}")
    gen = generate_recursive_corrected(model.apply, params, coarse4, 4,
                                       jax.random.PRNGKey(1300), std,
                                       offsets_wfm=offs, cond_fn=cond_fn,
                                       n_steps=80)
    out_npz[f"F_{arm}_e2e"] = np.asarray(gen[..., 0])
    # corrected hc: subtract model-unit offsets estimated on val
    hc_off = OFF["hc"][f"sandbox_{arm}@{PICKS[arm]}"]
    key = jax.random.PRNGKey(1400)
    hc = {}
    for j in (2, 3, 4):
        det_real, coarse = haar.octave_pair(test_n, j)
        cv = None if arm == "A" else jnp.broadcast_to(
            jnp.asarray(coords[j], jnp.float32), (coarse.shape[0], 2))
        key, k = jax.random.split(key)
        det = sample_tbase(model.apply, params, k, coarse, 3, n_steps=80,
                           cond_vec=cv)
        det = det - jnp.asarray(hc_off[str(j)]["mean"],
                                jnp.float32)[None, None, None, :]
        hc[str(j)] = score_full(pooled_wc(det, coarse), n_boot=200, seed=0)
    out_json["hc"][f"F_{arm}"] = hc
    log(f"F sandbox {arm}@{PICKS[arm]}: corrected e2e+hc done")

# J: uncorrected test gens at the joint pick (arm A)
jpick = JP["A"]["joint_pick"]
model = model_for("A")
params = load_params("ckpt_c1t_sandbox", "A", "sandbox", jpick)
gen = generate_recursive_tbase(model.apply, params, coarse4, 4,
                               jax.random.PRNGKey(1500), std, n_steps=80)
out_npz["J_A_e2e"] = np.asarray(gen[..., 0])
key = jax.random.PRNGKey(1600)
hc = {}
for j in (2, 3, 4):
    det_real, coarse = haar.octave_pair(test_n, j)
    key, k = jax.random.split(key)
    det = sample_tbase(model.apply, params, k, coarse, 3, n_steps=80,
                       cond_vec=None)
    hc[str(j)] = score_full(pooled_wc(det, coarse), n_boot=200, seed=0)
out_json["hc"]["J_A"] = hc
out_json["J_A_step"] = jpick
log(f"J sandbox A@{jpick}: uncorrected e2e+hc done")

# ---------------- gowerstreet echo (arm A, corrected) ----------------
gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
gstd = std_from(gtrain, [2, 3, 4])
gpools, _ = field_to_octaves(gtest, [4])
gcoarse4 = gpools[4][1]
model = model_for("A")
params = load_params("ckpt_c1t_gowerstreet", "A", "gowerstreet", GOW_PICK_A)
offs = wfm_offsets(f"gowerstreet_A@{GOW_PICK_A}")
gen = generate_recursive_corrected(model.apply, params, gcoarse4, 4,
                                   jax.random.PRNGKey(1700), gstd,
                                   offsets_wfm=offs, n_steps=80)
out_npz["F_gowA_e2e"] = np.asarray(gen[..., 0])
log(f"F gowerstreet A@{GOW_PICK_A}: corrected e2e done")

np.savez(os.path.join(RES, "fj_test_gen.npz"), **out_npz)
with open(os.path.join(RES, "fj_test_hc.json"), "w") as f:
    json.dump(out_json, f, indent=1)
log("SAMPLE phase 2 done -> fj_test_gen.npz + fj_test_hc.json")
