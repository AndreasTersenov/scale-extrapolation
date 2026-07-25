#!/usr/bin/env python
"""Arm F2 SAMPLE phase (R33; prereg 2026-07-25-f2-groupavg, committed pre-run).

Group-averaged (exact D4-equivariant-in-law) sampling, inference only:
  - sandbox test e2e + hc at the committed picks (A@7500 adjudicating,
    B@2500 descriptive);
  - gowerstreet arm A @16000 test e2e (the corrected real-field echo re-run).
Per field, per octave, g ~ Uniform(D4) (seeded); details sampled in the
g-frame, assembled there, mapped back (f2_group.assemble_in_frame).
Runtime gate: all-identity assignment must reproduce the committed sampler
EXACTLY at A@7500 (asserted). Outputs results_p2/f2_test_gen.npz +
results_p2/f2_test_hc.json.
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
from f2_group import D4_ELEMENTS, assemble_group_assigned, assemble_in_frame
from run_c1t_arms import pooled_wc, score_full
from wfm import haar
from wfm.dataset import d4_augment, field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
COORD_NORM = np.array([1.5, 13.0])
PICKS = {"A": 7500, "B": 2500}
GOW_PICK_A = 16000
SEED_G = 20260728


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


def gen_groupavg(apply_fn, params, coarse, j_start, key, std, grng,
                 cond_fn=None, identity_only=False):
    for j in range(j_start, 0, -1):
        key, k = jax.random.split(key)

        def model_fn(c_g, _k=k, _j=j, _s=std[j]):
            # cond sized to the per-group SUBSET (the full-batch closure broke
            # arm B's first run — job 17424070, fixed here): the vector is the
            # same for every field, so broadcasting to c_g's batch is exact.
            cond = None if cond_fn is None else jnp.broadcast_to(
                cond_fn(_j)[:1], (c_g.shape[0], cond_fn(_j).shape[1]))
            det_n = sample_tbase(apply_fn, params, _k, c_g, 3, n_steps=80,
                                 cond_vec=cond)
            return det_n * _s

        if identity_only:
            assign = np.zeros(coarse.shape[0], int)
        else:
            assign = grng.integers(0, len(D4_ELEMENTS), coarse.shape[0])
        coarse = assemble_group_assigned(coarse, model_fn, assign)
    return coarse


out_npz, out_json = {}, {"hc": {}, "identity_gate": None}

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

# runtime identity gate at A@7500
model = model_for("A")
params = load_params("ckpt_c1t_sandbox", "A", "sandbox", PICKS["A"])
ref = generate_recursive_tbase(model.apply, params, coarse4, 4,
                               jax.random.PRNGKey(2100), std, n_steps=80)
ident = gen_groupavg(model.apply, params, coarse4, 4, jax.random.PRNGKey(2100),
                     std, np.random.default_rng(0), identity_only=True)
diff = float(jnp.max(jnp.abs(ref - ident)))
out_json["identity_gate"] = diff
assert diff == 0.0, f"identity gate FAILED: {diff}"
log(f"identity gate: exact ({diff})")

for arm in ("A", "B"):
    model = model_for(arm)
    params = load_params("ckpt_c1t_sandbox", arm, "sandbox", PICKS[arm])
    cond_fn = None
    if arm == "B":
        def cond_fn(j):
            return jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32), (B, 2))
    grng = np.random.default_rng(SEED_G + (0 if arm == "A" else 1))
    gen = gen_groupavg(model.apply, params, coarse4, 4,
                       jax.random.PRNGKey(2200), std, grng, cond_fn=cond_fn)
    out_npz[f"F2_{arm}_e2e"] = np.asarray(gen[..., 0])
    # group-averaged hc: assemble in frame, re-analyze in the original frame
    key = jax.random.PRNGKey(2300)
    hc = {}
    for j in (2, 3, 4):
        det_real, coarse = haar.octave_pair(test_n, j)
        key, k = jax.random.split(key)

        def model_fn(c_g, _k=k, _j=j, _arm=arm):
            cv = None if _arm == "A" else jnp.broadcast_to(
                jnp.asarray(coords[_j], jnp.float32), (c_g.shape[0], 2))
            return sample_tbase(model.apply, params, _k, c_g, 3, n_steps=80,
                                cond_vec=cv)

        assign = grng.integers(0, len(D4_ELEMENTS), coarse.shape[0])
        f = assemble_group_assigned(coarse, model_fn, assign)
        _, dets = haar.dwt2(f)
        det = jnp.concatenate(dets, axis=-1)
        hc[str(j)] = score_full(pooled_wc(det, coarse), n_boot=200, seed=0)
    out_json["hc"][f"F2_{arm}"] = hc
    log(f"F2 sandbox {arm}@{PICKS[arm]}: group-averaged e2e+hc done")

# gowerstreet echo
gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
gstd = std_from(gtrain, [2, 3, 4])
gpools, _ = field_to_octaves(gtest, [4])
gcoarse4 = gpools[4][1]
model = model_for("A")
params = load_params("ckpt_c1t_gowerstreet", "A", "gowerstreet", GOW_PICK_A)
gen = gen_groupavg(model.apply, params, gcoarse4, 4, jax.random.PRNGKey(2400),
                   gstd, np.random.default_rng(SEED_G + 2))
out_npz["F2_gowA_e2e"] = np.asarray(gen[..., 0])
log(f"F2 gowerstreet A@{GOW_PICK_A}: group-averaged e2e done")

np.savez(os.path.join(RES, "f2_test_gen.npz"), **out_npz)
with open(os.path.join(RES, "f2_test_hc.json"), "w") as f:
    json.dump(out_json, f, indent=1)
log("F2 SAMPLE done -> f2_test_gen.npz + f2_test_hc.json")
