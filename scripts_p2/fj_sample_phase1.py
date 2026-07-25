#!/usr/bin/env python
"""Arms F/J, SAMPLE phase 1 (R32 GO; inference only, zero training).

Generates on VALIDATION tiles (never test — cage discipline):
 1. sandbox arm A: e2e from ALL dense checkpoints 500..20000 by 500 (J's
    T_coef_val curve) — the committed pick 7500 included;
    arm B: 2500-stride (descriptive) incl. its committed pick 2500.
 2. sandbox val head-conditional detail channel means per octave (2,3,4) at
    the committed picks (F's hc offsets; model units, wfm channel order) +
    4-fold means for the stability disclosure.
 3. gowerstreet arm A: val e2e at its committed pick 16000 (the corrected
    real-field echo's offset estimation).
 4. Equivalence gate: generate_recursive_corrected with ZERO offsets must
    reproduce generate_recursive_tbase exactly (same key) — asserted here.

Outputs: results_p2/fj_val_gen.npz + results_p2/fj_val_hc.json.
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
from wfm import haar
from wfm.dataset import d4_augment, field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

COORD_NORM = np.array([1.5, 13.0])
PICKS = {"A": 7500, "B": 2500}
GOW_PICK_A = 16000


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


out_npz, out_json = {}, {"hc_channel_means": {}, "equivalence": None}

# ---------------- sandbox ----------------
tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
train, val = tiles[:-64].astype(np.float32), tiles[-64:-32].astype(np.float32)
val_n = normalize_tiles(val)
std = std_from(train, [2, 3, 4])
pools, _ = field_to_octaves(val, [4])
coarse4 = pools[4][1]
coords_raw = json.load(open(os.path.join(
    REPO, "data_cache", "running_couplings_sandbox.json")))["sandbox"]
coords = {int(j): (np.asarray(v) / COORD_NORM) for j, v in coords_raw.items()}
B = coarse4.shape[0]

for arm, steps in (("A", list(range(500, 20001, 500))),
                   ("B", list(range(2500, 20001, 2500)))):
    model = model_for(arm)
    cond_fn = None
    if arm == "B":
        def cond_fn(j):
            return jnp.broadcast_to(jnp.asarray(coords[j], jnp.float32), (B, 2))
    for si in steps:
        t0 = time.time()
        params = load_params("ckpt_c1t_sandbox", arm, "sandbox", si)
        gen = generate_recursive_tbase(model.apply, params, coarse4, 4,
                                       jax.random.PRNGKey(900 + si), std,
                                       cond_fn=cond_fn, n_steps=80)
        out_npz[f"arm{arm}_s{si}"] = np.asarray(gen[..., 0])
        if arm == "A" and si == PICKS["A"]:
            gen2 = generate_recursive_corrected(
                model.apply, params, coarse4, 4, jax.random.PRNGKey(900 + si),
                std, offsets_wfm={}, cond_fn=cond_fn, n_steps=80)
            diff = float(jnp.max(jnp.abs(gen - gen2)))
            out_json["equivalence"] = diff
            assert diff == 0.0, f"equivalence gate FAILED: {diff}"
            log(f"equivalence gate: corrected(0)==reference exactly ({diff})")
        log(f"sandbox {arm} @{si}: val e2e [{time.time()-t0:.0f}s]")

# hc channel means at the committed picks (model units, wfm order)
for arm in ("A", "B"):
    model = model_for(arm)
    params = load_params("ckpt_c1t_sandbox", arm, "sandbox", PICKS[arm])
    key = jax.random.PRNGKey(1100)
    hc = {}
    for j in (2, 3, 4):
        det_real, coarse = haar.octave_pair(val_n, j)
        cv = None if arm == "A" else jnp.broadcast_to(
            jnp.asarray(coords[j], jnp.float32), (coarse.shape[0], 2))
        key, k = jax.random.split(key)
        det = np.asarray(sample_tbase(model.apply, params, k, coarse, 3,
                                      n_steps=80, cond_vec=cv))
        means = det.mean(axis=(0, 1, 2)).tolist()
        folds = [det[i::4].mean(axis=(0, 1, 2)).tolist() for i in range(4)]
        hc[str(j)] = {"mean": means, "folds": folds}
    out_json["hc_channel_means"][f"sandbox_{arm}@{PICKS[arm]}"] = hc
    log(f"sandbox {arm} hc channel means @ {PICKS[arm]} done")

# ---------------- gowerstreet (arm A echo) ----------------
gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gval = gtiles[:-64], gtiles[-64:-32]
gstd = std_from(gtrain, [2, 3, 4])
gpools, _ = field_to_octaves(gval, [4])
gcoarse4 = gpools[4][1]
model = model_for("A")
params = load_params("ckpt_c1t_gowerstreet", "A", "gowerstreet", GOW_PICK_A)
gen = generate_recursive_tbase(model.apply, params, gcoarse4, 4,
                               jax.random.PRNGKey(1200), gstd, n_steps=80)
out_npz[f"gowA_s{GOW_PICK_A}"] = np.asarray(gen[..., 0])
log(f"gowerstreet A @{GOW_PICK_A}: val e2e done")

np.savez(os.path.join(REPO, "results_p2", "fj_val_gen.npz"), **out_npz)
with open(os.path.join(REPO, "results_p2", "fj_val_hc.json"), "w") as f:
    json.dump(out_json, f, indent=1)
log("SAMPLE phase 1 done -> fj_val_gen.npz + fj_val_hc.json")
