#!/usr/bin/env python
"""NIGHT-ORDERS-3 TIDAL phase 1 (JAX): the white-base T-measurement stream +
the determinism gate, through the tidal (eigenframe-conditioned) F2 sampler.
stage3_b_white.py mirrored; prereg log/2026-08-05-prereg-night3.md. Usage:
night3_tidal_white.py <tag> <ckpt_path> <train_oct_lo> (tag in {tidal21,
tidal22}; TIDAL recipe lo=2). The ckpt pickle must carry feat_std_by_j (see
run_tidal_arm.py). Writes results_p2/stage3_<tag>_white.npz (+ gate record in
its json) so the UNCHANGED stage3_b_fit.py runs between white and final."""
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

from arms_p2.tidal.sample import feat_std_ladder, gen_groupavg_tidal
from l1p_lib import std_from
from wfm.dataset import field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
KEYS = {"tidal21": (5501, 20260888), "tidal22": (5601, 20260892)}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tag, ckpt_path, lo = sys.argv[1], sys.argv[2], int(sys.argv[3])
key_i, gseed = KEYS[tag]
gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
std = std_from(gtrain, list(range(lo, 5)))
coarse4 = field_to_octaves(gtest, [4])[0][4][1]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
with open(ckpt_path, "rb") as fh:
    ck = pickle.load(fh)
params = ck["params"]
fstd = feat_std_ladder(ck["feat_std_by_j"])

# determinism gate: same key twice -> identical maps
a = gen_groupavg_tidal(model.apply, params, coarse4[:4], 4,
                       jax.random.PRNGKey(999), std, fstd,
                       np.random.default_rng(7))
b = gen_groupavg_tidal(model.apply, params, coarse4[:4], 4,
                       jax.random.PRNGKey(999), std, fstd,
                       np.random.default_rng(7))
det = float(jax.numpy.max(jax.numpy.abs(a - b)))
assert det == 0.0, f"determinism gate FAILED: {det}"
log(f"determinism gate: exact ({det})")

gen = gen_groupavg_tidal(model.apply, params, coarse4, 4,
                         jax.random.PRNGKey(key_i), std, fstd,
                         np.random.default_rng(gseed))
np.savez(os.path.join(RES, f"stage3_{tag}_white.npz"),
         white=np.asarray(gen[..., 0], np.float32))
with open(os.path.join(RES, f"stage3_{tag}_white.json"), "w") as f:
    json.dump({"determinism_gate_maxabs": det, "ckpt": ckpt_path,
               "train_oct_lo": lo, "key": key_i, "grng": gseed,
               "sampler": "tidal", "sigma_h": ck.get("sigma_h")}, f, indent=1)
log(f"white stream done -> stage3_{tag}_white.npz")
