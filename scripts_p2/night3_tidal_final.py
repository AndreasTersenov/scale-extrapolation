#!/usr/bin/env python
"""NIGHT-ORDERS-3 TIDAL phase 3 (JAX): the three final-config streams from
the run's fitted filter, through the tidal (eigenframe-conditioned) F2
sampler. stage3_b_final.py mirrored; prereg log/2026-08-05-prereg-night3.md.
Usage: night3_tidal_final.py <tag> <ckpt_path> <train_oct_lo>. Reads
results_p2/stage3_<tag>_filter.npz (written by the UNCHANGED stage3_b_fit.py
from the white stream). Writes results_p2/stage3_<tag>_final.npz."""
from __future__ import annotations

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
from l1p_lib import make_colored_base, std_from
from wfm.dataset import field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
KEYS = {"tidal21": ((5511, 20260889), (5512, 20260890), (5513, 20260891)),
        "tidal22": ((5611, 20260893), (5612, 20260894), (5613, 20260895))}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tag, ckpt_path, lo = sys.argv[1], sys.argv[2], int(sys.argv[3])
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
F = np.load(os.path.join(RES, f"stage3_{tag}_filter.npz"))
cb = make_colored_base(np.asarray(F["filt"]), np.asarray(F["z_grid"]),
                       np.asarray(F["x_grid"]))
out = {}
for i, (key_i, gseed) in enumerate(KEYS[tag], 1):
    gen = gen_groupavg_tidal(model.apply, params, coarse4, 4,
                             jax.random.PRNGKey(key_i), std, fstd,
                             np.random.default_rng(gseed), base_by_j={1: cb})
    out[f"final{i}"] = np.asarray(gen[..., 0], np.float32)
    log(f"{tag} final{i}: done")
np.savez(os.path.join(RES, f"stage3_{tag}_final.npz"), **out)
log(f"-> stage3_{tag}_final.npz")
