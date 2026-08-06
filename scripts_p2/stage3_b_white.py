#!/usr/bin/env python
"""Stage-3 (b) phase 1 (JAX): the white-base T-measurement stream + the
determinism gate. Usage: stage3_b_white.py <tag> <ckpt_path> <train_oct_lo>
(tag in {dryrun, blind}; train octaves {lo..4}; stage-D recipe lo=3).
Writes results_p2/stage3_<tag>_white.npz (+ gate record in its json)."""
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

from l1p_lib import gen_groupavg_base, std_from
from wfm.dataset import field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
KEYS = {"dryrun": (4901, 20260841), "blind": (4701, 20260830),
        "seed1": (4101, 20260820), "seed2": (4201, 20260824),
        "oracle": (5201, 20260870),
        "aug11": (5301, 20260880), "aug12": (5401, 20260884),
        "oraclefix": (5801, 20261300),
        "d1_oracle": (5901, 20261400), "d1_seed1": (6001, 20261410),
        "d1_seed2": (6101, 20261420)}


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
    params = pickle.load(fh)["params"]

# determinism gate: same key twice -> identical maps
a = gen_groupavg_base(model.apply, params, coarse4[:4], 4,
                      jax.random.PRNGKey(999), std,
                      np.random.default_rng(7))
b = gen_groupavg_base(model.apply, params, coarse4[:4], 4,
                      jax.random.PRNGKey(999), std,
                      np.random.default_rng(7))
det = float(jax.numpy.max(jax.numpy.abs(a - b)))
assert det == 0.0, f"determinism gate FAILED: {det}"
log(f"determinism gate: exact ({det})")

gen = gen_groupavg_base(model.apply, params, coarse4, 4,
                        jax.random.PRNGKey(key_i), std,
                        np.random.default_rng(gseed))
np.savez(os.path.join(RES, f"stage3_{tag}_white.npz"),
         white=np.asarray(gen[..., 0], np.float32))
with open(os.path.join(RES, f"stage3_{tag}_white.json"), "w") as f:
    json.dump({"determinism_gate_maxabs": det, "ckpt": ckpt_path,
               "train_oct_lo": lo, "key": key_i, "grng": gseed}, f, indent=1)
log(f"white stream done -> stage3_{tag}_white.npz")
