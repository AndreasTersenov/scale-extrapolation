#!/usr/bin/env python
"""Stage-3 (b) phase 3 (JAX): the three final-config streams from the
run's fitted filter. Usage: stage3_b_final.py <tag> <ckpt_path>
<train_oct_lo>. Writes results_p2/stage3_<tag>_final.npz."""
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

from l1p_lib import gen_groupavg_base, make_colored_base, std_from
from wfm.dataset import field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
KEYS = {"dryrun": ((4911, 20260842), (4912, 20260843),
                   (4913, 20260844)),
        "blind": ((4711, 20260831), (4712, 20260832), (4713, 20260833)),
        "seed1": ((4111, 20260821), (4112, 20260822), (4113, 20260823)),
        "seed2": ((4211, 20260825), (4212, 20260826), (4213, 20260827)),
        "oracle": ((5211, 20260871), (5212, 20260872), (5213, 20260873)),
        "aug11": ((5311, 20260881), (5312, 20260882), (5313, 20260883)),
        "aug12": ((5411, 20260885), (5412, 20260886), (5413, 20260887)),
        "oraclefix": ((5811, 20261301), (5812, 20261302), (5813, 20261303)),
        "d1_oracle": ((5911, 20261401), (5912, 20261402), (5913, 20261403)),
        "d1_seed1": ((6011, 20261411), (6012, 20261412), (6013, 20261413)),
        "d1_seed2": ((6111, 20261421), (6112, 20261422), (6113, 20261423))}


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
    params = pickle.load(fh)["params"]
F = np.load(os.path.join(RES, f"stage3_{tag}_filter.npz"))
cb = make_colored_base(np.asarray(F["filt"]), np.asarray(F["z_grid"]),
                       np.asarray(F["x_grid"]))
out = {}
for i, (key_i, gseed) in enumerate(KEYS[tag], 1):
    gen = gen_groupavg_base(model.apply, params, coarse4, 4,
                            jax.random.PRNGKey(key_i), std,
                            np.random.default_rng(gseed), base_by_j={1: cb})
    out[f"final{i}"] = np.asarray(gen[..., 0], np.float32)
    log(f"{tag} final{i}: done")
np.savez(os.path.join(RES, f"stage3_{tag}_final.npz"), **out)
log(f"-> stage3_{tag}_final.npz")
