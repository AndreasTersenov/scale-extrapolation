#!/usr/bin/env python
"""L1' SANDBOX CANARY sample (pre-statement 67d7304; R39). Zero training.

Two streams from the committed sandbox A@7500 checkpoint:
  replay : WHITE base, key 2200 / grng 20260728 (the committed F2_A_e2e
           stream) — identity gate under the replay criterion, ASSERTED.
  canary : colored oct-1 base (sandbox oct2rescaled filter), key 3301 /
           grng 20260801 — the must-not-regress leg.

Outputs results_p2/l1p_canary_gen.npz + l1p_canary_sample.json.
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

from l1p_lib import gen_groupavg_base, make_colored_base, std_from
from wfm.dataset import field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
G2 = {"corr_min": 0.99, "ratio_tol": 5e-3, "maxabs_ceil": 5e-2}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tiles = np.load(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"))["sandbox"]
train, test = tiles[:-64].astype(np.float32), tiles[-32:].astype(np.float32)
std = std_from(train, [2, 3, 4])
pools, _ = field_to_octaves(test, [4])
coarse4 = pools[4][1]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
with open(os.path.join(REPO, "data_cache", "ckpt_c1t_sandbox",
                       "armA_sandbox_s7500.pkl"), "rb") as fh:
    params = pickle.load(fh)["params"]

out_json = {"streams": {"replay": [2200, 20260728], "canary": [3301, 20260801]},
            "filter": "l1p sandbox oct2rescaled", "replay_gate": {}}

# replay gate (white base reproduces the committed F2_A_e2e stream)
committed = np.asarray(
    np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_A_e2e"], np.float64)
rep = gen_groupavg_base(model.apply, params, coarse4, 4,
                        jax.random.PRNGKey(2200), std,
                        np.random.default_rng(20260728))
maps = np.asarray(rep[..., 0], np.float64)
corr = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1]
                 for a, b in zip(maps, committed)])
ratio = np.array([a.std() / b.std() for a, b in zip(maps, committed)])
rel = float(np.max(np.abs(maps - committed))) / float(committed.std())
gate = {"corr_min": float(corr.min()), "ratio_mean": float(ratio.mean()),
        "rel_maxabs": rel}
out_json["replay_gate"] = gate
log(f"replay gate: corr_min={corr.min():.8f} ratio_mean={ratio.mean():.6f} "
    f"rel={rel:.3e}")
assert corr.min() >= G2["corr_min"] and \
    abs(float(ratio.mean()) - 1) <= G2["ratio_tol"] and \
    rel <= G2["maxabs_ceil"], f"replay gate FAILED: {gate}"
log("replay gate: PASS")

# canary stream (colored oct-1 base)
F = np.load(os.path.join(RES, "l1p_filter_sandbox.npz")) if os.path.exists(
    os.path.join(RES, "l1p_filter_sandbox.npz")) else np.load(
    os.path.join(RES, "l1_filter_sandbox.npz"))
cb = make_colored_base(np.asarray(F["filt_oct2rescaled"]),
                       np.asarray(F["z_grid"]), np.asarray(F["x_grid"]))
gen = gen_groupavg_base(model.apply, params, coarse4, 4,
                        jax.random.PRNGKey(3301), std,
                        np.random.default_rng(20260801), base_by_j={1: cb})
np.savez(os.path.join(RES, "l1p_canary_gen.npz"),
         gen_canary=np.asarray(gen[..., 0], np.float32))
with open(os.path.join(RES, "l1p_canary_sample.json"), "w") as f:
    json.dump(out_json, f, indent=1)
log("canary stream done -> l1p_canary_gen.npz")
