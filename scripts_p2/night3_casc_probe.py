#!/usr/bin/env python
"""NIGHT-3 CASC probe (JAX; prereg 2026-08-05-prereg-night3 §CASC).
INFERENCE-ONLY, zero training: the committed trained-leg production chain
(l1pp_main_sample.py verbatim — gowerstreet A@16000 ckpt, l1pp_filter.npz
filt_adj + z/x tables, std_from(gtrain, [2,3,4]), gen_groupavg_base) with
the oct-1 copula base's white Gaussian INPUT replaced by casc_seed
(casc_base.py, lam documented there). Gates before the streams: committed
replay (white chain vs f2_test_gen.npz, l1pp_main's G2 bars) + determinism
on the CASC chain (same key twice -> exact). KEYS: prereg registry casc
row. Outputs results_p2/night3_casc_gen.npz + night3_casc_sample.json."""
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

from casc_base import LAM, casc_colored_base
from l1p_lib import gen_groupavg_base, std_from
from wfm.dataset import field_to_octaves
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")
G2 = {"corr_min": 0.99, "ratio_tol": 5e-3, "maxabs_ceil": 5e-2}
CKPT = os.path.join("data_cache", "ckpt_c1t_gowerstreet",
                    "armA_gowerstreet_s16000.pkl")
FILTER = os.path.join("results_p2", "l1pp_filter.npz")
STREAMS = (("casc1", 5711, 20260896), ("casc2", 5712, 20260897),
           ("casc3", 5713, 20260898))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
std = std_from(gtrain, [2, 3, 4])
coarse4 = field_to_octaves(gtest, [4])[0][4][1]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
with open(os.path.join(REPO, CKPT), "rb") as fh:
    params = pickle.load(fh)["params"]

# replay gate: l1pp_main_sample.py verbatim — the committed white chain
# must reproduce f2_test_gen.npz exactly enough (same ckpt/std/plumbing)
committed = np.asarray(
    np.load(os.path.join(RES, "f2_test_gen.npz"))["F2_gowA_e2e"], np.float64)
rep = gen_groupavg_base(model.apply, params, coarse4, 4,
                        jax.random.PRNGKey(2400), std,
                        np.random.default_rng(20260728 + 2))
maps = np.asarray(rep[..., 0], np.float64)
corr = np.array([np.corrcoef(a.ravel(), b.ravel())[0, 1]
                 for a, b in zip(maps, committed)])
ratio = np.array([a.std() / b.std() for a, b in zip(maps, committed)])
rel = float(np.max(np.abs(maps - committed))) / float(committed.std())
gate = {"corr_min": float(corr.min()), "ratio_mean": float(ratio.mean()),
        "rel_maxabs": rel}
log(f"gow replay gate: {gate}")
assert corr.min() >= G2["corr_min"] and \
    abs(float(ratio.mean()) - 1) <= G2["ratio_tol"] and \
    rel <= G2["maxabs_ceil"], f"gow replay gate FAILED: {gate}"
log("gow replay gate: PASS")

F = np.load(os.path.join(REPO, FILTER))
cb = casc_colored_base(np.asarray(F["filt_adj"]), np.asarray(F["z_grid"]),
                       np.asarray(F["x_grid"]), LAM)

# determinism gate on the CASC chain (stage3_b_white pattern)
a = gen_groupavg_base(model.apply, params, coarse4[:4], 4,
                      jax.random.PRNGKey(999), std,
                      np.random.default_rng(7), base_by_j={1: cb})
b = gen_groupavg_base(model.apply, params, coarse4[:4], 4,
                      jax.random.PRNGKey(999), std,
                      np.random.default_rng(7), base_by_j={1: cb})
det = float(jax.numpy.max(jax.numpy.abs(a - b)))
assert det == 0.0, f"determinism gate FAILED: {det}"
log(f"determinism gate: exact ({det})")

out = {}
for name, key_i, gseed in STREAMS:
    gen = gen_groupavg_base(model.apply, params, coarse4, 4,
                            jax.random.PRNGKey(key_i), std,
                            np.random.default_rng(gseed), base_by_j={1: cb})
    out[name] = np.asarray(gen[..., 0], np.float32)
    log(f"gow {name}: done")

np.savez(os.path.join(RES, "night3_casc_gen.npz"), **out)
with open(os.path.join(RES, "night3_casc_sample.json"), "w") as f:
    json.dump({"replay_gate": gate, "determinism_gate_maxabs": det,
               "ckpt": CKPT, "filter": FILTER, "filter_key": "filt_adj",
               "lam": LAM,
               "streams": {n: [k, g] for n, k, g in STREAMS}}, f, indent=1)
log("CASC probe done -> night3_casc_gen.npz")
