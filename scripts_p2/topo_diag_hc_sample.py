#!/usr/bin/env python
"""T1 hc-map generation (phase-3b prereg 2026-08-05; the licensed GPU).

hc map, per the pre-stated construction: for each real test tile, at every
octave j = 4..1 sample d_j^gen conditioned on the REAL coarse c_j^real
(F2 group-averaged; oct-1 base = the leg's committed deconvolved filter;
det scaled by std_j), then reconstruct from {c4^real, d4^gen .. d1^gen}.
Legs: blind config (ckpt_stage3_blind @10500, 3 streams, keys 5101-3 /
grngs 20260850-2) + stage-D lineage context (A@9000, 1 stream, 5105 /
20260853). Determinism gate asserted. Writes
results_p2/topo_diag_hc_gen.npz + topo_diag_hc_sample.json."""
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
from f2_group import D4_ELEMENTS, assemble_group_assigned
from l1p_lib import make_colored_base, sample_base_fn, std_from, white_base
from wfm import haar
from wfm.dataset import field_to_octaves, normalize_tiles
from wfm.model import ConditionalUNet

RES = os.path.join(REPO, "results_p2")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


gtiles = np.load(os.path.join(REPO, "data_cache",
                              "tiles_pnull.npz"))["gowerstreet"].astype(np.float32)
gtrain, gtest = gtiles[:-64], gtiles[-32:]
test_n = normalize_tiles(gtest)
std = std_from(gtrain, [3, 4])
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
coarse_by_j = {j: haar.octave_pair(test_n, j)[1] for j in (1, 2, 3, 4)}


def hc_map_stream(params, base_by_j, key, grng):
    """One hc stream: gen details on REAL coarse at every octave, then
    reconstruct from real c4 + the generated details."""
    dets_by_j = {}
    for j in (4, 3, 2, 1):
        key, k = jax.random.split(key)
        bf = base_by_j.get(j, white_base)

        def model_fn(c_g, _k=k, _s=std[j], _bf=bf):
            det_n = sample_base_fn(model.apply, params, _k, c_g, _bf,
                                   n_steps=80)
            return det_n * _s

        assign = grng.integers(0, len(D4_ELEMENTS),
                               coarse_by_j[j].shape[0])
        f = assemble_group_assigned(coarse_by_j[j], model_fn, assign)
        _, dets = haar.dwt2(f)
        dets_by_j[j] = dets
    recon = coarse_by_j[4]
    for j in (4, 3, 2, 1):
        d = dets_by_j[j]
        recon = haar.idwt2(recon, (d[0], d[1], d[2]))
    return np.asarray(recon[..., 0], np.float32)


LEGS = {
    "blind": {"ckpt": os.path.join(REPO, "data_cache", "ckpt_stage3_blind",
                                   "armA_gowerstreet_s10500.pkl"),
              "filter": os.path.join(RES, "stage3_blind_filter.npz"),
              "streams": [(5101, 20260850), (5102, 20260851),
                          (5103, 20260852)]},
    "stageD_context": {"ckpt": os.path.join(REPO, "data_cache",
                                            "ckpt_stageD",
                                            "armA_gowerstreet_s9000.pkl"),
                       "filter": os.path.join(
                           RES, "l1pp_decision_edge_filter.npz"),
                       "streams": [(5105, 20260853)]},
}
out_npz, meta = {}, {"streams": {}, "determinism": None}
for leg, cfg in LEGS.items():
    with open(cfg["ckpt"], "rb") as fh:
        params = pickle.load(fh)["params"]
    F = np.load(cfg["filter"])
    fkey = "filt" if "filt" in F.files else "filt_edge"
    cb = make_colored_base(np.asarray(F[fkey]), np.asarray(F["z_grid"]),
                           np.asarray(F["x_grid"]))
    if meta["determinism"] is None:
        a = hc_map_stream(params, {1: cb}, jax.random.PRNGKey(999),
                          np.random.default_rng(7))
        b = hc_map_stream(params, {1: cb}, jax.random.PRNGKey(999),
                          np.random.default_rng(7))
        det = float(np.max(np.abs(a - b)))
        assert det == 0.0, f"determinism gate FAILED: {det}"
        meta["determinism"] = det
        log(f"determinism gate: exact ({det})")
    for i, (key_i, gseed) in enumerate(cfg["streams"], 1):
        out_npz[f"{leg}_hc{i}"] = hc_map_stream(
            params, {1: cb}, jax.random.PRNGKey(key_i),
            np.random.default_rng(gseed))
        meta["streams"][f"{leg}_hc{i}"] = [key_i, gseed]
        log(f"{leg} hc{i}: done")

np.savez(os.path.join(RES, "topo_diag_hc_gen.npz"), **out_npz)
with open(os.path.join(RES, "topo_diag_hc_sample.json"), "w") as f:
    json.dump(meta, f, indent=1)
log("hc generation done -> topo_diag_hc_gen.npz")
