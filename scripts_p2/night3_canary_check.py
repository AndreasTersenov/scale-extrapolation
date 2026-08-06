#!/usr/bin/env python
"""NIGHT-3 canary dispersion check (JAX env). Usage:
night3_canary_check.py <ckpt> <data_npz> <field> <octave> <out_json>
Samples K=4 detail draws per VALIDATION field at the given octave from
the checkpoint and compares the pooled generated detail dispersion to
the real pooled detail dispersion (the phase-1 CFM under-dispersion
disease). KILL rule (standing): deficit > 40% (ratio < 0.6)."""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax

from arms_p2.c1t.flow import sample_tbase
from wfm import haar
from wfm.dataset import normalize_tiles
from wfm.model import ConditionalUNet

ckpt_path, data_npz, field, octave, out_json = sys.argv[1:6]
octave = int(octave)

tiles = np.load(data_npz)[field].astype(np.float32)
heldout = tiles[-64:]
val_n = normalize_tiles(heldout[:32])
det_real, coarse = haar.octave_pair(val_n, octave)
with open(ckpt_path, "rb") as fh:
    params = pickle.load(fh)["params"]
model = ConditionalUNet(out_channels=3, channels=(32, 64, 128),
                        bottleneck=256, cond_dim=0, cond_mode="film",
                        variance_head=False)
real_std = float(np.asarray(det_real).std())
key = jax.random.PRNGKey(424242)
gens = []
for k in range(4):
    key, kk = jax.random.split(key)
    det = sample_tbase(model.apply, params, kk, coarse, 3, n_steps=80,
                       cond_vec=None)
    gens.append(np.asarray(det))
# generated details are in STANDARDIZED space; real_std normalizes the
# real pool the same way the trainer's per-octave std does:
gen_std = float(np.concatenate(gens).std())
ratio = gen_std  # standardized-space gen dispersion vs 1.0 by construction
# compare vs the real pool standardized by its own std -> 1.0 reference
out = {"ckpt": ckpt_path, "octave": octave,
       "gen_std_standardized": gen_std, "real_std_raw": real_std,
       "dispersion_ratio": ratio, "kill": bool(ratio < 0.6)}
with open(out_json, "w") as f:
    json.dump(out, f, indent=1)
print(f"canary dispersion ratio {ratio:.3f} -> "
      f"{'KILL' if out['kill'] else 'PASS'}")
