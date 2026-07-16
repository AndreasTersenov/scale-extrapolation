"""Build the C1 sandbox data assets (deterministic; data_cache is gitignored so THIS
script + sandbox/recipe.py are the committed recipe).

- data_cache/tiles_sandbox.npz['sandbox']: 386 tiles = 322 training tiles (the
  Stage-A stream, seed 20260718, read from $SCRATCH) + 64 held-out tiles from the
  fresh stream SEED_HELDOUT=20260719.
- data_cache/running_couplings_sandbox.json: arm-B dial = TRUTH estimand values per
  octave from results_p2/sandbox_truth.json.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sandbox import recipe
from sandbox.lognormal import lognormal_map, sample_grf

SEED_HELDOUT = 20260719

train = np.load(os.path.expanduser(
    "~/links/scratch/scale-extrap-p2/sandbox_train_f32.npy"))
assert train.shape == (322, 128, 128), train.shape
rng = np.random.default_rng(SEED_HELDOUT)
sp = recipe.spec()
held = np.stack([lognormal_map(sample_grf(sp, rng), recipe.SIGMA_G)
                 for _ in range(64)]).astype(np.float32)
tiles = np.concatenate([train, held], axis=0)
np.savez(os.path.join(REPO, "data_cache", "tiles_sandbox.npz"), sandbox=tiles)

truth = json.load(open(os.path.join(REPO, "results_p2", "sandbox_truth.json")))["truth"]
coords = {"sandbox": {j: [truth[j]["var_slope"], truth[j]["kurtosis"]]
                      for j in ["1", "2", "3", "4"]}}
with open(os.path.join(REPO, "data_cache",
                       "running_couplings_sandbox.json"), "w") as f:
    json.dump(coords, f, indent=1)
print("wrote tiles_sandbox.npz", tiles.shape, "and running_couplings_sandbox.json")
