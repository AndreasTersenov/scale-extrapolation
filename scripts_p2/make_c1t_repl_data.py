"""R18 replication data: 64 FRESH sandbox test tiles from a new seed stream
(SEED_REPL=20260720 — disjoint from train 20260718 and the original held-out
20260719), exact law, same recipe. data_cache is gitignored; THIS script +
sandbox/recipe.py are the committed recipe (the make_c1_data.py convention)."""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sandbox import recipe
from sandbox.lognormal import lognormal_map, sample_grf

SEED_REPL = 20260720

rng = np.random.default_rng(SEED_REPL)
sp = recipe.spec()
fresh = np.stack([lognormal_map(sample_grf(sp, rng), recipe.SIGMA_G)
                  for _ in range(64)]).astype(np.float32)
np.savez(os.path.join(REPO, "data_cache", "tiles_sandbox_repl.npz"), sandbox=fresh)
print("wrote data_cache/tiles_sandbox_repl.npz", fresh.shape,
      f"(seed {SEED_REPL})")
