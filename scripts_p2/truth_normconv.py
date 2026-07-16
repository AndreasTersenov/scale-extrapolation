"""Normalized-convention truth: the exact estimand values under PER-TILE
normalization (the wfm pipeline's convention for training, generation and scoring).

Measured pre-readout (log/2026-07-16-c1-amendment-normconv.md): per-tile
normalization shifts the estimand by -3..-6% (var_slope) and -12..-19% (kurtosis)
relative to raw-field truth — comparable to the C1 bars, so generated fields (which
live in the normalized convention) must be adjudicated against truth computed under
the IDENTICAL transformation. Exactness is preserved: the ensembles are exact
samples; normalization is a deterministic per-field map.
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sandbox import recipe
from sandbox.truth_stats import truth_couplings

SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")
OUT = os.path.join(REPO, "results_p2", "sandbox_truth_normconv.json")

ens = np.load(os.path.join(SCRATCH, "sandbox_ens_f32.npy"), mmap_mode="r")
P, R = ens.shape[:2]
fields = []
for i in range(P):
    for r in range(R):
        f = np.asarray(ens[i, r], dtype=np.float64)
        fields.append((f - f.mean()) / f.std())
groups = np.repeat(np.arange(P), R)
print(f"[{time.strftime('%H:%M:%S')}] normconv truth on {len(fields)} fields",
      flush=True)
t0 = time.time()
truth = truth_couplings(fields, groups, recipe.OCTAVES, n_blocks=16)
with open(OUT, "w") as f:
    json.dump({"truth": {str(j): truth[j] for j in truth},
               "meta": {"convention": "per-tile zero-mean unit-variance",
                        "n_fields": len(fields), "n_blocks": 16}}, f, indent=1)
print(f"[{time.strftime('%H:%M:%S')}] done in {time.time()-t0:.0f}s -> {OUT}",
      flush=True)
