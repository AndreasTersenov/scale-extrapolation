"""Normalized-convention exact truth for the R10 condition-1 tail readout:
per-octave 99.9th-percentile absolute standardized coefficient (tail_q999) on the
sandbox conditional ensemble under per-tile normalization, with 16-block parent
batch-means SEs — the same convention and error protocol as
sandbox_truth_normconv.json. Descriptive instrument only (never adjudicating).
"""
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sandbox import recipe
from sandbox.truth_stats import _pool, tail_q999

SCRATCH = os.path.expanduser("~/links/scratch/scale-extrap-p2")
OUT = os.path.join(REPO, "results_p2", "sandbox_truth_q999.json")

ens = np.load(os.path.join(SCRATCH, "sandbox_ens_f32.npy"), mmap_mode="r")
P, R = ens.shape[:2]
fields = []
for i in range(P):
    for r in range(R):
        f = np.asarray(ens[i, r], dtype=np.float64)
        fields.append((f - f.mean()) / f.std())
groups = np.repeat(np.arange(P), R)
print(f"[{time.strftime('%H:%M:%S')}] normconv q999 truth on {len(fields)} fields",
      flush=True)

t0 = time.time()
parents = np.unique(groups)
blocks = np.array_split(parents, 16)
out = {}
for j in recipe.OCTAVES:
    w, _ = _pool(fields, j)
    point = tail_q999(w)
    del w
    bvals = []
    for blk in blocks:
        sel = np.isin(groups, blk)
        wb, _ = _pool([f for f, s in zip(fields, sel) if s], j)
        bvals.append(tail_q999(wb))
    se = float(np.nanstd(np.asarray(bvals), ddof=1) / np.sqrt(len(blocks)))
    out[str(j)] = {"q999": point, "q999_se": se, "n_blocks": len(blocks)}
    print(f"[{time.strftime('%H:%M:%S')}] oct {j}: q999={point:.3f} +- {se:.4f}",
          flush=True)

with open(OUT, "w") as f:
    json.dump({"truth": out,
               "meta": {"convention": "per-tile zero-mean unit-variance",
                        "estimator": "sandbox.truth_stats.tail_q999",
                        "n_fields": len(fields), "n_blocks": 16}}, f, indent=1)
print(f"[{time.strftime('%H:%M:%S')}] done in {time.time()-t0:.0f}s -> {OUT}",
      flush=True)
